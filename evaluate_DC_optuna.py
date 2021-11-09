
#%%
"""
training pipeline for novel classes
"""
#%%
import os
import torch
import argparse

import optuna
from optuna.trial import TrialState
import logging

import warnings
import pickle
import numpy as np
from tqdm import tqdm
import json
import scipy.stats
import time
from pytorch_utils import train
from few_shot_functions import *
import FSLTask

# Reproducibility
import random
import torch
from torch.backends import cudnn
torch.manual_seed(0)
random.seed(0)
np.random.seed(0)
cudnn.deterministic = True
cudnn.benchmark = False


warnings.simplefilter(action='ignore', category=FutureWarning)
#%%

parser = argparse.ArgumentParser(description="Few Shot Learning")
parser.add_argument("--n_shot", type=int, default=1)
parser.add_argument("--n_ways", type=int, default=5)
parser.add_argument("--n_queries", type=int, default=15)
parser.add_argument("--xpid", type=str, default='test')
parser.add_argument("--n_runs", type=int, default=200)
parser.add_argument("--print_iter", type=int, default=50)
parser.add_argument("--overwrite", action='store_true')
parser.add_argument("--dataset", type=str, default="miniImagenet", choices=["miniImagenet", "CUB", "StanfordDogs", "tieredImagenet", "WRN_MI_wo_relu"])
parser.add_argument("--n_trials", type=int, default=100)
parser.add_argument("--no_optuna", action='store_true', help='do not optimize with optuna, use given hyperparams')


# hyperparams, used only when --no_optuna flag is on
parser.add_argument("--beta", type=float, default=0.5, help='Tukey beta')
parser.add_argument("--m", type=float, default=1)
parser.add_argument("--k", type=int, default=2)
parser.add_argument("--alpha", type=float, default=0)
parser.add_argument("--alpha2", type=float, default=0)
parser.add_argument("--num_sampled", type=int, default=750, help='total no. of points to be sampled in each class')

# different experiments
parser.add_argument("--original_dc", action='store_true')                           # original free lunch paper
parser.add_argument("--original_dc_cc", action='store_true')                        # original free lunch paper with corrected covariance
parser.add_argument("--use_dc_from_scaling_v4_cc5", action='store_true')            # our method
# do not use any flags for no method

args = parser.parse_args()

#%%
# ---- data loading

if args.dataset == 'miniImagenet':
    args.base_features_path = 'ICLR2022/data/miniImagenet/base_features.plk'
    args.novel_features_path = 'ICLR2022/data/miniImagenet/novel_features.plk'
elif args.dataset == 'CUB':
    args.base_features_path = 'ICLR2022/data/CUB/base_features.plk'
    args.novel_features_path = 'ICLR2022/data/CUB/novel_features.plk'    
elif args.dataset == 'StanfordDogs':
    args.base_features_path = 'ICLR2022/data/StanfordDogs/base_features.plk'
    args.novel_features_path = 'ICLR2022/data/StanfordDogs/novel_features.plk'    
elif args.dataset == 'tieredImagenet':
    args.base_features_path = ''
    args.novel_features_path = ''    
else:
    raise Exception('Unrecognized dataset, check --dataset options')

condition_list = [ args.__dict__[el] for el in list(args.__dict__)[-3:] if isinstance(args.__dict__[el], bool) ]
if sum(condition_list) > 1:
    raise Exception('Invalid experiment combination, no two flags should be True at the same time')

log_file = os.path.join('ICLR2022/logs/', args.xpid+'.log')


if os.path.exists(log_file):
    if args.xpid != 'test' and not args.overwrite:
        raise Exception('log file already exists')
    elif args.overwrite:
        os.system('rm '+log_file)

#%%

print('Working on ', args.dataset)

n_lsamples = args.n_ways * args.n_shot
n_usamples = args.n_ways * args.n_queries
n_samples = n_lsamples + n_usamples

cfg = {'shot': args.n_shot, 'ways': args.n_ways, 'queries': args.n_queries}
FSLTask.loadDataSet(novel_features_file=args.novel_features_path)
FSLTask.setRandomStates(cfg)
ndatas = FSLTask.GenerateRunSet(end=args.n_runs, cfg=cfg)
ndatas = ndatas.permute(0, 2, 1, 3).reshape(args.n_runs, n_samples, -1)
labels = torch.arange(args.n_ways).view(1, 1, args.n_ways).expand(args.n_runs, args.n_shot + args.n_queries, 5).clone().view(args.n_runs,
                                                                                                    n_samples)
# ---- Base class statistics
base_cov = []

all_features = []
with open(args.base_features_path, 'rb') as f:
    data = pickle.load(f)
    for key in data.keys():
        feature = np.array(data[key])
        all_features.append(feature)
        mean = np.mean(feature, axis=0)
        cov = np.cov(feature.T)
        base_cov.append(cov)

# Transferring to GPU
device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

# base_means, base_cov should be calculated after all_features has been transformed
# base_means = torch.tensor(base_means, dtype=torch.float32).to(device)
base_cov = torch.tensor(base_cov, dtype=torch.float32).to(device)
labels = labels.to(device)
ndatas = ndatas.to(device)

if args.dataset == 'CUB':
    # CUB needs special handling since all classes do not have the same no. of points
    all_features = torch.tensor(np.stack([el[:44] for el in all_features]), dtype=torch.float32).to(device)
elif args.dataset == 'StanfordDogs':
    all_features = torch.tensor(np.stack([el[:148] for el in all_features]), dtype=torch.float32).to(device)
else:
    all_features = torch.tensor(np.stack(all_features), dtype=torch.float32).to(device)


# Used only in v4_cc5|cc7 and MD distance with covariance_shrinkage_2
identity = torch.eye(base_cov.shape[1]).to(all_features.device)
ones = torch.ones_like(base_cov[0])


all_covs_inverse = None
all_L_inverse = None


base_means = all_features.mean(dim=1)


# for el in range(64):
#     # Sanity checking inverse function
#     tmp = (all_covs_inverse @ N_covariances)
#     assert tmp[el].diag().mean()-1 < 1e-5
#     assert (tmp[el]*(ones - identity)).mean() < 1e-5


def transformation(data, beta, transform='tukey'):
    if transform == 'tukey':
        if args.beta != 0:
            data = torch.pow(data[:, ], beta)
        else:
            data = torch.log(data[:, ])
    elif transform == 'yj':
        if args.beta != 0:
            data = (torch.pow(data[:, ] + 1, beta) - 1) / beta
        else:
            data = torch.log(data[:, ] + 1)
    else:
        raise Exception('Unknown transform')
    return data


all_checked_params = []

if args.no_optuna:
    args.n_trials = 1                   # so that only one run with given hparams
else:
    args.print_iter = args.n_runs + 1   # so that no intermediate printing while tuning


#%%
def objective(trial):

    if not args.no_optuna:
        args.beta = trial.suggest_float('beta', 0, 10, step=0.25)
        args.m = trial.suggest_float('m', 0, 3, step=0.25)
        args.k = trial.suggest_int('k', 2, 10, step=1)
        args.alpha = trial.suggest_float('alpha', 0, 10000, step=1000)
        args.alpha2 = trial.suggest_categorical('alpha2', [0, 0.1, 1, 10])
        args.num_sampled = trial.suggest_int('num_sampled', 100, 1000, 100)

    num_sampled = torch.tensor([args.num_sampled/args.n_shot], dtype=torch.int32).to(device)

    # Prune unnecessary branch
    if trial.params in all_checked_params:
        raise optuna.exceptions.TrialPruned('Duplicate parameter set, pruning this search')
    all_checked_params.append(trial.params)

    # ---- classification for each task
    acc_list = []
    print('Start classification for %d tasks...'%(args.n_runs))

    iteration = 0
    for i in tqdm(range(args.n_runs)):

        support_data = ndatas[i][:n_lsamples]
        support_label = labels[i][:n_lsamples]

        query_data = ndatas[i][n_lsamples:]
        query_label = labels[i][n_lsamples:]


        # ---- Tukey's transform
        if args.dataset == 'StanfordDogs':
            transform = 'yj'
        else:
            transform = 'tukey'
        support_data = transformation(support_data, args.beta, transform=transform)
        query_data = transformation(query_data, args.beta, transform=transform)

        sampled_data = []
        sampled_label = []

        if any(condition_list):
            
            if args.use_dc_from_scaling_v4_cc5:
                for i in range(n_lsamples):
                    mean, cov, _ = distribution_calibration_from_scaling_v4_cc5(x_tilda=support_data[i], 
                                                                        m=args.m, alpha1=args.alpha, alpha2=args.alpha2*args.alpha, 
                                                                        k=args.k, all_features=all_features, 
                                                                        identity=identity, ones=ones, 
                                                                        all_covs_inverse=all_covs_inverse)
                    try:
                        distrib = construct_distribution(mean, cov)                                         # use this when you are not shrinking covariance
                        # distrib = torch.distributions.multivariate_normal.MultivariateNormal(mean, cov)   # This will error out without covariance shrinking
                    except RuntimeError:
                        raise optuna.exceptions.TrialPruned('Ran into runtime error with symeig')

                    sampled_data.append(distrib.sample(num_sampled))
                    sampled_label.extend([support_label[i]]*num_sampled)

            elif args.original_dc:
                for i in range(n_lsamples):
                    mean, cov = distribution_calibration(query=support_data[i], base_means=base_means, base_cov=base_cov, k=args.k, alpha=args.alpha, use_mahalanobis=False)# , all_covs_inverse = all_covs_inverse)
                    try:
                        distrib = construct_distribution(mean, cov)                                         
                        # distrib = torch.distributions.multivariate_normal.MultivariateNormal(mean, cov)   
                    except RuntimeError:
                        raise optuna.exceptions.TrialPruned('Ran into runtime error with symeig')
                    sampled_data.append(distrib.sample(num_sampled))
                    sampled_label.extend([support_label[i]]*num_sampled)

            elif args.original_dc_cc:
                for i in range(n_lsamples):
                    mean, cov = distribution_calibration_cc(support_data[i], base_means, base_cov, k=2, alpha=0.1, all_features=all_features)
                    # `TODO` : Not sure why cholesky error should come here at
                    distrib = construct_distribution(mean, cov)
                    # distrib = torch.distributions.multivariate_normal.MultivariateNormal(mean, cov)
                    sampled_data.append(distrib.sample(num_sampled))
                    sampled_label.extend([support_label[i]]*num_sampled)

            else:
                raise Exception('Corrupt condition list')

            sampled_data = torch.cat(sampled_data, dim=0)
            sampled_label = torch.stack(sampled_label)

            X_train = torch.cat([support_data, sampled_data], dim=0)
            Y_train = torch.cat([support_label, sampled_label], dim=0)

        else:
            # no_dc_tukey
            X_train = support_data
            Y_train = support_label

        acc = train(X_train=X_train, Y_train=Y_train, query_data=query_data, query_label=query_label)
        acc_list.append(acc)

        mean_acc = torch.stack(acc_list).mean()
        trial.report(mean_acc, iteration)


        if trial.should_prune():
            raise optuna.exceptions.TrialPruned('Accuracy less than median, pruning {}'.format(trial.params))

        if args.no_optuna and ((iteration+1) % args.print_iter == 0):
            out = mean_confidence_interval(torch.stack(acc_list).cpu().numpy())
            print('{} way {} shot  ACC : {:.3f} +- {:.3f}'.format(args.n_ways, args.n_shot, 100*out[0], 100*out[1]))

        iteration += 1

    out = mean_confidence_interval(torch.stack(acc_list).cpu().numpy())
    print('{} way {} shot  ACC : {:.3f} +- {:.3f}'.format(args.n_ways,args.n_shot, 100*out[0], 100*out[1]))

    return mean_acc
#%%
logger = logging.getLogger()

logger.setLevel(logging.INFO)  # Setup the root logger.
logger.addHandler(logging.FileHandler(log_file, mode="w"))

optuna.logging.enable_propagation()  # Propagate logs to the root logger.
optuna.logging.disable_default_handler()  # Stop showing logs in sys.stderr.

logger.info("Start optimization.")

study = optuna.create_study(direction="maximize",
                            pruner=optuna.pruners.MedianPruner(n_startup_trials=100,
                                                               n_warmup_steps=100,
                                                               interval_steps=10))

study.optimize(objective, n_trials=args.n_trials)

pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

logger.info("Study statistics: ")
logger.info("  Number of finished trials: {}".format(len(study.trials)))
logger.info("  Number of pruned trials: {}".format(len(pruned_trials)))
logger.info("  Number of complete trials: {}".format(len(complete_trials)))

logger.info("Best trial:")
trial = study.best_trial

logger.info("  Value: {}".format(trial.value))
logger.info("  Params: ")
for key, value in trial.params.items():
    logger.info("    {}: {}".format(key, value))

#%%
