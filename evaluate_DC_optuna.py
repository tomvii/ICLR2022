
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

try:
    os.chdir('/mnt/Few_Shot_DC')
except:
    pass

import warnings
import pickle
import numpy as np
from sklearn.linear_model import LogisticRegression
from tqdm import tqdm
import json
import scipy.stats
import time
from pytorch_utils import train
from few_shot_functions import *
import FSLTask
# from temperature_scaling.functions import find_optimal_T

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
##### Changed
DEBUG = False


if DEBUG:
    print(' *******************************************\n\
*******************************************\n\
******** DEBUG IS SET TO TRUE *************\n\
*******************************************\n\
*******************************************\n')

if not DEBUG:
    parser = argparse.ArgumentParser(description="Few Shot Learning")
    parser.add_argument("--n_shot", type=int, default=1)
    parser.add_argument("--n_ways", type=int, default=5)
    parser.add_argument("--n_queries", type=int, default=15)
    parser.add_argument("--xpid", type=str, default='test')
    parser.add_argument("--n_runs", type=int, default=200)
    parser.add_argument("--print_iter", type=int, default=50)
    parser.add_argument("--overwrite", action='store_true')
    parser.add_argument("--use_md", action='store_true', help='use mahalanobis distance instead of eucledian')
    parser.add_argument("--dataset", type=str, default="miniImagenet", choices=["miniImagenet", "CUB", "StanfordDogs", "tieredImagenet", "WRN_MI_wo_relu"])
    parser.add_argument("--n_trials", type=int, default=100)
    parser.add_argument("--no_optuna", action='store_true', help='do not optimize with optuna, use given hyperparams')


    # hyperparams, used only when --no_optuna flag is on
    parser.add_argument("--beta", type=float, default=0.5, help='Tukey beta')
    parser.add_argument("--m", type=float, default=1)
    parser.add_argument("--k", type=int, default=2)
    parser.add_argument("--alpha", type=float, default=0)
    parser.add_argument("--alpha2", type=float, default=0)
    parser.add_argument("--delta", type=float, default=1)
    parser.add_argument("--phi1", type=float, default=0)
    parser.add_argument("--num_sampled", type=int, default=750, help='total no. of points to be sampled in each class')


    ##### Changed
    # 2 layer NN LR instead of single layer LR
    parser.add_argument("--layer1dim", type=int, default=256)
    parser.add_argument("--layer2dim", type=int, default=64)


    # deprecated
    parser.add_argument("--gamma", type=float, default=1e-3, help='the shrinkage parameter gamma')
    parser.add_argument("--tscaling", type=float, default=1.0,              # original free lunch paper with T scaled features
                        help='T scaling for dc to check if it gives improvements')

    # different experiments
    parser.add_argument("--use_dc_from_scaling_v2", action='store_true')
    parser.add_argument("--use_scaling_factor", action='store_true')                    # our idea of weighted average of the base means
    parser.add_argument("--use_similarity_approach", action='store_true')               # our idea on mean, cov from support points + close base classes
    parser.add_argument("--use_calibration_from_mean", action='store_true')             # idea on distance from mean
    parser.add_argument("--use_dc_from_mean_with_scaling", action='store_true')         # our idea of weighted avg of the base means along with dc from mean
    parser.add_argument("--original_dc", action='store_true')                           # original free lunch paper
    parser.add_argument("--original_dc_cc", action='store_true')                        # original free lunch paper with corrected covariance

    parser.add_argument("--use_dc_from_scaling_v3", action='store_true')                # weighted avg mean and cov same as similarity approach
    parser.add_argument("--use_dc_from_scaling_v4", action='store_true')                # weighted avg mean but mean does not have u~ in denominator and cov same as similarity approach
    parser.add_argument("--use_dc_from_scaling_v4_cc", action='store_true')                # weighted avg mean and corrected covariance but mean does not have u~ in denominator
    parser.add_argument("--use_dc_from_scaling_v4_cc2", action='store_true')
    parser.add_argument("--use_dc_from_scaling_v4_cc3", action='store_true')
    parser.add_argument("--use_dc_from_scaling_v4_cc4", action='store_true')
    parser.add_argument("--use_dc_from_scaling_v4_cc5", action='store_true')
    parser.add_argument("--use_dc_from_scaling_v4_cc7", action='store_true')
    parser.add_argument("--use_dc_from_scaling_v4_cc8", action='store_true')
    parser.add_argument("--use_dc_from_scaling_v5", action='store_true')                # weighted avg mean and cov decaying exponentially but mean does not have u~ in denominator
    parser.add_argument("--use_dc_from_scaling_v6", action='store_true')                # weighted avg mean decaying exponentially but mean does not have u~ in denominator and cov same as similarity approach


class Bunch(object):
    def __init__(self, adict):
        self.__dict__.update(adict)

if not DEBUG:
    args = parser.parse_args()

else:

    args = Bunch({
        'n_shot': 1,
        'n_ways': 5,
        'n_queries': 15,
        'xpid': 'test',
        'n_runs': 21,
        'print_iter': 50,
        'n_trials': 1,
        'delta': 1,
        ##### Changed
        'beta': 1,
        'alpha': 600,
        'alpha2':400,
        'm':2.25,
        'k':4,
        'gamma':1e-1,
        'num_sampled':750,
        'tscaling': 1,
        'phi1':500,
        'use_md':True,
        'no_optuna': True,
        ##### Changed
        'dataset': 'StanfordDogs',
        # 'dataset': 'miniImagenet',
        'overwrite': False,
        'use_dc_from_scaling_v2': False,
        'use_scaling_factor': False,
        'use_similarity_approach': False,
        'use_calibration_from_mean': False,
        'use_dc_from_mean_with_scaling': False,
        'original_dc': False,
        'original_dc_cc': False,
        'use_dc_from_scaling_v3': False,
        'use_dc_from_scaling_v4': False,
        'use_dc_from_scaling_v4_cc': False,
        'use_dc_from_scaling_v4_cc2': False,
        'use_dc_from_scaling_v4_cc3': False, 
        'use_dc_from_scaling_v4_cc4': False, 
        'use_dc_from_scaling_v4_cc5': False, 
        'use_dc_from_scaling_v4_cc7': False, 
        'use_dc_from_scaling_v4_cc8': True, 
        'use_dc_from_scaling_v5': False,
        'use_dc_from_scaling_v6': False,
    })

#%%
# ---- data loading

if args.dataset == 'miniImagenet':
    args.base_features_path = '/mnt/checkpoints/base_features.plk'
    ##### Changed
    # args.novel_features_path = '/mnt/checkpoints/novel_features.plk'
    args.novel_features_path = '/mnt/checkpoints/val_features.plk'


elif args.dataset == 'CUB':
    args.base_features_path = '/domino/datasets/shakti_kumar/distribution-calibration/scratch/datasets/CUB/base_features.plk'
    args.novel_features_path = '/domino/datasets/shakti_kumar/distribution-calibration/scratch/datasets/CUB/novel_features.plk'    
elif args.dataset == 'StanfordDogs':
    ##### Changed
    # args.base_features_path = '/domino/datasets/shakti_kumar/distribution-calibration/scratch/datasets/StanfordDogs/saved_model_on_base_classes/features_wo_relu/base_features.plk'
    # args.novel_features_path = '/domino/datasets/shakti_kumar/distribution-calibration/scratch/datasets/StanfordDogs/saved_model_on_base_classes/features_wo_relu/novel_features.plk'    
    # args.base_features_path = '/domino/datasets/shakti_kumar/distribution-calibration/scratch/datasets/StanfordDogs/saved_features/model_on_train_classes/sfdogs_train_features.plk'
    # args.novel_features_path = '/domino/datasets/shakti_kumar/distribution-calibration/scratch/datasets/StanfordDogs/saved_features/model_on_train_classes/sfdogs_test_features.plk'    
    # args.base_features_path = '/domino/datasets/shakti_kumar/distribution-calibration/scratch/datasets/StanfordDogs/saved_features/inceptionv3/tmp_model_4/sfdogs_train_features.plk'
    # args.novel_features_path = '/domino/datasets/shakti_kumar/distribution-calibration/scratch/datasets/StanfordDogs/saved_features/inceptionv3/tmp_model_4/sfdogs_test_features.plk'    
    args.base_features_path = '/domino/datasets/shakti_kumar/distribution-calibration/scratch/checkpoints/64dims/StanfordDogs/WideResNet28_10_S2M2_R/last/base_features.plk'
    args.novel_features_path = '/domino/datasets/shakti_kumar/distribution-calibration/scratch/checkpoints/64dims/StanfordDogs/WideResNet28_10_S2M2_R/last/novel_features.plk'    
elif args.dataset == 'WRN_MI_wo_relu':
    args.base_features_path = '/domino/datasets/shakti_kumar/distribution-calibration/scratch/miniimagenet/miniimagenet_pretrained_from_fsl/WideResNet28_10_S2M2_R/last/base_features.plk'
    args.novel_features_path = '/domino/datasets/shakti_kumar/distribution-calibration/scratch/miniimagenet/miniimagenet_pretrained_from_fsl/WideResNet28_10_S2M2_R/last/novel_features.plk'    
elif args.dataset == 'tieredImagenet':
    args.base_features_path = ''
    args.novel_features_path = ''    
else:
    raise Exception('Unrecognized dataset, check --dataset options')

condition_list = [ args.__dict__[el] for el in list(args.__dict__)[-15:] if isinstance(args.__dict__[el], bool) ]
if sum(condition_list) > 1:
    raise Exception('Invalid experiment combination, no two flags should be True at the same time')


log_file = os.path.join('/domino/datasets/shakti_kumar/distribution-calibration/scratch/', args.xpid+'.log')

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

        feature = feature/args.tscaling

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

ndatas = ndatas / args.tscaling


if args.dataset == 'CUB':
    # CUB needs special handling since all classes do not have the same no. of points
    all_features = torch.tensor(np.stack([el[:44] for el in all_features]), dtype=torch.float32).to(device)
elif args.dataset == 'StanfordDogs':
    all_features = torch.tensor(np.stack([el[:148] for el in all_features]), dtype=torch.float32).to(device)
else:
    all_features = torch.tensor(np.stack(all_features), dtype=torch.float32).to(device)


def shift_features_for_tukey(all_features, ndatas):
    # the logits can be <0 if taken from the classification layer,
    # shifting it doesnt change the calibration error or the softmax but allows tukey to be done
    shift_value = torch.stack([all_features.min(), ndatas.min()]).min().abs()
    all_features = all_features + shift_value
    ndatas = ndatas + shift_value
    assert (all_features >= 0).all()
    assert (ndatas >= 0).all()
    return all_features, ndatas


# all_features, ndatas = shift_features_for_tukey(all_features, ndatas)

def select_gamma(cov):
    gamma_search = [4e-4, 6e-4, 8e-4, 10e-4, 12e-4]
    y1 = []
    y2 = []
    for gamma in gamma_search:
        new_cov = shrink_covariance(cov, gamma=gamma)
        prod = new_cov.inverse() @ new_cov
        d1 = torch.dist(new_cov, cov).item()
        d2 = torch.dist(prod, torch.eye(prod.shape[0]).expand_as(prod)).item()
        y1.append(d1)
        y2.append(d2)

    gamma = gamma_search[(torch.tensor(y1)-torch.tensor(y2)).abs().argmin().item()]
    # print((torch.tensor(y1)-torch.tensor(y2)).abs().min())
    eigval, eigvec = torch.symeig(shrink_covariance(cov, gamma=gamma), eigenvectors=True)
    assert (eigval > -1e-6).all()
    return gamma

# Used only in v4_cc5|cc7 and MD distance with covariance_shrinkage_2
identity = torch.eye(base_cov.shape[1]).to(all_features.device)
ones = torch.ones_like(base_cov[0])


##### Changed
if args.use_md:
    N_covariances = torch.stack([compute_cov_2(all_features[i]) for i in range(all_features.shape[0])])
    all_covs_inverse = torch.stack([el.inverse() for el in N_covariances])
    all_L_inverse = torch.stack([torch.cholesky(el).inverse() for el in N_covariances])
    logdet_of_N_covariances = torch.stack([el.logdet() for el in N_covariances])

    ##### CHanged, adding a value to make it positive to avoid negative wi in the formulation
    logdet_of_N_covariances = logdet_of_N_covariances + logdet_of_N_covariances.min().abs()

    ##### Changed
    # all_features = [torch.einsum("ij,jk->ik", all_L_inverse[el], all_features[el].T).T for el in range(all_features.shape[0])]
    # all_features = torch.stack(all_features)

    # sigma1 = torch.stack([N_covariances[i].diag() for i in range(all_features.shape[0])]).mean()
    # sigma2 = torch.stack([(N_covariances[i] * (ones-identity)) for i in range(all_features.shape[0])]).mean()
    # phi2 = args.phi1 * (sigma1 / sigma2)
    # N_covariances_tmp = torch.stack([shrink_covariance_2(N_covariances[i], alpha1=args.phi1, alpha2=phi2, identity=identity, ones=ones) for i in range(all_features.shape[0])])
    # all_covs_inverse = N_covariances_tmp.inverse()
    # print(all_covs_inverse.mean())
else:
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


def project_query_points(query_data):
    # Finds out the distances between query_points and all base means, 
    # then transforms query_points with the closest base mean's L_inv

    transformed_query = (all_L_inverse @ query_data.unsqueeze(0).transpose(1,2)).transpose(1,2)     # this is B,N,d
    # a = torch.einsum('ij,jk->ik', all_L_inverse[0], query_data.T).T

    distances = (base_means.unsqueeze(1) - transformed_query).norm(dim=2)
    closest_base_classes = distances.argmin(dim=0)

    transformed_query = all_L_inverse[closest_base_classes] @ query_data.unsqueeze(2)
    # torch.einsum('ij,jk->ik', all_L_inverse[7], query_data[0][:, None]).T

    return transformed_query.squeeze(), closest_base_classes


#%%
def objective(trial):

    if not args.no_optuna:
        ##### Changed, Setting optuna variables

        # args.beta = trial.suggest_float('beta', 0, 10, step=0.25)
        args.beta = 0.5
        # args.k = all_features.shape[0]
        # args.m = trial.suggest_float('m', 0, 3, step=0.25)

        args.m = 0
        args.k = trial.suggest_int('k', 2, 10, step=1)
        args.alpha = trial.suggest_float('alpha', 0, 0.4, step=0.05)
        args.alpha2 = 0

        # args.alpha2 = trial.suggest_categorical('alpha2', [0, 0.1, 1, 10])

        # args.delta = trial.suggest_float('delta', 0.1, 5, step=0.1)
        # args.delta = 1
        # args.phi1 = 0
        args.num_sampled = 750
        # args.num_sampled = trial.suggest_int('num_sampled', 100, 1000, 100)

        ##### Changed
        args.layer1dim = trial.suggest_categorical('layer1dim', [512, 256, 128])
        args.layer2dim = trial.suggest_categorical('layer2dim', [256, 128, 64])


    num_sampled = torch.tensor([args.num_sampled/args.n_shot], dtype=torch.int32).to(device)

    # Prune unnecessary branch
    ##### Changed, for checking results from all parameters during ablation studies
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
        ##### Changed
        # query_data, closest_base_classes = project_query_points(query_data)
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
            
            if args.use_dc_from_scaling_v4_cc4:
                for i in range(n_lsamples):
                    mean, cov = distribution_calibration_from_scaling_v4_cc4(x_tilda=support_data[i], m=args.m, alpha=args.alpha, k=args.k, all_features=all_features)
                    distrib = construct_distribution(mean, cov)                                         # use this when you are not shrinking covariance
                    # distrib = torch.distributions.multivariate_normal.MultivariateNormal(mean, cov)   # This will error out without covariance shrinking
                    sampled_data.append(distrib.sample(num_sampled))
                    sampled_label.extend([support_label[i]]*num_sampled)
            
            elif args.use_dc_from_scaling_v4_cc5:
                for i in range(n_lsamples):
                    mean, cov, _ = distribution_calibration_from_scaling_v4_cc5(x_tilda=support_data[i], 
                                                                        ##### Changed, alpha2 is args.alpha2 * args.alpha1
                                                                        m=args.m, alpha1=args.alpha, alpha2=args.alpha2*args.alpha, 
                                                                        k=args.k, all_features=all_features, 
                                                                        identity=identity, ones=ones, 
                                                                        use_md=args.use_md, all_covs_inverse=all_covs_inverse)
                    try:
                        distrib = construct_distribution(mean, cov)                                         # use this when you are not shrinking covariance
                        # distrib = torch.distributions.multivariate_normal.MultivariateNormal(mean, cov)   # This will error out without covariance shrinking
                    except RuntimeError:
                        raise optuna.exceptions.TrialPruned('Ran into runtime error with symeig')

                    sampled_data.append(distrib.sample(num_sampled))
                    sampled_label.extend([support_label[i]]*num_sampled)

            elif args.use_dc_from_scaling_v4_cc7:
                # print(args.delta)
                for i in range(n_lsamples):
                    mean, cov = distribution_calibration_from_scaling_v4_cc7(x_tilda=support_data[i], 
                                                                        ##### Changed, alpha2 is args.alpha2 * args.alpha1
                                                                        m=args.m, alpha1=args.alpha, alpha2=args.alpha2*args.alpha, 
                                                                        k=args.k, delta=args.delta, all_features=all_features, 
                                                                        identity=identity, ones=ones, 
                                                                        use_md=args.use_md, all_covs_inverse=all_covs_inverse)
                    try:
                        distrib = construct_distribution(mean, cov)                                         # use this when you are not shrinking covariance
                        # distrib = torch.distributions.multivariate_normal.MultivariateNormal(mean, cov)   # This will error out without covariance shrinking
                    except RuntimeError:
                        raise optuna.exceptions.TrialPruned('Ran into runtime error with symeig')

                    sampled_data.append(distrib.sample(num_sampled))
                    sampled_label.extend([support_label[i]]*num_sampled)

            elif args.use_dc_from_scaling_v4_cc8:
                # print(args.delta)
                for i in range(n_lsamples):
                    mean, cov, closest_found = distribution_calibration_from_scaling_v4_cc8(x_tilda=support_data[i], 
                                                                        ##### Changed, alpha2 is args.alpha2 * args.alpha1
                                                                        m=args.m, alpha1=args.alpha, alpha2=args.alpha2*args.alpha, 
                                                                        k=args.k, delta=args.delta, all_features=all_features, 
                                                                        identity=identity, ones=ones, 
                                                                        use_md=args.use_md, all_L_inverse=all_L_inverse,
                                                                        all_covs_inverse=all_covs_inverse, logdet_of_N_covariances=logdet_of_N_covariances)
                    try:
                        distrib = construct_distribution(mean, cov)                                         # use this when you are not shrinking covariance
                        # distrib = torch.distributions.multivariate_normal.MultivariateNormal(mean, cov)   # This will error out without covariance shrinking
                    except RuntimeError:
                        raise optuna.exceptions.TrialPruned('Ran into runtime error with symeig')

                    sampled_data.append(distrib.sample(num_sampled))
                    sampled_label.extend([support_label[i]]*num_sampled)
                    ##### Changed
                    # tmp = distrib.sample(num_sampled)
                    # project_tmp_to_eucledian = tmp @ project_l_inv.inverse().T
                    # sampled_data.append(project_tmp_to_eucledian)


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

        ##### Changed, check plot here to find if convergence is happening for CUB and Stanford Dogs
        # raise Exception('Find out convergence here')

        ##### Changed
        acc = train(X_train=X_train, Y_train=Y_train, query_data=query_data, query_label=query_label, layerdims=[args.layer1dim, args.layer2dim])
        acc_list.append(acc)

        mean_acc = torch.stack(acc_list).mean()
        trial.report(mean_acc, iteration)

        # print(trial.should_prune())

        ##### Changed, for checking results from all parameters during ablation studies
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

##### Changed, for checking results from all parameters during ablation studies
study = optuna.create_study(direction="maximize",
                            pruner=optuna.pruners.MedianPruner(n_startup_trials=100,
                                                               n_warmup_steps=100,
                                                               interval_steps=10))
# from optuna.samplers import GridSampler
# search_space = {"alpha2":np.arange(0,1001,100).tolist()}
# search_space = {"alpha":np.arange(0,1001,100).tolist()}
# search_space = {"m":np.arange(0,3.25, 0.25).tolist()}
# search_space = {"beta":np.arange(0,3.25,0.25).tolist()}
# search_space = {"k":np.arange(1,21,1).tolist()}
# study = optuna.create_study(direction="maximize",
#                             sampler=GridSampler(search_space))


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
