import torch
import numpy as np
import scipy


def distribution_calibration(query, base_means, base_cov, k, alpha, use_mahalanobis=False, all_covs_inverse=None):
    if use_mahalanobis:
        dist = ((base_means - query).unsqueeze(1) @ all_covs_inverse @ (base_means - query).unsqueeze(2)).squeeze()
    else:
        dist = (query - base_means).norm(dim=1)

    index = torch.topk(dist, largest=False, k=k).indices
    mean = torch.cat([base_means[index], query[None, :]], dim=0)
    calibrated_mean = mean.mean(dim=0)

    calibrated_cov = base_cov[index].mean(dim=0) + alpha
    # print('{:.5f}, {:.5f}'.format(calibrated_cov.mean().item(), calibrated_cov.std().item()))
    # random_number = torch.rand(640, 640)
    # calibrated_cov = ((random_number@random_number.T)/1e5).to(calibrated_mean.device) + alpha
    # print(calibrated_cov.diag().mean(), calibrated_cov.mean())

    return calibrated_mean, calibrated_cov


def distribution_calibration_from_scaling_v4_cc5(x_tilda, m, alpha1, alpha2, all_features, k, identity, ones, use_md=False, all_covs_inverse=None):
    """Same as v4_cc3 but with new_cov + alpha * avg_var * I + beta * avg_off_diag_cov * (1-I)
    """
    base_means = all_features.mean(dim=1)

    if use_md:
        ti = ((base_means - x_tilda).unsqueeze(1) @ all_covs_inverse @ (base_means - x_tilda).unsqueeze(2)).squeeze()
        # use this assert statement to verify the above unsqueeze(1|2) is correct
        # assert ((base_means - x_tilda).unsqueeze(1).transpose(1,2) == (base_means - x_tilda).unsqueeze(2)).all()
    else:
        # use eucledian distance
        ti = ((base_means - x_tilda)).norm(dim=1)        # this line is different from distribution_calibration_from_scaling_v3

    index = ti.topk(k, largest=False).indices
    ti = ti[index]

    selected_features = all_features[index]
    wi = 1 / (1 + ti.pow(m))

    ui = selected_features.mean(dim=1)
    mean_prime = (x_tilda + (wi[:, None] * ui).sum(dim=0)) / (1 + wi.sum(dim=0))

    new_rv = ( (wi[:, None, None] / wi.sum(0)) * selected_features).sum(0)         # This sum is basically summing the individual random variables with their weights
    new_cov = compute_cov_2(new_rv)

    sigma_prime = new_cov + alpha1*new_cov.diag().mean()*identity + alpha2*(new_cov*(ones-identity)).mean()*(ones-identity)

    return mean_prime, sigma_prime, index


def compute_cov_2(Z):
    # Z is NxD matrix
    N = Z.shape[0]
    Z_exp = Z - Z.mean(0)
    cov = (1/(N-1)) * Z_exp.T.mm(Z_exp)
    return cov


def construct_distribution(mean, cov):
    eigval, eigvec = torch.symeig(cov, eigenvectors=True)
    eigval_root = eigval.clamp_min(0.0).sqrt()
    corr_matrix = (eigvec * eigval_root)
    distrib = torch.distributions.multivariate_normal.MultivariateNormal(mean, scale_tril=corr_matrix)
    return distrib


def mean_confidence_interval(acc_data, confidence=0.95):
    a = 1.0 * np.array(acc_data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return m, 2*h
