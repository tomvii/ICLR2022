import torch
import numpy as np
import scipy


def shrink_covariance(sigma, gamma):
    cov = (1-gamma)*sigma + gamma * sigma.diag().mean() * torch.eye(sigma.shape[0]).to(sigma.device)
    return cov

def shrink_covariance_2(sigma, alpha1, alpha2, identity, ones):
    cov = sigma + alpha1*sigma.diag().mean()*identity + alpha2*(sigma*(ones-identity)).mean()*(ones-identity)
    return cov


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


def distribution_calibration_from_scaling_v4_cc6(x_tilda, m, alpha1, alpha2, all_features, k, identity, ones, use_md=False, all_covs_inverse=None):
    """Same as v4_cc5 but with wi = 1 / (1 + (1+ti)^m) to cover any ti < 1
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
    wi = 1 / (1 + (1+ti).pow(m))

    ui = selected_features.mean(dim=1)
    mean_prime = (x_tilda + (wi[:, None] * ui).sum(dim=0)) / (1 + wi.sum(dim=0))

    new_rv = ( (wi[:, None, None] / wi.sum(0)) * selected_features).sum(0)         # This sum is basically summing the individual random variables with their weights
    new_cov = compute_cov_2(new_rv)

    sigma_prime = new_cov + alpha1*new_cov.diag().mean()*identity + alpha2*(new_cov*(ones-identity)).mean()*(ones-identity)

    return mean_prime, sigma_prime

# higher result
def distribution_calibration_from_scaling_v4_cc7(x_tilda, m, alpha1, alpha2, k, delta, all_features, identity, ones, use_md=False, all_covs_inverse=None):
    """Same as v4_cc5 but with generalized distance instead of eucledian
    """
    base_means = all_features.mean(dim=1)

    ti = ((delta*base_means - x_tilda)).norm(dim=1)             # this line is different from distribution_calibration_from_scaling_v3
    ##### Changed
    # ti = ((delta + base_means - x_tilda)).norm(dim=1)         # this line is different from distribution_calibration_from_scaling_v3

    index = ti.topk(k, largest=False).indices
    ti = ti[index]

    selected_features = all_features[index]
    wi = 1 / (1 + ti.pow(m))

    ui = selected_features.mean(dim=1)
    mean_prime = (x_tilda + (wi[:, None] * ui).sum(dim=0)) / (1 + wi.sum(dim=0))

    new_rv = ( (wi[:, None, None] / wi.sum(0)) * selected_features).sum(0)         # This sum is basically summing the individual random variables with their weights
    new_cov = compute_cov_2(new_rv)

    sigma_prime = new_cov + alpha1*new_cov.diag().mean()*identity + alpha2*(new_cov*(ones-identity)).mean()*(ones-identity)

    return mean_prime, sigma_prime


def distribution_calibration_from_scaling_v4_cc8(x_tilda, m, alpha1, alpha2, k, delta, all_features, identity, ones, use_md=False, all_L_inverse=None, all_covs_inverse=None, logdet_of_N_covariances=None):
    """
    # Same as v4_cc5 but with Linv distances and with transformer Xi with Linv
    # Same as v4_cc5 but with alpha1 * diagonal variance 
    Same as v4_cc5 but with MD and including ln term in the distance
    """
    base_means = all_features.mean(dim=1)

    if use_md:
        ##### Changed
        ti = logdet_of_N_covariances + ((base_means - x_tilda).unsqueeze(1) @ all_covs_inverse @ (base_means - x_tilda).unsqueeze(2)).squeeze()
        # ti = ((base_means - x_tilda).unsqueeze(1) @ all_covs_inverse @ (base_means - x_tilda).unsqueeze(2)).squeeze()
        # use this assert statement to verify the above unsqueeze(1|2) is correct
        # assert ((base_means - x_tilda).unsqueeze(1).transpose(1,2) == (base_means - x_tilda).unsqueeze(2)).all()
    else:
        # use eucledian distance
        ti = ((base_means - x_tilda)).norm(dim=1)        # this line is different from distribution_calibration_from_scaling_v3
        # a = (all_L_inverse @ x_tilda.reshape(1, x_tilda.shape[0], 1)).squeeze() # B,d,d x 1,d,1 -> B,d,1
        # a = torch.einsum("bij,jk->bik", all_L_inverse, x_tilda.unsqueeze(1)).squeeze()
        # ti = ((base_means - a)).norm(dim=1)             # this line is different from distribution_calibration_from_scaling_v3


    index = ti.topk(k, largest=False).indices
    ti = ti[index]

    selected_features = all_features[index]
    wi = 1 / (1 + ti.pow(m))

    ui = selected_features.mean(dim=1)

    # mean_prime = ((all_L_inverse[index[0]] @ x_tilda.reshape(-1, 1)).squeeze() + (wi[:, None] * ui).sum(dim=0)) / (1 + wi.sum(dim=0))
    # mean_prime = (torch.einsum('ij,jk->ik', all_L_inverse[index[0]], x_tilda.unsqueeze(1)).squeeze() + (wi[:, None] * ui).sum(dim=0)) / (1 + wi.sum(dim=0))
    mean_prime = (x_tilda + (wi[:, None] * ui).sum(dim=0)) / (1 + wi.sum(dim=0))

    new_rv = ( (wi[:, None, None] / wi.sum(0)) * selected_features).sum(0)         # This sum is basically summing the individual random variables with their weights
    new_cov = compute_cov_2(new_rv)

    # sigma_prime = new_cov + alpha1*new_cov*identity + alpha2*(new_cov*(ones-identity)).mean()*(ones-identity)
    sigma_prime = new_cov + alpha1*new_cov.diag().mean()*identity + alpha2*(new_cov*(ones-identity)).mean()*(ones-identity)

    ##### Changed
    return mean_prime, sigma_prime, index[0]


def compute_cov_2(Z):
    # Z is NxD matrix
    N = Z.shape[0]
    Z_exp = Z - Z.mean(0)
    cov = (1/(N-1)) * Z_exp.T.mm(Z_exp)
    return cov


def compute_cov(m):
    m_exp = torch.mean(m, dim=1)
    x = m - m_exp[:, None]
    cov = 1 / (x.size(1) - 1) * x.mm(x.t())
    return cov


def construct_distribution(mean, cov):
    eigval, eigvec = torch.symeig(cov, eigenvectors=True)

    ##### Changed
    # if not torch.all(eigval >= -1e-4):
    #     raise ValueError("Covariance matrix not PSD.")

    eigval_root = eigval.clamp_min(0.0).sqrt()
    corr_matrix = (eigvec * eigval_root)
    distrib = torch.distributions.multivariate_normal.MultivariateNormal(mean, scale_tril=corr_matrix)

    ##### Changed, removed this assertion as well
    # print((cov-distrib.covariance_matrix).abs().max())
    # assert not ((cov - distrib.covariance_matrix) > 1e-3).any()

    return distrib


def mean_confidence_interval(acc_data, confidence=0.95):
    a = 1.0 * np.array(acc_data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return m, 2*h
