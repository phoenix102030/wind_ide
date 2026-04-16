import torch


def gaussian_nll_vector(y_true, y_pred, sigma_eps):
    """
    y_true, y_pred: [B,3,2]
    sigma_eps: scalar
    """
    resid = y_true - y_pred
    n = resid[0].numel()
    var = sigma_eps**2 + 1e-8
    sq = (resid**2).reshape(resid.shape[0], -1).sum(dim=1)
    return (0.5 * (sq / var + n * torch.log(var))).mean()


def total_loss(y_true, y_pred, sigma_eps, Sigma_v, lambda_cov=1e-4):
    nll = gaussian_nll_vector(y_true, y_pred, sigma_eps)
    cov_reg = lambda_cov * (Sigma_v ** 2).mean()
    return nll + cov_reg