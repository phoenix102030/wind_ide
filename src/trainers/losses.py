import math

import torch


def gaussian_nll_isotropic(y_true, y_pred, sigma_eps):
    resid = y_true - y_pred
    if resid.ndim < 2:
        raise ValueError(f"Expected batched residuals, got shape {tuple(resid.shape)}")

    n = resid[0].numel()
    var = sigma_eps.square() + 1e-8
    sq = resid.reshape(resid.shape[0], -1).square().sum(dim=1)
    return (0.5 * (sq / var + n * torch.log(var) + n * math.log(2.0 * math.pi))).mean()


def gaussian_nll_vector(y_true, y_pred, sigma_eps):
    return gaussian_nll_isotropic(y_true, y_pred, sigma_eps)


def total_loss(y_true, y_pred, sigma_eps, Sigma_v, lambda_cov=1e-4):
    nll = gaussian_nll_vector(y_true, y_pred, sigma_eps)
    cov_reg = lambda_cov * (Sigma_v ** 2).mean()
    return nll + cov_reg


def total_objective(
    y_true,
    y_pred,
    sigma_eps,
    delta=None,
    temporal_penalty=None,
    lambda_time=1.0,
    lambda_ml=1e-4,
):
    loss = gaussian_nll_isotropic(y_true, y_pred, sigma_eps)
    if temporal_penalty is not None:
        loss = loss + float(lambda_time) * temporal_penalty
    if delta is not None:
        loss = loss + float(lambda_ml) * delta.square().mean()
    return loss
