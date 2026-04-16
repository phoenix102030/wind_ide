import torch
from src.trainers.losses import gaussian_nll_isotropic


def test_gaussian_nll_isotropic_runs():
    y = torch.randn(2, 4, 4)
    pred = torch.randn(2, 4, 4)
    sigma = torch.tensor(0.3)
    loss = gaussian_nll_isotropic(y, pred, sigma)
    assert loss.ndim == 0