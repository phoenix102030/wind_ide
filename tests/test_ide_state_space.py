import torch

from src.models.advection_mean_net import AdvectionMeanNet
from src.models.ide_state_space import IDEStateSpaceModel


def test_sequence_nll_accepts_tensor_and_flat_state():
    model = IDEStateSpaceModel(num_sites=3)
    z_tensor = torch.randn(2, 5, 3, 2)
    z_flat = z_tensor.reshape(2, 5, 6)
    site_lon = torch.tensor([120.0, 120.1, 120.2])
    site_lat = torch.tensor([30.0, 30.1, 30.2])

    nll_tensor = model.sequence_nll(z_tensor, site_lon=site_lon, site_lat=site_lat)
    nll_flat = model.sequence_nll(z_flat, site_lon=site_lon, site_lat=site_lat)

    assert torch.isfinite(nll_tensor)
    assert torch.isfinite(nll_flat)
    assert torch.allclose(nll_tensor, nll_flat, atol=1e-6)


def test_transition_matrix_uses_component_pair_kernels():
    model = IDEStateSpaceModel(num_sites=3, dt=0.5, total_steps=8, param_window=2)
    site_lon = torch.tensor([120.0, 120.1, 120.2]).unsqueeze(0)
    site_lat = torch.tensor([30.0, 30.1, 30.2]).unsqueeze(0)
    sigma = torch.tensor(
        [
            [0.40, 0.05, 0.18, -0.04],
            [0.05, 0.30, 0.02, 0.08],
            [0.18, 0.02, 0.35, 0.06],
            [-0.04, 0.08, 0.06, 0.28],
        ],
        dtype=site_lon.dtype,
    ).unsqueeze(0)
    dynamics_t = {
        "mu": torch.tensor([[1.2, 0.4, -0.3, 0.8]], dtype=site_lon.dtype),
        "sigma": sigma,
    }

    kernels = model.build_component_kernels(
        site_lon=site_lon,
        site_lat=site_lat,
        dynamics_t=dynamics_t,
        device=site_lon.device,
        dtype=site_lon.dtype,
    )
    A, Q = model.build_transition_matrix(
        site_lon=site_lon,
        site_lat=site_lat,
        dynamics_t=dynamics_t,
        transition_idx=torch.tensor([3]),
        device=site_lon.device,
        dtype=site_lon.dtype,
    )

    assert kernels.shape == (1, 2, 2, 3, 3)
    assert A.shape == (1, 6, 6)
    assert Q.shape == (1, 6, 6)
    assert not torch.allclose(kernels[:, 0, 0], kernels[:, 0, 1])
    assert torch.all(Q.diagonal(dim1=-2, dim2=-1) > 0)


def test_advection_mean_net_outputs_matrix_mean_and_spd_covariance():
    model = AdvectionMeanNet()
    x_seq = torch.randn(2, 4, 6, 8, 8)
    out = model(x_seq)

    assert out["mu"].shape == (2, 4)
    assert out["mu_matrix"].shape == (2, 2, 2)
    assert out["sigma"].shape == (2, 4, 4)
    assert out["chol_factor"].shape == (2, 4, 4)
    assert torch.allclose(out["mu"], out["mu_matrix"].reshape(2, 4), atol=1e-6)

    sigma = out["sigma"]
    assert torch.allclose(sigma, sigma.transpose(-1, -2), atol=1e-6)
    assert torch.all(torch.linalg.eigvalsh(sigma) > 0)


def test_forecast_multistep_runs_with_dynamic_advection_only():
    model = IDEStateSpaceModel(num_sites=3, total_steps=16, param_window=2)
    z_hist = torch.randn(2, 6, 3, 2)
    site_lon = torch.tensor([120.0, 120.1, 120.2]).unsqueeze(0).expand(2, -1)
    site_lat = torch.tensor([30.0, 30.1, 30.2]).unsqueeze(0).expand(2, -1)

    dynamics_hist = {
        "mu": torch.randn(2, 5, 4),
        "sigma": 0.05 * torch.eye(4).reshape(1, 1, 4, 4).expand(2, 5, -1, -1),
    }
    dynamics_future = {
        "mu": torch.randn(2, 3, 4),
        "sigma": 0.05 * torch.eye(4).reshape(1, 1, 4, 4).expand(2, 3, -1, -1),
    }

    pred = model.forecast_multistep(
        z_hist=z_hist,
        site_lon=site_lon,
        site_lat=site_lat,
        dynamics_hist=dynamics_hist,
        dynamics_future=dynamics_future,
        start_idx=torch.tensor([2, 4]),
    )

    assert pred.shape == (2, 3, 3, 2)


def test_time_varying_ide_params_follow_absolute_index():
    model = IDEStateSpaceModel(num_sites=3, total_steps=8, param_window=2)
    with torch.no_grad():
        model.log_q_proc_knots.copy_(torch.tensor([-2.0, -1.0, 0.0, 1.0]))

    start_idx = torch.tensor([0, 4])
    gathered = model._get_time_params(start_idx[:, None])["q_proc"][:, 0]
    expected = torch.tensor([torch.exp(torch.tensor(-2.0)), torch.exp(torch.tensor(0.0))])
    assert torch.allclose(gathered, expected)


def test_legacy_coupling_key_is_ignored_on_load():
    model = IDEStateSpaceModel(num_sites=3, total_steps=8, param_window=2)
    state_dict = model.state_dict()
    state_dict["coupling_raw"] = torch.zeros(2, 2)

    model.load_state_dict(state_dict, strict=True)
    assert not hasattr(model, "coupling_raw")


def test_legacy_length_scale_keys_are_ignored_on_load():
    model = IDEStateSpaceModel(num_sites=3, total_steps=8, param_window=2)
    state_dict = model.state_dict()
    state_dict["log_ell_par"] = torch.tensor(0.3)
    state_dict["log_ell_perp"] = torch.tensor(-0.2)

    model.load_state_dict(state_dict, strict=True)
    assert float(model.ell_par) == 1.0
    assert float(model.ell_perp) == 1.0
