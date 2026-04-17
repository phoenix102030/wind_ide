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


def test_component_kernels_remain_finite_for_ill_conditioned_sigma():
    model = IDEStateSpaceModel(num_sites=3, dt=1.0, total_steps=8, param_window=2)
    site_lon = torch.tensor([120.0, 120.1, 120.2]).unsqueeze(0)
    site_lat = torch.tensor([30.0, 30.1, 30.2]).unsqueeze(0)
    sigma = torch.tensor(
        [
            [1e6, -1e6, 5e5, -5e5],
            [-1e6, 1e6, -5e5, 5e5],
            [5e5, -5e5, 2.5e5, -2.5e5],
            [-5e5, 5e5, -2.5e5, 2.5e5],
        ],
        dtype=site_lon.dtype,
    ).unsqueeze(0)

    kernels = model.build_component_kernels(
        site_lon=site_lon,
        site_lat=site_lat,
        dynamics_t={"mu": torch.zeros(1, 4), "sigma": sigma},
        device=site_lon.device,
        dtype=site_lon.dtype,
    )

    assert torch.isfinite(kernels).all()
    assert torch.allclose(kernels.sum(dim=-1), torch.ones_like(kernels.sum(dim=-1)), atol=1e-5)


def test_component_kernels_respond_to_learned_base_length_scales():
    site_lon = torch.tensor([120.0, 120.1, 120.2]).unsqueeze(0)
    site_lat = torch.tensor([30.0, 30.1, 30.2]).unsqueeze(0)
    dynamics_t = {
        "mu": torch.tensor([[1.0, 0.0, 1.0, 0.0]], dtype=site_lon.dtype),
        "sigma": torch.zeros(1, 4, 4, dtype=site_lon.dtype),
    }

    narrow = IDEStateSpaceModel(num_sites=3, dt=1.0, total_steps=8, param_window=2)
    wide = IDEStateSpaceModel(num_sites=3, dt=1.0, total_steps=8, param_window=2)
    with torch.no_grad():
        narrow.log_ell_par_knots.fill_(-1.5)
        narrow.log_ell_perp_knots.fill_(-1.5)
        wide.log_ell_par_knots.fill_(1.0)
        wide.log_ell_perp_knots.fill_(1.0)

    narrow_kernels = narrow.build_component_kernels(
        site_lon=site_lon,
        site_lat=site_lat,
        dynamics_t=dynamics_t,
        device=site_lon.device,
        dtype=site_lon.dtype,
        transition_idx=torch.tensor([0]),
    )
    wide_kernels = wide.build_component_kernels(
        site_lon=site_lon,
        site_lat=site_lat,
        dynamics_t=dynamics_t,
        device=site_lon.device,
        dtype=site_lon.dtype,
        transition_idx=torch.tensor([0]),
    )

    assert wide_kernels[:, 0, 0].amax() < narrow_kernels[:, 0, 0].amax()


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


def test_global_ide_params_are_shared_across_absolute_time():
    model = IDEStateSpaceModel(num_sites=3, total_steps=8, param_window=2, param_mode="global")
    with torch.no_grad():
        model.log_q_proc_knots.copy_(torch.tensor([-1.25]))

    gathered = model._get_time_params(torch.tensor([[0], [7]]))["q_proc"][:, 0]
    expected = torch.full((2,), torch.exp(torch.tensor(-1.25)))
    assert model.num_knots == 1
    assert torch.allclose(gathered, expected)


def test_loading_absolute_knots_into_global_mode_reduces_to_shared_value():
    absolute_model = IDEStateSpaceModel(num_sites=3, total_steps=8, param_window=2)
    with torch.no_grad():
        absolute_model.log_q_proc_knots.copy_(torch.tensor([-2.0, -1.0, 0.0, 1.0]))

    global_model = IDEStateSpaceModel(num_sites=3, total_steps=8, param_window=2, param_mode="global")
    state_dict = absolute_model.state_dict()
    global_model.load_state_dict(state_dict, strict=True)

    assert global_model.log_q_proc_knots.shape == (1,)
    assert torch.allclose(global_model.log_q_proc_knots, torch.tensor([-0.5]))


def test_damping_is_clamped_to_configured_range():
    model = IDEStateSpaceModel(num_sites=3, total_steps=8, param_window=2, damping_min=0.05, damping_max=1.0)
    with torch.no_grad():
        model.log_damping_knots.fill_(5.0)
        model.clamp_parameters_()
    assert torch.allclose(model.damping, torch.tensor(1.0), atol=1e-6)


def test_q_and_r_are_clamped_to_configured_ranges():
    model = IDEStateSpaceModel(
        num_sites=3,
        total_steps=8,
        param_window=2,
        q_proc_min=0.05,
        q_proc_max=0.4,
        r_obs_min=0.06,
        r_obs_max=0.7,
    )
    with torch.no_grad():
        model.log_q_proc_knots.fill_(5.0)
        model.log_r_obs_knots.fill_(5.0)
        model.clamp_parameters_()
    assert torch.allclose(model.q_proc, torch.tensor(0.4), atol=1e-6)
    assert torch.allclose(model.r_obs, torch.tensor(0.7), atol=1e-6)


def test_legacy_coupling_key_is_ignored_on_load():
    model = IDEStateSpaceModel(num_sites=3, total_steps=8, param_window=2)
    state_dict = model.state_dict()
    state_dict["coupling_raw"] = torch.zeros(2, 2)

    model.load_state_dict(state_dict, strict=True)
    assert not hasattr(model, "coupling_raw")


def test_legacy_global_ide_params_expand_to_knots():
    model = IDEStateSpaceModel(num_sites=3, total_steps=8, param_window=2)
    state_dict = model.state_dict()
    state_dict.pop("log_q_proc_knots")
    state_dict.pop("log_r_obs_knots")
    state_dict.pop("log_p0_knots")
    state_dict.pop("log_damping_knots")
    state_dict.pop("init_mean_knots")
    state_dict["log_q_proc"] = torch.tensor(-1.5)
    state_dict["log_r_obs"] = torch.tensor(-0.5)
    state_dict["log_p0"] = torch.tensor(0.25)
    state_dict["log_damping"] = torch.tensor(-0.75)
    state_dict["init_mean"] = torch.arange(6, dtype=torch.float32)

    model.load_state_dict(state_dict, strict=True)
    assert torch.allclose(model.log_q_proc_knots, torch.full((4,), -1.5))
    assert torch.allclose(model.log_r_obs_knots, torch.full((4,), -0.5))
    assert torch.allclose(model.log_p0_knots, torch.full((4,), 0.25))
    assert torch.allclose(model.log_damping_knots, torch.full((4,), -0.75))
    assert torch.allclose(model.init_mean_knots[0], torch.arange(6, dtype=torch.float32))


def test_legacy_length_scale_keys_are_ignored_on_load():
    model = IDEStateSpaceModel(num_sites=3, total_steps=8, param_window=2)
    state_dict = model.state_dict()
    state_dict["log_ell_par"] = torch.tensor(0.3)
    state_dict["log_ell_perp"] = torch.tensor(-0.2)
    state_dict.pop("log_ell_par_knots")
    state_dict.pop("log_ell_perp_knots")

    model.load_state_dict(state_dict, strict=True)
    assert torch.allclose(model.log_ell_par_knots, torch.full((4,), 0.3))
    assert torch.allclose(model.log_ell_perp_knots, torch.full((4,), -0.2))


def test_missing_length_scale_keys_fall_back_to_model_defaults():
    model = IDEStateSpaceModel(num_sites=3, total_steps=8, param_window=2)
    state_dict = model.state_dict()
    state_dict.pop("log_ell_par_knots")
    state_dict.pop("log_ell_perp_knots")

    model.load_state_dict(state_dict, strict=True)
    assert torch.allclose(model.log_ell_par_knots, torch.full((4,), 0.5))
    assert torch.allclose(model.log_ell_perp_knots, torch.full((4,), 0.0))
