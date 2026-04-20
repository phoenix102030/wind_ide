import math

import torch

from src.models.advection_mean_net import AdvectionMeanNet
from src.models.ide_state_space import IDEStateSpaceModel


def _joint_sigma(diag_values, cross_block=None, dtype=torch.float32):
    sigma = torch.zeros(4, 4, dtype=dtype)
    sigma[:2, :2] = torch.diag(torch.tensor(diag_values[:2], dtype=dtype))
    sigma[2:, 2:] = torch.diag(torch.tensor(diag_values[2:], dtype=dtype))
    if cross_block is not None:
        cross = torch.tensor(cross_block, dtype=dtype)
        sigma[:2, 2:] = cross
        sigma[2:, :2] = cross.transpose(0, 1)
    return sigma.unsqueeze(0)


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


def test_transition_matrix_uses_pair_specific_kernels():
    model = IDEStateSpaceModel(num_sites=3, dt=0.5, total_steps=8, param_window=2)
    site_lon = torch.tensor([120.0, 120.1, 120.2]).unsqueeze(0)
    site_lat = torch.tensor([30.0, 30.1, 30.2]).unsqueeze(0)
    dynamics_t = {
        "mu": torch.tensor([[1.2, 0.4, -0.3, 0.8]], dtype=site_lon.dtype),
        "sigma": _joint_sigma([0.40, 0.30, 0.25, 0.35], cross_block=[[0.05, 0.01], [0.02, 0.04]], dtype=site_lon.dtype),
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
    assert not torch.allclose(kernels[:, 0, 0], kernels[:, 1, 1])
    assert not torch.allclose(kernels[:, 0, 1], kernels[:, 1, 0])
    assert torch.all(Q.diagonal(dim1=-2, dim2=-1) > 0)


def test_cross_component_kernel_prefers_same_site_when_distance_is_zero():
    model = IDEStateSpaceModel(num_sites=3, dt=1.0, total_steps=8, param_window=2)
    site_lon = torch.tensor([120.0, 120.1, 120.2]).unsqueeze(0)
    site_lat = torch.tensor([30.0, 30.1, 30.2]).unsqueeze(0)
    dynamics_t = {
        "mu": torch.zeros(1, 4, dtype=site_lon.dtype),
        "sigma": torch.zeros(1, 4, 4, dtype=site_lon.dtype),
    }

    kernels = model.build_component_kernels(
        site_lon=site_lon,
        site_lat=site_lat,
        dynamics_t=dynamics_t,
        device=site_lon.device,
        dtype=site_lon.dtype,
    )

    uv_kernel = kernels[0, 0, 1]
    assert torch.all(torch.diagonal(uv_kernel) >= uv_kernel.max(dim=-1).values - 1e-6)


def test_transition_matrix_respects_site_component_state_order():
    model = IDEStateSpaceModel(
        num_sites=3,
        dt=1.0,
        total_steps=4,
        param_window=1,
        damping_min=1e-6,
        damping_max=1.0,
    )
    site_lon = torch.tensor([120.0, 120.1, 120.2]).unsqueeze(0)
    site_lat = torch.tensor([30.0, 30.1, 30.2]).unsqueeze(0)
    kernels = torch.arange(36, dtype=site_lon.dtype).reshape(1, 2, 2, 3, 3)

    with torch.no_grad():
        model.log_damping_knots.fill_(-20.0)

    model.build_component_kernels = lambda **kwargs: kernels
    A, _ = model.build_transition_matrix(
        site_lon=site_lon,
        site_lat=site_lat,
        dynamics_t={"mu": torch.zeros(1, 4), "sigma": torch.zeros(1, 4, 4)},
        transition_idx=torch.tensor([0]),
        device=site_lon.device,
        dtype=site_lon.dtype,
    )

    damping = model._get_time_params(torch.tensor([[0]]))["damping"][:, 0].to(dtype=site_lon.dtype)
    operator = kernels.permute(0, 3, 1, 4, 2).reshape(1, model.state_dim, model.state_dim)
    operator = operator / operator.abs().sum(dim=-1, keepdim=True).clamp_min(1.0)
    expected = (1.0 - model.dt * damping)[:, None, None] * operator

    assert torch.allclose(A, expected)


def test_transition_matrix_contracts_constant_mode_for_row_stochastic_operator():
    model = IDEStateSpaceModel(
        num_sites=3,
        dt=1.0,
        total_steps=4,
        param_window=1,
        damping_min=1e-6,
        damping_max=1.0,
    )
    site_lon = torch.tensor([120.0, 120.1, 120.2]).unsqueeze(0)
    site_lat = torch.tensor([30.0, 30.1, 30.2]).unsqueeze(0)

    with torch.no_grad():
        model.log_damping_knots.fill_(torch.log(torch.tensor(0.3)).item())

    A, _ = model.build_transition_matrix(
        site_lon=site_lon,
        site_lat=site_lat,
        dynamics_t={"mu": torch.zeros(1, 4), "sigma": torch.zeros(1, 4, 4)},
        transition_idx=torch.tensor([0]),
        device=site_lon.device,
        dtype=site_lon.dtype,
    )

    ones = torch.ones(1, model.state_dim, 1, dtype=site_lon.dtype)
    damping = model._get_time_params(torch.tensor([[0]]))["damping"][:, 0].to(dtype=site_lon.dtype)
    propagated = A @ ones
    expected = (1.0 - damping)[:, None, None] * ones
    assert torch.allclose(propagated, expected, atol=1e-5)


def test_transition_matrix_can_skip_damping_for_open_loop_forecast():
    model = IDEStateSpaceModel(
        num_sites=3,
        dt=1.0,
        total_steps=4,
        param_window=1,
        damping_min=1e-6,
        damping_max=1.0,
    )
    site_lon = torch.tensor([120.0, 120.1, 120.2]).unsqueeze(0)
    site_lat = torch.tensor([30.0, 30.1, 30.2]).unsqueeze(0)

    with torch.no_grad():
        model.log_damping_knots.fill_(torch.log(torch.tensor(0.8)).item())

    A, _ = model.build_transition_matrix(
        site_lon=site_lon,
        site_lat=site_lat,
        dynamics_t={"mu": torch.zeros(1, 4), "sigma": torch.zeros(1, 4, 4)},
        transition_idx=torch.tensor([0]),
        device=site_lon.device,
        dtype=site_lon.dtype,
        apply_damping=False,
    )

    ones = torch.ones(1, model.state_dim, 1, dtype=site_lon.dtype)
    propagated = A @ ones
    assert torch.allclose(propagated, ones, atol=1e-5)


def test_component_kernels_remain_finite_for_large_joint_sigma():
    model = IDEStateSpaceModel(num_sites=3, dt=1.0, total_steps=8, param_window=2)
    site_lon = torch.tensor([120.0, 120.1, 120.2]).unsqueeze(0)
    site_lat = torch.tensor([30.0, 30.1, 30.2]).unsqueeze(0)
    sigma = _joint_sigma([1e4, 2e4, 1.5e4, 2.5e4], cross_block=[[5e3, 1e3], [1e3, 4e3]], dtype=site_lon.dtype)

    kernels = model.build_component_kernels(
        site_lon=site_lon,
        site_lat=site_lat,
        dynamics_t={"mu": torch.zeros(1, 4), "sigma": sigma},
        device=site_lon.device,
        dtype=site_lon.dtype,
    )

    assert torch.isfinite(kernels).all()
    assert torch.allclose(kernels[:, 0, 0].sum(dim=-1), torch.ones_like(kernels[:, 0, 0].sum(dim=-1)), atol=1e-5)


def test_legacy_two_dimensional_dynamics_are_still_accepted():
    model = IDEStateSpaceModel(num_sites=3, dt=1.0, total_steps=8, param_window=2)
    site_lon = torch.tensor([120.0, 120.1, 120.2]).unsqueeze(0)
    site_lat = torch.tensor([30.0, 30.1, 30.2]).unsqueeze(0)
    dynamics_t = {
        "mu": torch.tensor([[1.0, -0.5]], dtype=site_lon.dtype),
        "sigma": 0.05 * torch.eye(2, dtype=site_lon.dtype).unsqueeze(0),
    }

    kernels = model.build_component_kernels(
        site_lon=site_lon,
        site_lat=site_lat,
        dynamics_t=dynamics_t,
        device=site_lon.device,
        dtype=site_lon.dtype,
    )

    assert kernels.shape == (1, 2, 2, 3, 3)
    assert torch.isfinite(kernels).all()


def test_advection_mean_net_outputs_four_dimensional_mean_and_spd_covariance():
    model = AdvectionMeanNet()
    x_seq = torch.randn(2, 4, 6, 8, 8)
    out = model(x_seq)

    assert out["mu"].shape == (2, 4)
    assert out["mu_pairs"].shape == (2, 2, 2)
    assert out["base_scales"].shape == (2, 2)
    assert out["sigma"].shape == (2, 4, 4)
    assert out["chol_factor"].shape == (2, 4, 4)
    assert torch.all(out["base_scales"] > 0)
    assert torch.allclose(out["sigma"], out["sigma"].transpose(-1, -2), atol=1e-6)
    assert torch.all(torch.linalg.eigvalsh(out["sigma"]) > 0)


def test_advection_mean_net_free_mode_outputs_direct_pair_advections():
    model = AdvectionMeanNet(mu_mode="free")
    with torch.no_grad():
        model.mu_head.weight.zero_()
        model.mu_head.bias.copy_(torch.tensor([1.0, -0.5, 2.0, -3.0]))

    x_seq = torch.randn(2, 4, 6, 8, 8)
    out = model(x_seq)
    expected = model.mu_scale * torch.tanh(torch.tensor([1.0, -0.5, 2.0, -3.0]))
    assert torch.allclose(out["mu"], expected.reshape(1, 4).expand_as(out["mu"]))


def test_advection_mean_net_can_anchor_mu_and_share_global_sigma():
    model = AdvectionMeanNet(mu_mode="anchored", sigma_mode="global", mu_scale=0.5, init_global_sigma_diag=0.15)
    with torch.no_grad():
        model.mu_head.weight.zero_()
        model.mu_head.bias.zero_()

    x_seq = torch.randn(2, 4, 6, 8, 8)
    x_seq[:, -1, 2] = 3.0
    x_seq[:, -1, 3] = -2.0

    out = model(x_seq)

    assert out["mu"].shape == (2, 4)
    assert out["mu_pairs"].shape == (2, 2, 2)
    assert out["mu_coeff_matrix"].shape == (2, 4, 2)
    assert torch.allclose(out["wind_anchor"], torch.tensor([[3.0, -2.0], [3.0, -2.0]]), atol=1e-6)
    assert torch.allclose(out["mu"], torch.zeros_like(out["mu"]), atol=1e-6)
    assert torch.allclose(out["sigma"][0], out["sigma"][1], atol=1e-6)
    assert torch.all(torch.linalg.eigvalsh(out["sigma"]) > 0)
    assert not any(param.requires_grad for param in model.chol_head.parameters())
    assert model.global_chol_params.requires_grad


def test_advection_mean_net_all12_uses_140m_uv_anchor_indices():
    model = AdvectionMeanNet(in_channels=12, mu_mode="anchored")
    x_seq = torch.zeros(1, 4, 12, 6, 6)
    x_seq[:, -1, 8] = 3.0
    x_seq[:, -1, 9] = -2.0

    out = model(x_seq)

    assert torch.allclose(out["wind_anchor"], torch.tensor([[3.0, -2.0]]), atol=1e-6)


def test_forecast_multistep_runs_with_dynamic_advection_only():
    model = IDEStateSpaceModel(num_sites=3, total_steps=16, param_window=2)
    z_hist = torch.randn(2, 6, 3, 2)
    site_lon = torch.tensor([120.0, 120.1, 120.2]).unsqueeze(0).expand(2, -1)
    site_lat = torch.tensor([30.0, 30.1, 30.2]).unsqueeze(0).expand(2, -1)

    dynamics_hist = {
        "mu": torch.randn(2, 5, 4),
        "base_scales": torch.ones(2, 5, 2),
        "sigma": 0.05 * torch.eye(4).reshape(1, 1, 4, 4).expand(2, 5, -1, -1),
    }
    dynamics_future = {
        "mu": torch.randn(2, 3, 4),
        "base_scales": torch.ones(2, 3, 2),
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


def test_forecast_multistep_remains_finite_with_nearly_singular_filter_covariance():
    model = IDEStateSpaceModel(
        num_sites=3,
        total_steps=16,
        param_window=2,
        q_proc_min=1e-8,
        q_proc_max=1e-6,
        r_obs_min=1e-8,
        r_obs_max=1e-6,
    )
    with torch.no_grad():
        model.log_q_proc_knots.fill_(math.log(1e-6))
        model.log_r_obs_knots.fill_(math.log(1e-6))
        model.log_p0_knots.fill_(math.log(1e-6))

    z_hist = torch.zeros(1, 6, 3, 2)
    site_lon = torch.tensor([120.0, 120.1, 120.2]).unsqueeze(0)
    site_lat = torch.tensor([30.0, 30.1, 30.2]).unsqueeze(0)
    dynamics_hist = {
        "mu": torch.zeros(1, 5, 4),
        "base_scales": torch.ones(1, 5, 2),
        "sigma": torch.zeros(1, 5, 4, 4),
    }
    dynamics_future = {
        "mu": torch.zeros(1, 3, 4),
        "base_scales": torch.ones(1, 3, 2),
        "sigma": torch.zeros(1, 3, 4, 4),
    }

    pred = model.forecast_multistep(
        z_hist=z_hist,
        site_lon=site_lon,
        site_lat=site_lat,
        dynamics_hist=dynamics_hist,
        dynamics_future=dynamics_future,
        start_idx=torch.tensor([0]),
    )

    assert torch.isfinite(pred).all()


def test_small_base_scales_produce_sharper_kernel_than_unit_floor():
    model = IDEStateSpaceModel(num_sites=3, dt=1.0, total_steps=8, param_window=2)
    site_lon = torch.tensor([120.0, 120.1, 120.2]).unsqueeze(0)
    site_lat = torch.tensor([30.0, 30.1, 30.2]).unsqueeze(0)

    wide = model.build_component_kernels(
        site_lon=site_lon,
        site_lat=site_lat,
        dynamics_t={
            "mu": torch.zeros(1, 4),
            "base_scales": torch.ones(1, 2),
            "sigma": torch.zeros(1, 4, 4),
        },
        device=site_lon.device,
        dtype=site_lon.dtype,
    )
    sharp = model.build_component_kernels(
        site_lon=site_lon,
        site_lat=site_lat,
        dynamics_t={
            "mu": torch.zeros(1, 4),
            "base_scales": torch.full((1, 2), 0.2),
            "sigma": torch.zeros(1, 4, 4),
        },
        device=site_lon.device,
        dtype=site_lon.dtype,
    )

    assert torch.all(torch.diagonal(sharp[0, 0, 0]) >= torch.diagonal(wide[0, 0, 0]))


def test_predict_sequence_keeps_gradients_for_advection_supervision():
    model = IDEStateSpaceModel(num_sites=3, dt=1.0, total_steps=8, param_window=2)
    z_seq = torch.randn(1, 4, 3, 2)
    site_lon = torch.tensor([120.0, 120.1, 120.2]).unsqueeze(0)
    site_lat = torch.tensor([30.0, 30.1, 30.2]).unsqueeze(0)
    dynamics_seq = {
        "mu": torch.zeros(1, 3, 4, requires_grad=True),
        "base_scales": torch.ones(1, 3, 2, requires_grad=True),
        "sigma": (0.05 * torch.eye(4).reshape(1, 1, 4, 4).expand(1, 3, -1, -1)).clone().requires_grad_(True),
    }

    pred = model.predict_sequence(
        z_seq=z_seq,
        site_lon=site_lon,
        site_lat=site_lat,
        dynamics_seq=dynamics_seq,
        start_idx=torch.tensor([0]),
    )
    loss = pred.square().mean()
    loss.backward()

    assert dynamics_seq["mu"].grad is not None
    assert dynamics_seq["base_scales"].grad is not None
    assert dynamics_seq["sigma"].grad is not None



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
    expected = torch.tensor([max(-0.5, math.log(global_model.q_proc_min))]).clamp_max(math.log(global_model.q_proc_max))
    assert torch.allclose(global_model.log_q_proc_knots, expected)


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


def test_legacy_row_selector_key_is_ignored_on_load():
    model = IDEStateSpaceModel(num_sites=3, total_steps=8, param_window=2)
    state_dict = model.state_dict()
    state_dict["row_selector"] = torch.eye(2)

    model.load_state_dict(state_dict, strict=True)


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
