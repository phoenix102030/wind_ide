import numpy as np
import torch

from dataset.vector_data_utils import (
    build_z_from_measurements,
    impute_time_series_columns,
    imputed_measurement_path,
    resolve_measurement_path,
)
from model.vector_attcnn import VectorAdvectionNet
from model.vector_dstm import VectorMIDE
from model.vector_kernel import VectorLagrangianKernel
from train.train_vector_offline import training_loss_kwargs


def test_advection_net_shapes_and_constraints():
    T = 5
    x = torch.randn(T, 6, 40, 40)
    net = VectorAdvectionNet(
        in_channels=6,
        hidden_dim=32,
        network_type="cnn_transformer",
        transformer_d_model=32,
        transformer_nhead=4,
        transformer_layers=1,
        transformer_dim_feedforward=64,
    )
    out = net(x)

    assert out["mu"].shape == (T, 4)
    assert out["L"].shape == (T, 4, 4)
    assert out["Sigma"].shape == (T, 4, 4)
    assert out["A"].shape == (T, 2, 2)
    assert torch.allclose(out["A"].sum(dim=-1), torch.ones(T, 2), atol=1.0e-5)
    assert torch.all(torch.linalg.eigvalsh(out["Sigma"]) > 0)


def test_advection_net_cnn_mode_still_works():
    x = torch.randn(3, 6, 40, 40)
    net = VectorAdvectionNet(in_channels=6, hidden_dim=32, network_type="cnn")
    out = net(x)

    assert out["mu"].shape == (3, 4)
    assert out["A"].shape == (3, 2, 2)


def test_component_specific_mu_uses_shared_full_input_with_separate_heads():
    x = torch.randn(3, 6, 40, 40)
    net = VectorAdvectionNet(
        in_channels=6,
        hidden_dim=32,
        network_type="cnn",
        component_specific_mu=True,
    )
    out = net(x)

    assert out["mu"].shape == (3, 4)
    assert net.mu_head is None
    assert net.mu_u_head is not None
    assert net.mu_v_head is not None
    assert net.mu_u_head is not net.mu_v_head
    assert net.mu_u_head.in_features == net.mu_v_head.in_features


def test_shared_flow_deformation_outputs_and_transition_are_finite():
    T = 4
    x = torch.randn(T, 6, 40, 40)
    coords = torch.tensor([[0.0, 0.0], [3.0, 0.5], [1.5, 2.0]])
    model = VectorMIDE(
        n_sites=3,
        in_channels=6,
        hidden_dim=32,
        network_type="cnn",
        advection_mode="shared_flow_deformation",
        deformation_scale=0.3,
    )
    out = model(x, coords)

    assert out["flow_mu"].shape == (T, 2)
    assert out["flow_Sigma"].shape == (T, 2, 2)
    assert out["B"].shape == (T, 2, 2)
    assert torch.allclose(out["mu"][:, :2], out["mu"][:, 2:], atol=1.0e-6)
    assert torch.all(torch.linalg.eigvalsh(out["flow_Sigma"]) > 0)
    assert torch.allclose(
        out["B"],
        torch.eye(2).expand(T, 2, 2),
        atol=1.0e-6,
    )
    assert out["M"].shape == (T, 6, 6)
    assert torch.isfinite(out["M"]).all()
    assert torch.allclose(out["M_base"].sum(dim=-1), torch.ones(T, 6), atol=1.0e-4)


def test_advection_net_transition_heads_are_bounded_and_zero_control_init():
    x = torch.randn(3, 6, 40, 40)
    net = VectorAdvectionNet(
        in_channels=6,
        hidden_dim=32,
        network_type="cnn",
        transition_kernel_weight=True,
        transition_kernel_weight_init=0.2,
        transition_residual_decay=True,
        transition_residual_decay_init=0.97,
        transition_control_dim=6,
        transition_control_scale=0.5,
    )
    out = net(x)

    assert out["kernel_weight"].shape == (3, 1)
    assert out["residual_decay"].shape == (3, 1)
    assert out["transition_control"].shape == (3, 6)
    assert torch.all((0.0 <= out["kernel_weight"]) & (out["kernel_weight"] <= 1.0))
    assert torch.all((0.0 <= out["residual_decay"]) & (out["residual_decay"] <= 1.0))
    assert torch.allclose(out["kernel_weight"], torch.full((3, 1), 0.2), atol=1.0e-5)
    assert torch.allclose(out["residual_decay"], torch.full((3, 1), 0.97), atol=1.0e-5)
    assert torch.allclose(out["transition_control"], torch.zeros(3, 6), atol=1.0e-6)


def test_vector_kernel_transition_shape_and_row_sums():
    T = 4
    coords = torch.tensor([[0.0, 0.0], [3.0, 0.5], [1.5, 2.0]])
    mu = torch.randn(T, 4) * 0.1
    raw = torch.randn(T, 4, 4)
    sigma = raw @ raw.transpose(-1, -2) + 0.05 * torch.eye(4)
    A = torch.softmax(torch.randn(T, 2, 2), dim=-1)

    kernel = VectorLagrangianKernel(n_dim=3, dt=1.0, gamma=0.0)
    M = kernel(coords, mu, sigma, A)

    assert M.shape == (T, 6, 6)
    assert torch.allclose(M.sum(dim=-1), torch.ones(T, 6), atol=1.0e-4)
    assert torch.all(M >= 0)


def test_vector_kernel_vectorized_matches_single_step():
    T = 5
    coords = torch.tensor([[0.0, 0.0], [3.0, 0.5], [1.5, 2.0]])
    mu = torch.randn(T, 4) * 0.1
    raw = torch.randn(T, 4, 4)
    sigma = raw @ raw.transpose(-1, -2) + 0.05 * torch.eye(4)
    A = torch.softmax(torch.randn(T, 2, 2), dim=-1)

    kernel = VectorLagrangianKernel(n_dim=3, dt=1.0, gamma=0.0)
    vectorized = kernel(coords, mu, sigma, A)
    stepwise = torch.stack(
        [kernel.forward_single(coords, mu[t], sigma[t], A[t]) for t in range(T)],
        dim=0,
    )

    assert torch.allclose(vectorized, stepwise, atol=1.0e-6, rtol=1.0e-5)


def test_vector_mide_kalman_loss_is_finite():
    T = 6
    x = torch.randn(T, 6, 40, 40)
    z = torch.randn(T, 6)
    coords = torch.tensor([[0.0, 0.0], [3.0, 0.5], [1.5, 2.0]])

    model = VectorMIDE(n_sites=3, in_channels=6, hidden_dim=32)
    losses = model.training_losses(x=x, z=z, coords=coords, v_star=torch.randn(T, 4) * 0.1)

    assert torch.isfinite(losses["loss"])
    assert torch.isfinite(losses["loss_kf"])
    assert losses["M"].shape == (T, 6, 6)


def test_vector_mide_transition_modifiers_and_control_loss_are_finite():
    T = 6
    x = torch.randn(T, 6, 40, 40)
    z = torch.randn(T, 6)
    coords = torch.tensor([[0.0, 0.0], [3.0, 0.5], [1.5, 2.0]])

    model = VectorMIDE(
        n_sites=3,
        in_channels=6,
        hidden_dim=32,
        network_type="cnn",
        transition_kernel_weight=True,
        transition_kernel_weight_init=0.2,
        transition_residual_decay=True,
        transition_residual_decay_init=0.97,
        transition_control=True,
        transition_control_scale=0.5,
    )
    losses = model.training_losses(
        x=x,
        z=z,
        coords=coords,
        v_star=torch.randn(T, 4) * 0.1,
        lambda_multistep=0.1,
        multistep_horizons=[1, 3],
    )

    assert torch.isfinite(losses["loss"])
    assert losses["M"].shape == (T, 6, 6)
    assert losses["transition_control"].shape == (T, 6)
    assert torch.allclose(losses["M"].sum(dim=-1), torch.full((T, 6), 0.97), atol=1.0e-4)
    assert torch.allclose(losses["M_base"].sum(dim=-1), torch.ones(T, 6), atol=1.0e-4)


def test_measurement_columns_build_140m_state_order():
    ws_uv = np.arange(2 * 18, dtype=np.float32).reshape(2, 18)
    z = build_z_from_measurements(ws_uv)

    assert z.shape == (2, 6)
    np.testing.assert_array_equal(z[0], ws_uv[0, [6, 8, 10, 7, 9, 11]])


def test_measurement_path_prefers_imputed_sibling(tmp_path):
    raw_path = tmp_path / "wv_h100_180_offline.mat"
    imputed_path = tmp_path / "wv_h100_180_offline_imputed.mat"
    raw_path.touch()
    imputed_path.touch()

    configured, resolved = resolve_measurement_path(
        {"offline_measurement_path": str(raw_path)},
        "offline",
    )

    assert configured == raw_path
    assert resolved == imputed_path
    assert imputed_measurement_path(raw_path) == imputed_path
    assert imputed_measurement_path(imputed_path) == imputed_path


def test_measurement_path_can_disable_imputed_preference(tmp_path):
    raw_path = tmp_path / "wv_h100_180_offline.mat"
    imputed_path = tmp_path / "wv_h100_180_offline_imputed.mat"
    raw_path.touch()
    imputed_path.touch()

    _, resolved = resolve_measurement_path(
        {
            "offline_measurement_path": str(raw_path),
            "prefer_imputed_measurements": False,
        },
        "offline",
    )

    assert resolved == raw_path


def test_impute_time_series_columns_fills_all_missing_values():
    values = np.array(
        [
            [1.0, np.nan, np.nan],
            [np.nan, 2.0, np.nan],
            [3.0, np.nan, np.nan],
        ],
        dtype=np.float32,
    )

    filled = impute_time_series_columns(values)

    assert np.isfinite(filled).all()
    np.testing.assert_allclose(filled[:, 0], [1.0, 2.0, 3.0])
    np.testing.assert_allclose(filled[:, 1], [2.0, 2.0, 2.0])
    np.testing.assert_allclose(filled[:, 2], [0.0, 0.0, 0.0])


def test_online_training_loss_kwargs_can_override_offline_objective():
    config = {
        "lambda_adv": 0.1,
        "lambda_deform": 0.05,
        "lambda_multistep": 0.2,
        "multistep_stages": ["joint", "online"],
        "multistep_horizons": [1, 2, 3],
        "online_lambda_adv": 0.0,
        "online_lambda_deform": 0.0,
        "online_lambda_multistep": 0.0,
        "online_multistep_horizons": [1],
    }

    joint = training_loss_kwargs(config, "joint")
    online = training_loss_kwargs(config, "online")

    assert joint["lambda_adv"] == 0.1
    assert joint["lambda_deform"] == 0.05
    assert joint["lambda_multistep"] == 0.2
    assert online["lambda_adv"] == 0.0
    assert online["lambda_deform"] == 0.0
    assert online["lambda_multistep"] == 0.0
    assert online["multistep_horizons"] == [1]
