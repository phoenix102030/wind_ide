import numpy as np
import torch

from dataset.vector_data_utils import build_z_from_measurements
from model.vector_attcnn import VectorAdvectionNet
from model.vector_dstm import VectorMIDE
from model.vector_kernel import VectorLagrangianKernel


def test_advection_net_shapes_and_constraints():
    T = 5
    x = torch.randn(T, 6, 40, 40)
    net = VectorAdvectionNet(in_channels=6, hidden_dim=32)
    out = net(x)

    assert out["mu"].shape == (T, 4)
    assert out["L"].shape == (T, 4, 4)
    assert out["Sigma"].shape == (T, 4, 4)
    assert out["A"].shape == (T, 2, 2)
    assert torch.allclose(out["A"].sum(dim=-1), torch.ones(T, 2), atol=1.0e-5)
    assert torch.all(torch.linalg.eigvalsh(out["Sigma"]) > 0)


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


def test_measurement_columns_build_140m_state_order():
    ws_uv = np.arange(2 * 18, dtype=np.float32).reshape(2, 18)
    z = build_z_from_measurements(ws_uv)

    assert z.shape == (2, 6)
    np.testing.assert_array_equal(z[0], ws_uv[0, [6, 8, 10, 7, 9, 11]])
