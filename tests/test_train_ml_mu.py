import torch

from src.models.advection_mean_net import AdvectionMeanNet
from src.models.ide_state_space import IDEStateSpaceModel
from src.trainers.train_ml_mu import _deterministic_prediction_nll, build_dynamics_sequence


def test_build_dynamics_sequence_chunked_matches_stepwise():
    torch.manual_seed(0)
    model = AdvectionMeanNet(
        in_channels=6,
        hidden_dim=8,
        embed_dim=8,
        num_heads=2,
        num_layers=1,
        ff_dim=16,
        dropout=0.0,
    )
    model.eval()

    nwp_seq_full = torch.randn(2, 7, 6, 4, 4)
    seq_len = 4
    keys = ("mu", "base_scales", "transport_gates", "state_bias", "sigma")

    with torch.no_grad():
        got = build_dynamics_sequence(model, nwp_seq_full, seq_len=seq_len, window_batch_size=2)
        outputs = []
        for end_t in range(seq_len - 1, nwp_seq_full.shape[1]):
            outputs.append(model(nwp_seq_full[:, end_t - seq_len + 1:end_t + 1]))

    expected = {key: torch.stack([out[key] for out in outputs], dim=1) for key in keys}
    for key in keys:
        assert torch.allclose(got[key], expected[key], atol=1e-6), key


def test_deterministic_prediction_nll_matches_sequence_nll():
    torch.manual_seed(0)
    model = IDEStateSpaceModel(num_sites=3, total_steps=8, param_window=2)

    z_seq = torch.randn(2, 5, 3, 2)
    site_lon = torch.tensor([120.0, 120.1, 120.2]).unsqueeze(0).expand(2, -1)
    site_lat = torch.tensor([30.0, 30.1, 30.2]).unsqueeze(0).expand(2, -1)
    dynamics_seq = {
        "mu": torch.randn(2, 4, 4),
        "base_scales": torch.ones(2, 4, 2),
        "transport_gates": torch.zeros(2, 4, 2),
        "state_bias": torch.zeros(2, 4, 6),
        "sigma": 0.05 * torch.eye(4).reshape(1, 1, 4, 4).expand(2, 4, -1, -1),
    }
    start_idx = torch.tensor([0, 1], dtype=torch.long)

    with torch.no_grad():
        preds = model.deterministic_predict_sequence(
            z_seq=z_seq,
            site_lon=site_lon,
            site_lat=site_lat,
            dynamics_seq=dynamics_seq,
            start_idx=start_idx,
            apply_damping=True,
        )
        expected = model.deterministic_sequence_nll(
            z_seq=z_seq,
            site_lon=site_lon,
            site_lat=site_lat,
            dynamics_seq=dynamics_seq,
            start_idx=start_idx,
            apply_damping=True,
        )
        got = _deterministic_prediction_nll(
            raw_ide=model,
            preds=preds.reshape(preds.shape[0], preds.shape[1], model.state_dim),
            target=z_seq[:, 1:].reshape(z_seq.shape[0], z_seq.shape[1] - 1, model.state_dim),
            start_idx=start_idx,
            dtype=z_seq.dtype,
        )

    assert torch.allclose(got, expected, atol=1e-6)
