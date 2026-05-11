import torch

from dataset.grid_residual_data_utils import (
    bilinear_sample_grid,
    build_grid_residual_features,
    feature_channel_count,
)
from model.grid_residual import GridResidualCNN
from train.grid_residual_losses import (
    distance_weighted_prior_loss,
    masked_mse,
    spatial_smoothness_loss,
)


def _static(height=4, width=5, n_stations=3):
    y, x = torch.meshgrid(torch.arange(height), torch.arange(width), indexing="ij")
    grid_x = x.float()
    grid_y = y.float()
    station_xy = torch.tensor([[0.5, 0.5], [2.0, 1.0], [3.0, 2.5]], dtype=torch.float32)
    distances = []
    for sx, sy in station_xy:
        distances.append(torch.sqrt((grid_x - sx) ** 2 + (grid_y - sy) ** 2))
    distance_km = torch.stack(distances)
    return {
        "grid_x": grid_x,
        "grid_y": grid_y,
        "coord_features": torch.stack([grid_x / width, grid_y / height]),
        "distance_km": distance_km,
        "distance_features": distance_km / distance_km.max().clamp_min(1.0e-6),
        "station_xy": station_xy,
        "sample_yx": torch.tensor([[0.5, 0.5], [1.0, 2.0], [2.5, 3.0]], dtype=torch.float32),
    }


def test_bilinear_sample_grid_center_value():
    field = torch.tensor([[[[[0.0, 2.0], [4.0, 6.0]]]]])
    sampled = bilinear_sample_grid(field, torch.tensor([[0.5, 0.5]]))

    assert sampled.shape == (1, 1, 1)
    assert torch.allclose(sampled, torch.tensor([[[3.0]]]))


def test_grid_residual_features_shape():
    bsz, steps, channels, height, width = 2, 3, 6, 4, 5
    n_stations = 3
    cfg = {
        "use_coords": True,
        "use_station_residual_broadcast": True,
        "use_distance_features": True,
        "use_advection_alignment": True,
        "use_advective_weight": True,
        "advective_length_scale_km": 10.0,
    }
    features = build_grid_residual_features(
        nwp_input=torch.randn(bsz, steps, channels, height, width),
        station_residuals=torch.randn(bsz, steps, n_stations),
        u=torch.ones(bsz, steps, height, width),
        v=torch.zeros(bsz, steps, height, width),
        static=_static(height, width, n_stations),
        feature_cfg=cfg,
    )

    assert features.shape == (
        bsz,
        steps,
        feature_channel_count(channels, n_stations, cfg),
        height,
        width,
    )
    assert torch.isfinite(features).all()


def test_grid_residual_model_and_losses_are_finite():
    bsz, steps, channels, height, width = 2, 2, 17, 4, 5
    model = GridResidualCNN(in_channels=channels, hidden_dim=16, num_blocks=2, max_residual=5.0)
    pred = model(torch.randn(bsz, steps, channels, height, width))
    sampled = bilinear_sample_grid(pred, _static(height, width)["sample_yx"])
    target = torch.randn_like(sampled)
    mask = torch.ones_like(sampled, dtype=torch.bool)
    distance = _static(height, width)["distance_km"]
    loss = (
        masked_mse(sampled, target, mask)
        + spatial_smoothness_loss(pred)
        + distance_weighted_prior_loss(pred, distance)
    )

    assert pred.shape == (bsz, steps, 1, height, width)
    assert torch.isfinite(loss)
