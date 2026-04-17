import torch
from src.models.cnn_transformer import CNNTransformerAdjuster
from src.models.ide_system import WindSpeedIDEModel


def test_adjuster_shape():
    model = CNNTransformerAdjuster()
    z = torch.randn(2, 4, 3, 40, 40)
    out = model(z)
    assert out.shape == (2, 2, 40, 40)


def test_joint_model_shape():
    model = WindSpeedIDEModel(local_radius=1)
    batch = {
        "y_t": torch.randn(1, 8, 8),
        "y_next": torch.randn(1, 8, 8),
        "u_t": torch.randn(1, 8, 8),
        "v_t": torch.randn(1, 8, 8),
        "z_seq": torch.randn(1, 4, 3, 8, 8),
        "lat": torch.randn(1, 8, 8),
        "lon": torch.randn(1, 8, 8),
    }
    pred, alpha, _ = model(batch)
    assert pred.shape == (1, 8, 8)
    assert alpha.shape == (1, 2, 8, 8)
