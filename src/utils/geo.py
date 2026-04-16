import torch


def make_grid_coordinates(lat, lon):
    if lat.shape != lon.shape:
        raise ValueError(f"lat/lon mismatch: {lat.shape} vs {lon.shape}")
    return torch.stack([lon.reshape(-1), lat.reshape(-1)], dim=-1)