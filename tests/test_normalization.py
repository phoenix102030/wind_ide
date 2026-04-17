import numpy as np

from src.data.aligned_measurement_nwp import (
    AlignedDataBundle,
    apply_bundle_normalization,
    denormalize_nwp,
    denormalize_z,
    fit_bundle_normalization,
    normalization_stats_from_config,
)


def test_bundle_normalization_round_trip():
    rng = np.random.default_rng(0)
    z = rng.normal(loc=5.0, scale=3.0, size=(12, 3, 2)).astype(np.float32)
    nwp = rng.normal(loc=-2.0, scale=4.0, size=(12, 6, 4, 5)).astype(np.float32)
    bundle = AlignedDataBundle(
        z_meas=z,
        meas_lat=np.array([1.0, 2.0, 3.0], dtype=np.float32),
        meas_lon=np.array([4.0, 5.0, 6.0], dtype=np.float32),
        nwp_uv=nwp,
        nwp_lat=np.zeros((4, 5), dtype=np.float32),
        nwp_lon=np.zeros((4, 5), dtype=np.float32),
    )

    stats = fit_bundle_normalization(bundle, end_time=8)
    normalized = apply_bundle_normalization(bundle, stats)

    assert np.allclose(normalized.z_meas[:8].mean(axis=0), 0.0, atol=1e-5)
    assert np.allclose(normalized.nwp_uv[:8].mean(axis=(0, 2, 3)), 0.0, atol=1e-5)

    restored_z = denormalize_z(normalized.z_meas, stats)
    restored_nwp = denormalize_nwp(normalized.nwp_uv, stats)
    assert np.allclose(restored_z, bundle.z_meas, atol=1e-5)
    assert np.allclose(restored_nwp, bundle.nwp_uv, atol=1e-5)


def test_normalization_stats_round_trip_via_config():
    rng = np.random.default_rng(1)
    bundle = AlignedDataBundle(
        z_meas=rng.normal(size=(6, 3, 2)).astype(np.float32),
        meas_lat=np.array([1.0, 2.0, 3.0], dtype=np.float32),
        meas_lon=np.array([4.0, 5.0, 6.0], dtype=np.float32),
        nwp_uv=rng.normal(size=(6, 6, 2, 2)).astype(np.float32),
        nwp_lat=np.zeros((2, 2), dtype=np.float32),
        nwp_lon=np.zeros((2, 2), dtype=np.float32),
    )
    stats = fit_bundle_normalization(bundle)
    restored = normalization_stats_from_config({"normalization": stats.to_config_dict()})

    assert np.allclose(restored.z_mean, stats.z_mean)
    assert np.allclose(restored.z_std, stats.z_std)
    assert np.allclose(restored.nwp_mean, stats.nwp_mean)
    assert np.allclose(restored.nwp_std, stats.nwp_std)
