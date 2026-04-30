# VectorMIDE

This project implements a single-height bivariate vector-wind version of DeepMIDE
for 140m wind components. The latent state is ordered as:

```text
[U(s1), U(s2), U(s3), V(s1), V(s2), V(s3)]
```

The neural network maps NWP grids to a four-dimensional random advection
distribution and a 2x2 component mixing matrix. A Lagrangian kernel turns those
outputs into a time-varying transition matrix for a Cholesky-based Kalman
filter likelihood.

## Main Files

- `model/vector_attcnn.py`: CNN/attention head for `(mu, Sigma, A)`.
- `model/vector_kernel.py`: 4D random-advection Lagrangian transition kernel.
- `model/vector_dstm.py`: Kalman filtering, losses, and the combined model.
- `model/covariance.py`: Cholesky covariance utilities and losses.
- `dataset/vector_data_utils.py`: MATLAB data loading and vector wind assembly.
- `train/train_vector_offline.py`: offline pretraining/Kalman/finetuning flow.
- `train/train_vector_online.py`: rolling online adaptation flow.
- `yml_files/VectorMIDE.yaml`: default configuration.
- `tests/test_vector_shapes.py`: Stage 1 shape and finite-loss checks.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
pytest
```

## Device Selection

Use the YAML files to switch platforms:

```bash
# Mac local debugging
python train/train_vector_offline.py --config yml_files/VectorMIDE_mps.yaml --limit 128 --dry-run

# CUDA server, e.g. A100
python train/train_vector_offline.py --config yml_files/VectorMIDE_cuda.yaml

# Temporary override without editing YAML
python train/train_vector_offline.py --config yml_files/VectorMIDE.yaml --device cuda:0
```

`yml_files/VectorMIDE.yaml` keeps `device: auto`, which chooses CUDA first,
then MPS, then CPU. Set `allow_device_fallback: false` when you want the script
to fail loudly if the requested backend is unavailable.

## Data

The current `data/` folder is expected to contain:

```text
data/measurement/wv_h100_180_offline.mat
data/measurement/wv_h100_180_online.mat
data/nwp/data_grid_offline.mat
data/nwp/data_grid_online.mat
```

Measurement rows are converted to `Z` using the 140m `U,V` columns. NWP maps
use the channels `[u100, v100, u140, v140, u180, v180]` by default.

The measurement `.mat` files include `LatValue_vec` and `LonValue_vec`, and the
loader uses them by default for the three station coordinates. You can override
that with `station_latlon` or `station_grid_indices` in
`yml_files/VectorMIDE.yaml`.
