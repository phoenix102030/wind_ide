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

## Evaluation

After training, evaluate one-step Kalman forecasts against held-out or online
observations:

```bash
python train/evaluate_vector.py \
  --config yml_files/VectorMIDE_cuda.yaml \
  --checkpoint checkpoints/vector_mide_offline_cuda.pt \
  --split online
```

The script reports RMSE/MAE overall, separately for U and V, per station and
component, Kalman NLL per observation, and a persistence baseline where
``Z_hat[t] = Z[t-1]``.

Evaluation artifacts are saved by default under:

```text
outputs/evaluation/<checkpoint-name>_<split>/
```

Key files:

- `results.json`: metrics and metadata.
- `forecasts.npz`: target, model prediction, persistence prediction.
- `transition_matrices.npz`: all evaluated transition matrices `M[t,6,6]`.
- `advection_parameters.npz`: `mu`, `Sigma`, `A`, `ell`, `Q`, `R`, station coordinates.
- `time_parameters.csv`: flattened time-series parameters for quick inspection.
- `plots/transition_matrix.gif`: animated transition matrix over sampled times.
- `plots/*.png`: parameter time-series and heatmaps.

Offline training saves the best checkpoint during the configured monitor stage
to `offline_checkpoint_name` and the final epoch checkpoint to
`last_offline_checkpoint_name`. By default, the best checkpoint is selected from
the joint finetuning stage using `loss_kf`.

## Neural Encoder

The default encoder is now:

```text
NWP maps [T,C,H,W]
  -> CNN spatial encoder
  -> temporal Transformer encoder
  -> separate mu / Cholesky / A heads
```

Use `network_type: cnn_transformer` for the temporal model or `network_type:
cnn` for the older independent-map baseline. `transformer_causal: true` keeps
the encoder online-safe by preventing each time step from attending to future
NWP maps.

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
