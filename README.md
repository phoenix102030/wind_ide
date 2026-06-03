# VectorMIDE

This project implements a single-height bivariate vector-wind version of DeepMIDE
for 140m wind components. The latent state is ordered as:

```text
[U(s1), U(s2), U(s3), V(s1), V(s2), V(s3)]
```

The neural network maps NWP grids to a four-dimensional component-advection
distribution and a 2x2 component mixing matrix `alpha`. In the default component mode,
the four advection entries are ordered as `[u_x, u_y, v_x, v_y]`: two
two-dimensional spatial displacements for the U and V component fields. A
Lagrangian kernel turns those outputs into a base transition matrix. In
residual-NWP mode, the model can then blend that base transition with identity
persistence, decay the residual toward zero, and add a small learned control
term before the Cholesky-based Kalman filter likelihood.

## Main Files

- `model/vector_attcnn.py`: CNN/attention head for `(mu, Sigma, alpha)` plus optional transition modifiers.
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
  --split online \
  --forecast-horizon 12
```

The script reports RMSE/MAE overall, separately for U and V, per station and
component, Kalman NLL per observation, a measurement persistence baseline, a
nearest-grid NWP baseline, and an NWP-plus-persisted-residual baseline. It also
reports multi-step forecasts from horizon 1 to `forecast_horizon`; for horizon
`h`, the model filters through time `t` and then forecasts `t+h` without using
observations from `t+1 ... t+h`.

Evaluation artifacts are saved by default under:

```text
outputs/evaluation/<checkpoint-name>_<split>/
```

Key files:

- `results.json`: metrics and metadata.
- `forecasts.npz`: measurement target, residual target, NWP baseline, model prediction, and baselines.
- `multi_step_metrics.npz`: RMSE/MAE curves for model vs persistence/NWP baselines by horizon.
- `transition_matrices.npz`: all evaluated transition matrices `M[t,6,6]`.
- `advection_parameters.npz`: `mu`, `Sigma`, component mixing `alpha`, optional shared-flow `flow_mu`/`flow_Sigma`, deformation `B`, transition modifiers, `ell`, `Q`, `R`, station coordinates.
- `time_parameters.csv`: flattened time-series parameters for quick inspection.
- `plots/transition_matrix.gif`: animated transition matrix over sampled times.
- `plots/*.png`: parameter time-series and heatmaps.

Offline training saves the best checkpoint during the configured monitor stage
to `offline_checkpoint_name` and the final epoch checkpoint to
`last_offline_checkpoint_name`. By default, the best checkpoint is selected from
the joint finetuning stage using fixed validation-window `val_loss_forecast`.

Validation is configurable in YAML:

```yaml
validation_enabled: true
validation_fraction: 0.15
validation_window_size: null
validation_num_windows: 8
validation_every_epochs: 5
checkpoint_metric: val_loss_forecast
```

The validation segment is taken from the tail of the offline split, while
training windows are sampled from the preceding segment.

For multi-step forecasting experiments, the joint stage can add a direct
forecast loss from filtered state at time `t` to future observations:

```yaml
lambda_multistep: 0.2
multistep_horizons: [1, 2, 3, 6, 9, 12]
multistep_max_origins: 256
multistep_stages: [joint, online]
checkpoint_metric: val_loss_forecast
```

`loss_forecast` is `loss_kf + lambda_multistep * loss_multistep`, so checkpoint
selection stays focused on Kalman fit plus multi-step forecast quality, without
including auxiliary advection or regularization terms.

## Residual Target

The default target is now the residual to the nearest-grid 140m NWP baseline:

```text
b_t = NWP_140m(nearest_grid(s1,s2,s3), t)
r_t = y_t - b_t
```

The Kalman/IDE model trains on `r_t`, while advection labels are still computed
from the original NWP fields exactly as before. Forecasts are reported back in
measurement space:

```text
y_hat_{t+h|t} = b_{t+h} + r_hat_{t+h|t}
```

This is enabled by:

```yaml
data:
  target_mode: residual_nwp
```

For residual targets, the default transition is:

```text
r_{t+1} = rho_t * ((1 - pi_t) I + pi_t M_base,t) r_t + c_t + noise
```

`M_base,t` is still the original Lagrangian spatial kernel. `pi_t` lets the
model keep short horizons close to persistence, `rho_t` lets long horizons
return toward the NWP baseline, and `c_t` is a small NWP-dependent residual
control initialized to zero. The defaults are conservative:

```yaml
transition_kernel_weight: true
transition_kernel_weight_init: 0.2
transition_residual_decay: true
transition_residual_decay_init: 0.97
transition_control: true
transition_control_scale: 0.5
```

## Online Adaptation

Online adaptation starts from an offline checkpoint and updates only the
adaptation-sensitive parameters by default: neural output heads, `Q/R`, and
optionally `ell`.

```bash
python train/train_vector_online.py \
  --config yml_files/VectorMIDE_cuda.yaml \
  --checkpoint outputs/cnn_transformer_new/vector_mide_offline_cuda.pt
```

The script saves two checkpoints:

- `online_checkpoint_name`: best online checkpoint selected by validation loss.
- `last_online_checkpoint_name`: final adapted checkpoint after the last update.

Online validation uses the future window immediately after each adapted training
window, so it is a model-selection signal rather than a strict deployment
metric. The interval and monitor are configurable:

```yaml
online_validation_window_size: 168
online_validation_every_updates: 10
online_validation_gap: 0
online_checkpoint_metric: val_loss_forecast
online_early_stop_patience: 0
online_min_delta: 0.0
```

## Full-Grid Residual Analysis Model

The original VectorMIDE model can be extended to the full NWP grid without
changing its three-station training objective. First train/evaluate VectorMIDE
as usual, then use the same learned IDE kernel to map the filtered/predicted
three-station residual state to all 40x40 grid locations:

```bash
python train/infer_vector_grid_residual.py \
  --config yml_files/VectorMIDE_cuda.yaml \
  --checkpoint checkpoints/vector_mide_offline_cuda.pt \
  --split offline

python train/visualize_vector_grid_residual.py \
  --npz outputs/vector_grid_residual/vector_mide_offline_cuda_offline/grid_residual_extension.npz \
  --state prediction \
  --num-points 40
```

The exported file contains both `prediction_*` fields, matching the one-step
Kalman prior used for station forecast metrics, and `analysis_*` fields, which
use same-time station observations after filtering. This path keeps the
three-point IDE/Kalman model as the primary model; off-station residuals are a
spatial extension of that model, not a replacement.

A separate experimental analysis-correction CNN is also available. It trains a
full-grid residual field directly while applying supervised loss only at the
three measurement stations:

```bash
python train/train_grid_residual.py --config yml_files/GridResidual_u140.yaml --dry-run
python train/train_grid_residual.py --config yml_files/GridResidual_u140.yaml
python train/train_grid_residual.py --config yml_files/GridResidual_v140.yaml
```

The model input combines NWP fields, projected grid coordinates, station
residual broadcasts, station distance fields, wind-alignment features, and
advective weights. The V0 objective is:

```text
loss = loss_obs + 0.05 * loss_smooth + 0.01 * loss_prior
```

where `loss_obs` is computed after bilinear sampling the predicted residual
grid back to the three stations. No fake full-grid residual labels are built.
Leave-one-station-out runs can be launched by withholding one station from both
the residual conditioning channels and supervised loss:

```bash
python train/train_grid_residual.py \
  --config yml_files/GridResidual_u140.yaml \
  --holdout-station 2
```

Evaluate a checkpoint at station locations with raw-NWP skill metrics:

```bash
python train/evaluate_grid_residual.py \
  --checkpoint checkpoints/grid_residual_u140.pt \
  --split offline
```

Export full-grid residual and corrected NWP fields:

```bash
python train/infer_grid_residual.py \
  --checkpoint checkpoints/grid_residual_u140.pt \
  --split online
```

After training both U and V residual models, visualize corrected wind speed and
direction maps, representative grid-point time series, and station comparisons:

```bash
python train/visualize_grid_residual.py \
  --u-checkpoint checkpoints/grid_residual_u140.pt \
  --v-checkpoint checkpoints/grid_residual_v140.pt \
  --split offline \
  --num-points 40
```

## Neural Encoder

The default encoder is now:

```text
NWP maps [T,C,H,W]
  -> CNN spatial encoder
  -> temporal Transformer encoder
  -> separate mu / Cholesky / alpha heads
```

Use `network_type: cnn_transformer` for the temporal model or `network_type:
cnn` for the older independent-map baseline. `transformer_causal: true` keeps
the encoder online-safe by preventing each time step from attending to future
NWP maps.

## Data

The current `data/` folder is expected to contain:

```text
data/measurement/wv_h100_180_offline.mat
data/measurement/wv_h100_180_offline_imputed.mat
data/measurement/wv_h100_180_online.mat
data/measurement/wv_h100_180_online_imputed.mat
data/nwp/data_grid_offline.mat
data/nwp/data_grid_online.mat
```

Measurement rows are converted to raw `Y` using the 140m `U,V` columns. In
`target_mode: residual_nwp`, model target `Z` is `Y - nwp_baseline`. NWP maps use
the channels `[u100, v100, u140, v140, u180, v180]` by default.
Training configs point to the imputed measurement files, and the loader also
prefers a same-directory `*_imputed.mat` sibling by default when a raw
measurement path is configured. If the resolved measurement file still contains
missing values, training fails loudly unless `data.measurement_missing_policy:
interpolate` is set.
With `advection_mode: component`, the model predicts
`mu=[u_x,u_y,v_x,v_y]` and a full 4x4 advection covariance. The U rows of the
transition matrix use the U-field 2D shift, and the V rows use the V-field 2D
shift. With the default one-step `time_mode: target_only`, cross-component
blocks also use the target component's projected advection, corresponding to
the relative-time convention `t1=dt, t2=0`. The same projection is used for both
the kernel mean shift and the projected covariance inside the dispersion
matrix. Cross-component interaction strength is represented by the learned 2x2
mixing weights `alpha`; the advection itself remains a spatial displacement in the
same two-dimensional coordinate system as `s_i - s_j`. The optional
source-side term `dt * gamma * E_source` can be enabled for full
cross-component projection experiments. The default advection label mode is
`optical_flow`, which estimates separate component-field motion targets for the
NWP 140m U and V fields. The older `simple` label mode uses the same local NWP
wind displacement as the pseudo-target for both component fields and can
collapse `A_u` and `A_v` toward the same learned mean and covariance; keep it
only as a shared-flow baseline.
The older `shared_flow_deformation` and `shared_flow_component_kernel` modes
are retained for ablation experiments.

The measurement `.mat` files include `LatValue_vec` and `LonValue_vec`, and the
loader uses them by default for the three station coordinates. You can override
that with `station_latlon` or `station_grid_indices` in
`yml_files/VectorMIDE.yaml`.
