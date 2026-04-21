import math

import torch
import torch.distributed as dist

from src.models.ide_state_space import project_lon_lat


def move_batch_to_device(batch, device):
    return {k: v.to(device) for k, v in batch.items()}


def unwrap_model(model):
    return getattr(model, "module", model)


def reduce_metric_means(metrics, device):
    reduced = {}
    for key, values in metrics.items():
        local_sum = float(sum(values))
        local_count = float(len(values))
        total = torch.tensor([local_sum, local_count], device=device)
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(total, op=dist.ReduceOp.SUM)
        reduced[key] = float((total[0] / total[1].clamp_min(1.0)).item())
    return reduced


def build_dynamics_sequence(mean_model, nwp_seq_full, seq_len, start_at=None, window_batch_size=8):
    """
    nwp_seq_full: [B, seq_len+chunk_len-1, 6, Y, X]
    returns:
        mu:            [B, chunk_len, 4]
        transport_gates:[B, chunk_len, 2]
        state_bias:    [B, chunk_len, state_dim]
        sigma:         [B, chunk_len, 4, 4]
    """
    if start_at is None:
        start_at = seq_len - 1

    total_frames = nwp_seq_full.shape[1]
    first_start = start_at - seq_len + 1
    if first_start < 0:
        raise ValueError(f"Need start_at >= seq_len - 1, got start_at={start_at}, seq_len={seq_len}")
    if total_frames < seq_len:
        raise ValueError(f"Need at least seq_len={seq_len} NWP frames, got {total_frames}")

    window_starts = torch.arange(
        first_start,
        total_frames - seq_len + 1,
        device=nwp_seq_full.device,
        dtype=torch.long,
    )
    num_windows = int(window_starts.numel())
    if num_windows <= 0:
        raise ValueError("No valid NWP windows available to build the dynamics sequence.")

    if window_batch_size is None or int(window_batch_size) <= 0:
        window_batch_size = num_windows
    window_batch_size = min(int(window_batch_size), num_windows)

    batch_size = nwp_seq_full.shape[0]
    offsets = torch.arange(seq_len, device=nwp_seq_full.device, dtype=torch.long)
    keys = ("mu", "base_scales", "transport_gates", "state_bias", "sigma")
    stacked = {key: [] for key in keys}

    for chunk_start in range(0, num_windows, window_batch_size):
        starts_chunk = window_starts[chunk_start:chunk_start + window_batch_size]
        gather_idx = starts_chunk[:, None] + offsets[None, :]
        x = nwp_seq_full[:, gather_idx]
        x = x.reshape(batch_size * starts_chunk.shape[0], *x.shape[2:])
        out = mean_model(x)
        for key in keys:
            value = out[key]
            stacked[key].append(value.reshape(batch_size, starts_chunk.shape[0], *value.shape[1:]))

    return {key: torch.cat(parts, dim=1) for key, parts in stacked.items()}


def zero_dynamics_sequence(batch_size, chunk_len, device, dtype, state_dim=6):
    return {
        "mu": torch.zeros(batch_size, chunk_len, 4, device=device, dtype=dtype),
        "base_scales": torch.ones(batch_size, chunk_len, 2, device=device, dtype=dtype),
        "transport_gates": torch.zeros(batch_size, chunk_len, 2, device=device, dtype=dtype),
        "state_bias": torch.zeros(batch_size, chunk_len, state_dim, device=device, dtype=dtype),
        "sigma": torch.zeros(batch_size, chunk_len, 4, 4, device=device, dtype=dtype),
    }


def dynamics_smoothness_penalty(dynamics_seq):
    penalty = 0.0
    for name in ("mu", "base_scales", "transport_gates", "state_bias", "sigma"):
        tensor = dynamics_seq[name]
        if tensor.shape[1] <= 1:
            continue
        diff = tensor[:, 1:] - tensor[:, :-1]
        penalty = penalty + diff.square().mean()
    return penalty


def sigma_summary(dynamics_seq):
    sigma = dynamics_seq["sigma"]
    return float(sigma.diagonal(dim1=-2, dim2=-1).mean().detach().cpu())


def sigma_trace_summary(dynamics_seq):
    sigma = dynamics_seq["sigma"]
    return float(sigma.diagonal(dim1=-2, dim2=-1).sum(dim=-1).mean().detach().cpu())


def sigma_diag_min_summary(dynamics_seq):
    sigma = dynamics_seq["sigma"]
    return float(sigma.diagonal(dim1=-2, dim2=-1).amin().detach().cpu())


def mu_norm_summary(dynamics_seq):
    mu = dynamics_seq["mu"]
    return float(mu.norm(dim=-1).mean().detach().cpu())


def mu_abs_summary(dynamics_seq):
    mu = dynamics_seq["mu"]
    return float(mu.abs().mean().detach().cpu())


def base_scale_summary(dynamics_seq):
    return float(dynamics_seq["base_scales"].mean().detach().cpu())


def base_scale_anisotropy_summary(dynamics_seq):
    scales = dynamics_seq["base_scales"]
    ratio = scales[..., 0] / scales[..., 1].clamp_min(1e-6)
    return float(ratio.mean().detach().cpu())


def transport_gate_summary(dynamics_seq):
    gates = dynamics_seq["transport_gates"]
    return float(gates.mean().detach().cpu())


def state_bias_summary(dynamics_seq):
    bias = dynamics_seq["state_bias"]
    return float(bias.abs().mean().detach().cpu())


def sigma_offdiag_summary(dynamics_seq):
    sigma = dynamics_seq["sigma"]
    diag = torch.diagonal(sigma, dim1=-2, dim2=-1)
    offdiag_mass = sigma.abs().sum(dim=(-1, -2)) - diag.abs().sum(dim=-1)
    return float(offdiag_mass.mean().detach().cpu())


def sigma_cross_reg_summary(dynamics_seq):
    sigma_cross = dynamics_seq["sigma"][..., :2, 2:]
    return float(sigma_cross.square().mean().detach().cpu())


def _sigma_state_var_mean(raw_ide, dynamics_seq):
    sigma_seq = dynamics_seq["sigma"]
    gates_seq = dynamics_seq["transport_gates"]
    state_var = []
    for t in range(sigma_seq.shape[1]):
        state_var.append(raw_ide._sigma_to_state_noise_diag(sigma_seq[:, t], gates_seq[:, t], dtype=sigma_seq.dtype))
    sigma_state_var = torch.stack(state_var, dim=1)
    return float(sigma_state_var.mean().detach().cpu())


def _transport_floor_penalty(dynamics_seq, min_transport_gate):
    if min_transport_gate <= 0.0:
        return dynamics_seq["transport_gates"].new_tensor(0.0)
    shortfall = torch.relu(float(min_transport_gate) - dynamics_seq["transport_gates"])
    return shortfall.mean()


def _sigma_floor_penalty(dynamics_seq, min_sigma_diag):
    if min_sigma_diag <= 0.0:
        return dynamics_seq["sigma"].new_tensor(0.0)
    diag = dynamics_seq["sigma"].diagonal(dim1=-2, dim2=-1)
    shortfall = torch.relu(float(min_sigma_diag) - diag)
    return shortfall.mean()


def _scale_floor_penalty(raw_ide, min_q_adv_scale, min_nll_sigma_scale):
    penalty = raw_ide.q_adv_scale.new_tensor(0.0)
    if min_q_adv_scale > 0.0:
        penalty = penalty + torch.relu(float(min_q_adv_scale) - raw_ide.q_adv_scale)
    if min_nll_sigma_scale > 0.0:
        penalty = penalty + torch.relu(float(min_nll_sigma_scale) - raw_ide.nll_sigma_scale)
    return penalty


def _advection_metric_names():
    return [
        "loss",
        "prob_nll",
        "one_step_mse",
        "rollout_mse",
        "step_nll",
        "noise_reg",
        "smoothness",
        "transport_reg",
        "transport_floor_pen",
        "state_bias_reg",
        "sigma_cross_reg",
        "sigma_floor_pen",
        "scale_floor_pen",
        "asym_loss",
        "asym_active_frac",
        "sigma_mean",
        "sigma_trace",
        "sigma_diag_min",
        "sigma_offdiag_mean",
        "sigma_state_var_mean",
        "mu_norm",
        "mu_abs_mean",
        "base_scale_mean",
        "base_scale_anisotropy",
        "transport_gate_mean",
        "state_bias_abs_mean",
        "q_adv_mean",
        "nll_sigma_scale_mean",
    ]


def _aligned_inputs(batch, seq_len, mean_model=None):
    z_full = batch["z_seq_full"]
    nwp_full = batch["nwp_seq_full"]
    start_idx = batch["time_idx_start"]
    batch_size = z_full.shape[0]
    chunk_len = nwp_full.shape[1] - seq_len + 1

    if mean_model is None:
        dynamics_seq = zero_dynamics_sequence(
            batch_size=batch_size,
            chunk_len=chunk_len,
            device=z_full.device,
            dtype=z_full.dtype,
            state_dim=z_full.shape[-2] * z_full.shape[-1],
        )
    else:
        dynamics_seq = build_dynamics_sequence(mean_model, nwp_full, seq_len=seq_len)

    z_aligned = z_full[:, -(chunk_len + 1):]
    aligned_start_idx = start_idx + seq_len - 1
    return z_aligned, dynamics_seq, aligned_start_idx


def _stats_loss(ide_model, z_seq, site_lon, site_lat, dynamics_seq, start_idx, noise_reg_weight):
    raw_ide = unwrap_model(ide_model)
    nll = ide_model(
        z_seq=z_seq,
        site_lon=site_lon,
        site_lat=site_lat,
        dynamics_seq=dynamics_seq,
        start_idx=start_idx,
    )
    noise_reg = raw_ide.noise_regularization()
    loss = nll + noise_reg_weight * noise_reg
    return loss, {
        "loss": float(loss.detach().cpu()),
        "nll": float(nll.detach().cpu()),
        "noise_reg": float(noise_reg.detach().cpu()),
    }


def _slice_dynamics_sequence(dynamics_seq, start, end):
    return {key: value[:, start:end] for key, value in dynamics_seq.items()}


def _prediction_mse(pred, target):
    return (pred - target).square().mean()


def _deterministic_prediction_nll(raw_ide, preds, target, dynamics_seq, start_idx, dtype):
    _, steps_minus_one, _ = target.shape
    next_obs_idx = raw_ide._time_indices(start_idx + 1, steps_minus_one)
    trans_idx = raw_ide._time_indices(start_idx, steps_minus_one)

    q_proc = raw_ide._gather_scalar(raw_ide.q_proc_series, trans_idx).to(dtype=dtype)
    r_obs = raw_ide._gather_scalar(raw_ide.r_obs_series, next_obs_idx).to(dtype=dtype)
    sigma_seq = dynamics_seq["sigma"]
    gates_seq = dynamics_seq["transport_gates"]
    sigma_state_var = []
    for t in range(steps_minus_one):
        sigma_state_var.append(
            raw_ide._sigma_to_state_noise_diag(
                sigma_seq[:, t],
                gates_seq[:, t],
                dtype=dtype,
            )
        )
    sigma_state_var = torch.stack(sigma_state_var, dim=1)
    variance = (
        q_proc[:, :, None].square()
        + r_obs[:, :, None].square()
        + raw_ide.nll_sigma_scale.to(dtype=dtype) * sigma_state_var
    ).clamp_min(1e-6)

    residual = target - preds
    nll = 0.5 * (
        residual.square() / variance
        + torch.log(variance)
        + math.log(2.0 * math.pi)
    )
    return nll.sum(dim=-1).mean()


def _directional_asymmetry_loss(pred_next, target_next, site_lon, site_lat, dynamics_seq):
    residual = target_next - pred_next
    mu = dynamics_seq["mu"]
    mu_pairs = mu.reshape(mu.shape[0], mu.shape[1], 2, 2)
    mean_dir = mu_pairs.mean(dim=2)

    coords = project_lon_lat(site_lon, site_lat)
    h = coords[:, :, None, :] - coords[:, None, :, :]
    dir_unit = mean_dir / mean_dir.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    proj = (h[:, None] * dir_unit[:, :, None, None, :]).sum(dim=-1)
    num_sites = coords.shape[1]
    pair_mask = ~torch.eye(num_sites, device=coords.device, dtype=torch.bool)[None, None]
    downwind_mask = (proj > 1e-6) & pair_mask
    upwind_mask = (proj < -1e-6) & pair_mask

    emp_cov = torch.einsum("btsc,btuc->btsu", residual, residual) / residual.shape[-1]

    downwind_vals = emp_cov.masked_select(downwind_mask)
    upwind_vals = emp_cov.masked_select(upwind_mask)
    if downwind_vals.numel() == 0 or upwind_vals.numel() == 0:
        return emp_cov.new_tensor(0.0), 0.0

    asym = (downwind_vals.abs().mean() - upwind_vals.abs().mean()).abs()
    active_frac = float((downwind_mask | upwind_mask).float().mean().detach().cpu())
    return asym, active_frac


def _advection_prediction_loss(
    ide_model,
    z_seq,
    site_lon,
    site_lat,
    dynamics_seq,
    start_idx,
    smoothness_weight,
    one_step_weight,
    rollout_weight,
    rollout_history,
    step_nll_weight,
    prob_nll_weight,
    noise_reg_weight,
    transport_reg_weight,
    transport_floor_weight,
    min_transport_gate,
    state_bias_reg_weight,
    asymmetry_weight,
    sigma_cross_reg_weight,
    sigma_floor_weight,
    min_sigma_diag,
    scale_floor_weight,
    min_q_adv_scale,
    min_nll_sigma_scale,
):
    raw_ide = unwrap_model(ide_model)
    dtype = z_seq.dtype

    if z_seq.shape[1] < 2:
        raise ValueError("Advection supervision requires at least two observations per window.")

    prob_nll = z_seq.new_tensor(0.0)
    if prob_nll_weight > 0.0:
        prob_nll = ide_model(
            z_seq=z_seq,
            site_lon=site_lon,
            site_lat=site_lat,
            dynamics_seq=dynamics_seq,
            start_idx=start_idx,
        )

    one_step = z_seq.new_tensor(0.0)
    pred_next = None
    if one_step_weight > 0.0 or step_nll_weight > 0.0:
        pred_next = raw_ide.deterministic_predict_sequence(
            z_seq=z_seq,
            site_lon=site_lon,
            site_lat=site_lat,
            dynamics_seq=dynamics_seq,
            start_idx=start_idx,
            apply_damping=True,
        )
    if one_step_weight > 0.0:
        one_step = _prediction_mse(pred_next, z_seq[:, 1:])
    step_nll = z_seq.new_tensor(0.0)
    if step_nll_weight > 0.0:
        step_nll = _deterministic_prediction_nll(
            raw_ide=raw_ide,
            preds=pred_next.reshape(pred_next.shape[0], pred_next.shape[1], raw_ide.state_dim),
            target=z_seq[:, 1:].reshape(z_seq.shape[0], z_seq.shape[1] - 1, raw_ide.state_dim),
            dynamics_seq=dynamics_seq,
            start_idx=start_idx,
            dtype=dtype,
        )

    steps = z_seq.shape[1]
    history_len = min(max(int(rollout_history), 1), steps - 1)
    future_horizon = steps - history_len
    if future_horizon <= 0:
        raise ValueError("Need at least one open-loop target step for advection supervision.")

    forecast_start_idx = start_idx + history_len - 1
    rollout_pred = raw_ide.deterministic_forecast_multistep(
        z_start=z_seq[:, history_len - 1],
        site_lon=site_lon,
        site_lat=site_lat,
        dynamics_future=_slice_dynamics_sequence(dynamics_seq, history_len - 1, dynamics_seq["mu"].shape[1]),
        start_idx=forecast_start_idx,
        apply_damping=True,
    )
    rollout_target = z_seq[:, history_len:]
    rollout = _prediction_mse(rollout_pred, rollout_target)

    smoothness = dynamics_smoothness_penalty(dynamics_seq)
    transport_reg = dynamics_seq["transport_gates"].square().mean()
    transport_floor_pen = _transport_floor_penalty(dynamics_seq, min_transport_gate)
    state_bias_reg = dynamics_seq["state_bias"].square().mean()
    sigma_cross_reg = dynamics_seq["sigma"][..., :2, 2:].square().mean()
    sigma_floor_pen = _sigma_floor_penalty(dynamics_seq, min_sigma_diag)
    scale_floor_pen = _scale_floor_penalty(raw_ide, min_q_adv_scale, min_nll_sigma_scale)
    asym_loss = z_seq.new_tensor(0.0)
    asym_active_frac = 0.0
    if pred_next is not None and asymmetry_weight > 0.0:
        asym_loss, asym_active_frac = _directional_asymmetry_loss(
            pred_next=pred_next,
            target_next=z_seq[:, 1:],
            site_lon=site_lon,
            site_lat=site_lat,
            dynamics_seq=dynamics_seq,
        )
    noise_reg = raw_ide.noise_regularization()
    loss = (
        prob_nll_weight * prob_nll
        + one_step_weight * one_step
        + rollout_weight * rollout
        + step_nll_weight * step_nll
        + asymmetry_weight * asym_loss
        + noise_reg_weight * noise_reg
        + smoothness_weight * smoothness
        + transport_reg_weight * transport_reg
        + transport_floor_weight * transport_floor_pen
        + state_bias_reg_weight * state_bias_reg
        + sigma_cross_reg_weight * sigma_cross_reg
        + sigma_floor_weight * sigma_floor_pen
        + scale_floor_weight * scale_floor_pen
    )
    return loss, {
        "loss": float(loss.detach().cpu()),
        "prob_nll": float(prob_nll.detach().cpu()),
        "one_step_mse": float(one_step.detach().cpu()),
        "rollout_mse": float(rollout.detach().cpu()),
        "step_nll": float(step_nll.detach().cpu()),
        "noise_reg": float(noise_reg.detach().cpu()),
        "smoothness": float(smoothness.detach().cpu()),
        "transport_reg": float(transport_reg.detach().cpu()),
        "transport_floor_pen": float(transport_floor_pen.detach().cpu()),
        "state_bias_reg": float(state_bias_reg.detach().cpu()),
        "sigma_cross_reg": float(sigma_cross_reg.detach().cpu()),
        "sigma_floor_pen": float(sigma_floor_pen.detach().cpu()),
        "scale_floor_pen": float(scale_floor_pen.detach().cpu()),
        "asym_loss": float(asym_loss.detach().cpu()),
        "asym_active_frac": float(asym_active_frac),
        "sigma_mean": sigma_summary(dynamics_seq),
        "sigma_trace": sigma_trace_summary(dynamics_seq),
        "sigma_diag_min": sigma_diag_min_summary(dynamics_seq),
        "sigma_offdiag_mean": sigma_offdiag_summary(dynamics_seq),
        "sigma_state_var_mean": _sigma_state_var_mean(raw_ide, dynamics_seq),
        "mu_norm": mu_norm_summary(dynamics_seq),
        "mu_abs_mean": mu_abs_summary(dynamics_seq),
        "base_scale_mean": base_scale_summary(dynamics_seq),
        "base_scale_anisotropy": base_scale_anisotropy_summary(dynamics_seq),
        "transport_gate_mean": transport_gate_summary(dynamics_seq),
        "state_bias_abs_mean": state_bias_summary(dynamics_seq),
        "q_adv_mean": float(raw_ide.q_adv_scale.detach().cpu()),
        "nll_sigma_scale_mean": float(raw_ide.nll_sigma_scale.detach().cpu()),
    }


def train_statistical_one_epoch(
    mean_model,
    ide_model,
    loader,
    optimizer,
    device,
    seq_len,
    max_steps=None,
    noise_reg_weight=1e-4,
    use_advection=True,
):
    ide_model.train()
    if mean_model is not None:
        mean_model.eval()

    metrics = {"loss": [], "nll": [], "noise_reg": []}

    for step, batch in enumerate(loader):
        if max_steps is not None and step >= max_steps:
            break

        raw_ide = unwrap_model(ide_model)
        batch = move_batch_to_device(batch, device)
        optimizer.zero_grad()

        if use_advection and mean_model is not None:
            with torch.no_grad():
                z_aligned, dynamics_seq, aligned_start_idx = _aligned_inputs(batch, seq_len, mean_model=mean_model)
        else:
            z_aligned, dynamics_seq, aligned_start_idx = _aligned_inputs(batch, seq_len, mean_model=None)

        loss, stats = _stats_loss(
            ide_model=ide_model,
            z_seq=z_aligned,
            site_lon=batch["site_lon"],
            site_lat=batch["site_lat"],
            dynamics_seq=dynamics_seq,
            start_idx=aligned_start_idx,
            noise_reg_weight=noise_reg_weight,
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(ide_model.parameters(), 1.0)
        optimizer.step()
        raw_ide.clamp_parameters_()

        for key in metrics:
            metrics[key].append(stats[key])

    if not metrics["loss"]:
        return {key: float("nan") for key in metrics}
    return reduce_metric_means(metrics, device)


@torch.no_grad()
def eval_statistical(
    mean_model,
    ide_model,
    loader,
    device,
    seq_len,
    max_steps=None,
    noise_reg_weight=1e-4,
    use_advection=True,
):
    ide_model.eval()
    if mean_model is not None:
        mean_model.eval()

    metrics = {"loss": [], "nll": [], "noise_reg": []}

    for step, batch in enumerate(loader):
        if max_steps is not None and step >= max_steps:
            break

        batch = move_batch_to_device(batch, device)
        if use_advection and mean_model is not None:
            z_aligned, dynamics_seq, aligned_start_idx = _aligned_inputs(batch, seq_len, mean_model=mean_model)
        else:
            z_aligned, dynamics_seq, aligned_start_idx = _aligned_inputs(batch, seq_len, mean_model=None)

        _, stats = _stats_loss(
            ide_model=ide_model,
            z_seq=z_aligned,
            site_lon=batch["site_lon"],
            site_lat=batch["site_lat"],
            dynamics_seq=dynamics_seq,
            start_idx=aligned_start_idx,
            noise_reg_weight=noise_reg_weight,
        )
        for key in metrics:
            metrics[key].append(stats[key])

    if not metrics["loss"]:
        return {key: float("nan") for key in metrics}
    return reduce_metric_means(metrics, device)


def train_advection_one_epoch(
    mean_model,
    ide_model,
    loader,
    optimizer,
    device,
    seq_len,
    max_steps=None,
    smoothness_weight=1e-3,
    one_step_weight=0.25,
    rollout_weight=1.0,
    rollout_history=6,
    step_nll_weight=0.1,
    prob_nll_weight=1.0,
    noise_reg_weight=1e-4,
    transport_reg_weight=1e-2,
    transport_floor_weight=0.0,
    min_transport_gate=0.0,
    state_bias_reg_weight=1e-3,
    asymmetry_weight=0.01,
    sigma_cross_reg_weight=1e-4,
    sigma_floor_weight=0.0,
    min_sigma_diag=0.0,
    scale_floor_weight=0.0,
    min_q_adv_scale=0.0,
    min_nll_sigma_scale=0.0,
):
    mean_model.train()
    ide_model.eval()

    metrics = {key: [] for key in _advection_metric_names()}

    for step, batch in enumerate(loader):
        if max_steps is not None and step >= max_steps:
            break

        batch = move_batch_to_device(batch, device)
        optimizer.zero_grad()
        z_aligned, dynamics_seq, aligned_start_idx = _aligned_inputs(batch, seq_len, mean_model=mean_model)

        loss, stats = _advection_prediction_loss(
            ide_model=ide_model,
            z_seq=z_aligned,
            site_lon=batch["site_lon"],
            site_lat=batch["site_lat"],
            dynamics_seq=dynamics_seq,
            start_idx=aligned_start_idx,
            smoothness_weight=smoothness_weight,
            one_step_weight=one_step_weight,
            rollout_weight=rollout_weight,
            rollout_history=rollout_history,
            step_nll_weight=step_nll_weight,
            prob_nll_weight=prob_nll_weight,
            noise_reg_weight=noise_reg_weight,
            transport_reg_weight=transport_reg_weight,
            transport_floor_weight=transport_floor_weight,
            min_transport_gate=min_transport_gate,
            state_bias_reg_weight=state_bias_reg_weight,
            asymmetry_weight=asymmetry_weight,
            sigma_cross_reg_weight=sigma_cross_reg_weight,
            sigma_floor_weight=sigma_floor_weight,
            min_sigma_diag=min_sigma_diag,
            scale_floor_weight=scale_floor_weight,
            min_q_adv_scale=min_q_adv_scale,
            min_nll_sigma_scale=min_nll_sigma_scale,
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(mean_model.parameters(), 1.0)
        optimizer.step()

        for key in metrics:
            metrics[key].append(stats[key])

    if not metrics["loss"]:
        return {key: float("nan") for key in metrics}
    return reduce_metric_means(metrics, device)


@torch.no_grad()
def eval_advection(
    mean_model,
    ide_model,
    loader,
    device,
    seq_len,
    max_steps=None,
    smoothness_weight=1e-3,
    one_step_weight=0.25,
    rollout_weight=1.0,
    rollout_history=6,
    step_nll_weight=0.1,
    prob_nll_weight=1.0,
    noise_reg_weight=1e-4,
    transport_reg_weight=1e-2,
    transport_floor_weight=0.0,
    min_transport_gate=0.0,
    state_bias_reg_weight=1e-3,
    asymmetry_weight=0.01,
    sigma_cross_reg_weight=1e-4,
    sigma_floor_weight=0.0,
    min_sigma_diag=0.0,
    scale_floor_weight=0.0,
    min_q_adv_scale=0.0,
    min_nll_sigma_scale=0.0,
):
    mean_model.eval()
    ide_model.eval()

    metrics = {key: [] for key in _advection_metric_names()}

    for step, batch in enumerate(loader):
        if max_steps is not None and step >= max_steps:
            break

        batch = move_batch_to_device(batch, device)
        z_aligned, dynamics_seq, aligned_start_idx = _aligned_inputs(batch, seq_len, mean_model=mean_model)

        _, stats = _advection_prediction_loss(
            ide_model=ide_model,
            z_seq=z_aligned,
            site_lon=batch["site_lon"],
            site_lat=batch["site_lat"],
            dynamics_seq=dynamics_seq,
            start_idx=aligned_start_idx,
            smoothness_weight=smoothness_weight,
            one_step_weight=one_step_weight,
            rollout_weight=rollout_weight,
            rollout_history=rollout_history,
            step_nll_weight=step_nll_weight,
            prob_nll_weight=prob_nll_weight,
            noise_reg_weight=noise_reg_weight,
            transport_reg_weight=transport_reg_weight,
            transport_floor_weight=transport_floor_weight,
            min_transport_gate=min_transport_gate,
            state_bias_reg_weight=state_bias_reg_weight,
            asymmetry_weight=asymmetry_weight,
            sigma_cross_reg_weight=sigma_cross_reg_weight,
            sigma_floor_weight=sigma_floor_weight,
            min_sigma_diag=min_sigma_diag,
            scale_floor_weight=scale_floor_weight,
            min_q_adv_scale=min_q_adv_scale,
            min_nll_sigma_scale=min_nll_sigma_scale,
        )
        for key in metrics:
            metrics[key].append(stats[key])

    if not metrics["loss"]:
        return {key: float("nan") for key in metrics}
    return reduce_metric_means(metrics, device)


def train_joint_one_epoch(
    mean_model,
    ide_model,
    loader,
    optimizer,
    device,
    seq_len,
    max_steps=None,
    smoothness_weight=1e-3,
    one_step_weight=0.25,
    rollout_weight=1.0,
    rollout_history=6,
    step_nll_weight=0.1,
    prob_nll_weight=1.0,
    noise_reg_weight=1e-4,
    transport_reg_weight=1e-2,
    transport_floor_weight=0.0,
    min_transport_gate=0.0,
    state_bias_reg_weight=1e-3,
    asymmetry_weight=0.01,
    sigma_cross_reg_weight=1e-4,
    sigma_floor_weight=0.0,
    min_sigma_diag=0.0,
    scale_floor_weight=0.0,
    min_q_adv_scale=0.0,
    min_nll_sigma_scale=0.0,
):
    mean_model.train()
    ide_model.train()
    metrics = {key: [] for key in _advection_metric_names()}

    for step, batch in enumerate(loader):
        if max_steps is not None and step >= max_steps:
            break

        raw_ide = unwrap_model(ide_model)
        batch = move_batch_to_device(batch, device)
        optimizer.zero_grad()
        z_aligned, dynamics_seq, aligned_start_idx = _aligned_inputs(batch, seq_len, mean_model=mean_model)

        loss, stats = _advection_prediction_loss(
            ide_model=ide_model,
            z_seq=z_aligned,
            site_lon=batch["site_lon"],
            site_lat=batch["site_lat"],
            dynamics_seq=dynamics_seq,
            start_idx=aligned_start_idx,
            smoothness_weight=smoothness_weight,
            one_step_weight=one_step_weight,
            rollout_weight=rollout_weight,
            rollout_history=rollout_history,
            step_nll_weight=step_nll_weight,
            prob_nll_weight=prob_nll_weight,
            noise_reg_weight=noise_reg_weight,
            transport_reg_weight=transport_reg_weight,
            transport_floor_weight=transport_floor_weight,
            min_transport_gate=min_transport_gate,
            state_bias_reg_weight=state_bias_reg_weight,
            asymmetry_weight=asymmetry_weight,
            sigma_cross_reg_weight=sigma_cross_reg_weight,
            sigma_floor_weight=sigma_floor_weight,
            min_sigma_diag=min_sigma_diag,
            scale_floor_weight=scale_floor_weight,
            min_q_adv_scale=min_q_adv_scale,
            min_nll_sigma_scale=min_nll_sigma_scale,
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(mean_model.parameters(), 1.0)
        torch.nn.utils.clip_grad_norm_(ide_model.parameters(), 1.0)
        optimizer.step()
        raw_ide.clamp_parameters_()

        for key in metrics:
            metrics[key].append(stats[key])

    if not metrics["loss"]:
        return {key: float("nan") for key in metrics}
    return reduce_metric_means(metrics, device)


@torch.no_grad()
def eval_joint(
    mean_model,
    ide_model,
    loader,
    device,
    seq_len,
    max_steps=None,
    smoothness_weight=1e-3,
    one_step_weight=0.25,
    rollout_weight=1.0,
    rollout_history=6,
    step_nll_weight=0.1,
    prob_nll_weight=1.0,
    noise_reg_weight=1e-4,
    transport_reg_weight=1e-2,
    transport_floor_weight=0.0,
    min_transport_gate=0.0,
    state_bias_reg_weight=1e-3,
    asymmetry_weight=0.01,
    sigma_cross_reg_weight=1e-4,
    sigma_floor_weight=0.0,
    min_sigma_diag=0.0,
    scale_floor_weight=0.0,
    min_q_adv_scale=0.0,
    min_nll_sigma_scale=0.0,
):
    mean_model.eval()
    ide_model.eval()
    metrics = {key: [] for key in _advection_metric_names()}

    for step, batch in enumerate(loader):
        if max_steps is not None and step >= max_steps:
            break

        batch = move_batch_to_device(batch, device)
        z_aligned, dynamics_seq, aligned_start_idx = _aligned_inputs(batch, seq_len, mean_model=mean_model)
        _, stats = _advection_prediction_loss(
            ide_model=ide_model,
            z_seq=z_aligned,
            site_lon=batch["site_lon"],
            site_lat=batch["site_lat"],
            dynamics_seq=dynamics_seq,
            start_idx=aligned_start_idx,
            smoothness_weight=smoothness_weight,
            one_step_weight=one_step_weight,
            rollout_weight=rollout_weight,
            rollout_history=rollout_history,
            step_nll_weight=step_nll_weight,
            prob_nll_weight=prob_nll_weight,
            noise_reg_weight=noise_reg_weight,
            transport_reg_weight=transport_reg_weight,
            transport_floor_weight=transport_floor_weight,
            min_transport_gate=min_transport_gate,
            state_bias_reg_weight=state_bias_reg_weight,
            asymmetry_weight=asymmetry_weight,
            sigma_cross_reg_weight=sigma_cross_reg_weight,
            sigma_floor_weight=sigma_floor_weight,
            min_sigma_diag=min_sigma_diag,
            scale_floor_weight=scale_floor_weight,
            min_q_adv_scale=min_q_adv_scale,
            min_nll_sigma_scale=min_nll_sigma_scale,
        )
        for key in metrics:
            metrics[key].append(stats[key])

    if not metrics["loss"]:
        return {key: float("nan") for key in metrics}
    return reduce_metric_means(metrics, device)


train_ml_mu_one_epoch = train_joint_one_epoch
eval_ml_mu = eval_joint
