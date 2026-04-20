import torch
import torch.distributed as dist


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


def build_dynamics_sequence(mean_model, nwp_seq_full, seq_len, start_at=None):
    """
    nwp_seq_full: [B, seq_len+chunk_len-1, 6, Y, X]
    returns:
        mu:            [B, chunk_len, 4]
        sigma:         [B, chunk_len, 4, 4]
    """
    if start_at is None:
        start_at = seq_len - 1

    outputs = []
    for end_t in range(start_at, nwp_seq_full.shape[1]):
        x = nwp_seq_full[:, end_t - seq_len + 1:end_t + 1]
        outputs.append(mean_model(x))

    return {
        "mu": torch.stack([out["mu"] for out in outputs], dim=1),
        "base_scales": torch.stack([out["base_scales"] for out in outputs], dim=1),
        "sigma": torch.stack([out["sigma"] for out in outputs], dim=1),
    }


def zero_dynamics_sequence(batch_size, chunk_len, device, dtype):
    return {
        "mu": torch.zeros(batch_size, chunk_len, 4, device=device, dtype=dtype),
        "base_scales": torch.ones(batch_size, chunk_len, 2, device=device, dtype=dtype),
        "sigma": torch.zeros(batch_size, chunk_len, 4, 4, device=device, dtype=dtype),
    }


def dynamics_smoothness_penalty(dynamics_seq):
    penalty = 0.0
    for name in ("mu", "base_scales", "sigma"):
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


def sigma_offdiag_summary(dynamics_seq):
    sigma = dynamics_seq["sigma"]
    diag = torch.diagonal(sigma, dim1=-2, dim2=-1)
    offdiag_mass = sigma.abs().sum(dim=(-1, -2)) - diag.abs().sum(dim=-1)
    return float(offdiag_mass.mean().detach().cpu())


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
):
    raw_ide = unwrap_model(ide_model)

    if z_seq.shape[1] < 2:
        raise ValueError("Advection supervision requires at least two observations per window.")

    one_step = z_seq.new_tensor(0.0)
    if one_step_weight > 0.0:
        pred_next = raw_ide.predict_sequence(
            z_seq=z_seq,
            site_lon=site_lon,
            site_lat=site_lat,
            dynamics_seq=dynamics_seq,
            start_idx=start_idx,
        )
        one_step = _prediction_mse(pred_next, z_seq[:, 1:])

    steps = z_seq.shape[1]
    history_len = min(max(int(rollout_history), 1), steps - 1)
    future_horizon = steps - history_len
    if future_horizon <= 0:
        raise ValueError("Need at least one open-loop target step for advection supervision.")

    hist_dyn_len = max(history_len - 1, 0)
    rollout_pred = raw_ide.forecast_multistep(
        z_hist=z_seq[:, :history_len],
        site_lon=site_lon,
        site_lat=site_lat,
        dynamics_hist=_slice_dynamics_sequence(dynamics_seq, 0, hist_dyn_len),
        dynamics_future=_slice_dynamics_sequence(dynamics_seq, hist_dyn_len, dynamics_seq["mu"].shape[1]),
        start_idx=start_idx,
    )
    rollout_target = z_seq[:, history_len:]
    rollout = _prediction_mse(rollout_pred, rollout_target)

    smoothness = dynamics_smoothness_penalty(dynamics_seq)
    loss = one_step_weight * one_step + rollout_weight * rollout + smoothness_weight * smoothness
    return loss, {
        "loss": float(loss.detach().cpu()),
        "one_step_mse": float(one_step.detach().cpu()),
        "rollout_mse": float(rollout.detach().cpu()),
        "smoothness": float(smoothness.detach().cpu()),
        "sigma_mean": sigma_summary(dynamics_seq),
        "sigma_trace": sigma_trace_summary(dynamics_seq),
        "sigma_diag_min": sigma_diag_min_summary(dynamics_seq),
        "sigma_offdiag_mean": sigma_offdiag_summary(dynamics_seq),
        "mu_norm": mu_norm_summary(dynamics_seq),
        "mu_abs_mean": mu_abs_summary(dynamics_seq),
        "base_scale_mean": base_scale_summary(dynamics_seq),
        "base_scale_anisotropy": base_scale_anisotropy_summary(dynamics_seq),
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
):
    mean_model.train()
    ide_model.eval()

    metrics = {
        "loss": [],
        "one_step_mse": [],
        "rollout_mse": [],
        "smoothness": [],
        "sigma_mean": [],
        "sigma_trace": [],
        "sigma_diag_min": [],
        "sigma_offdiag_mean": [],
        "mu_norm": [],
        "mu_abs_mean": [],
        "base_scale_mean": [],
        "base_scale_anisotropy": [],
    }

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
):
    mean_model.eval()
    ide_model.eval()

    metrics = {
        "loss": [],
        "one_step_mse": [],
        "rollout_mse": [],
        "smoothness": [],
        "sigma_mean": [],
        "sigma_trace": [],
        "sigma_diag_min": [],
        "sigma_offdiag_mean": [],
        "mu_norm": [],
        "mu_abs_mean": [],
        "base_scale_mean": [],
        "base_scale_anisotropy": [],
    }

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
        )
        for key in metrics:
            metrics[key].append(stats[key])

    if not metrics["loss"]:
        return {key: float("nan") for key in metrics}
    return reduce_metric_means(metrics, device)


train_ml_mu_one_epoch = train_advection_one_epoch
eval_ml_mu = eval_advection
