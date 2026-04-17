import torch
import torch.distributed as dist


def move_batch_to_device(batch, device):
    return {k: v.to(device) for k, v in batch.items()}


def unwrap_model(model):
    return getattr(model, "module", model)


def reduce_scalar_mean(value, device):
    total = torch.tensor([float(value), 1.0], device=device)
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(total, op=dist.ReduceOp.SUM)
    return float((total[0] / total[1].clamp_min(1.0)).item())


def train_ide_baseline_one_epoch(
    ide_model,
    loader,
    optimizer,
    device,
    max_steps=None,
    noise_reg_weight=1e-4,
):
    """
    Stage 1:
    IDE only, with mu_t fixed to zero.
    """
    ide_model.train()
    losses = []

    for step, batch in enumerate(loader):
        if max_steps is not None and step >= max_steps:
            break

        raw_ide = unwrap_model(ide_model)
        batch = move_batch_to_device(batch, device)
        optimizer.zero_grad()

        z = batch["z_seq_full"]                         # [B, chunk_len+1, 3, 2]
        nll = ide_model(
            z_seq=z,
            site_lon=batch["site_lon"],
            site_lat=batch["site_lat"],
            start_idx=batch["time_idx_start"],
        )
        loss = nll + noise_reg_weight * raw_ide.noise_regularization()

        loss.backward()
        torch.nn.utils.clip_grad_norm_(ide_model.parameters(), 1.0)
        optimizer.step()
        raw_ide.clamp_parameters_()

        losses.append(float(loss.detach().cpu()))

    mean_loss = float(sum(losses) / max(1, len(losses)))
    return reduce_scalar_mean(mean_loss, device)


@torch.no_grad()
def eval_ide_baseline(ide_model, loader, device, max_steps=None, noise_reg_weight=1e-4):
    ide_model.eval()
    losses = []

    for step, batch in enumerate(loader):
        if max_steps is not None and step >= max_steps:
            break

        raw_ide = unwrap_model(ide_model)
        batch = move_batch_to_device(batch, device)

        z = batch["z_seq_full"]
        nll = ide_model(
            z_seq=z,
            site_lon=batch["site_lon"],
            site_lat=batch["site_lat"],
            start_idx=batch["time_idx_start"],
        )
        loss = nll + noise_reg_weight * raw_ide.noise_regularization()
        losses.append(float(loss.cpu()))

    mean_loss = float(sum(losses) / max(1, len(losses)))
    return reduce_scalar_mean(mean_loss, device)
