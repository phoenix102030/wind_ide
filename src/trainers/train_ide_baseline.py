import torch


def move_batch_to_device(batch, device):
    return {k: v.to(device) for k, v in batch.items()}


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

        batch = move_batch_to_device(batch, device)
        optimizer.zero_grad()

        z = batch["z_seq_full"]                         # [B, chunk_len+1, 3, 2]
        nll = ide_model.sequence_nll(
            z_seq=z,
            site_lon=batch["site_lon"],
            site_lat=batch["site_lat"],
        )
        loss = nll + noise_reg_weight * ide_model.noise_regularization()

        loss.backward()
        torch.nn.utils.clip_grad_norm_(ide_model.parameters(), 1.0)
        optimizer.step()
        ide_model.clamp_parameters_()

        losses.append(float(loss.detach().cpu()))

    return float(sum(losses) / max(1, len(losses)))


@torch.no_grad()
def eval_ide_baseline(ide_model, loader, device, max_steps=None, noise_reg_weight=1e-4):
    ide_model.eval()
    losses = []

    for step, batch in enumerate(loader):
        if max_steps is not None and step >= max_steps:
            break

        batch = move_batch_to_device(batch, device)

        z = batch["z_seq_full"]
        nll = ide_model.sequence_nll(
            z_seq=z,
            site_lon=batch["site_lon"],
            site_lat=batch["site_lat"],
        )
        loss = nll + noise_reg_weight * ide_model.noise_regularization()
        losses.append(float(loss.cpu()))

    return float(sum(losses) / max(1, len(losses)))
