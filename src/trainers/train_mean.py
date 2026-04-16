import torch


def move_batch_to_device(batch, device):
    return {k: v.to(device) for k, v in batch.items()}


def train_mean_one_epoch(model, loader, optimizer, device, max_steps=None):
    model.train()
    losses = []

    for step, batch in enumerate(loader):
        if max_steps is not None and step >= max_steps:
            break

        batch = move_batch_to_device(batch, device)
        optimizer.zero_grad()

        mu_pred = model(batch["nwp_seq"])                   # [B,2]
        loss = ((mu_pred - batch["mu_target"]) ** 2).mean()

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        losses.append(float(loss.detach().cpu()))

    return float(sum(losses) / max(1, len(losses)))


@torch.no_grad()
def eval_mean(model, loader, device, max_steps=None):
    model.eval()
    losses = []

    for step, batch in enumerate(loader):
        if max_steps is not None and step >= max_steps:
            break

        batch = move_batch_to_device(batch, device)
        mu_pred = model(batch["nwp_seq"])
        loss = ((mu_pred - batch["mu_target"]) ** 2).mean()
        losses.append(float(loss.cpu()))

    return float(sum(losses) / max(1, len(losses)))