import math
import torch
from .losses import total_objective


def move_batch_to_device(batch, device):
    return {k: v.to(device) for k, v in batch.items()}


def set_requires_grad(module, flag: bool):
    for p in module.parameters():
        p.requires_grad = flag


def _compute_grad_norm(model):
    total_norm_sq = 0.0
    for p in model.parameters():
        if p.grad is not None:
            n = p.grad.detach().data.norm(2).item()
            total_norm_sq += n**2
    return math.sqrt(total_norm_sq)


def alternating_train_one_epoch(
    model,
    loader,
    ide_optimizer,
    ml_optimizer,
    device,
    max_steps=None,
    ide_inner_steps=3,
    ml_inner_steps=1,
    lambda_time=1.0,
    lambda_ml=1e-4,
    grad_clip=1.0,
):
    model.train()
    losses = []

    for step, batch in enumerate(loader):
        if max_steps is not None and step >= max_steps:
            break

        batch = move_batch_to_device(batch, device)

        # -------------------------
        # Phase A: update IDE base
        # -------------------------
        set_requires_grad(model.ide, True)
        set_requires_grad(model.adjuster, False)

        for _ in range(ide_inner_steps):
            ide_optimizer.zero_grad()

            with torch.no_grad():
                delta = model.adjuster(batch["z_seq"])

            y_pred, diag = model.ide.forward_step(
                y_t=batch["y_t"],
                u_t=batch["u_t"],
                v_t=batch["v_t"],
                lat=batch["lat"],
                lon=batch["lon"],
                delta=delta,
            )

            loss = total_objective(
                y_true=batch["y_next"],
                y_pred=y_pred,
                sigma_eps=model.ide.sigma_eps,
                delta=delta,
                temporal_penalty=model.ide.temporal_smoothness_penalty(),
                lambda_time=lambda_time,
                lambda_ml=lambda_ml,
            )

            loss.backward()
            if grad_clip is not None and grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.ide.parameters(), max_norm=grad_clip)
            ide_optimizer.step()

        # -------------------------
        # Phase B: update ML refine
        # -------------------------
        set_requires_grad(model.ide, False)
        set_requires_grad(model.adjuster, True)

        for _ in range(ml_inner_steps):
            ml_optimizer.zero_grad()

            delta = model.adjuster(batch["z_seq"])
            y_pred, diag = model.ide.forward_step(
                y_t=batch["y_t"],
                u_t=batch["u_t"],
                v_t=batch["v_t"],
                lat=batch["lat"],
                lon=batch["lon"],
                delta=delta,
            )

            loss = total_objective(
                y_true=batch["y_next"],
                y_pred=y_pred,
                sigma_eps=model.ide.sigma_eps,
                delta=delta,
                temporal_penalty=model.ide.temporal_smoothness_penalty(),
                lambda_time=lambda_time,
                lambda_ml=lambda_ml,
            )

            loss.backward()
            if grad_clip is not None and grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.adjuster.parameters(), max_norm=grad_clip)
            ml_optimizer.step()

        # 这一 step 完成后记录
        losses.append(float(loss.detach().cpu()))

    # epoch 结束后，把当前 IDE 参数记成下一轮参考值
    model.ide.update_prev_params()

    if len(losses) == 0:
        return {"train_loss_mean": float("nan")}

    return {
        "train_loss_mean": float(sum(losses) / len(losses)),
        "train_loss_min": float(min(losses)),
        "train_loss_max": float(max(losses)),
        "delta_par_mean": float(diag.delta_par_mean.cpu()),
        "delta_perp_mean": float(diag.delta_perp_mean.cpu()),
        "ell_par_mean": float(diag.ell_par_mean.cpu()),
        "ell_perp_mean": float(diag.ell_perp_mean.cpu()),
        "sigma_eps": float(diag.sigma_eps.cpu()),
    }


@torch.no_grad()
def evaluate(model, loader, device, max_steps=None, lambda_time=1.0, lambda_ml=1e-4):
    model.eval()
    losses = []

    for step, batch in enumerate(loader):
        if max_steps is not None and step >= max_steps:
            break

        batch = move_batch_to_device(batch, device)
        delta = model.adjuster(batch["z_seq"])
        y_pred, _ = model.ide.forward_step(
            y_t=batch["y_t"],
            u_t=batch["u_t"],
            v_t=batch["v_t"],
            lat=batch["lat"],
            lon=batch["lon"],
            delta=delta,
        )

        loss = total_objective(
            y_true=batch["y_next"],
            y_pred=y_pred,
            sigma_eps=model.ide.sigma_eps,
            delta=delta,
            temporal_penalty=model.ide.temporal_smoothness_penalty(),
            lambda_time=lambda_time,
            lambda_ml=lambda_ml,
        )
        losses.append(float(loss.cpu()))

    return {"loss": float(sum(losses) / max(1, len(losses)))}