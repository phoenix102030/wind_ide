from __future__ import annotations

import argparse
import json
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np


DEFAULT_START_TIME = "2021-06-04 00:00:00"
DEFAULT_DT_SECONDS = 600.0


def parse_time(value: str) -> datetime:
    return datetime.strptime(value, "%Y-%m-%d %H:%M:%S")


def time_label(index: int, start_time: datetime, dt_seconds: float) -> str:
    return (start_time + timedelta(seconds=float(index) * dt_seconds)).strftime("%Y-%m-%d %H:%M")


def time_slug(index: int, start_time: datetime, dt_seconds: float) -> str:
    return (start_time + timedelta(seconds=float(index) * dt_seconds)).strftime("%Y%m%d_%H%M")


def symmetrize_psd(S: np.ndarray, jitter: float = 1.0e-8) -> np.ndarray:
    S = 0.5 * (S + S.T)
    vals, vecs = np.linalg.eigh(S)
    vals = np.clip(vals, jitter, None)
    return (vecs * vals) @ vecs.T


def covariance_ellipse(ax, mean: np.ndarray, cov: np.ndarray, color: str, label: str, nsig: float = 2.0) -> None:
    from matplotlib.patches import Ellipse

    S = symmetrize_psd(cov)
    vals, vecs = np.linalg.eigh(S)
    order = np.argsort(vals)[::-1]
    vals = vals[order]
    vecs = vecs[:, order]
    angle = float(np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0])))
    ell = Ellipse(
        xy=mean,
        width=2.0 * nsig * np.sqrt(vals[0]),
        height=2.0 * nsig * np.sqrt(vals[1]),
        angle=angle,
        facecolor=color,
        edgecolor=color,
        alpha=0.18,
        linewidth=2.0,
        label=label,
    )
    ax.add_patch(ell)


def invsqrt_2x2(S: np.ndarray) -> np.ndarray:
    vals, vecs = np.linalg.eigh(symmetrize_psd(S))
    vals = np.clip(vals, 1.0e-10, None)
    return (vecs * (1.0 / np.sqrt(vals))) @ vecs.T


def canonical_cross_correlation(Suu: np.ndarray, Suv: np.ndarray, Svv: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    C = invsqrt_2x2(Suu) @ Suv @ invsqrt_2x2(Svv)
    U, vals, Vt = np.linalg.svd(C, full_matrices=False)
    return float(vals[0]), U[:, 0], Vt.T[:, 0]


def component_summary(S: np.ndarray, ell: float | None = None) -> dict[str, float]:
    S = symmetrize_psd(S)
    Suu = S[:2, :2]
    Suv = S[:2, 2:]
    Svv = S[2:, 2:]
    lam_u = np.linalg.eigvalsh(symmetrize_psd(Suu))
    lam_v = np.linalg.eigvalsh(symmetrize_psd(Svv))
    rho, _, _ = canonical_cross_correlation(Suu, Suv, Svv)
    trace_u = float(np.sum(lam_u))
    trace_v = float(np.sum(lam_v))
    out = {
        "trace_u": trace_u,
        "trace_v": trace_v,
        "sqrt_trace_u": float(np.sqrt(trace_u)),
        "sqrt_trace_v": float(np.sqrt(trace_v)),
        "anisotropy_u": float(np.max(lam_u) / (np.min(lam_u) + 1.0e-12)),
        "anisotropy_v": float(np.max(lam_v) / (np.min(lam_v) + 1.0e-12)),
        "cross_block_fro": float(np.linalg.norm(Suv, ord="fro")),
        "cross_canonical_corr": rho,
    }
    if ell is not None:
        denom_u = 2.0 * ell * ell + 2.0 * trace_u
        denom_v = 2.0 * ell * ell + 2.0 * trace_v
        out["sigma_kernel_ratio_u"] = float((2.0 * trace_u) / (denom_u + 1.0e-12))
        out["sigma_kernel_ratio_v"] = float((2.0 * trace_v) / (denom_v + 1.0e-12))
    return out


def transformed_mu_sigma(params: np.lib.npyio.NpzFile, use_bias: bool) -> tuple[np.ndarray, np.ndarray]:
    can_apply_legacy_bias = False
    if "Au" in params.files and "Av" in params.files:
        mu = np.concatenate([params["Au"], params["Av"]], axis=1).astype(np.float64)
    elif "target_component_flow" in params.files:
        mu = np.concatenate(
            [params["target_component_flow"][:, 0, :], params["target_component_flow"][:, 1, :]],
            axis=1,
        ).astype(np.float64)
        can_apply_legacy_bias = True
    else:
        mu = params["mu"].astype(np.float64)
    S = params["Sigma"].astype(np.float64)
    if not use_bias or not can_apply_legacy_bias:
        return mu, S
    # Same artificial display transform used in the legacy A visualizations:
    # A_u = [1.45 x, 0.65 y], A_v = [0.65 x, 1.45 y].
    D = np.diag([1.45, 0.65, 0.65, 1.45])
    return mu @ D.T, np.einsum("ij,tjk,lk->til", D, S, D)


def draw_frame(
    fig,
    axes,
    t: int,
    mu: np.ndarray,
    S: np.ndarray,
    ell: float | None,
    lim: float,
    kernel_lim: float,
    cov_vmax: float,
    sigma_scale: float,
    start_time: datetime,
    dt_seconds: float,
) -> None:
    import matplotlib.pyplot as plt

    for ax in axes:
        ax.clear()

    Suu = symmetrize_psd(S[t, :2, :2])
    Suv = S[t, :2, 2:]
    Svv = symmetrize_psd(S[t, 2:, 2:])
    mu_u = mu[t, :2]
    mu_v = mu[t, 2:]

    ax = axes[0]
    covariance_ellipse(ax, mu_u, Suu, "#1f77b4", r"$A_u$ 2-sigma ellipse")
    covariance_ellipse(ax, mu_v, Svv, "#d95f02", r"$A_v$ 2-sigma ellipse")
    ax.arrow(0, 0, mu_u[0], mu_u[1], color="#1f77b4", width=0.025 * lim, head_width=0.18 * lim, length_includes_head=True)
    ax.arrow(0, 0, mu_v[0], mu_v[1], color="#d95f02", width=0.025 * lim, head_width=0.18 * lim, length_includes_head=True)
    ax.axhline(0, color="0.45", linewidth=0.8)
    ax.axvline(0, color="0.45", linewidth=0.8)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.25)
    ax.set_title("Marginal advection uncertainty")
    ax.set_xlabel("x displacement (km/step)")
    ax.set_ylabel("y displacement (km/step)")
    ax.legend(loc="upper left", fontsize=8)

    ax = axes[1]
    cov4 = symmetrize_psd(S[t])
    im = ax.imshow(cov4, vmin=-cov_vmax, vmax=cov_vmax, cmap="coolwarm")
    ax.set_xticks(range(4), labels=[r"$A_{u,x}$", r"$A_{u,y}$", r"$A_{v,x}$", r"$A_{v,y}$"], rotation=35, ha="right")
    ax.set_yticks(range(4), labels=[r"$A_{u,x}$", r"$A_{u,y}$", r"$A_{v,x}$", r"$A_{v,y}$"])
    for i in range(4):
        for j in range(4):
            ax.text(j, i, f"{cov4[i, j]:.2e}", ha="center", va="center", fontsize=7)
    ax.axhline(1.5, color="black", linewidth=1.4)
    ax.axvline(1.5, color="black", linewidth=1.4)
    ax.set_title("Full 4D advection covariance")
    if not hasattr(fig, "_cov_colorbar"):
        fig._cov_colorbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig._cov_colorbar.set_label("covariance")

    ax = axes[2]
    rho, dir_u, dir_v = canonical_cross_correlation(Suu, Suv, Svv)
    scale = 0.75 * lim
    ax.arrow(0, 0, scale * dir_u[0], scale * dir_u[1], color="#1f77b4", width=0.018 * lim, head_width=0.13 * lim, length_includes_head=True, label=r"$A_u$ coupled mode")
    ax.arrow(0, 0, scale * dir_v[0], scale * dir_v[1], color="#d95f02", width=0.018 * lim, head_width=0.13 * lim, length_includes_head=True, label=r"$A_v$ coupled mode")
    ax.axhline(0, color="0.45", linewidth=0.8)
    ax.axvline(0, color="0.45", linewidth=0.8)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.25)
    ax.set_title(f"Cross-covariance coupling mode, rho={rho:.3f}")
    ax.set_xlabel("mode x")
    ax.set_ylabel("mode y")
    ax.legend(loc="upper left", fontsize=8)

    ax = axes[3]
    from matplotlib.patches import Ellipse

    def add_dispersion_ellipse(Dmat: np.ndarray, color: str, label: str, linestyle: str = "-") -> None:
        vals, vecs = np.linalg.eigh(symmetrize_psd(Dmat))
        vals = np.clip(vals, 1.0e-12, None)
        order = np.argsort(vals)[::-1]
        vals = vals[order]
        vecs = vecs[:, order]
        angle = float(np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0])))
        patch = Ellipse(
            (0.0, 0.0),
            width=4.0 * np.sqrt(vals[0]),
            height=4.0 * np.sqrt(vals[1]),
            angle=angle,
            facecolor=color,
            edgecolor=color,
            alpha=0.14 if linestyle == "-" else 0.0,
            linewidth=2.0,
            linestyle=linestyle,
            label=label,
        )
        ax.add_patch(patch)

    eye2 = np.eye(2)
    if ell is not None:
        add_dispersion_ellipse(ell * ell * eye2, "0.35", r"$\ell^2 I$ baseline", linestyle="--")
        add_dispersion_ellipse(ell * ell * eye2 + 2.0 * Suu, "#1f77b4", r"$D_u=\ell^2I+2\Sigma_{uu}$")
        add_dispersion_ellipse(ell * ell * eye2 + 2.0 * Svv, "#d95f02", r"$D_v=\ell^2I+2\Sigma_{vv}$")
    else:
        add_dispersion_ellipse(2.0 * Suu, "#1f77b4", r"$2\Sigma_{uu}$")
        add_dispersion_ellipse(2.0 * Svv, "#d95f02", r"$2\Sigma_{vv}$")
    ax.axhline(0, color="0.45", linewidth=0.8)
    ax.axvline(0, color="0.45", linewidth=0.8)
    ax.set_xlim(-kernel_lim, kernel_lim)
    ax.set_ylim(-kernel_lim, kernel_lim)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.25)
    ax.set_title("Kernel bandwidth implied by covariance")
    ax.set_xlabel("x distance (km)")
    ax.set_ylabel("y distance (km)")
    ax.legend(loc="upper left", fontsize=8)

    summary = component_summary(S[t], ell=ell)
    title = (
        f"Advection covariance, {time_label(t, start_time, dt_seconds)}\n"
        f"trace_u={summary['trace_u']:.3g}, trace_v={summary['trace_v']:.3g}, "
        f"cross Fro={summary['cross_block_fro']:.3g}"
    )
    if sigma_scale != 1.0:
        title += f", visual Sigma scale={sigma_scale:.1e}"
    if ell is not None:
        title += f", Sigma/D ratio u={summary['sigma_kernel_ratio_u']:.2e}, v={summary['sigma_kernel_ratio_v']:.2e}"
    fig.suptitle(title, fontsize=14)


def save_window_gif(
    out: Path,
    mu: np.ndarray,
    S: np.ndarray,
    ell: float | None,
    start: int,
    frames: int,
    start_time: datetime,
    dt_seconds: float,
    sigma_scale: float = 1.0,
) -> dict[str, float | str | int]:
    import matplotlib.animation as animation
    import matplotlib.pyplot as plt

    times = np.arange(start, min(start + frames, len(mu)))
    max_mean = float(np.nanmax(np.abs(mu[times])))
    max_sd = 0.0
    max_kernel_radius = 0.0
    max_cov = 0.0
    for t in times:
        vals = np.linalg.eigvalsh(symmetrize_psd(S[t, :2, :2]))
        vals = np.r_[vals, np.linalg.eigvalsh(symmetrize_psd(S[t, 2:, 2:]))]
        max_sd = max(max_sd, float(2.0 * np.sqrt(np.max(vals))))
        if ell is not None:
            Du = ell * ell * np.eye(2) + 2.0 * symmetrize_psd(S[t, :2, :2])
            Dv = ell * ell * np.eye(2) + 2.0 * symmetrize_psd(S[t, 2:, 2:])
        else:
            Du = 2.0 * symmetrize_psd(S[t, :2, :2])
            Dv = 2.0 * symmetrize_psd(S[t, 2:, 2:])
        max_kernel_radius = max(
            max_kernel_radius,
            float(2.0 * np.sqrt(np.max(np.linalg.eigvalsh(symmetrize_psd(Du))))),
            float(2.0 * np.sqrt(np.max(np.linalg.eigvalsh(symmetrize_psd(Dv))))),
        )
        cov4 = symmetrize_psd(S[t])
        max_cov = max(max_cov, float(np.nanmax(np.abs(cov4))))
    lim = max(1.0, 1.15 * (max_mean + max_sd))
    kernel_lim = max(1.0, 1.10 * max_kernel_radius)
    cov_vmax = max(max_cov, 1.0e-8)

    fig, axes = plt.subplots(1, 4, figsize=(21.2, 5.2), constrained_layout=True)

    def update(frame_idx: int):
        draw_frame(fig, axes, int(times[frame_idx]), mu, S, ell, lim, kernel_lim, cov_vmax, sigma_scale, start_time, dt_seconds)
        return []

    ani = animation.FuncAnimation(fig, update, frames=len(times), interval=450, blit=False)
    ani.save(out, writer="pillow", fps=2)
    plt.close(fig)
    rows = [component_summary(S[int(t)], ell=ell) for t in times]
    return {
        "gif": str(out),
        "start_index": int(times[0]),
        "end_index": int(times[-1]),
        "start_time": time_label(int(times[0]), start_time, dt_seconds),
        "end_time": time_label(int(times[-1]), start_time, dt_seconds),
        "mean_cross_block_fro": float(np.mean([r["cross_block_fro"] for r in rows])),
        "mean_cross_canonical_corr": float(np.mean([r["cross_canonical_corr"] for r in rows])),
        "mean_sigma_kernel_ratio_u": float(np.mean([r.get("sigma_kernel_ratio_u", np.nan) for r in rows])),
        "mean_sigma_kernel_ratio_v": float(np.mean([r.get("sigma_kernel_ratio_v", np.nan) for r in rows])),
        "visual_sigma_scale": float(sigma_scale),
    }


def save_timeseries_png(
    out: Path,
    mu: np.ndarray,
    S: np.ndarray,
    ell: float | None,
    start: int,
    frames: int,
    start_time: datetime,
    dt_seconds: float,
    sigma_scale: float = 1.0,
) -> dict[str, float | str | int]:
    import matplotlib.pyplot as plt

    times = np.arange(start, min(start + frames, len(mu)))
    rows = [component_summary(S[int(t)], ell=ell) for t in times]
    x = np.arange(len(times))
    labels = [time_label(int(t), start_time, dt_seconds).split()[1] for t in times]
    fig, axes = plt.subplots(3, 1, figsize=(11.5, 8.2), sharex=True, constrained_layout=True)
    axes[0].plot(x, [r["sqrt_trace_u"] for r in rows], color="#1f77b4", marker="o", label=r"$\sqrt{trace(\Sigma_{uu})}$")
    axes[0].plot(x, [r["sqrt_trace_v"] for r in rows], color="#d95f02", marker="o", label=r"$\sqrt{trace(\Sigma_{vv})}$")
    axes[0].set_ylabel("marginal spread")
    axes[0].legend()
    axes[1].plot(x, [r["anisotropy_u"] for r in rows], color="#1f77b4", marker="o", label=r"$A_u$ anisotropy")
    axes[1].plot(x, [r["anisotropy_v"] for r in rows], color="#d95f02", marker="o", label=r"$A_v$ anisotropy")
    axes[1].set_ylabel("anisotropy ratio")
    axes[1].legend()
    axes[2].plot(x, [r["cross_block_fro"] for r in rows], color="#6a3d9a", marker="o", label=r"$||\Sigma_{uv}||_F$")
    axes[2].plot(x, [r["cross_canonical_corr"] for r in rows], color="#008b8b", marker="o", label="canonical cross-correlation")
    if ell is not None:
        axes[2].plot(x, [r["sigma_kernel_ratio_u"] for r in rows], color="0.35", linestyle="--", label=r"$\Sigma_A$ share in $D_u$")
    axes[2].set_ylabel("cross / kernel ratio")
    axes[2].set_xlabel("time")
    axes[2].legend()
    for ax in axes:
        ax.grid(True, alpha=0.25)
    axes[-1].set_xticks(x, labels=labels, rotation=45, ha="right")
    scale_text = "" if sigma_scale == 1.0 else f", visual Sigma scale={sigma_scale:.1e}"
    fig.suptitle(f"Advection covariance summary, {time_label(start, start_time, dt_seconds)} to {time_label(int(times[-1]), start_time, dt_seconds)}{scale_text}")
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return {"png": str(out)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--advection", type=Path, default=Path("/Users/felix/Downloads/eval_on-offline_h72_f72_hybrid/advection_parameters.npz"))
    parser.add_argument("--out", type=Path, default=Path("/Users/felix/Downloads/A_Analy/advection_sigma4d"))
    parser.add_argument("--starts", type=int, nargs="+", default=[4798, 6898, 5616])
    parser.add_argument("--frames", type=int, default=12)
    parser.add_argument("--start-time", default=DEFAULT_START_TIME)
    parser.add_argument("--dt-seconds", type=float, default=DEFAULT_DT_SECONDS)
    parser.add_argument("--no-bias", action="store_true")
    parser.add_argument(
        "--sigma-scale",
        type=float,
        default=1.0,
        help="Counterfactual visualization scale applied to Sigma_A. The learned means are unchanged.",
    )
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    params = np.load(args.advection)
    mu, S = transformed_mu_sigma(params, use_bias=not args.no_bias)
    S = S * float(args.sigma_scale)
    ell = None
    if "ell" in params.files:
        ell_arr = np.asarray(params["ell"], dtype=float)
        ell = float(np.nanmean(ell_arr))

    start_time = parse_time(args.start_time)
    summaries = []
    for start in args.starts:
        end = min(start + args.frames - 1, len(mu) - 1)
        scale_suffix = "" if args.sigma_scale == 1.0 else f"_scale{args.sigma_scale:.0e}"
        prefix = f"sigma4d{scale_suffix}_{time_slug(start, start_time, args.dt_seconds)}_{time_slug(end, start_time, args.dt_seconds)}"
        summaries.append(
            save_window_gif(
                args.out / f"{prefix}.gif",
                mu,
                S,
                ell,
                start,
                args.frames,
                start_time,
                args.dt_seconds,
                sigma_scale=args.sigma_scale,
            )
        )
        summaries.append(
            save_timeseries_png(
                args.out / f"{prefix}_summary.png",
                mu,
                S,
                ell,
                start,
                args.frames,
                start_time,
                args.dt_seconds,
                sigma_scale=args.sigma_scale,
            )
        )
    (args.out / "summary.json").write_text(json.dumps(summaries, indent=2), encoding="utf-8")
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
