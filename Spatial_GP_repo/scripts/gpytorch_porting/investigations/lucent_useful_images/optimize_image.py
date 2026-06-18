"""
Optimize a synthetic image to MAXIMIZE the distribution-aware (DA) utility of a
trained default_gpy GP model, using lucent's Fourier parameterization as a
natural-image prior + sigmoid bounding (no pixel clipping, no rescaling).

Pipeline per optimization step:
    fft params --(lucent fft_image, 1/f^decay_power spectral prior)--> raw
             --(sigmoid)--> img01 in (0,1)
             --(affine GP_MIN..GP_MAX)--> x_cand (1, 11664) in GP space
             --(distribution_aware_utility)--> utility
    loss = -utility ; loss.backward() ; Adam.step()

Why this tames the runaway that raw pixel-space ascent suffers:
  - sigmoid bounding => pixels in (GP_MIN, GP_MAX) => bounded C-norm => bounded
    marginal entropy => bounded utility (no divergence to infinity / borders).
  - 1/f^decay_power spectral prior => smooth, low-frequency images preferred in
    parameter space => suppresses the high-contrast checkerboard solutions.
  - DA utility's conditional term rewards angular alignment with the natural pool.

The DA objective is run with sample_lambda=False (deterministic posterior-mean
conditioning) so the optimization is reproducible and the gradient is smooth.

Reproducible: same args => same result (training seed, lucent init seed, and the
pool subset are all seeded; sample_lambda=False removes the only stochastic op).
"""
import argparse
import os
import sys
import time

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Vector-editor-friendly font (Affinity/Illustrator); metric-compatible with Arial.
matplotlib.rcParams["font.sans-serif"] = ["Liberation Sans", "Arial", "Helvetica", "DejaVu Sans"]
matplotlib.rcParams["svg.fonttype"] = "none"

import gp_models
import lucent_param
import acquisition  # GP_PORT is on sys.path via gp_models
from lut.lut_utility import lut_standard_utility  # LUT-backed standard utility (vendored)

IMG_SIDE = 108
R_MAX = 100  # Laplace truncation (default_params.json utility.r_max)


# --------------------------------------------------------------------------- #
# Monitoring helpers
# --------------------------------------------------------------------------- #
def marginal_logf_moments(model, likelihood, x_cand):
    """Return (mu_g, sigma2_g) = log-firing-rate mean/var at x_cand (no grad)."""
    with torch.no_grad():
        lam_m, lam_v = acquisition.get_gp_marginal_moments(model, x_cand)
        A = likelihood.A.squeeze()
        lambda0 = likelihood.lambda0.squeeze()
        mu_g = (A * lam_m + lambda0).item()
        sigma2_g = (A ** 2 * lam_v).item()
    return mu_g, sigma2_g


def z_safe(mu_g, sigma2_g, r_max=R_MAX):
    """Laplace-validity pre-check (da_utility_theory.md): >2 safe, <1.5 unreliable."""
    return (np.log(r_max) - mu_g) / np.sqrt(max(sigma2_g, 1e-12))


# --------------------------------------------------------------------------- #
# Core optimization
# --------------------------------------------------------------------------- #
def optimize_image(model, likelihood, X_pool, gp_min, gp_max, *,
                   decay_power=1.0, lr=0.05, n_steps=400, n_mc=48, sd=0.01,
                   init_seed=0, pool_seed=0, sample_lambda=False, fft=True,
                   start_image01=None, r_max=R_MAX, utility_mode="da", verbose=True):
    """Run image optimization (DA or standard utility). Returns a results dict.

    start_image01: optional (W,W) array in (0,1) to initialize from (e.g. the
    most-useful natural image). None -> near-gray random Fourier init.
    utility_mode: 'da' (distribution-aware, uses the x_samples conditioning set),
    'standard' (H_marg - E[H_noise] via the live r_max=100 Laplace utility,
    deterministic, ignores x_samples; blows up past the ~4-sigma gate), or
    'standard_lut' (the SAME standard utility read from the numerically-stable LUT,
    lut/lut_utility.py -- finite/monotone everywhere, no r_max=100 blow-up).
    NOTE: DA results so far use sample_lambda=False (biased; see why_sample_lambda.tex).
    """
    device = next(model.parameters()).device

    # Fixed pool subset for the MC estimate of H_cond (seeded -> reproducible).
    torch.manual_seed(pool_seed)
    perm = torch.randperm(X_pool.shape[0], device=X_pool.device)[:n_mc]
    x_samples = X_pool[perm].to(device)

    # lucent grayscale Fourier parameterization + affine to GP space.
    params, img01_fn = lucent_param.make_grayscale_param(
        W=IMG_SIDE, decay_power=decay_power, sd=sd, seed=init_seed, fft=fft,
        start_image01=start_image01)
    gp_fn = lucent_param.make_gp_image_fn(img01_fn, gp_min, gp_max)

    img01_start = img01_fn().detach().reshape(IMG_SIDE, IMG_SIDE).cpu().numpy()

    optimizer = torch.optim.Adam(params, lr=lr)

    traj = {k: [] for k in ("utility", "H_marg", "H_cond", "mu_g", "sigma2_g",
                            "firing_mean", "z_safe", "grad_norm")}
    t0 = time.time()
    for step in range(n_steps):
        optimizer.zero_grad()
        x_cand = gp_fn()  # (1, 11664), requires grad through params
        if utility_mode == "standard":
            out = acquisition.standard_utility(
                model, likelihood, x_cand, r_max=r_max, adaptive_r_max=False)
            util_val = out["utility"]
            h_marg_val, h_cond_val = util_val, torch.zeros_like(util_val)
        elif utility_mode == "standard_lut":
            # Same standard utility, but read from the numerically-stable LUT (no r_max=100
            # blow-up) in a torch-differentiable form. Marginal-only, like 'standard'.
            out = lut_standard_utility(model, likelihood, x_cand)
            util_val = out["utility"]
            h_marg_val, h_cond_val = util_val, torch.zeros_like(util_val)
        else:
            out = acquisition.distribution_aware_utility(
                model, likelihood, x_cand, x_samples,
                r_max=r_max, adaptive_r_max=False, sample_lambda=sample_lambda)
            util_val, h_marg_val, h_cond_val = out["utility"], out["H_marg"], out["H_cond"]
        utility = util_val.sum()
        loss = -utility
        loss.backward()
        gnorm = float(params[0].grad.norm())
        optimizer.step()

        mu_g, sigma2_g = marginal_logf_moments(model, likelihood, gp_fn())
        traj["utility"].append(float(util_val.item()))
        traj["H_marg"].append(float(h_marg_val.item()))
        traj["H_cond"].append(float(h_cond_val.item()))
        traj["mu_g"].append(mu_g)
        traj["sigma2_g"].append(sigma2_g)
        traj["firing_mean"].append(float(np.exp(mu_g + 0.5 * sigma2_g)))
        traj["z_safe"].append(float(z_safe(mu_g, sigma2_g, r_max)))
        traj["grad_norm"].append(gnorm)

        if verbose and (step % max(1, n_steps // 8) == 0 or step == n_steps - 1):
            print(f"  step {step:4d}: U={traj['utility'][-1]:+.4f}  "
                  f"H_marg={traj['H_marg'][-1]:.3f}  H_cond={traj['H_cond'][-1]:.3f}  "
                  f"mu_g={mu_g:+.2f}  sig2_g={sigma2_g:.3f}  "
                  f"fr~{traj['firing_mean'][-1]:.1f}  z_safe={traj['z_safe'][-1]:.2f}")
            sys.stdout.flush()

    with torch.no_grad():
        img01_final = img01_fn().reshape(IMG_SIDE, IMG_SIDE).cpu().numpy()
        x_final = gp_fn()
        img_gp_final = x_final.reshape(IMG_SIDE, IMG_SIDE).cpu().numpy()

    return {
        "img01_start": img01_start,
        "img01_final": img01_final,
        "img_gp_final": img_gp_final,
        "traj": {k: np.asarray(v) for k, v in traj.items()},
        "x_samples_idx": perm.detach().cpu().numpy(),
        "params": dict(decay_power=decay_power, lr=lr, n_steps=n_steps, n_mc=n_mc,
                       sd=sd, init_seed=init_seed, pool_seed=pool_seed,
                       sample_lambda=sample_lambda, fft=fft, r_max=r_max,
                       utility_mode=utility_mode,
                       start=("natural" if start_image01 is not None else "gray")),
        "elapsed_s": time.time() - t0,
        "gp_min": gp_min, "gp_max": gp_max,
    }


def best_natural_image(model, likelihood, X_pool, x_samples, gp_min, gp_max,
                       n_eval=256, eval_seed=1, r_max=R_MAX, sample_lambda=False):
    """Find the natural pool image with the highest DA utility (reference).

    Evaluated over a random subset of the pool for speed.
    """
    device = next(model.parameters()).device
    torch.manual_seed(eval_seed)
    idx = torch.randperm(X_pool.shape[0], device=X_pool.device)[:n_eval]
    best_u, best_img, best_i = -1e30, None, -1
    with torch.no_grad():
        for j in range(idx.shape[0]):
            xc = X_pool[idx[j]].unsqueeze(0).to(device)
            out = acquisition.distribution_aware_utility(
                model, likelihood, xc, x_samples,
                r_max=r_max, adaptive_r_max=False, sample_lambda=sample_lambda)
            u = float(out["utility"].item())
            if u > best_u:
                best_u, best_i = u, int(idx[j].item())
                best_img = xc.reshape(IMG_SIDE, IMG_SIDE).cpu().numpy()
    return best_img, best_u, best_i


# --------------------------------------------------------------------------- #
# Plotting
# --------------------------------------------------------------------------- #
def _imshow_bounded(ax, img, gp_min, gp_max, title):
    """imshow with FIXED vmin/vmax = dataset global range.

    OOB (red): pixels strictly beyond the physical display range [gp_min, gp_max]
    (uses a float tolerance so the sigmoid-saturated rail is not falsely flagged).
    SAT (orange): pixels pinned within 1% of a rail -- the honest "overblown
    contrast" signal, since lucent's sigmoid keeps values in-range by construction.
    The actual [min,max] is always printed.
    """
    span = gp_max - gp_min
    tol = 1e-5 * span
    oob = np.mean((img < gp_min - tol) | (img > gp_max + tol)) * 100.0
    sat = np.mean((img < gp_min + 0.01 * span) | (img > gp_max - 0.01 * span)) * 100.0
    ax.imshow(img, cmap="gray", vmin=gp_min, vmax=gp_max)
    ax.set_xticks([]); ax.set_yticks([])
    rng = f"  [{img.min():+.2f},{img.max():+.2f}]"
    if oob > 0:
        ax.set_title(f"{title}{rng}\n[OOB {oob:.1f}%]", color="red", fontsize=9)
    elif sat > 10:
        ax.set_title(f"{title}{rng}  [sat {sat:.0f}%]", color="darkorange", fontsize=9)
    else:
        ax.set_title(f"{title}{rng}", fontsize=9)


def _rf_bbox(diff, gp_min, gp_max, frac=0.03, pad=6):
    """Bounding box of pixels the optimizer actually changed (the active RF)."""
    thresh = frac * (gp_max - gp_min)
    active = np.abs(diff) > thresh
    if not active.any():
        return None
    rows = np.where(active.any(axis=1))[0]
    cols = np.where(active.any(axis=0))[0]
    r0, r1 = max(0, rows.min() - pad), min(diff.shape[0], rows.max() + pad + 1)
    c0, c1 = max(0, cols.min() - pad), min(diff.shape[1], cols.max() + pad + 1)
    return r0, r1, c0, c1


def plot_diagnostic(res, model_info, best_nat=None, out_path=None):
    """8-panel diagnostic for one optimization run."""
    gp_min, gp_max = res["gp_min"], res["gp_max"]
    tr = res["traj"]
    steps = np.arange(len(tr["utility"]))
    start_gp = res["img01_start"] * (gp_max - gp_min) + gp_min
    final_gp = res["img_gp_final"]
    diff = final_gp - start_gp

    fig, axes = plt.subplots(2, 4, figsize=(17, 8.5))

    _imshow_bounded(axes[0, 0], start_gp, gp_min, gp_max, "start (near-gray)")
    _imshow_bounded(axes[0, 1], final_gp, gp_min, gp_max, "DA-optimized (lucent)")
    if best_nat is not None:
        _imshow_bounded(axes[0, 2], best_nat[0], gp_min, gp_max,
                        f"best natural (U={best_nat[1]:+.3f})")
    else:
        axes[0, 2].axis("off")

    # Difference (final - start) reveals the active RF region; symmetric colormap.
    ax = axes[0, 3]
    dmax = max(1e-6, np.abs(diff).max())
    ax.imshow(diff, cmap="RdBu_r", vmin=-dmax, vmax=dmax)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title("difference (final - start)", fontsize=9)

    # Zoom on the active RF region of the optimized image.
    ax = axes[1, 3]
    bbox = _rf_bbox(diff, gp_min, gp_max)
    if bbox is not None:
        r0, r1, c0, c1 = bbox
        ax.imshow(final_gp[r0:r1, c0:c1], cmap="gray", vmin=gp_min, vmax=gp_max)
        ax.set_title(f"RF zoom [{r1-r0}x{c1-c0}px]", fontsize=9)
    else:
        ax.set_title("RF zoom (no active region)", fontsize=9)
    ax.set_xticks([]); ax.set_yticks([])

    # Utility decomposition
    ax = axes[1, 0]
    ax.plot(steps, tr["utility"], label="U = H_marg - H_cond", color="C0")
    ax.plot(steps, tr["H_marg"], label="H_marg", color="C1", ls="--")
    ax.plot(steps, tr["H_cond"], label="H_cond", color="C2", ls=":")
    ax.set_xlabel("step"); ax.set_ylabel("nats"); ax.set_title("utility decomposition")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # Firing rate + log-firing mean
    ax = axes[1, 1]
    ax.plot(steps, tr["firing_mean"], color="C3")
    ax.set_xlabel("step"); ax.set_ylabel("mean firing exp(mu_g+sig2/2)")
    ax.set_title("predicted firing rate"); ax.grid(alpha=0.3)
    ax2 = ax.twinx()
    ax2.plot(steps, tr["mu_g"], color="C4", ls="--", alpha=0.7)
    ax2.set_ylabel("mu_g (log-firing mean)", color="C4")

    # Laplace validity: sigma2_g + z_safe
    ax = axes[1, 2]
    ax.plot(steps, tr["sigma2_g"], color="C5")
    ax.axhline(1.0, color="green", ls=":", alpha=0.6, label="sig2=1 (<1% Laplace err)")
    ax.axhline(6.0, color="orange", ls=":", alpha=0.6, label="sig2=6 (~5% err)")
    ax.set_xlabel("step"); ax.set_ylabel("sigma2_g", color="C5")
    ax.set_title("Laplace validity monitor"); ax.legend(fontsize=7); ax.grid(alpha=0.3)
    ax3 = ax.twinx()
    ax3.plot(steps, tr["z_safe"], color="C6", ls="--", alpha=0.7)
    ax3.axhline(2.0, color="C6", ls=":", alpha=0.3)
    ax3.set_ylabel("z_safe (>2 safe)", color="C6")

    sup = (f"{model_info}  |  decay_power={res['params']['decay_power']} "
           f"lr={res['params']['lr']} steps={res['params']['n_steps']} "
           f"n_mc={res['params']['n_mc']}  |  final U={tr['utility'][-1]:+.4f} "
           f"sig2_g={tr['sigma2_g'][-1]:.2f} fr~{tr['firing_mean'][-1]:.1f}")
    fig.suptitle(sup, fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    if out_path:
        fig.savefig(out_path, dpi=140, bbox_inches="tight")
        print(f"  saved {out_path}")
    plt.close(fig)


# --------------------------------------------------------------------------- #
# CLI: optimize one model
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", type=int, default=8)
    ap.add_argument("--n-train", type=int, default=300)
    ap.add_argument("--M", type=int, default=None,
                    help="inducing points; default None => M=n_train (full inducing set)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--decay-power", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--n-steps", type=int, default=400)
    ap.add_argument("--n-mc", type=int, default=48)
    ap.add_argument("--start", choices=["gray", "natural"], default="gray",
                    help="gray: near-gray Fourier init. natural: start from the "
                         "most-useful natural image (init params from its FFT).")
    ap.add_argument("--no-best-natural", action="store_true")
    ap.add_argument("--tag", type=str, default="")
    args = ap.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    out_dir = os.path.join(here, "out", "single_runs")  # keep out/ root tidy
    os.makedirs(out_dir, exist_ok=True)

    m_label = args.M if args.M is not None else "n_train"
    print(f"Loading pool + training default_gpy cell={args.cell} "
          f"n_train={args.n_train} M={m_label} seed={args.seed} ...")
    X_all, gp_min, gp_max = gp_models.load_pool()
    model, lik, idx_train, test_r = gp_models.train_default_gpy(
        cell_id=args.cell, n_train=args.n_train, M=args.M, seed=args.seed)
    X_pool, _ = gp_models.pool_complement(X_all, idx_train)
    print(f"  test_r={test_r:.4f}  A={float(lik.A):.4f}  lambda0={float(lik.lambda0):+.3f}")

    # Best natural image: a reference, and the start point when --start natural.
    # x_samples matches the optimizer's pool subset (pool_seed=0) for consistency.
    best_nat = None
    if (not args.no_best_natural) or args.start == "natural":
        torch.manual_seed(0)
        perm = torch.randperm(X_pool.shape[0], device=X_pool.device)[:args.n_mc]
        x_samples = X_pool[perm]
        print("Finding best natural image (DA utility over pool subset) ...")
        best_nat = best_natural_image(model, lik, X_pool, x_samples, gp_min, gp_max)
        print(f"  best natural U={best_nat[1]:+.4f} (pool idx {best_nat[2]})")

    start_image01 = None
    if args.start == "natural":
        start_image01 = (best_nat[0] - gp_min) / (gp_max - gp_min)  # (108,108) in (0,1)

    print(f"Optimizing (start={args.start} decay_power={args.decay_power} lr={args.lr} "
          f"n_steps={args.n_steps} n_mc={args.n_mc}) ...")
    res = optimize_image(model, lik, X_pool, gp_min, gp_max,
                         decay_power=args.decay_power, lr=args.lr,
                         n_steps=args.n_steps, n_mc=args.n_mc,
                         start_image01=start_image01)
    print(f"  done in {res['elapsed_s']:.1f}s")

    tag = ("_" + args.tag) if args.tag else ""
    out_path = os.path.join(
        out_dir,
        f"opt_cell{args.cell}_nt{args.n_train}_dp{args.decay_power}_{args.start}{tag}.png")
    info = (f"cell {args.cell} | n_train={args.n_train} M={m_label} | test_r={test_r:.3f}"
            f" | start={args.start}")
    plot_diagnostic(res, info, best_nat=best_nat, out_path=out_path)


if __name__ == "__main__":
    main()
