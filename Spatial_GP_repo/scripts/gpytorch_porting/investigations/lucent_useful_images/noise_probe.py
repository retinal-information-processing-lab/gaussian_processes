"""
Fig B / n_mc probe: how much does a single sample_lambda=True (unbiased DA) realization
vary, and does raising n_mc shrink that variance? (Answers "is n_mc=48 too small for the
unbiased utility?" -> yes if the cross-seed spread shrinks as n_mc grows.)

For each cell in {3,13,36}: train ONCE at n_train=300 (M=300, seed 42), then run
sample_lambda=True optimizations over n_mc x seed, reusing the model. All optimization
params match the ladder (n_steps=150, decay_power=1.0, lr=0.05, gray start).

Cache: cache/noise_probe.pkl (incremental/resumable). Figures:
  cell{N}_sl_noise.png   per-cell: rows=n_mc, cols=[seed0..seedK, mean]  (the spread)
  sl_noise_summary.png   cross-seed spread (mean pairwise RMS/range) + conc std vs n_mc

Usage: python noise_probe.py
"""
import gc
import os
import pickle
import time

import numpy as np
import torch
from scipy.ndimage import gaussian_filter
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
matplotlib.rcParams["font.sans-serif"] = ["Liberation Sans", "Arial", "Helvetica", "DejaVu Sans"]
matplotlib.rcParams["svg.fonttype"] = "none"

import gp_models
import optimize_image as oi

CELLS = [3, 13, 36]
N_TRAIN = 300           # confident regime (RF well-formed)
N_MCS = [48, 96, 192]   # the n_mc lever
SEEDS = [0, 1, 2]       # realizations (pool_seed)
N_STEPS = 150
DECAY_POWER = 1.0
LR = 0.05
SEED_TRAIN = 42
R_DISK = 15
SMOOTH = 2.0
COLORS = {3: "C0", 13: "C1", 36: "C2"}

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "cache")
OUTP = os.path.join(HERE, "out", "panels")


def localize(diff, R=R_DISK, smooth=SMOOTH):
    ad = np.abs(diff)
    pr, pc = np.unravel_index(np.argmax(gaussian_filter(ad, smooth)), ad.shape)
    E = diff ** 2
    yy, xx = np.mgrid[0:diff.shape[0], 0:diff.shape[1]]
    within = (yy - pr) ** 2 + (xx - pc) ** 2 <= R * R
    return float(E[within].sum() / E.sum()), float(ad.max() / max(np.median(ad), 1e-9))


def pairwise_rms(imgs, span):
    """Mean pairwise RMS between realizations, as a fraction of dataset range."""
    vals = []
    for i in range(len(imgs)):
        for j in range(i + 1, len(imgs)):
            vals.append(np.sqrt(np.mean((imgs[i] - imgs[j]) ** 2)) / span)
    return float(np.mean(vals)) if vals else 0.0


def run(cache_path, X_all, gp_min, gp_max):
    span = gp_max - gp_min
    results = pickle.load(open(cache_path, "rb")) if os.path.exists(cache_path) else \
        {"meta": dict(gp_min=gp_min, gp_max=gp_max, n_train=N_TRAIN, n_mcs=N_MCS,
                      seeds=SEEDS, n_steps=N_STEPS), "cells": {}}
    for cell in CELLS:
        model, lik, idx, test_r = gp_models.train_default_gpy(cell, N_TRAIN, M=N_TRAIN, seed=SEED_TRAIN)
        X_pool, _ = gp_models.pool_complement(X_all, idx)
        rec = results["cells"].get(cell, {"test_r": float(test_r), "runs": {}})
        for n_mc in N_MCS:
            for seed in SEEDS:
                key = f"nmc{n_mc}_seed{seed}"
                if key in rec["runs"]:
                    print(f"cell {cell} {key}: cached"); continue
                tt = time.time()
                res = oi.optimize_image(model, lik, X_pool, gp_min, gp_max,
                                        decay_power=DECAY_POWER, lr=LR, n_steps=N_STEPS,
                                        n_mc=n_mc, pool_seed=seed, start_image01=None,
                                        sample_lambda=True, verbose=False)
                final = res["img_gp_final"]
                gray0 = res["img01_start"] * span + gp_min
                conc, contrast = localize(final - gray0)
                rec["runs"][key] = dict(n_mc=n_mc, seed=seed, final_img=final.astype(np.float32),
                                        conc=conc, contrast=contrast,
                                        sig2=float(res["traj"]["sigma2_g"][-1]),
                                        fr=float(res["traj"]["firing_mean"][-1]))
                results["cells"][cell] = rec
                pickle.dump(results, open(cache_path, "wb"))
                print(f"cell {cell} {key}: conc={conc:.3f} contrast={contrast:.0f} "
                      f"sig2={rec['runs'][key]['sig2']:.3f} ({time.time()-tt:.0f}s)")
        del model, lik
        gc.collect(); torch.cuda.empty_cache()
    return results


def figures(results, gp_min, gp_max):
    span = gp_max - gp_min
    # Per-cell spread panels: rows = n_mc, cols = [seed0..seedK, mean].
    for cell in CELLS:
        rec = results["cells"][cell]
        nc = len(SEEDS) + 1
        fig, ax = plt.subplots(len(N_MCS), nc, figsize=(1.7 * nc, 1.7 * len(N_MCS) + 0.6),
                               squeeze=False)
        for i, n_mc in enumerate(N_MCS):
            imgs = [rec["runs"][f"nmc{n_mc}_seed{s}"]["final_img"] for s in SEEDS]
            for j, s in enumerate(SEEDS):
                ax[i][j].imshow(imgs[j], cmap="gray", vmin=gp_min, vmax=gp_max)
                if i == 0:
                    ax[i][j].set_title(f"seed {s}", fontsize=8)
            mean_img = np.mean(imgs, axis=0)
            ax[i][-1].imshow(mean_img, cmap="gray", vmin=gp_min, vmax=gp_max)
            if i == 0:
                ax[i][-1].set_title("mean", fontsize=8)
            ax[i][0].set_ylabel(f"n_mc={n_mc}\nspread={pairwise_rms(imgs, span):.2f}", fontsize=8)
            for j in range(nc):
                ax[i][j].set_xticks([]); ax[i][j].set_yticks([])
        fig.suptitle(f"cell {cell}: sample_lambda=True realizations at n_train={N_TRAIN} "
                     f"(test_r={rec['test_r']:+.2f}) -- does spread shrink with n_mc?", fontsize=10)
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        out = os.path.join(OUTP, f"cell{cell}_sl_noise.png")
        fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
        print(f"  saved {out}")

    # Summary: cross-seed spread + concentration std vs n_mc.
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11, 4.2))
    for cell in CELLS:
        rec = results["cells"][cell]
        spreads, concstd = [], []
        for n_mc in N_MCS:
            imgs = [rec["runs"][f"nmc{n_mc}_seed{s}"]["final_img"] for s in SEEDS]
            concs = [rec["runs"][f"nmc{n_mc}_seed{s}"]["conc"] for s in SEEDS]
            spreads.append(pairwise_rms(imgs, span))
            concstd.append(float(np.std(concs)))
        axL.plot(N_MCS, spreads, "-o", color=COLORS[cell], label=f"cell {cell}", ms=5)
        axR.plot(N_MCS, concstd, "-o", color=COLORS[cell], label=f"cell {cell}", ms=5)
    axL.set_xlabel("n_mc (MC conditioning images)"); axL.set_ylabel("cross-seed image spread (RMS/range)")
    axL.set_title(f"realization spread vs n_mc (n_train={N_TRAIN})")
    axL.set_xticks(N_MCS); axL.legend(fontsize=8); axL.grid(alpha=0.3)
    axR.set_xlabel("n_mc"); axR.set_ylabel("concentration std across seeds")
    axR.set_title("concentration stability vs n_mc")
    axR.set_xticks(N_MCS); axR.legend(fontsize=8); axR.grid(alpha=0.3)
    fig.suptitle("Is n_mc=48 too small for the unbiased utility? (lower spread = more stable)",
                 fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out = os.path.join(OUTP, "sl_noise_summary.png")
    fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"  saved {out}")


def main():
    os.makedirs(CACHE, exist_ok=True); os.makedirs(OUTP, exist_ok=True)
    X_all, gp_min, gp_max = gp_models.load_pool()
    print(f"pool {tuple(X_all.shape)}  n_train={N_TRAIN}  n_mcs={N_MCS}  seeds={SEEDS}")
    t0 = time.time()
    results = run(os.path.join(CACHE, "noise_probe.pkl"), X_all, gp_min, gp_max)
    figures(results, gp_min, gp_max)
    print(f"DONE in {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
