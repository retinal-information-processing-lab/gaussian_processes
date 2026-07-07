"""
GRAY-START version of the most-useful-image ladder for the testbed trio (3, 13, 36).

Companion to useful_image_panels.py (which starts from the best natural image). Here
every column starts from the SAME near-gray Fourier init, so the optimized receptive
field stands out against a flat background -- the cleanest VISUAL of "does the optimized
stimulus crystallize into a compact RF as the model improves?" It directly checks the
energy-vs-contrast split found from the natural-start cache (rf_localization.py): cell 13
(reaches test_r ~0.96) should form a clean compact RF; cells 3 / 36 (plateau ~0.73-0.83)
should show a sharp peak embedded in a diffuse field.

Per (cell, n_train) in {50,75,...,300}:
  1. Train default_gpy (M = n_train, seed=42)  -- same convention as the natural panels.
  2. Optimize from GRAY (start_image01=None) with the SAME DA-utility settings.
  3. Cache: final image, gray reference, sigma2_g, firing, localization (concentration +
     contrast of final-gray), AND the full per-step trajectory (so convergence can be
     inspected later with no refit).

All optimization params match useful_image_panels.py for direct comparability.
Cache: cache/gray_panels_results.pkl (gitignored, incremental/resumable).
Figures -> out/panels/cell{ID}_gray.png (3 rows: optimized | change | numbers).

Usage:  python gray_panels.py --cells 3 13 36
"""
import argparse
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

# Match useful_image_panels.py exactly (comparability).
N_TRAINS = list(range(50, 301, 25))   # 50,75,...,300
SEED = 42
DECAY_POWER = 1.0
LR = 0.05
N_STEPS = 150
N_MC = 48
POOL_SEED = 0
R_DISK = 15          # localization radius (px), matches rf_localization.py
SMOOTH = 2.0

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "cache")
OUTP = os.path.join(HERE, "out", "panels")
IMG_SIDE = 108


def localize(diff, R=R_DISK, smooth=SMOOTH):
    """Concentration of change-energy within R px of the peak + peak/median contrast."""
    ad = np.abs(diff)
    pr, pc = np.unravel_index(np.argmax(gaussian_filter(ad, smooth)), ad.shape)
    E = diff ** 2
    yy, xx = np.mgrid[0:diff.shape[0], 0:diff.shape[1]]
    within = (yy - pr) ** 2 + (xx - pc) ** 2 <= R * R
    conc = float(E[within].sum() / E.sum())
    contrast = float(ad.max() / max(np.median(ad), 1e-9))
    return conc, contrast


def run_one(cell, nt, X_all, gp_min, gp_max, sample_lambda=False):
    model, lik, idx_train, test_r = gp_models.train_default_gpy(
        cell_id=cell, n_train=nt, M=nt, seed=SEED)
    X_pool, _ = gp_models.pool_complement(X_all, idx_train)

    res = oi.optimize_image(model, lik, X_pool, gp_min, gp_max,
                            decay_power=DECAY_POWER, lr=LR, n_steps=N_STEPS,
                            n_mc=N_MC, pool_seed=POOL_SEED, start_image01=None,
                            sample_lambda=sample_lambda, verbose=False)
    span = gp_max - gp_min
    gray_start_gp = res["img01_start"] * span + gp_min      # the near-gray init, GP space
    final_gp = res["img_gp_final"]
    conc, contrast = localize(final_gp - gray_start_gp)

    rec = dict(
        test_r=float(test_r), n_pool=int(X_pool.shape[0]),
        gray_start_img=gray_start_gp.astype(np.float32),
        final_img=final_gp.astype(np.float32),
        U_opt=float(res["traj"]["utility"][-1]),
        sig2_final=float(res["traj"]["sigma2_g"][-1]),
        fr_final=float(res["traj"]["firing_mean"][-1]),
        conc=conc, contrast=contrast,
        traj={k: np.asarray(v, dtype=np.float32) for k, v in res["traj"].items()},
    )
    del model, lik
    gc.collect(); torch.cuda.empty_cache()
    return rec


def build_figure(cell, cell_rec, gp_min, gp_max, out_path, sample_lambda=False):
    span = gp_max - gp_min
    nts = sorted(cell_rec.keys())
    nc = len(nts)
    fig, axes = plt.subplots(3, nc, figsize=(1.7 * nc, 6.2), squeeze=False)
    for j, nt in enumerate(nts):
        d = cell_rec[nt]
        final = d["final_img"]; diff = final - d["gray_start_img"]

        ax = axes[0][j]
        ax.imshow(final, cmap="gray", vmin=gp_min, vmax=gp_max)
        ax.set_xticks([]); ax.set_yticks([])
        tol = 1e-5 * span
        oob = float(np.mean((final < gp_min - tol) | (final > gp_max + tol)) * 100)
        ax.set_title(f"n_train={nt}\ntest_r={d['test_r']:+.2f}",
                     fontsize=8.5, color=("red" if oob > 0 else "black"))

        ax = axes[1][j]
        dmax = max(1e-6, np.abs(diff).max())
        ax.imshow(diff, cmap="RdBu_r", vmin=-dmax, vmax=dmax)
        ax.set_xticks([]); ax.set_yticks([])

        ax = axes[2][j]; ax.axis("off")
        txt = (f"sigma2_g\n  {d['sig2_final']:.3f}\n"
               f"conc_R\n  {d['conc']:.3f}\n"
               f"contrast\n  {d['contrast']:.0f}\n"
               f"fr~{d['fr_final']:.0f}")
        ax.text(0.02, 0.98, txt, transform=ax.transAxes, ha="left", va="top",
                fontsize=6.6, family="monospace")
    axes[0][0].set_ylabel("optimized\n(gray start)", fontsize=9)
    axes[1][0].set_ylabel("change\n(opt - gray)", fontsize=9)
    axes[2][0].set_ylabel("RF sharpness", fontsize=9)
    sl_txt = "sample_lambda=True (unbiased)" if sample_lambda else "sample_lambda=False"
    fig.suptitle(f"cell {cell}: gray-start most-useful-image vs model quality "
                 f"(M=n_train, DA utility, {sl_txt}; RF on flat gray)", fontsize=10.5, y=1.0)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(f"{out_path}.png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out_path}.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", type=int, nargs="*", default=[3, 13, 36])
    ap.add_argument("--sample-lambda", action="store_true",
                    help="unbiased DA: sample lambda from the posterior at each conditioning "
                         "image instead of using its mean (adds MC noise; separate cache/figures)")
    args = ap.parse_args()
    sl = args.sample_lambda
    suffix = "_sl" if sl else ""
    os.makedirs(CACHE, exist_ok=True); os.makedirs(OUTP, exist_ok=True)
    cache_path = os.path.join(CACHE, f"gray_panels{suffix}_results.pkl")

    X_all, gp_min, gp_max = gp_models.load_pool()
    print(f"pool {tuple(X_all.shape)}  GP_MIN={gp_min:.4f} GP_MAX={gp_max:.4f}  sample_lambda={sl}")

    if os.path.exists(cache_path):
        results = pickle.load(open(cache_path, "rb"))
    else:
        results = dict(meta=dict(gp_min=gp_min, gp_max=gp_max, M="n_train",
                                 n_trains=N_TRAINS, n_steps=N_STEPS, start="gray",
                                 sample_lambda=sl), cells={})

    t0 = time.time()
    for cell in args.cells:
        print(f"\n===== cell {cell} (gray start, sample_lambda={sl}) =====")
        cell_rec = results["cells"].get(cell, {})
        for nt in N_TRAINS:
            if nt in cell_rec:
                print(f"  nt={nt}: cached, skipping"); continue
            tt = time.time()
            rec = run_one(cell, nt, X_all, gp_min, gp_max, sample_lambda=sl)
            cell_rec[nt] = rec
            print(f"  nt={nt:3d}: test_r={rec['test_r']:+.2f} | Uopt={rec['U_opt']:+.4f} "
                  f"sig2={rec['sig2_final']:.3f} conc={rec['conc']:.3f} "
                  f"contrast={rec['contrast']:.0f} fr~{rec['fr_final']:.0f} "
                  f"({time.time()-tt:.0f}s)")
            results["cells"][cell] = cell_rec
            pickle.dump(results, open(cache_path, "wb"))
        build_figure(cell, cell_rec, gp_min, gp_max,
                     os.path.join(OUTP, f"cell{cell}_gray{suffix}"), sample_lambda=sl)
    print(f"\nDONE in {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
