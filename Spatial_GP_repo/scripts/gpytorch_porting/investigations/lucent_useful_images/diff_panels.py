"""
RF-change ("does it sharpen?") view of the most-useful-image panels, built
PURELY from the cached natural-start results (cache/panels_results.pkl) -- no GPU,
no refit. Companion to useful_image_panels.py.

Why a difference view: the natural-start optimization only retunes the RF region
(non-RF pixels get ~no kernel gradient, so they stay at the natural scene). In the
plain optimized image the RF change is buried inside a busy natural scene. The
difference (optimized - start) cancels the unchanged background and isolates the RF
modification, so you can SEE whether it sharpens from diffuse -> compact as the model
improves (test_r up, sigma2_g down).

Per cell, 3 rows x n_train cols:
  row 1: optimized image (gray, fixed vmin/vmax = dataset range; OOB flagged red)
  row 2: difference optimized-start (symmetric RdBu_r; the RF-localized change)
  row 3: quantified sharpness text -- test_r, sigma2_g, active-pixel count,
         active bounding-box WxH, peak |diff| (RF contrast)

ACTIVE pixel = |diff| > 0.03 * (gp_max - gp_min), the same 3%-of-range threshold
optimize_image._rf_bbox uses. As the model gets confident the RF change is expected
to get higher-contrast (peak up) and more localized (bbox shrinks).

Reads only numpy arrays + matplotlib -> env-agnostic, instant.
Usage:  python diff_panels.py --cells 3 13 36
"""
import argparse
import os
import pickle

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
matplotlib.rcParams["font.sans-serif"] = ["Liberation Sans", "Arial", "Helvetica", "DejaVu Sans"]
matplotlib.rcParams["svg.fonttype"] = "none"

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "cache", "panels_results.pkl")
OUTP = os.path.join(HERE, "out", "panels")
ACTIVE_FRAC = 0.03  # |diff| > 3% of dataset range = "meaningfully changed" (matches optimize_image)


def active_stats(diff, span):
    """Active-RF descriptors: #px changed, bbox HxW, peak |diff| (RF contrast)."""
    thresh = ACTIVE_FRAC * span
    active = np.abs(diff) > thresh
    n_active = int(active.sum())
    peak = float(np.abs(diff).max())
    if n_active == 0:
        return n_active, peak, None
    rows = np.where(active.any(axis=1))[0]
    cols = np.where(active.any(axis=0))[0]
    h = int(rows.max() - rows.min() + 1)
    w = int(cols.max() - cols.min() + 1)
    return n_active, peak, (h, w)


def build(cell, cell_rec, gp_min, gp_max, out_path):
    span = gp_max - gp_min
    nts = sorted(cell_rec.keys())
    nc = len(nts)
    fig, axes = plt.subplots(3, nc, figsize=(1.7 * nc, 6.2), squeeze=False)

    for j, nt in enumerate(nts):
        d = cell_rec[nt]
        final = d["final_img"]
        start = d["start_img"]
        diff = final - start

        # Row 1: optimized image, fixed dataset-range scaling, OOB flagged.
        ax = axes[0][j]
        ax.imshow(final, cmap="gray", vmin=gp_min, vmax=gp_max)
        ax.set_xticks([]); ax.set_yticks([])
        tol = 1e-5 * span
        oob = float(np.mean((final < gp_min - tol) | (final > gp_max + tol)) * 100)
        ttl = f"n_train={nt}\ntest_r={d['test_r']:+.2f}"
        ax.set_title(ttl, fontsize=8.5, color=("red" if oob > 0 else "black"))

        # Row 2: difference (optimized - start) = the RF-localized change.
        ax = axes[1][j]
        dmax = max(1e-6, np.abs(diff).max())
        ax.imshow(diff, cmap="RdBu_r", vmin=-dmax, vmax=dmax)
        ax.set_xticks([]); ax.set_yticks([])

        # Row 3: quantified sharpness.
        n_active, peak, bbox = active_stats(diff, span)
        bbox_s = f"{bbox[0]}x{bbox[1]}" if bbox else "--"
        ax = axes[2][j]
        ax.axis("off")
        txt = (f"sigma2_g\n  {d['sig2_final']:.3f}\n"
               f"active px\n  {n_active}\n"
               f"bbox\n  {bbox_s}\n"
               f"peak|d|\n  {peak:.2f}\n"
               f"fr~{d['fr_final']:.0f}")
        ax.text(0.02, 0.98, txt, transform=ax.transAxes, ha="left", va="top",
                fontsize=6.6, family="monospace")

    axes[0][0].set_ylabel("optimized", fontsize=9)
    axes[1][0].set_ylabel("change\n(opt - start)", fontsize=9)
    axes[2][0].set_ylabel("RF sharpness", fontsize=9)
    fig.suptitle(f"cell {cell}: RF-change view -- does the optimized stimulus sharpen as the model improves? "
                 f"(M=n_train, DA utility, natural-start; from cache)", fontsize=10.5, y=1.0)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(f"{out_path}.png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out_path}.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", type=int, nargs="*", default=[3, 13, 36])
    args = ap.parse_args()

    with open(CACHE, "rb") as f:
        results = pickle.load(f)
    gp_min = results["meta"]["gp_min"]
    gp_max = results["meta"]["gp_max"]
    os.makedirs(OUTP, exist_ok=True)

    for cell in args.cells:
        if cell not in results["cells"]:
            print(f"cell {cell}: not in cache yet, skipping")
            continue
        print(f"cell {cell}:")
        build(cell, results["cells"][cell], gp_min, gp_max,
              os.path.join(OUTP, f"cell{cell}_diff"))


if __name__ == "__main__":
    main()
