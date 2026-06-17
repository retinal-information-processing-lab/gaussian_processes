"""
Build the deliverable figures from cache/grid_results.pkl (produced by run_grid.py).

Figures (rows = cells, cols = n_train = model-quality axis):
  A. fig_useful_natural   -- the most-useful image at each model quality, started
     from a fixed per-cell natural image. Headline: bounded, natural, not overblown.
  B. fig_useful_rf        -- gray-start optimized image: the isolated preferred
     stimulus / RF the model wants, on a neutral background. Shows the
     epistemic->firing transition as the model becomes confident.
  C. fig_rf_difference    -- (natural-final - natural-start): the RF-localized change
     the optimization makes to the natural image (what the optimizer "adds").

All panels: fixed vmin/vmax = dataset global range [GP_MIN, GP_MAX]; per-panel OOB
red-title check + saturation flag; Liberation Sans for vector-editor compatibility.
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
matplotlib.rcParams["pdf.fonttype"] = 42

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "cache", "grid_results.pkl")


def _panel(ax, img, gp_min, gp_max, cmap="gray", symmetric=False):
    """imshow with fixed range + OOB/sat reporting. Returns (oob%, sat%)."""
    span = gp_max - gp_min
    tol = 1e-5 * span
    oob = float(np.mean((img < gp_min - tol) | (img > gp_max + tol)) * 100)
    sat = float(np.mean((img < gp_min + 0.01 * span) | (img > gp_max - 0.01 * span)) * 100)
    if symmetric:
        m = max(1e-6, np.abs(img).max())
        ax.imshow(img, cmap=cmap, vmin=-m, vmax=m)
    else:
        ax.imshow(img, cmap=cmap, vmin=gp_min, vmax=gp_max)
    ax.set_xticks([]); ax.set_yticks([])
    return oob, sat


def _corner(ax, text, loc="upper left", color="white"):
    x, y, ha, va = (0.03, 0.97, "left", "top") if loc == "upper left" else (0.97, 0.03, "right", "bottom")
    ax.text(x, y, text, transform=ax.transAxes, ha=ha, va=va, fontsize=7.5,
            color=color, weight="bold",
            bbox=dict(facecolor="black", alpha=0.45, pad=1.2, edgecolor="none"))


def build_grid(results, which, out_base, title, cells=None):
    """which in {'nat','gray','diff'}."""
    meta = results["meta"]
    gp_min, gp_max = meta["gp_min"], meta["gp_max"]
    if cells is None:
        cells = [c for c in meta["cells"] if c in results["cells"]]
    n_trains = meta["n_trains"]
    nr, nc = len(cells), len(n_trains)

    fig, axes = plt.subplots(nr, nc, figsize=(2.5 * nc, 2.7 * nr), squeeze=False)

    for i, cell in enumerate(cells):
        rec = results["cells"][cell]
        for j, nt in enumerate(n_trains):
            ax = axes[i][j]
            d = rec["by_nt"].get(nt)
            if d is None:
                ax.axis("off"); continue
            if which == "nat":
                img = d["nat"]["img_gp"]; fin = d["nat"]["final"]
                oob, sat = _panel(ax, img, gp_min, gp_max)
            elif which == "gray":
                img = d["gray"]["img_gp"]; fin = d["gray"]["final"]
                oob, sat = _panel(ax, img, gp_min, gp_max)
            else:  # diff = natural-final - natural-start(ref)
                img = d["nat"]["img_gp"] - rec["ref_img_gp"]; fin = d["nat"]["final"]
                oob, sat = _panel(ax, img, gp_min, gp_max, cmap="RdBu_r", symmetric=True)

            _corner(ax, f"r={d['test_r']:.2f}", "upper left")
            _corner(ax, f"fr {fin['firing']:.0f}\nσ² {fin['sigma2_g']:.2f}", "lower right")
            # OOB/sat frame only meaningful for actual images (not signed diffs).
            if which != "diff":
                if oob > 0:
                    for s in ax.spines.values():
                        s.set_color("red"); s.set_linewidth(2.5)
                elif sat > 10:
                    for s in ax.spines.values():
                        s.set_color("darkorange"); s.set_linewidth(2.0)

            if i == 0:
                ax.set_title(f"n_train = {nt}", fontsize=11)
            if j == 0:
                ax.set_ylabel(f"cell {cell}", fontsize=11)

    fig.suptitle(title + "   (model quality increases left → right;  "
                 "vmin/vmax = dataset global range; no panel exceeds it)",
                 fontsize=11, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    p = f"{out_base}.png"
    fig.savefig(p, dpi=200, bbox_inches="tight")
    print(f"  saved {p}")
    plt.close(fig)


def build_hero(results, cell, out_base):
    """Per-cell summary: 3 rows (natural / preferred-stimulus / RF-change) x n_train."""
    meta = results["meta"]
    gp_min, gp_max = meta["gp_min"], meta["gp_max"]
    rec = results["cells"][cell]
    n_trains = meta["n_trains"]
    rows = [("natural-start (bounded, looks natural)", "nat", "gray", False),
            ("gray-start (preferred stimulus)", "gray", "gray", False),
            ("RF change (optimized − natural)", "diff", "RdBu_r", True)]
    nr, nc = len(rows), len(n_trains)
    fig, axes = plt.subplots(nr, nc, figsize=(2.6 * nc, 2.8 * nr), squeeze=False)

    for i, (label, which, cmap, sym) in enumerate(rows):
        for j, nt in enumerate(n_trains):
            ax = axes[i][j]
            d = rec["by_nt"].get(nt)
            if d is None:
                ax.axis("off"); continue
            if which == "nat":
                img = d["nat"]["img_gp"]
            elif which == "gray":
                img = d["gray"]["img_gp"]
            else:
                img = d["nat"]["img_gp"] - rec["ref_img_gp"]
            _panel(ax, img, gp_min, gp_max, cmap=cmap, symmetric=sym)
            fin = d["nat"]["final"] if which != "gray" else d["gray"]["final"]
            if i == 0:
                ax.set_title(f"n_train = {nt}\n(test_r = {d['test_r']:.2f})", fontsize=10)
            if i == 2:
                _corner(ax, f"σ²={fin['sigma2_g']:.2f}  fr={fin['firing']:.0f}",
                        "lower right")
            if j == 0:
                ax.set_ylabel(label, fontsize=9.5)

    fig.suptitle(f"cell {cell}: most-useful image vs model quality "
                 f"(DA utility, lucent; bounded by construction)", fontsize=12, y=0.998)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    p = f"{out_base}.png"
    fig.savefig(p, dpi=200, bbox_inches="tight")
    print(f"  saved {p}")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default=CACHE)
    ap.add_argument("--out-dir", default=os.path.join(HERE, "refined"))
    ap.add_argument("--cells", type=int, nargs="*", default=None,
                    help="cells (and order) to show in the grids (default: all in cache)")
    ap.add_argument("--hero-cells", type=int, nargs="*", default=None,
                    help="cells to make per-cell hero figures for (default: --cells)")
    args = ap.parse_args()

    with open(args.cache, "rb") as f:
        results = pickle.load(f)
    os.makedirs(args.out_dir, exist_ok=True)

    cells = args.cells  # None -> all in meta order
    build_grid(results, "nat", os.path.join(args.out_dir, "fig_useful_natural"),
               "Most-useful image (DA utility, lucent) — started from a natural image", cells)
    build_grid(results, "gray", os.path.join(args.out_dir, "fig_useful_rf"),
               "Most-useful image (DA utility, lucent) — preferred stimulus on neutral gray", cells)
    build_grid(results, "diff", os.path.join(args.out_dir, "fig_rf_difference"),
               "RF-localized change to the natural image (optimized − natural start)", cells)

    hero_cells = args.hero_cells if args.hero_cells is not None else (cells or list(results["cells"].keys()))
    for cell in hero_cells:
        if cell in results["cells"]:
            build_hero(results, cell, os.path.join(args.out_dir, f"fig_hero_cell{cell}"))


if __name__ == "__main__":
    main()
