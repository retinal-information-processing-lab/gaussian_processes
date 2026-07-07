"""
Fig A: dedicated sample_lambda False-vs-True comparison for the gray-start ladder.
Reads cache/gray_panels_results.pkl (False, mean-lambda) and
cache/gray_panels_sl_results.pkl (True, one sampled-lambda realization). No GPU.

Per cell -> cell{N}_sl_compare.png (3 rows x n_train):
  row1 False-optimized | row2 True-optimized | row3 (True-False) map (blank=agree).
Summary -> sl_divergence.png: RMS(True-False)/range vs n_train (left) and concentration
  False (dashed) vs True (solid) vs n_train (right).
"""
import os
import pickle

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
matplotlib.rcParams["font.sans-serif"] = ["Liberation Sans", "Arial", "Helvetica", "DejaVu Sans"]
matplotlib.rcParams["svg.fonttype"] = "none"

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "cache")
OUTP = os.path.join(HERE, "out", "panels")
CELLS = [3, 13, 36]
COLORS = {3: "C0", 13: "C1", 36: "C2"}


def per_cell(cell, F, T, gp_min, gp_max):
    span = gp_max - gp_min
    fc, tc = F["cells"][cell], T["cells"][cell]
    nts = sorted(fc)
    nc = len(nts)
    fig, ax = plt.subplots(3, nc, figsize=(1.7 * nc, 5.8), squeeze=False)
    for j, nt in enumerate(nts):
        f, t = fc[nt], tc[nt]
        d = t["final_img"] - f["final_img"]
        rms = float(np.sqrt(np.mean(d ** 2))) / span
        ax[0][j].imshow(f["final_img"], cmap="gray", vmin=gp_min, vmax=gp_max)
        ax[0][j].set_title(f"n={nt}\ntest_r={f['test_r']:+.2f}", fontsize=8)
        ax[1][j].imshow(t["final_img"], cmap="gray", vmin=gp_min, vmax=gp_max)
        dm = max(1e-6, np.abs(d).max())
        ax[2][j].imshow(d, cmap="RdBu_r", vmin=-dm, vmax=dm)
        ax[2][j].set_title(f"RMS {rms:.2f}", fontsize=8, color=("red" if rms > 0.15 else "black"))
        for r in range(3):
            ax[r][j].set_xticks([]); ax[r][j].set_yticks([])
    ax[0][0].set_ylabel("False\n(mean-lambda)", fontsize=9)
    ax[1][0].set_ylabel("True\n(sampled)", fontsize=9)
    ax[2][0].set_ylabel("True - False", fontsize=9)
    fig.suptitle(f"cell {cell}: sample_lambda False vs True (gray-start, M=n_train, DA utility)",
                 fontsize=11, y=1.0)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out = os.path.join(OUTP, f"cell{cell}_sl_compare.png")
    fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"  saved {out}")


def summary(F, T, gp_min, gp_max):
    span = gp_max - gp_min
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11, 4.2))
    for cell in CELLS:
        fc, tc = F["cells"][cell], T["cells"][cell]
        nts = sorted(fc)
        rms = [float(np.sqrt(np.mean((tc[nt]["final_img"] - fc[nt]["final_img"]) ** 2))) / span
               for nt in nts]
        axL.plot(nts, rms, "-o", color=COLORS[cell], label=f"cell {cell}", ms=4)
        axR.plot(nts, [fc[nt]["conc"] for nt in nts], "--o", color=COLORS[cell], ms=3, alpha=0.55)
        axR.plot(nts, [tc[nt]["conc"] for nt in nts], "-s", color=COLORS[cell], ms=4,
                 label=f"cell {cell}")
    axL.axhline(0.15, color="red", ls=":", alpha=0.5, label="0.15 (material)")
    axL.set_xlabel("n_train"); axL.set_ylabel("RMS(True-False) / range")
    axL.set_title("single-realization divergence"); axL.legend(fontsize=8); axL.grid(alpha=0.3)
    axR.set_xlabel("n_train"); axR.set_ylabel("concentration (dashed=False, solid=True)")
    axR.set_title("concentration agreement"); axR.legend(fontsize=8); axR.grid(alpha=0.3)
    fig.suptitle("sample_lambda False vs True: divergence + concentration", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out = os.path.join(OUTP, "sl_divergence.png")
    fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"  saved {out}")


def main():
    F = pickle.load(open(os.path.join(CACHE, "gray_panels_results.pkl"), "rb"))
    T = pickle.load(open(os.path.join(CACHE, "gray_panels_sl_results.pkl"), "rb"))
    gp_min, gp_max = F["meta"]["gp_min"], F["meta"]["gp_max"]
    os.makedirs(OUTP, exist_ok=True)
    for cell in CELLS:
        per_cell(cell, F, T, gp_min, gp_max)
    summary(F, T, gp_min, gp_max)


if __name__ == "__main__":
    main()
