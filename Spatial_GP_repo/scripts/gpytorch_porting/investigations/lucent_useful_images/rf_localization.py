"""
Quantify whether the optimized RF change is LOCALIZED (sharp) or DIFFUSE, cleanly
separating the receptive-field-concentrated change from the global Fourier ripple
that the lucent parameterization spreads over the whole frame.

Motivation: under the lucent FFT parameterization the optimizer moves global Fourier
coefficients, so |optimized - start| has above-threshold pixels across most of the
frame (bbox / active-pixel-count metrics saturate). A radius-based energy concentration
is robust to that diffuse floor: it asks what FRACTION of the change-energy sits in a
small disk around the RF, so a uniform ripple scores ~disk_area/frame_area (~0.06) while
a compact RF scores high.

Per (cell, n_train), from cache/panels_results.pkl (no GPU, no refit):
  diff      = optimized - start                       (the change)
  peak      = argmax of gaussian-smoothed |diff|       (RF location)
  conc_R    = sum(diff**2 within R px of peak) / sum(diff**2)   (concentration; R=15)
  contrast  = max|diff| / median|diff|                 (RF peak above the ripple floor)
Hypothesis: as the model improves (test_r up, sigma2_g down) the change becomes more
concentrated (conc_R up) and stands out more (contrast up).

Usage:  python rf_localization.py --cells 3 13 36
"""
import argparse
import os
import pickle

import numpy as np
from scipy.ndimage import gaussian_filter
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
matplotlib.rcParams["font.sans-serif"] = ["Liberation Sans", "Arial", "Helvetica", "DejaVu Sans"]
matplotlib.rcParams["svg.fonttype"] = "none"

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "cache", "panels_results.pkl")
OUTP = os.path.join(HERE, "out", "panels")
R_DISK = 15        # concentration radius (px); pi*R^2/108^2 ~ 0.06 = uniform-ripple baseline
SMOOTH = 2.0       # gaussian sigma for robust peak location
UNIFORM_BASE = np.pi * R_DISK ** 2 / (108 * 108)


def localize(diff, R=R_DISK, smooth=SMOOTH):
    ad = np.abs(diff)
    peak_rc = np.unravel_index(np.argmax(gaussian_filter(ad, smooth)), ad.shape)
    E = diff ** 2
    yy, xx = np.mgrid[0:diff.shape[0], 0:diff.shape[1]]
    within = (yy - peak_rc[0]) ** 2 + (xx - peak_rc[1]) ** 2 <= R * R
    conc = float(E[within].sum() / E.sum())
    contrast = float(ad.max() / max(np.median(ad), 1e-9))
    return peak_rc, conc, contrast


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", type=int, nargs="*", default=[3, 13, 36])
    args = ap.parse_args()

    d = pickle.load(open(CACHE, "rb"))
    cells = [c for c in args.cells if c in d["cells"]]
    os.makedirs(OUTP, exist_ok=True)

    series = {}  # cell -> (nts, test_r, conc, contrast, sig2)
    for cell in cells:
        rec = d["cells"][cell]
        nts = sorted(rec)
        rows = []
        print(f"\n=== cell {cell} (R={R_DISK}px, uniform-ripple baseline conc={UNIFORM_BASE:.3f}) ===")
        print(f"{'nt':>4} {'test_r':>7} {'sig2_g':>8} {'conc_R':>7} {'contrast':>9}")
        for nt in nts:
            diff = rec[nt]["final_img"] - rec[nt]["start_img"]
            _, conc, contrast = localize(diff)
            rows.append((nt, rec[nt]["test_r"], conc, contrast, rec[nt]["sig2_final"]))
            print(f"{nt:>4} {rec[nt]['test_r']:>+7.2f} {rec[nt]['sig2_final']:>8.3f} "
                  f"{conc:>7.3f} {contrast:>9.1f}")
        series[cell] = np.array(rows)

    # Figure: concentration vs n_train (left) and vs test_r (right).
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11, 4.2))
    colors = {3: "C0", 13: "C1", 36: "C2"}
    for cell, arr in series.items():
        nt, tr, conc = arr[:, 0], arr[:, 1], arr[:, 2]
        axL.plot(nt, conc, "-o", color=colors.get(cell, "C3"), label=f"cell {cell}", ms=4)
        axR.scatter(tr, conc, color=colors.get(cell, "C3"), label=f"cell {cell}", s=30)
    axL.axhline(UNIFORM_BASE, color="gray", ls=":", label=f"uniform ripple ({UNIFORM_BASE:.2f})")
    axL.set_xlabel("n_train (M = n_train)"); axL.set_ylabel(f"change-energy concentration (R={R_DISK}px)")
    axL.set_title("concentration vs training size"); axL.legend(fontsize=8); axL.grid(alpha=0.3)
    axR.axhline(UNIFORM_BASE, color="gray", ls=":")
    axR.set_xlabel("test_r (model quality)"); axR.set_ylabel(f"concentration (R={R_DISK}px)")
    axR.set_title("concentration vs model quality"); axR.legend(fontsize=8); axR.grid(alpha=0.3)
    fig.suptitle("RF-change localization: is the optimized stimulus a compact RF (high) or diffuse (~ripple)?",
                 fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out = os.path.join(OUTP, "rf_localization.png")
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
