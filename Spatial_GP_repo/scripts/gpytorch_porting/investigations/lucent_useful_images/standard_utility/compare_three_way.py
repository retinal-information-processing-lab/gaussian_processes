"""Three-way comparison: DA vs Laplace-standard vs LUT-standard, per cell.

Reads three caches (all gitignored, produced by the runners in this investigation):
  - DA                : ../cache/panels_results.pkl          (useful_image_panels.py)
  - Laplace-standard  : ./cache/panels_results.pkl           (standard_panels.py, live r_max=100)
  - LUT-standard      : ./cache/panels_results_lut.pkl        (lut_panels.py, numerically stable)

Produces, for cells 13 (the blow-up cell) and 3 (the content-difference cell):
  - compare_cell{N}_3way.png : 3-row image grid (DA / Laplace-std / LUT-std), cols = n_train,
    each panel annotated with optimized utility U and firing fr. Laplace U>100 -> red title
    (numerical blow-up); LUT U is finite everywhere.
  - compare_3way_curves.png  : the decisive picture. Top row = optimized utility U vs n_train
    (log y); bottom = firing vs n_train (log y); cols = cell 13, cell 3; one line per method.
    Shows (a) Laplace U spikes to ~1e5 while LUT stays ~1-4 nats (the NUMERICS are fixed),
    and (b) firing is high for BOTH standard variants at badly-fit points (the utility's real
    high-firing taste) while DA stays low (the conditioning term is what tames firing).

Output -> standard_utility/panels/ (gitignored). Additive; no engine/existing-file edits.
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
PARENT = os.path.dirname(HERE)
OUT = os.path.join(HERE, "panels")

CELLS = [13, 3]
BLOWUP_U = 100.0   # Laplace U above this = the r_max=100 numerical blow-up (flag red)


def _load(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"missing cache: {path} (run the corresponding runner first)")
    return pickle.load(open(path, "rb"))


da = _load(os.path.join(PARENT, "cache", "panels_results.pkl"))
lap = _load(os.path.join(HERE, "cache", "panels_results.pkl"))
lut = _load(os.path.join(HERE, "cache", "panels_results_lut.pkl"))

gp_min, gp_max = da["meta"]["gp_min"], da["meta"]["gp_max"]
span = gp_max - gp_min
nts_all = da["meta"]["n_trains"]

METHODS = [("DA utility", da), ("standard (Laplace r_max=100)", lap), ("standard (LUT)", lut)]


def _panel(ax, img, title, flag_red=False):
    sat = np.mean((img < gp_min + 0.01 * span) | (img > gp_max - 0.01 * span)) * 100
    ax.imshow(img, cmap="gray", vmin=gp_min, vmax=gp_max)   # FIXED range (no adaptive scaling)
    ax.set_xticks([]); ax.set_yticks([])
    color = "red" if (flag_red or sat > 10) else "black"
    ax.set_title(title, fontsize=6.6, color=color)


def _fmtU(u):
    return f"{u:.2f}" if abs(u) < 1e3 else f"{u:.1e}"


def image_grid(cell):
    rows = [d["cells"][cell] for _, d in METHODS if cell in d["cells"]]
    labels = [lab for lab, d in METHODS if cell in d["cells"]]
    cols = [nt for nt in nts_all if all(nt in r for r in rows)]
    fig, axes = plt.subplots(len(rows), len(cols),
                             figsize=(1.45 * len(cols), 1.7 * len(rows)), squeeze=False)
    for i, rec in enumerate(rows):
        for j, nt in enumerate(cols):
            d = rec[nt]
            u = d["U_opt"]; fr = d["fr_final"]
            red = (labels[i].startswith("standard (Laplace") and u > BLOWUP_U)
            top = f"n={nt}\n" if i == 0 else ""
            _panel(axes[i][j], d["final_img"], f"{top}U={_fmtU(u)}\nfr={fr:.0f}", flag_red=red)
        axes[i][0].set_ylabel(labels[i].replace(" (", "\n("), fontsize=8)
    fig.suptitle(f"cell {cell}: optimized image, DA vs Laplace-standard vs LUT-standard "
                 f"(U=utility, fr=firing; red = Laplace r_max=100 blow-up / >10% pixels at rails)",
                 fontsize=9, y=1.0)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    p = os.path.join(OUT, f"compare_cell{cell}_3way.png")
    fig.savefig(p, dpi=160, bbox_inches="tight"); plt.close(fig)
    print(f"saved {p}")


def curves():
    fig, axes = plt.subplots(2, len(CELLS), figsize=(5.2 * len(CELLS), 7.2), squeeze=False)
    colors = {"DA utility": "C2", "standard (Laplace r_max=100)": "C3", "standard (LUT)": "C0"}
    for jc, cell in enumerate(CELLS):
        for lab, d in METHODS:
            if cell not in d["cells"]:
                continue
            rec = d["cells"][cell]
            nts = [nt for nt in nts_all if nt in rec]
            U = [rec[nt]["U_opt"] for nt in nts]
            FR = [rec[nt]["fr_final"] for nt in nts]
            # clamp to a positive floor for the log axis (U/fr can be ~0 for DA / well-fit).
            Up = [max(u, 1e-3) for u in U]
            FRp = [max(f, 1e-2) for f in FR]
            axes[0][jc].plot(nts, Up, "o-", color=colors[lab], label=lab, ms=4)
            axes[1][jc].plot(nts, FRp, "o-", color=colors[lab], label=lab, ms=4)
        axes[0][jc].axhline(BLOWUP_U, color="grey", ls=":", alpha=0.6)
        axes[0][jc].set_yscale("log"); axes[1][jc].set_yscale("log")
        axes[0][jc].set_title(f"cell {cell}: optimized utility U", fontsize=10)
        axes[1][jc].set_title(f"cell {cell}: firing at optimum", fontsize=10)
        axes[1][jc].set_xlabel("n_train")
        for r in (0, 1):
            axes[r][jc].grid(alpha=0.3, which="both")
        axes[0][jc].legend(fontsize=7.5)
    axes[0][0].set_ylabel("U_opt (nats, log)")
    axes[1][0].set_ylabel("firing exp(mu+sig2/2) (log)")
    fig.suptitle("LUT removes the standard-utility NUMERICAL blow-up (top: Laplace U -> ~1e5, "
                 "LUT U stays ~1-4 nats), not its high-firing TASTE\n(bottom: both standard "
                 "variants reach high firing at badly-fit points; DA's conditioning keeps it low)",
                 fontsize=10, y=1.0)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    p = os.path.join(OUT, "compare_3way_curves.png")
    fig.savefig(p, dpi=160, bbox_inches="tight"); plt.close(fig)
    print(f"saved {p}")


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    for cell in CELLS:
        image_grid(cell)
    curves()
