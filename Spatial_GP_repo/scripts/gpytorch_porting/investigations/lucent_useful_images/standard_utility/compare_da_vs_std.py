"""
Direct side-by-side: optimized image from DA utility vs standard utility, per cell.
Top row = DA, bottom row = standard; columns = n_train. Firing rate annotated (the key
difference). Reads both caches (parent DA cache + this folder's standard cache).
Output -> standard_utility/panels/compare_cell{N}.png
"""
import os
import pickle

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
matplotlib.rcParams["font.sans-serif"] = ["Liberation Sans", "Arial", "Helvetica", "DejaVu Sans"]

HERE = os.path.dirname(os.path.abspath(__file__))
PARENT = os.path.dirname(HERE)
da = pickle.load(open(os.path.join(PARENT, "cache", "panels_results.pkl"), "rb"))
st = pickle.load(open(os.path.join(HERE, "cache", "panels_results.pkl"), "rb"))
gp_min, gp_max = da["meta"]["gp_min"], da["meta"]["gp_max"]
nts = da["meta"]["n_trains"]
OUT = os.path.join(HERE, "panels")


def panel(ax, img, title):
    span = gp_max - gp_min
    sat = np.mean((img < gp_min + 0.01 * span) | (img > gp_max - 0.01 * span)) * 100
    ax.imshow(img, cmap="gray", vmin=gp_min, vmax=gp_max)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(title, fontsize=7.5, color=("red" if sat > 10 else "black"))


for cell in da["cells"]:
    if cell not in st["cells"]:
        continue
    dr, sr = da["cells"][cell], st["cells"][cell]
    cols = [nt for nt in nts if nt in dr and nt in sr]
    fig, axes = plt.subplots(2, len(cols), figsize=(1.6 * len(cols), 3.8), squeeze=False)
    for j, nt in enumerate(cols):
        panel(axes[0][j], dr[nt]["final_img"], f"n={nt}\nfr {dr[nt]['fr_final']:.0f}")
        panel(axes[1][j], sr[nt]["final_img"], f"fr {sr[nt]['fr_final']:.0f}")
    axes[0][0].set_ylabel("DA\nutility", fontsize=10)
    axes[1][0].set_ylabel("standard\nutility", fontsize=10)
    fig.suptitle(f"cell {cell}: optimized image — DA vs standard utility "
                 f"(firing rate fr annotated; both lucent-bounded; red title = >10% pixels at the rails)",
                 fontsize=10, y=1.0)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(os.path.join(OUT, f"compare_cell{cell}.png"), dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"saved compare_cell{cell}.png")
