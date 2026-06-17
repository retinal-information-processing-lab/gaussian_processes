"""
Per-cell utility-optimization panels across a fine model-quality (n_train) axis.

For each cell and each n_train in {50,75,...,300}:
  1. Train a default_gpy model (M = n_train, full inducing set; seed=42).
  2. Compute the DA utility of EVERY available (pool) image  -> the utility
     distribution; the highest-utility image is the optimization START.
  3. Optimize from that start (lucent Fourier+sigmoid param, bounded) -> FINAL image.
  4. Record the utility gain, in absolute terms AND relative to the pool utility
     distribution (z-score / sigma units).

Per-cell figure (3 rows x 11 cols, cols = n_train):
  row 1: start image (best available natural image)
  row 2: optimized image (OOB% in RED in the title if any pixel exceeds the range)
  row 3: text box with the utility-gain numbers (no plot)

All utilities use the SAME conditioning subset x_samples (seeded, sample_lambda=False)
so start / optimized / pool values are directly comparable. Output -> out/panels/.
"""
import argparse
import gc
import os
import pickle
import time

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
matplotlib.rcParams["font.sans-serif"] = ["Liberation Sans", "Arial", "Helvetica", "DejaVu Sans"]
matplotlib.rcParams["svg.fonttype"] = "none"

import gp_models
import optimize_image as oi
import acquisition

CELLS = [13, 3, 1, 11, 12]
N_TRAINS = list(range(50, 301, 25))   # 50,75,...,300  (11 values)
# M = n_train at all times: every training point is an inducing point => the full
# (non-sparse) variational GP at each training size. So M is set per-run to nt below.
SEED = 42
DECAY_POWER = 1.0
LR = 0.05
N_STEPS = 150
N_MC = 48
POOL_SEED = 0
R_MAX = 100
CHUNK = 512  # candidate-batch size for the pool utility distribution

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "cache")
OUTP = os.path.join(HERE, "out", "panels")
IMG_SIDE = 108


def pool_utilities(model, lik, X_pool, x_samples, chunk=CHUNK):
    """DA utility of every pool image (batched over candidates). Returns (n_pool,)."""
    outs = []
    with torch.no_grad():
        for i in range(0, X_pool.shape[0], chunk):
            xc = X_pool[i:i + chunk]
            res = acquisition.distribution_aware_utility(
                model, lik, xc, x_samples, r_max=R_MAX, adaptive_r_max=False,
                sample_lambda=False)
            outs.append(res["utility"].detach())
    return torch.cat(outs).cpu().numpy()


def run_one(cell, nt, X_all, gp_min, gp_max):
    span = gp_max - gp_min
    # M = n_train: full (non-sparse) inducing set at this training size.
    model, lik, idx_train, test_r = gp_models.train_default_gpy(
        cell_id=cell, n_train=nt, M=nt, seed=SEED)
    X_pool, _ = gp_models.pool_complement(X_all, idx_train)

    # Same conditioning subset that optimize_image will use (pool_seed=POOL_SEED).
    torch.manual_seed(POOL_SEED)
    perm = torch.randperm(X_pool.shape[0], device=X_pool.device)[:N_MC]
    x_samples = X_pool[perm]

    # Utility distribution over ALL available pool images -> best = start.
    pu = pool_utilities(model, lik, X_pool, x_samples)
    best_i = int(np.argmax(pu))
    U_start = float(pu[best_i])
    start_img_gp = X_pool[best_i].reshape(IMG_SIDE, IMG_SIDE).cpu().numpy()
    start_img01 = (start_img_gp - gp_min) / span

    # Optimize from the best available image.
    res = oi.optimize_image(model, lik, X_pool, gp_min, gp_max,
                            decay_power=DECAY_POWER, lr=LR, n_steps=N_STEPS,
                            n_mc=N_MC, pool_seed=POOL_SEED, start_image01=start_img01,
                            verbose=False)
    U_opt = float(res["traj"]["utility"][-1])
    final_img = res["img_gp_final"]
    sig2_final = float(res["traj"]["sigma2_g"][-1])
    fr_final = float(res["traj"]["firing_mean"][-1])

    rec = dict(
        test_r=float(test_r), n_pool=int(X_pool.shape[0]),
        start_img=start_img_gp.astype(np.float32),
        final_img=final_img.astype(np.float32),
        U_start=U_start, U_opt=U_opt,
        pool_mean=float(pu.mean()), pool_std=float(pu.std()),
        pool_min=float(pu.min()), pool_max=float(pu.max()),
        pool_p50=float(np.percentile(pu, 50)), pool_p95=float(np.percentile(pu, 95)),
        sig2_final=sig2_final, fr_final=fr_final,
    )
    del model, lik
    gc.collect(); torch.cuda.empty_cache()
    return rec


def _oob_sat(img, gp_min, gp_max):
    span = gp_max - gp_min
    tol = 1e-5 * span
    oob = float(np.mean((img < gp_min - tol) | (img > gp_max + tol)) * 100)
    sat = float(np.mean((img < gp_min + 0.01 * span) | (img > gp_max - 0.01 * span)) * 100)
    return oob, sat


def build_figure(cell, cell_rec, gp_min, gp_max, out_path, utility_label="DA utility"):
    nts = [nt for nt in N_TRAINS if nt in cell_rec]
    nc = len(nts)
    fig, axes = plt.subplots(3, nc, figsize=(1.7 * nc, 5.6), squeeze=False)

    for j, nt in enumerate(nts):
        d = cell_rec[nt]
        # Row 1: start (best available natural image)
        ax = axes[0][j]
        ax.imshow(d["start_img"], cmap="gray", vmin=gp_min, vmax=gp_max)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"n_train={nt}\n(test_r={d['test_r']:.2f})", fontsize=9)

        # Row 2: optimized (OOB red if present)
        ax = axes[1][j]
        ax.imshow(d["final_img"], cmap="gray", vmin=gp_min, vmax=gp_max)
        ax.set_xticks([]); ax.set_yticks([])
        oob, sat = _oob_sat(d["final_img"], gp_min, gp_max)
        if oob > 0:
            ax.set_title(f"OOB {oob:.1f}%", color="red", fontsize=8.5)
        elif sat > 10:
            ax.set_title(f"sat {sat:.0f}%", color="darkorange", fontsize=8.5)
        else:
            ax.set_title("in-range", fontsize=8, color="green")

        # Row 3: utility-gain text box
        ax = axes[2][j]
        ax.axis("off")
        U0, U1 = d["U_start"], d["U_opt"]
        dU = U1 - U0
        dU_pct = 100 * dU / abs(U0) if abs(U0) > 1e-9 else float("nan")
        sigma = d["pool_std"] if d["pool_std"] > 1e-12 else 1e-12
        z_opt = (U1 - d["pool_mean"]) / sigma
        z_start = (U0 - d["pool_mean"]) / sigma
        dU_sig = dU / sigma
        txt = (f"best avail U\n  {U0:+.4f}\n"
               f"optimized U\n  {U1:+.4f}\n"
               f"ΔU = {dU:+.4f}\n  ({dU_pct:+.0f}%)\n"
               f"——\npool U dist\n μ={d['pool_mean']:+.4f}\n σ={d['pool_std']:.4f}\n"
               f"best @ {z_start:+.1f}σ\nopt  @ {z_opt:+.1f}σ\nΔU = {dU_sig:+.1f}σ")
        ax.text(0.02, 0.98, txt, transform=ax.transAxes, ha="left", va="top",
                fontsize=6.3, family="monospace")

    axes[0][0].set_ylabel("start\n(best available)", fontsize=9)
    axes[1][0].set_ylabel("optimized", fontsize=9)
    axes[2][0].set_ylabel("utility gain", fontsize=9)
    fig.suptitle(f"cell {cell}: most-useful-image optimization vs model quality "
                 f"(M = n_train, full inducing set; {utility_label}; bounded by lucent param)",
                 fontsize=11, y=1.0)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(f"{out_path}.png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out_path}.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", type=int, nargs="*", default=CELLS)
    ap.add_argument("--preview", action="store_true",
                    help="1 cell, n_train in {50,150,300} only (fast validation)")
    args = ap.parse_args()

    os.makedirs(CACHE, exist_ok=True)
    os.makedirs(OUTP, exist_ok=True)
    cache_path = os.path.join(CACHE, "panels_results.pkl")

    cells = [args.cells[0]] if args.preview else args.cells
    nts = [50, 150, 300] if args.preview else N_TRAINS

    X_all, gp_min, gp_max = gp_models.load_pool()
    print(f"pool {tuple(X_all.shape)}  GP_MIN={gp_min:.4f} GP_MAX={gp_max:.4f}")

    if os.path.exists(cache_path) and not args.preview:
        with open(cache_path, "rb") as f:
            results = pickle.load(f)
    else:
        results = dict(meta=dict(gp_min=gp_min, gp_max=gp_max, M="n_train",
                                 n_trains=N_TRAINS), cells={})

    t0 = time.time()
    for cell in cells:
        print(f"\n===== cell {cell} =====")
        cell_rec = results["cells"].get(cell, {})
        for nt in nts:
            if nt in cell_rec:
                print(f"  nt={nt}: cached, skipping"); continue
            tt = time.time()
            rec = run_one(cell, nt, X_all, gp_min, gp_max)
            cell_rec[nt] = rec
            dU = rec["U_opt"] - rec["U_start"]
            print(f"  nt={nt:3d}: test_r={rec['test_r']:+.2f} n_pool={rec['n_pool']} | "
                  f"Ustart={rec['U_start']:+.4f} Uopt={rec['U_opt']:+.4f} "
                  f"ΔU={dU:+.4f} | poolμ={rec['pool_mean']:+.4f} σ={rec['pool_std']:.4f} "
                  f"({time.time()-tt:.0f}s)")
            results["cells"][cell] = cell_rec
            if not args.preview:
                with open(cache_path, "wb") as f:
                    pickle.dump(results, f)
        tag = "_preview" if args.preview else ""
        build_figure(cell, cell_rec, gp_min, gp_max,
                     os.path.join(OUTP, f"cell{cell}_panels{tag}"))
    print(f"\nDONE in {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
