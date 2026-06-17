"""
Standard-utility version of the per-cell panels — a CONTROL for the DA utility.

Same pipeline as the parent `useful_image_panels.py` (M = n_train, same 5 cells, same
n_train grid, FRESH fits / no rank-1 warm start, lucent Fourier+sigmoid parameterization)
but the acquisition is the STANDARD utility  U = H_marg - E[H_noise]  — no
distribution-aware conditioning, no Monte-Carlo over images, no lambda: it is DETERMINISTIC.

Purpose: see whether the optimized images are really different from the DA ones, i.e.
isolate what the distribution-aware conditioning adds on top of lucent's bounding +
smoothness. The standard utility favors high firing / high contrast, BUT lucent's sigmoid
keeps every pixel in [GP_MIN, GP_MAX], so it cannot blow up — worst case is
high-contrast-but-in-range (flagged in the figure).

Reuses the parent's `optimize_image` (with utility_mode='standard') and `build_figure`.
Output -> standard_utility/panels/ + standard_utility/cache/ (gitignored).
"""
import argparse
import gc
import os
import pickle
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
PARENT = os.path.dirname(HERE)
if PARENT not in sys.path:
    sys.path.insert(0, PARENT)

import gp_models
import optimize_image as oi
import useful_image_panels as panels   # reuse build_figure
import acquisition

CELLS = [13, 3, 1, 11, 12]
N_TRAINS = list(range(50, 301, 25))   # 50,75,...,300  -- same grid as the DA run
SEED = 42
DECAY_POWER = 1.0
LR = 0.05
N_STEPS = 150
POOL_SEED = 0
R_MAX = 100
CHUNK = 512
IMG_SIDE = 108

CACHE = os.path.join(HERE, "cache")
OUTP = os.path.join(HERE, "panels")


def pool_utilities_standard(model, lik, X_pool, chunk=CHUNK):
    """STANDARD utility of every pool image (batched, deterministic). Returns (n_pool,)."""
    outs = []
    with torch.no_grad():
        for i in range(0, X_pool.shape[0], chunk):
            res = acquisition.standard_utility(model, lik, X_pool[i:i + chunk],
                                               r_max=R_MAX, adaptive_r_max=False)
            outs.append(res["utility"].detach())
    return torch.cat(outs).cpu().numpy()


def run_one(cell, nt, X_all, gp_min, gp_max):
    span = gp_max - gp_min
    model, lik, idx_train, test_r = gp_models.train_default_gpy(cell, nt, seed=SEED)  # M=n_train
    X_pool, _ = gp_models.pool_complement(X_all, idx_train)

    pu = pool_utilities_standard(model, lik, X_pool)          # distribution over all images
    best_i = int(np.argmax(pu))
    U_start = float(pu[best_i])
    start_img_gp = X_pool[best_i].reshape(IMG_SIDE, IMG_SIDE).cpu().numpy()
    start01 = (start_img_gp - gp_min) / span

    res = oi.optimize_image(model, lik, X_pool, gp_min, gp_max, decay_power=DECAY_POWER,
                            lr=LR, n_steps=N_STEPS, n_mc=1, pool_seed=POOL_SEED,
                            start_image01=start01, utility_mode="standard", verbose=False)
    U_opt = float(res["traj"]["utility"][-1])

    rec = dict(
        test_r=float(test_r), n_pool=int(X_pool.shape[0]),
        start_img=start_img_gp.astype(np.float32),
        final_img=res["img_gp_final"].astype(np.float32),
        U_start=U_start, U_opt=U_opt,
        pool_mean=float(pu.mean()), pool_std=float(pu.std()),
        pool_min=float(pu.min()), pool_max=float(pu.max()),
        pool_p50=float(np.percentile(pu, 50)), pool_p95=float(np.percentile(pu, 95)),
        sig2_final=float(res["traj"]["sigma2_g"][-1]),
        fr_final=float(res["traj"]["firing_mean"][-1]),
    )
    del model, lik
    gc.collect(); torch.cuda.empty_cache()
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", type=int, nargs="*", default=CELLS)
    ap.add_argument("--preview", action="store_true")
    args = ap.parse_args()

    os.makedirs(CACHE, exist_ok=True); os.makedirs(OUTP, exist_ok=True)
    cache_path = os.path.join(CACHE, "panels_results.pkl")
    cells = [args.cells[0]] if args.preview else args.cells
    nts = [50, 150, 300] if args.preview else N_TRAINS

    X_all, gp_min, gp_max = gp_models.load_pool()
    print(f"pool {tuple(X_all.shape)}  GP_MIN={gp_min:.4f} GP_MAX={gp_max:.4f}  | STANDARD utility")

    if os.path.exists(cache_path) and not args.preview:
        results = pickle.load(open(cache_path, "rb"))
    else:
        results = dict(meta=dict(gp_min=gp_min, gp_max=gp_max, M="n_train",
                                 n_trains=N_TRAINS, utility="standard"), cells={})

    t0 = time.time()
    for cell in cells:
        print(f"\n===== cell {cell} =====")
        cell_rec = results["cells"].get(cell, {})
        for nt in nts:
            if nt in cell_rec:
                print(f"  nt={nt}: cached"); continue
            tt = time.time()
            rec = run_one(cell, nt, X_all, gp_min, gp_max)
            cell_rec[nt] = rec
            dU = rec["U_opt"] - rec["U_start"]
            print(f"  nt={nt:3d}: test_r={rec['test_r']:+.2f} n_pool={rec['n_pool']} | "
                  f"Ustart={rec['U_start']:+.4f} Uopt={rec['U_opt']:+.4f} ΔU={dU:+.4f} | "
                  f"poolμ={rec['pool_mean']:+.4f} σ={rec['pool_std']:.4f} "
                  f"fr={rec['fr_final']:.1f} ({time.time()-tt:.0f}s)")
            results["cells"][cell] = cell_rec
            if not args.preview:
                pickle.dump(results, open(cache_path, "wb"))
        tag = "_preview" if args.preview else ""
        panels.build_figure(cell, cell_rec, gp_min, gp_max,
                            os.path.join(OUTP, f"cell{cell}_panels_std{tag}"),
                            utility_label="standard utility")
    print(f"\nDONE in {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
