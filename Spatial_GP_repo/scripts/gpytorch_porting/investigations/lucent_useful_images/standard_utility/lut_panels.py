"""LUT-backed standard-utility panels -- the numerically-stable re-run of standard_panels.py.

IDENTICAL pipeline to standard_panels.py (same cells, n_train grid, M=n_train, seed=42,
lucent Fourier+sigmoid param, fresh fits, start = best-available image) EXCEPT the utility
is read from the precomputed LUT (lut/lut_utility.py) instead of the live r_max=100 Laplace
utility. The LUT utility is finite/monotone everywhere, so the documented blow-up
(cell 13: U=295,800 nats / firing 26,605 at n=275) cannot happen -- the standard-vs-DA
comparison is then fair (numerics removed as a confound).

The pool ranking that picks the START image uses the SAME LUT utility (pool_utilities_lut),
so "start = best-available" is consistent with the utility being optimized.

Single-sources the experimental constants from standard_panels (no drift); only the utility
BACKEND differs. Output -> standard_utility/panels/cell{N}_panels_lut.png +
standard_utility/cache/panels_results_lut.pkl (both gitignored). Additive: edits no engine
file and no existing file (standard_panels.py is imported, not modified).

Run:  <pytorch_gpytorch python> lut_panels.py --cells 13 3
      <pytorch_gpytorch python> lut_panels.py --preview        # 1 cell, n in {50,150,300}
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
import useful_image_panels as panels          # reuse build_figure
from lut.lut_utility import lut_standard_utility
# Single-source the experimental knobs from the Laplace-standard runner (no drift).
from standard_panels import (CELLS, N_TRAINS, SEED, DECAY_POWER, LR, N_STEPS,
                             POOL_SEED, CHUNK, IMG_SIDE)

CACHE = os.path.join(HERE, "cache")
OUTP = os.path.join(HERE, "panels")
CACHE_NAME = "panels_results_lut.pkl"


def pool_utilities_lut(model, lik, X_pool, chunk=CHUNK):
    """LUT standard utility of every pool image (batched, deterministic). Returns (n_pool,)."""
    outs = []
    with torch.no_grad():
        for i in range(0, X_pool.shape[0], chunk):
            res = lut_standard_utility(model, lik, X_pool[i:i + chunk])
            outs.append(res["utility"].detach())
    return torch.cat(outs).cpu().numpy()


def run_one(cell, nt, X_all, gp_min, gp_max):
    span = gp_max - gp_min
    model, lik, idx_train, test_r = gp_models.train_default_gpy(cell, nt, seed=SEED)  # M=n_train
    X_pool, _ = gp_models.pool_complement(X_all, idx_train)

    pu = pool_utilities_lut(model, lik, X_pool)               # LUT utility over all images
    best_i = int(np.argmax(pu))
    U_start = float(pu[best_i])
    start_img_gp = X_pool[best_i].reshape(IMG_SIDE, IMG_SIDE).cpu().numpy()
    start01 = (start_img_gp - gp_min) / span

    res = oi.optimize_image(model, lik, X_pool, gp_min, gp_max, decay_power=DECAY_POWER,
                            lr=LR, n_steps=N_STEPS, n_mc=1, pool_seed=POOL_SEED,
                            start_image01=start01, utility_mode="standard_lut", verbose=False)
    U_opt = float(res["traj"]["utility"][-1])
    sig2_final = float(res["traj"]["sigma2_g"][-1])

    rec = dict(
        test_r=float(test_r), n_pool=int(X_pool.shape[0]),
        start_img=start_img_gp.astype(np.float32),
        final_img=res["img_gp_final"].astype(np.float32),
        U_start=U_start, U_opt=U_opt,
        pool_mean=float(pu.mean()), pool_std=float(pu.std()),
        pool_min=float(pu.min()), pool_max=float(pu.max()),
        pool_p50=float(np.percentile(pu, 50)), pool_p95=float(np.percentile(pu, 95)),
        sig2_final=sig2_final,
        fr_final=float(res["traj"]["firing_mean"][-1]),
        # LUT diagnostic: fraction of optimization steps whose sigma2_g exceeded the LUT
        # cap (=6), i.e. used the high-rate fallback (a ranking proxy, not a trusted entropy).
        oob_frac_steps=_oob_frac_from_sig2(res),
        mu_g_final=float(res["traj"]["mu_g"][-1]),
    )
    del model, lik
    gc.collect(); torch.cuda.empty_cache()
    return rec


def _oob_frac_from_sig2(res):
    """Fraction of optimization steps whose sigma2_g exceeded the LUT cap (=6)."""
    s2 = np.asarray(res["traj"]["sigma2_g"])
    return float(np.mean(s2 > 6.0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", type=int, nargs="*", default=[13, 3],
                    help="default 13 3 -- the blow-up cell + the content-difference cell")
    ap.add_argument("--preview", action="store_true",
                    help="1 cell, n_train in {50,150,300} only (fast smoke test)")
    args = ap.parse_args()

    os.makedirs(CACHE, exist_ok=True); os.makedirs(OUTP, exist_ok=True)
    cache_path = os.path.join(CACHE, CACHE_NAME)
    cells = [args.cells[0]] if args.preview else args.cells
    nts = [50, 150, 300] if args.preview else N_TRAINS

    X_all, gp_min, gp_max = gp_models.load_pool()
    print(f"pool {tuple(X_all.shape)}  GP_MIN={gp_min:.4f} GP_MAX={gp_max:.4f}  | LUT standard utility")

    if os.path.exists(cache_path) and not args.preview:
        results = pickle.load(open(cache_path, "rb"))
    else:
        results = dict(meta=dict(gp_min=gp_min, gp_max=gp_max, M="n_train",
                                 n_trains=N_TRAINS, utility="standard_lut"), cells={})

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
                  f"Ustart={rec['U_start']:+.4f} Uopt={rec['U_opt']:+.4f} dU={dU:+.4f} | "
                  f"poolmu={rec['pool_mean']:+.4f} sig2={rec['sig2_final']:.3f} "
                  f"fr={rec['fr_final']:.1f} oob={rec['oob_frac_steps']*100:.0f}% "
                  f"({time.time()-tt:.0f}s)")
            results["cells"][cell] = cell_rec
            if not args.preview:
                pickle.dump(results, open(cache_path, "wb"))
        tag = "_preview" if args.preview else ""
        panels.build_figure(cell, cell_rec, gp_min, gp_max,
                            os.path.join(OUTP, f"cell{cell}_panels_lut{tag}"),
                            utility_label="LUT standard utility")
    print(f"\nDONE in {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
