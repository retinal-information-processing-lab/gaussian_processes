"""
Run the full cell x n_train grid of DA-utility image optimizations and cache the
results for figure-building. Two start modes per (cell, n_train):
  - gray:    near-gray Fourier init  -> reveals the isolated RF / preferred stimulus
  - natural: start from a FIXED per-cell natural image (the n_train=300 model's
             most-useful natural image) -> shows the optimization stays natural and
             bounded, with only the RF region modified.

All images are bounded by construction (lucent sigmoid + affine to [GP_MIN,GP_MAX]).
Results cached incrementally to cache/grid_results.pkl (resumable, so a long run is
never lost). Per-(cell,n_train) diagnostic PNGs go to out/grid/.

Reproducible: fixed seeds throughout (training seed=42, lucent init_seed=0,
pool_seed=0, sample_lambda=False).

Usage:
    python run_grid.py                      # default cells
    python run_grid.py --cells 13 1 11 8    # explicit
    python run_grid.py --preview            # 1 cell, fast sanity check
"""
import argparse
import gc
import os
import pickle
import time

import numpy as np
import torch

import gp_models
import optimize_image as oi

# Default roster: high default_gpy test_r at the top corner, varied A / firing
# regimes and RF locations (excludes STA-edge-artifact cells).
DEFAULT_CELLS = [13, 1, 11, 8]
N_TRAINS = [50, 100, 200, 300]
# M = n_train at all times (full inducing set); train_default_gpy defaults M to n_train.
SEED = 42
DECAY_POWER = 1.0
LR = 0.05
N_STEPS = 200
N_MC = 48
REF_NT = 300        # which model's best-natural image is the fixed natural backdrop
REF_N_EVAL = 256    # pool subset size for picking the reference natural image

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "cache")
OUTG = os.path.join(HERE, "out", "grid")


def _final_metrics(res):
    """Distill final-step metrics + pixel stats from an optimize_image result."""
    tr = res["traj"]
    img = res["img_gp_final"]
    gp_min, gp_max = res["gp_min"], res["gp_max"]
    span = gp_max - gp_min
    sat = float(np.mean((img < gp_min + 0.01 * span) | (img > gp_max - 0.01 * span)) * 100)
    return dict(
        U=float(tr["utility"][-1]), H_marg=float(tr["H_marg"][-1]),
        H_cond=float(tr["H_cond"][-1]), mu_g=float(tr["mu_g"][-1]),
        sigma2_g=float(tr["sigma2_g"][-1]), firing=float(tr["firing_mean"][-1]),
        z_safe=float(tr["z_safe"][-1]),
        pix_min=float(img.min()), pix_max=float(img.max()), sat_pct=sat,
    )


def _store(res):
    """Lean per-run record for the cache."""
    return dict(
        img_gp=res["img_gp_final"].astype(np.float32),
        img01_start=res["img01_start"].astype(np.float32),
        traj_U=res["traj"]["utility"].astype(np.float32),
        traj_sig2=res["traj"]["sigma2_g"].astype(np.float32),
        traj_firing=res["traj"]["firing_mean"].astype(np.float32),
        final=_final_metrics(res),
    )


def run_cell(cell, X_all, gp_min, gp_max, results, cache_path):
    span = gp_max - gp_min
    print(f"\n===== cell {cell} =====")

    # Reference natural backdrop: the REF_NT model's most-useful natural image.
    print(f"  training ref model (n_train={REF_NT}) + best-natural backdrop ...")
    model_ref, lik_ref, idx_ref, tr_ref = gp_models.train_default_gpy(
        cell_id=cell, n_train=REF_NT, seed=SEED)  # M=n_train
    X_pool_ref, _ = gp_models.pool_complement(X_all, idx_ref)
    torch.manual_seed(0)
    perm = torch.randperm(X_pool_ref.shape[0], device=X_pool_ref.device)[:N_MC]
    x_samples_ref = X_pool_ref[perm]
    ref_img_gp, ref_u, ref_idx = oi.best_natural_image(
        model_ref, lik_ref, X_pool_ref, x_samples_ref, gp_min, gp_max,
        n_eval=REF_N_EVAL)
    ref_img01 = (ref_img_gp - gp_min) / span
    print(f"    ref natural U={ref_u:+.4f} (pool idx {ref_idx})")

    cell_rec = dict(ref_img_gp=ref_img_gp.astype(np.float32), ref_u=float(ref_u),
                    ref_idx=int(ref_idx), by_nt={})

    for nt in N_TRAINS:
        t0 = time.time()
        if nt == REF_NT:
            model, lik, idx_train, test_r = model_ref, lik_ref, idx_ref, tr_ref
        else:
            model, lik, idx_train, test_r = gp_models.train_default_gpy(
                cell_id=cell, n_train=nt, seed=SEED)  # M=n_train
        X_pool, _ = gp_models.pool_complement(X_all, idx_train)

        res_gray = oi.optimize_image(
            model, lik, X_pool, gp_min, gp_max, decay_power=DECAY_POWER, lr=LR,
            n_steps=N_STEPS, n_mc=N_MC, start_image01=None, verbose=False)
        res_nat = oi.optimize_image(
            model, lik, X_pool, gp_min, gp_max, decay_power=DECAY_POWER, lr=LR,
            n_steps=N_STEPS, n_mc=N_MC, start_image01=ref_img01, verbose=False)

        cell_rec["by_nt"][nt] = dict(
            test_r=float(test_r), A=float(lik.A), lambda0=float(lik.lambda0),
            gray=_store(res_gray), nat=_store(res_nat))

        # Per-(cell,nt) diagnostic PNGs (gitignored, for inspection).
        info = f"cell {cell} | n_train={nt} M=n_train | test_r={test_r:.3f}"
        oi.plot_diagnostic(res_gray, info + " | start=gray",
                           out_path=os.path.join(OUTG, f"cell{cell}_nt{nt}_gray.png"))
        oi.plot_diagnostic(res_nat, info + " | start=natural",
                           best_nat=(ref_img_gp, ref_u, ref_idx),
                           out_path=os.path.join(OUTG, f"cell{cell}_nt{nt}_nat.png"))

        g, n = cell_rec["by_nt"][nt]["gray"]["final"], cell_rec["by_nt"][nt]["nat"]["final"]
        print(f"  nt={nt:3d}: test_r={test_r:.3f} | gray U={g['U']:+.4f} sig2={g['sigma2_g']:.3f} "
              f"fr={g['firing']:.1f} sat={g['sat_pct']:.0f}% | nat U={n['U']:+.4f} "
              f"sig2={n['sigma2_g']:.3f} fr={n['firing']:.1f} sat={n['sat_pct']:.0f}% "
              f"({time.time()-t0:.0f}s)")

        # Incremental save after every (cell, nt).
        results["cells"][cell] = cell_rec
        with open(cache_path, "wb") as f:
            pickle.dump(results, f)

    del model_ref, lik_ref
    gc.collect(); torch.cuda.empty_cache()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", type=int, nargs="*", default=None)
    ap.add_argument("--preview", action="store_true", help="1 cell, quick sanity check")
    args = ap.parse_args()

    cells = args.cells or ([DEFAULT_CELLS[0]] if args.preview else DEFAULT_CELLS)
    os.makedirs(CACHE, exist_ok=True)
    os.makedirs(OUTG, exist_ok=True)
    cache_path = os.path.join(CACHE, "grid_results.pkl")

    print(f"Grid: cells={cells}  n_trains={N_TRAINS}  M={M} decay_power={DECAY_POWER} "
          f"lr={LR} n_steps={N_STEPS} n_mc={N_MC}")
    X_all, gp_min, gp_max = gp_models.load_pool()
    print(f"pool {tuple(X_all.shape)}  GP_MIN={gp_min:.4f} GP_MAX={gp_max:.4f}")

    # Append-aware: load existing cache and add only cells not already complete.
    if os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            results = pickle.load(f)
        print(f"loaded existing cache: cells {list(results['cells'].keys())}")
    else:
        results = dict(meta=dict(gp_min=gp_min, gp_max=gp_max, decay_power=DECAY_POWER,
                                 lr=LR, n_steps=N_STEPS, n_mc=N_MC, M="n_train", seed=SEED,
                                 cells=[], n_trains=N_TRAINS, ref_nt=REF_NT),
                       cells={})
    # meta['cells'] = union (order: existing then new) -- figure order set separately.
    for c in cells:
        if c not in results["meta"]["cells"]:
            results["meta"]["cells"].append(c)

    t0 = time.time()
    for cell in cells:
        rec = results["cells"].get(cell)
        if rec is not None and len(rec.get("by_nt", {})) == len(N_TRAINS):
            print(f"\n===== cell {cell}: already complete, skipping =====")
            continue
        run_cell(cell, X_all, gp_min, gp_max, results, cache_path)
    with open(cache_path, "wb") as f:
        pickle.dump(results, f)
    print(f"\nDONE in {(time.time()-t0)/60:.1f} min -> {cache_path}")


if __name__ == "__main__":
    main()
