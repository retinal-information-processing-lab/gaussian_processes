"""41-cell scaled-ES NGD test at low n_train.

Hypothesis: NGD's gap to vargp at low n_train (Exp C, Δ = -0.06 to -0.12)
is mostly an ES-config issue. With vargp-scale ES (patience=15,
min_delta_rel=1e-3, min_iterations=10) instead of NGD's default
(patience=200, min_delta_rel=1e-2, min_iterations=50), NGD should match
vargp.

Grid: 41 cells × 3 (M=n_train ∈ {50, 150, 300}) × seed=1 = 123 runs.
Compare against Exp C vargp baseline (same (cell, M, seed=1) tuples).

Crash-safe: skips (cell, M) tuples already in results_scaled_es.jsonl.
"""
from __future__ import annotations
import datetime, json, os, sys, time, traceback
from pathlib import Path
import numpy as np, torch

EXP_DIR = Path(__file__).parent
ROOT    = EXP_DIR.parent.parent
sys.path.insert(0, str(ROOT))
from run_single_mode import build_config_from_defaults, run_single_config

DATA_PATH = 'datasets/PNAS_64x64_center_crop_no_renorm.npz'
SEED = 1                         # matches Exp C vargp baseline
FIX_AMP = True                   # matches Exp C
CELLS = list(range(41))
GRID = [(50, 50), (150, 150), (300, 300)]   # (M, n_train)
RESULTS = EXP_DIR / 'results_scaled_es.jsonl'

# === The ES override being tested ===
NGD_ES_PATIENCE = 15        # was 200 (vargp's)
NGD_ES_MIN_DELTA_REL = 1e-3 # was 1e-2 (vargp's)
NGD_ES_MIN_ITERATIONS = 10  # was 50 (vargp's)
NGD_N_ITERATIONS = 200      # was 1500 (safety cap; ES should fire earlier)


def _nan_to_none(x):
    try: return float(x) if np.isfinite(float(x)) else None
    except: return None


def run_one(cell, M, n_train):
    if torch.cuda.is_available():
        torch.cuda.empty_cache(); torch.cuda.synchronize()
    config = build_config_from_defaults(
        mode='ngd',
        data_path=DATA_PATH,
        M=M, n_train=n_train,
        cell=cell, seed=SEED,
        ip_selection='random', fix_Amp=FIX_AMP,
        ngd_n_iterations=NGD_N_ITERATIONS,
        ngd_es_patience=NGD_ES_PATIENCE,
        ngd_es_min_delta_rel=NGD_ES_MIN_DELTA_REL,
        ngd_es_min_iterations=NGD_ES_MIN_ITERATIONS,
    )
    t0 = time.time()
    try:
        result = run_single_config(config); err = None
    except Exception as e:
        traceback.print_exc(); result = None; err = repr(e)
    wall = time.time() - t0
    if result is None:
        return {
            'cell': cell, 'M': M, 'n_train': n_train, 'seed': SEED,
            'mode': 'ngd_scaled_es', 'status': 'crashed', 'error': err,
            'wall_time_s': wall,
        }
    return {
        'cell': cell, 'M': M, 'n_train': n_train, 'seed': SEED,
        'mode': 'ngd_scaled_es',
        'status': result.get('status'), 'wall_time_s': wall,
        'n_iterations_run': result.get('n_iterations_run'),
        'stopped_early': result.get('stopped_early'),
        'best_iteration': result.get('best_iteration'),
        'test_r': _nan_to_none(result.get('test_r')),
        'explained_var': _nan_to_none(result.get('explained_var')),
        'final_A': result.get('final_A'),
        'final_Amp': result.get('final_Amp'),
        'final_beta': result.get('final_beta'),
        'es_patience': NGD_ES_PATIENCE,
        'es_min_delta_rel': NGD_ES_MIN_DELTA_REL,
        'es_min_iterations': NGD_ES_MIN_ITERATIONS,
        'n_iterations_cap': NGD_N_ITERATIONS,
    }


def main():
    os.chdir(ROOT)
    done = set()
    if RESULTS.exists():
        for l in open(RESULTS):
            try:
                r = json.loads(l)
                done.add((r['cell'], r['M']))
            except: pass

    all_runs = [(c, M, n) for c in CELLS for (M, n) in GRID]
    remaining = [r for r in all_runs if (r[0], r[1]) not in done]
    total = len(all_runs)
    print(f"NGD scaled-ES sweep: {total} runs, {len(done)} done, "
          f"{len(remaining)} remaining", flush=True)
    print(f"  ES: patience={NGD_ES_PATIENCE}, min_delta_rel={NGD_ES_MIN_DELTA_REL}, "
          f"min_iters={NGD_ES_MIN_ITERATIONS}, cap={NGD_N_ITERATIONS}", flush=True)

    t0 = time.time()
    for i, (c, M, n) in enumerate(remaining, 1):
        print(f"  [{len(done)+i}/{total}] cell={c} M=n={M}", end='', flush=True)
        rec = run_one(c, M, n)
        with open(RESULTS, 'a') as f: f.write(json.dumps(rec) + '\n')
        done.add((c, M))
        tr = rec.get('test_r')
        print(f"  test_r={tr if tr is None else f'{tr:.4f}'}  "
              f"n_iter={rec.get('n_iterations_run')}  "
              f"wall={rec['wall_time_s']:.0f}s", flush=True)

    print(f"\nDone. {len(done)}/{total} in {(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == '__main__':
    main()
