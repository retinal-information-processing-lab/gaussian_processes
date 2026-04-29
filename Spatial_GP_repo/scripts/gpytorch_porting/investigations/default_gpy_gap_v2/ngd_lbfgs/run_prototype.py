"""Phase 3E prototype: NGD+LBFGS on 5 cells × 1 seed.

Spec: investigations/default_gpy_gap_v2/SCRAPBOOK.md §40 Step 1.

Cells  : {40, 38, 16, 30, 29}  (same as Phase 3 prototype)
Seed   : 42
Config : matches Phase 3C final-verdict sweep exactly except the
         optimizer — M=250, n_train=3160, fix_Amp=True, ip_selection=random.

Pass/fail gate (§40):
  - No run crashes or produces NaN.
  - No cell regresses by more than 0.05 vs NGD+Adam at matched iter count
    (we compare against the Phase 3C sweep's seed-42 numbers — wait: the
    Phase 3C sweep used seeds {1,2,3}, not 42. The Phase 3 (pre-ES) sweep
    had seed 42 — see `ngd_results.noES_1000iter.jsonl` in the
    superseded ngd/ folder. We use those as the seed-42 reference.)

Strategy: monkey-patch `ngd_training.train_ngd` to our `train_ngd_lbfgs`,
then call `run_single_config` with `mode='ngd'`. The config's ngd_* keys
are overridden to the vargp-scale ES values (patience=15, min_delta_rel=1e-3,
min_iterations=10) per Phase 3E Q1(a).

Output: prototype_results.jsonl in this folder.
"""
from __future__ import annotations

import datetime
import json
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch

INV_DIR = Path(__file__).parent
ROOT = INV_DIR.parent.parent.parent  # .../gpytorch_porting
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(INV_DIR))

# Monkey-patch BEFORE importing run_single_mode / run_single_config uses it.
import ngd_training as _ngd_training_mod
from ngd_lbfgs_training import train_ngd_lbfgs
_ngd_training_mod.train_ngd = train_ngd_lbfgs

from run_single_mode import build_config_from_defaults, run_single_config  # noqa: E402


DATA_PATH = 'datasets/PNAS_64x64_center_crop_no_renorm.npz'
M = 250
N_TRAIN = 3160
N_ITERATIONS = 200  # cap; at vargp ES scale (patience=15) we expect stop << 100.
CELLS = [40, 38, 16, 30, 29]
SEEDS = [42, 123, 789]  # matches Phase 3B seed set — enables 3-seed paired comparison
FIX_AMP = True
RESULTS_PATH = INV_DIR / 'prototype_results.jsonl'


def _nan_to_none(x):
    if x is None:
        return None
    try:
        xf = float(x)
    except (TypeError, ValueError):
        return None
    return xf if np.isfinite(xf) else None


def run_one(cell, seed):
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    config = build_config_from_defaults(
        mode='ngd',
        data_path=DATA_PATH,
        M=M,
        n_train=N_TRAIN,
        cell=cell,
        seed=seed,
        ip_selection='random',
        fix_Amp=FIX_AMP,
        ngd_n_iterations=N_ITERATIONS,
        # Phase 3E Q1(a) — vargp-scale ES.
        ngd_es_patience=15,
        ngd_es_min_delta_rel=1e-3,
        ngd_es_min_iterations=10,
    )

    t0 = time.time()
    started = datetime.datetime.now().isoformat(timespec='seconds')
    try:
        result = run_single_config(config)
        err = None
    except Exception as e:
        traceback.print_exc()
        result = None
        err = repr(e)
    wall = time.time() - t0

    if result is None:
        return {
            'cell': cell, 'seed': seed, 'mode': 'ngd_lbfgs', 'status': 'crashed',
            'error': err, 'wall_time_s': wall, 'started_at': started,
            'M': M, 'n_train': N_TRAIN, 'fix_Amp': FIX_AMP,
        }

    curves = result.get('curves') or {}
    return {
        'cell': cell,
        'seed': seed,
        'mode': 'ngd_lbfgs',
        'status': result.get('status', 'unknown'),
        'started_at': started,
        'wall_time_s': wall,
        'M': M,
        'n_train': N_TRAIN,
        'fix_Amp': FIX_AMP,
        'n_iterations_requested': N_ITERATIONS,
        'n_iterations_run': result.get('n_iterations_run'),
        'stopped_early': result.get('stopped_early'),
        'best_iteration': result.get('best_iteration'),
        'train_time_s': result.get('train_time'),
        'test_r': _nan_to_none(result.get('test_r')),
        'train_r': _nan_to_none(result.get('train_r')),
        'explained_var': _nan_to_none(result.get('explained_var')),
        'adjusted_r2': _nan_to_none(result.get('adjusted_r2')),
        'reliability': _nan_to_none(result.get('reliability')),
        'final_loss': _nan_to_none(result.get('final_loss')),
        'final_A': result.get('final_A'),
        'final_lambda0': result.get('final_lambda0'),
        'final_Amp': result.get('final_Amp'),
        'final_beta': result.get('final_beta'),
        'final_rho': result.get('final_rho'),
        'final_eps_0x': result.get('final_eps_0x'),
        'final_eps_0y': result.get('final_eps_0y'),
        'final_sigma_0': result.get('final_sigma_0'),
        # Full trajectory curves (including LBFGS-specific diagnostics).
        'train_loss_curve': curves.get('train_loss'),
        'A_curve': curves.get('A'),
        'beta_curve': curves.get('beta'),
        'rho_curve': curves.get('rho'),
        'lambda0_curve': curves.get('lambda0'),
        'iter_time_curve': curves.get('iter_time'),
        'nat_vec_norm_curve': curves.get('nat_vec_norm'),
        'nat_tril_offdiag_norm_curve': curves.get('nat_tril_offdiag_norm'),
        'lbfgs_n_func_evals_curve': curves.get('lbfgs_n_func_evals'),
        'lbfgs_closure_infs_curve': curves.get('lbfgs_closure_infs'),
    }


def main():
    os.chdir(ROOT)  # run_single_config expects data path relative to cwd

    if RESULTS_PATH.exists():
        ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        backup = RESULTS_PATH.with_suffix(f'.backup_{ts}.jsonl')
        RESULTS_PATH.rename(backup)
        print(f"[renamed existing results to {backup.name}]", flush=True)

    total = len(CELLS) * len(SEEDS)
    print(f"NGD+LBFGS prototype: {len(CELLS)} cells × {len(SEEDS)} seed "
          f"= {total} runs", flush=True)
    print(f"  output: {RESULTS_PATH}", flush=True)

    wall_start = time.time()
    idx = 0
    for cell in CELLS:
        for seed in SEEDS:
            idx += 1
            print(f"\n[{idx}/{total}] cell={cell} seed={seed} "
                  f"=================================", flush=True)
            rec = run_one(cell, seed)
            with open(RESULTS_PATH, 'a') as f:
                f.write(json.dumps(rec) + '\n')
            tr = rec.get('test_r')
            print(f"  -> test_r={tr if tr is None else f'{tr:.4f}'}  "
                  f"stopped_early={rec.get('stopped_early')}  "
                  f"n_iter={rec.get('n_iterations_run')}  "
                  f"wall={rec['wall_time_s']:.1f}s",
                  flush=True)

    wall_elapsed = time.time() - wall_start
    print(f"\n=== Prototype done. {idx}/{total} runs in "
          f"{wall_elapsed / 60:.1f} min. ===", flush=True)


if __name__ == '__main__':
    main()
