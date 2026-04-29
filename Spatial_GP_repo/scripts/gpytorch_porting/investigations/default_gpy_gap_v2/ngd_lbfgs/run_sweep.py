"""Phase 3E verdict sweep: NGD+LBFGS on 41 cells × 3 seeds.

Spec: investigations/default_gpy_gap_v2/SCRAPBOOK.md §40 Step 2.

Mirrors Phase 3C (experiments/2026-04-22_ngd_final_verdict_64x64/run_sweep.py)
exactly except the optimizer — swap NGD+Adam for NGD+LBFGS via monkey-patch.

Cells   : 0..40 (all 41)
Seeds   : {1, 2, 3} — matches Phase 3C and vargp best sweep, enables paired-Δ.
Config  : M=250, n_train=3160, fix_Amp=True, ip_selection='random',
          n_iterations cap=300 (ES expected to fire ~iter 50-150 at
          vargp-scale patience),
          ES patience=15 min_delta_rel=1e-3 min_iterations=10 (Phase 3E Q1a).

Output: results.jsonl in this folder.
"""
from __future__ import annotations

import datetime
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch

INV_DIR = Path(__file__).parent
ROOT = INV_DIR.parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(INV_DIR))

import ngd_training as _ngd_training_mod
from ngd_lbfgs_training import train_ngd_lbfgs
_ngd_training_mod.train_ngd = train_ngd_lbfgs

from run_single_mode import build_config_from_defaults, run_single_config  # noqa: E402


DATA_PATH = 'datasets/PNAS_64x64_center_crop_no_renorm.npz'
M = 250
N_TRAIN = 3160
# Cap on outer iterations. Prototype showed ES firing at 53-101 outer iters
# on hard cells; 300 gives generous headroom.
N_ITERATIONS = 300
CELLS = list(range(41))
SEEDS = [1, 2, 3]
FIX_AMP = True
RESULTS_PATH = INV_DIR / 'results.jsonl'
METADATA_PATH = INV_DIR / 'metadata.json'


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
        # Phase 3E Q1(a) — vargp-scale ES defaults.
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


def write_metadata(git_commit, total_runs):
    meta = {
        'started_at': datetime.datetime.now().isoformat(timespec='seconds'),
        'git_commit': git_commit,
        'branch': 'pietro/investigate-default-gpy',
        'spec_section': 'SCRAPBOOK.md §40 (Phase 3E)',
        'config': {
            'mode': 'ngd_lbfgs (NGD + LBFGS)',
            'optimizer_swap': 'Adam -> LBFGS on hyperparameters',
            'data_path': DATA_PATH,
            'M': M,
            'n_train': N_TRAIN,
            'cells': CELLS,
            'seeds': SEEDS,
            'total_runs': total_runs,
            'fix_Amp': FIX_AMP,
            'n_iterations_cap': N_ITERATIONS,
            'es': {'patience': 15, 'min_delta_rel': 1e-3, 'min_iterations': 10,
                   'restore_best': True},
        },
    }
    with open(METADATA_PATH, 'w') as f:
        json.dump(meta, f, indent=2)


def main():
    os.chdir(ROOT)

    if RESULTS_PATH.exists():
        ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        backup = RESULTS_PATH.with_suffix(f'.backup_{ts}.jsonl')
        RESULTS_PATH.rename(backup)
        print(f"[renamed existing results to {backup.name}]", flush=True)

    try:
        commit = subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'], cwd=str(ROOT), text=True
        ).strip()
    except Exception:
        commit = 'unknown'

    total = len(CELLS) * len(SEEDS)
    write_metadata(commit, total)
    print(f"NGD+LBFGS verdict sweep: {len(CELLS)} cells × {len(SEEDS)} seeds "
          f"= {total} runs", flush=True)
    print(f"  git_commit: {commit}", flush=True)
    print(f"  output:     {RESULTS_PATH}", flush=True)

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
    print(f"\n=== Sweep done. {idx}/{total} runs in {wall_elapsed / 60:.1f} min. ===",
          flush=True)


if __name__ == '__main__':
    main()
