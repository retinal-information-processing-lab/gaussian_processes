"""Phase 3C final-verdict sweep for NGD.

Runs 41 cells × 3 seeds = 123 NGD fits to produce the paired-Δ
comparison against vargp_direct's best-ever config (`intl_fixAmp` + ELBO
ES p=15 on the same 41-cell × 3-seed grid).

Spec: investigations/default_gpy_gap_v2/SCRAPBOOK.md §26–33.

Config (locked — do not modify without re-reading SCRAPBOOK §26):

  mode         = 'ngd'
  dataset      = datasets/PNAS_64x64_center_crop_no_renorm.npz
  cells        = 0..40  (all 41)
  seeds        = {1, 2, 3}   (MATCHES vargp best-sweep seeds, not Phase 3's {42,123,789})
  M            = 250   (MATCHES vargp best-sweep; Phase 3 used M=300)
  n_train      = 3160  (full pool; n_val_split=0)
  ip_selection = 'random'
  kernel       = arc_cosine (default)
  rf_init      = ground_truth (default)
  beta, rho    = 0.1, 0.1 (defaults)
  A_init       = 0.01      (NGD default; differs from vargp's 1e-4, see §26)
  lambda0_init = 1.0       (NGD default)
  Amp          = 1.0 FROZEN (fix_Amp=True) — matches vargp's intl_fixAmp
  n_iterations = 1500      (cap; ES expected to fire earlier)
  ngd_lr       = 0.1       (_constants.NGD_LR)
  adam_lr      = 0.01      (_constants.NGD_ADAM_LR)
  ES           = ELBO, patience=200, min_delta_rel=1e-2, min_iters=50, restore_best=True
                 (all from _constants / default_params.json["ngd"])
  dtype        = float32
  device       = cuda
  probe_test_r = True (diagnostic; NOT used for ES decisions)

Output: results.jsonl in this folder.
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

EXP_DIR = Path(__file__).parent
ROOT = EXP_DIR.parent.parent  # .../gpytorch_porting
sys.path.insert(0, str(ROOT))

from run_single_mode import build_config_from_defaults, run_single_config

DATA_PATH = 'datasets/PNAS_64x64_center_crop_no_renorm.npz'
M = 250
N_TRAIN = 3160
N_ITERATIONS = 1500  # cap; ES fires earlier on most cells
CELLS = list(range(41))
SEEDS = [1, 2, 3]
FIX_AMP = True
RESULTS_PATH = EXP_DIR / 'results.jsonl'
LOG_PATH = EXP_DIR / 'run.log'
METADATA_PATH = EXP_DIR / 'metadata.json'


def _nan_to_none(x):
    if x is None:
        return None
    try:
        xf = float(x)
    except (TypeError, ValueError):
        return None
    return xf if np.isfinite(xf) else None


def run_one(cell, seed):
    """Run a single NGD fit with the verdict-sweep config."""
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
        # ngd_lr, ngd_adam_lr, ngd_es_* come from default_params.json["ngd"]
        # via build_config_from_defaults.
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
            'cell': cell, 'seed': seed, 'mode': 'ngd', 'status': 'crashed',
            'error': err, 'wall_time_s': wall, 'started_at': started,
            'M': M, 'n_train': N_TRAIN, 'fix_Amp': FIX_AMP,
        }

    # Keep only serializable fields — `result` has _model/_predictions
    # references we don't want in JSONL.
    curves = result.get('curves') or {}
    rec = {
        'cell': cell,
        'seed': seed,
        'mode': 'ngd',
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
        # Per-iter training curves for post-hoc analysis.
        'train_loss_curve': curves.get('train_loss'),
        'A_curve': curves.get('A'),
        'beta_curve': curves.get('beta'),
        'rho_curve': curves.get('rho'),
        'lambda0_curve': curves.get('lambda0'),
        'iter_time_curve': curves.get('iter_time'),
        'nat_vec_norm_curve': curves.get('nat_vec_norm'),
        'nat_tril_offdiag_norm_curve': curves.get('nat_tril_offdiag_norm'),
    }
    return rec


def write_metadata(git_commit, seeds, cells, total_runs):
    meta = {
        'started_at': datetime.datetime.now().isoformat(timespec='seconds'),
        'git_commit': git_commit,
        'branch': 'pietro/investigate-default-gpy',
        'spec_section': 'SCRAPBOOK.md §26–33 (Phase 3C)',
        'config': {
            'mode': 'ngd',
            'data_path': DATA_PATH,
            'M': M,
            'n_train': N_TRAIN,
            'cells': cells,
            'seeds': seeds,
            'total_runs': total_runs,
            'fix_Amp': FIX_AMP,
            'n_iterations_cap': N_ITERATIONS,
        },
    }
    with open(METADATA_PATH, 'w') as f:
        json.dump(meta, f, indent=2)


def main():
    os.chdir(ROOT)  # run_single_config resolves data path relative to cwd

    if RESULTS_PATH.exists():
        ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        backup = RESULTS_PATH.with_suffix(f'.backup_{ts}.jsonl')
        RESULTS_PATH.rename(backup)
        print(f"[renamed existing results to {backup.name}]", flush=True)

    import subprocess
    try:
        commit = subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'], cwd=str(ROOT), text=True
        ).strip()
    except Exception:
        commit = 'unknown'

    total = len(CELLS) * len(SEEDS)
    write_metadata(commit, SEEDS, CELLS, total)
    print(f"NGD final-verdict sweep: {len(CELLS)} cells × {len(SEEDS)} seeds "
          f"= {total} runs", flush=True)
    print(f"  git_commit: {commit}", flush=True)
    print(f"  output:     {RESULTS_PATH}", flush=True)

    wall_start = time.time()
    idx = 0
    for cell in CELLS:
        for seed in SEEDS:
            idx += 1
            print(f"\n[{idx}/{total}] cell={cell} seed={seed} "
                  f"============================================", flush=True)
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
    print(f"Results: {RESULTS_PATH}", flush=True)


if __name__ == '__main__':
    main()
