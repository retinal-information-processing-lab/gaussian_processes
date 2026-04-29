"""NGD+Adam M-sweep — 41 cells × 9 M values × 3 seeds = 1107 runs.

Spec: experiments/2026-04-23_ngd_M_sweep_64x64/README.md
Grid: cells 0..40, M ∈ {50,100,200,250,300,500,750,1000,1500}, seeds {0,1,2}
Config: build_config_from_defaults(mode='ngd', ...) — see README §Configuration.

Crash-safe: on restart, any (cell, M, seed) already present in results.jsonl
is skipped. Safe to Ctrl-C and re-run without data loss or duplicate records.
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

EXP_DIR = Path(__file__).parent
ROOT = EXP_DIR.parent.parent
sys.path.insert(0, str(ROOT))

from run_single_mode import build_config_from_defaults, run_single_config

# ---------------------------------------------------------------------------
# Frozen grid (spec: README §Grid)
# ---------------------------------------------------------------------------
DATA_PATH = 'datasets/PNAS_64x64_center_crop_no_renorm.npz'
M_VALUES = [50, 100, 200, 250, 300, 500, 750, 1000, 1500]
CELLS = list(range(41))
SEEDS = [0, 1, 2]
N_TRAIN = 3160
FIX_AMP = True
# See README §"Two explicit non-default overrides"
NGD_N_ITERATIONS_CAP = 1500

RESULTS_PATH = EXP_DIR / 'results.jsonl'
METADATA_PATH = EXP_DIR / 'metadata.json'
EFFECTIVE_CONFIG_PATH = EXP_DIR / 'effective_config.json'


def _nan_to_none(x):
    if x is None:
        return None
    try:
        xf = float(x)
    except (TypeError, ValueError):
        return None
    return xf if np.isfinite(xf) else None


def build_run_config(cell, M, seed):
    return build_config_from_defaults(
        mode='ngd',
        data_path=DATA_PATH,
        M=M,
        n_train=N_TRAIN,
        cell=cell,
        seed=seed,
        ip_selection='random',
        fix_Amp=FIX_AMP,
        # Explicit non-default overrides — see README §"Two explicit non-default overrides".
        ngd_n_iterations=NGD_N_ITERATIONS_CAP,
    )


def run_one(cell, M, seed):
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    config = build_run_config(cell, M, seed)
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
            'cell': cell, 'M': M, 'seed': seed,
            'mode': 'ngd', 'status': 'crashed',
            'error': err, 'wall_time_s': wall, 'started_at': started,
            'n_train': N_TRAIN, 'fix_Amp': FIX_AMP,
        }

    curves = result.get('curves') or {}
    return {
        'cell': cell,
        'M': M,
        'seed': seed,
        'mode': 'ngd',
        'status': result.get('status', 'unknown'),
        'started_at': started,
        'wall_time_s': wall,
        'n_train': N_TRAIN,
        'fix_Amp': FIX_AMP,
        'n_iterations_cap': NGD_N_ITERATIONS_CAP,
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
        'iter_time_curve': curves.get('iter_time'),
    }


def write_metadata(git_commit):
    meta = {
        'started_at': datetime.datetime.now().isoformat(timespec='seconds'),
        'git_commit': git_commit,
        'branch': 'pietro/investigate-default-gpy',
        'spec': 'experiments/2026-04-23_ngd_M_sweep_64x64/README.md',
        'config': {
            'mode': 'ngd',
            'data_path': DATA_PATH,
            'M_values': M_VALUES,
            'cells': CELLS,
            'seeds': SEEDS,
            'total_runs': len(M_VALUES) * len(CELLS) * len(SEEDS),
            'n_train': N_TRAIN,
            'fix_Amp': FIX_AMP,
            'ngd_n_iterations_cap': NGD_N_ITERATIONS_CAP,
            'config_source': 'build_config_from_defaults(mode="ngd", ...)',
            'note': (
                'All params from default_params.json["ngd"] EXCEPT two explicit '
                'overrides: data_path (64x64 not 108x108 default) and '
                'ngd_n_iterations (1500 not 1000 default). '
                'See README §"Two explicit non-default overrides".'
            ),
        },
    }
    with open(METADATA_PATH, 'w') as f:
        json.dump(meta, f, indent=2)


def write_effective_config(git_commit):
    """Freeze the fully-resolved config for one representative run at launch time."""
    config = build_run_config(cell=0, M=250, seed=0)
    # Strip non-serialisable items
    serializable = {k: v for k, v in config.items() if not callable(v)}
    out = {
        'description': (
            'Fully-resolved config for cell=0, M=250, seed=0 at sweep launch. '
            'Frozen here so the exact parameter values used in every run are '
            'recoverable even if default_params.json changes later.'
        ),
        'git_commit': git_commit,
        'frozen_at': datetime.datetime.now().isoformat(timespec='seconds'),
        'config': serializable,
    }
    with open(EFFECTIVE_CONFIG_PATH, 'w') as f:
        json.dump(out, f, indent=2)


def load_completed():
    """Return set of (cell, M, seed) already in results.jsonl."""
    if not RESULTS_PATH.exists():
        return set()
    done = set()
    with open(RESULTS_PATH) as f:
        for line in f:
            try:
                r = json.loads(line)
                done.add((r['cell'], r['M'], r['seed']))
            except Exception:
                pass
    return done


def main():
    os.chdir(ROOT)

    try:
        commit = subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'], cwd=str(ROOT), text=True
        ).strip()
    except Exception:
        commit = 'unknown'

    # Write metadata and effective config at launch (idempotent).
    write_metadata(commit)
    write_effective_config(commit)

    completed = load_completed()
    total = len(M_VALUES) * len(CELLS) * len(SEEDS)
    remaining = total - len(completed)

    print(f"NGD+Adam M-sweep: {len(M_VALUES)} M-values × {len(CELLS)} cells × "
          f"{len(SEEDS)} seeds = {total} runs", flush=True)
    print(f"  git_commit:   {commit}", flush=True)
    print(f"  output:       {RESULTS_PATH}", flush=True)
    print(f"  already done: {len(completed)} / {total}  "
          f"({'resuming' if completed else 'fresh start'})", flush=True)

    wall_start = time.time()
    idx_run = len(completed)

    for M in M_VALUES:
        m_start = time.time()
        m_done_before = sum(1 for (_, mm, _) in completed if mm == M)
        print(f"\n{'='*60}", flush=True)
        print(f"M = {M}  ({m_done_before}/{len(CELLS)*len(SEEDS)} already done)", flush=True)

        for cell in CELLS:
            for seed in SEEDS:
                if (cell, M, seed) in completed:
                    continue
                idx_run += 1
                print(
                    f"  [{idx_run}/{total}] cell={cell} M={M} seed={seed}",
                    end='', flush=True,
                )
                rec = run_one(cell, M, seed)
                with open(RESULTS_PATH, 'a') as f:
                    f.write(json.dumps(rec) + '\n')
                completed.add((cell, M, seed))
                tr = rec.get('test_r')
                print(
                    f"  test_r={tr if tr is None else f'{tr:.4f}'}  "
                    f"n_iter={rec.get('n_iterations_run')}  "
                    f"wall={rec['wall_time_s']:.1f}s",
                    flush=True,
                )

        m_elapsed = time.time() - m_start
        m_done_now = sum(1 for (_, mm, _) in completed if mm == M)
        newly_run = m_done_now - m_done_before
        if newly_run > 0:
            print(
                f"  M={M} done: {m_done_now}/{len(CELLS)*len(SEEDS)} runs  "
                f"in {m_elapsed/60:.1f} min  "
                f"({m_elapsed/newly_run:.1f}s/run mean)",
                flush=True,
            )

    wall_elapsed = time.time() - wall_start
    print(f"\n{'='*60}", flush=True)
    print(f"Sweep done. {len(completed)}/{total} runs in "
          f"{wall_elapsed/3600:.2f} h ({wall_elapsed/60:.1f} min).", flush=True)
    print(f"Results: {RESULTS_PATH}", flush=True)


if __name__ == '__main__':
    main()
