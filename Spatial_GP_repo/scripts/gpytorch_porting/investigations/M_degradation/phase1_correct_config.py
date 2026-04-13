"""
Corrected M sweep using the ES sweep training config.

The previous Phase 1 sweep used build_config_from_defaults() without overriding
the 5 parameters that the ES sweep explicitly sets for interleaved training.
This script uses the correct ES sweep config:

    A_init=1e-4            (not 0.01  — prevents E-step Newton overshoot)
    lambda0_init=-1.0      (not 1.0   — better starting point for bias)
    n_estep=50             (not 10    — more E-step iterations per outer loop)
    n_mstep=20             (not 10    — more M-step iterations)
    n_iterations=80        (not 50    — more outer iterations)

Source: experiments/2026-04-06_es_sweeps_64x64/run_sweep_elbo_es_64x64.py
Config: 64_elbo_intl_fixAmp (lines 154-173, 222-256)

All other parameters come from default_params.json (beta=0.1, rho=0.1, ELBO ES
with patience=15, ip_selection='random', rf_init='ground_truth', etc.).

Cells: [0, 1, 3, 5, 8, 9, 12, 25, 26, 28, 30, 35, 36] (same 13 as previous)
Seeds: [0, 1, 2]
M: [50, 100, 200, 300, 500, 750, 1000, 1500]
Total: 312 runs. Cells 12, 30 expected to fail (~48 fast failures).

Run from gpytorch_porting/:
  python investigations/M_degradation/phase1_correct_config.py

Resume-safe: skips (cell, M, seed) already in the JSONL.
"""
import sys
import os
import json
import time
import gc
from pathlib import Path
from datetime import datetime

PROJ = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJ)

from run_single_mode import build_config_from_defaults, run_single_config

# =============================================================================
# ES sweep config overrides for interleaved training
# Source: run_sweep_elbo_es_64x64.py lines 154-163, 230-252
# =============================================================================
ES_CONFIG = dict(
    A_init=1e-4,           # physical constraint: prevents E-step Newton overshoot
    lambda0_init=-1.0,     # ES sweep value
    n_estep=50,            # 5x the default (10)
    n_mstep=20,            # 2x the default (10)
    n_iterations=80,       # 1.6x the default (50)
)

# =============================================================================
# Experiment matrix
# =============================================================================

# Same 13 cells as the previous (wrong-config) investigation
# Strong: 1 (0.981), 3 (0.943)
# Medium: 8 (0.867), 9 (0.890), 25, 26, 28, 35, 36
# Weak: 5 (0.634), 0 (0.583, works with correct A_init)
# Known to fail even with correct config: 12, 30 (included for completeness)
CELLS = [0, 1, 3, 5, 8, 9, 12, 25, 26, 28, 30, 35, 36]

M_VALUES = [50, 100, 200, 300, 500, 750, 1000, 1500]
SEEDS = [0, 1, 2]

N_TRAIN = 3160
DATA_64 = 'datasets/PNAS_64x64_center_crop_no_renorm.npz'

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
RESULTS_FILE = os.path.join(RESULTS_DIR, 'phase1_correct_config.jsonl')


# =============================================================================
# Helpers
# =============================================================================

def load_completed(results_file):
    """Return set of (cell, M, seed) already in the JSONL."""
    completed = set()
    if not os.path.exists(results_file):
        return completed
    with open(results_file) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
                completed.add((r['cell'], r['M'], r['seed']))
            except (json.JSONDecodeError, KeyError):
                pass
    return completed


def extract_record(result, config):
    """Build serializable record from run_single_config output."""
    if result is None:
        return {
            'cell': config['cell'], 'M': config['M'], 'seed': config['seed'],
            'status': 'failed',
            'timestamp': datetime.now().isoformat(timespec='seconds'),
        }

    model = result.get('_model')
    n_b = None
    eigval_min = None
    eigval_max = None
    if model is not None and hasattr(model, 'state') and model.state is not None:
        try:
            eigvals = model.state.eigvals_b
            n_b = int(len(eigvals))
            eigval_min = float(eigvals.min().item())
            eigval_max = float(eigvals.max().item())
        except Exception:
            pass

    M = config['M']
    curves = result.get('curves') or {}
    ll_curve = curves.get('train_log_lik') or []
    final_train_log_lik = float(ll_curve[-1]) if ll_curve else None

    return {
        'cell': config['cell'],
        'M': M,
        'seed': config['seed'],
        'n_train': result.get('n_train'),
        'config': 'es_intl_fixAmp',
        # Performance
        'test_r': result.get('test_r'),
        'train_r': result.get('train_r'),
        'adjusted_r2': result.get('adjusted_r2'),
        'final_loss': result.get('final_loss'),
        'final_train_log_lik': final_train_log_lik,
        # Optimization
        'n_iterations_run': result.get('n_iterations_run'),
        'stopped_early': result.get('stopped_early'),
        'best_iteration': result.get('best_iteration'),
        'train_time': result.get('train_time'),
        # Final hyperparameters
        'final_A': result.get('final_A'),
        'final_lambda0': result.get('final_lambda0'),
        'final_beta': result.get('final_beta'),
        'final_rho': result.get('final_rho'),
        'final_sigma_0': result.get('final_sigma_0'),
        'final_eps_0x': result.get('final_eps_0x'),
        'final_eps_0y': result.get('final_eps_0y'),
        # Eigenspace
        'n_b': n_b,
        'n_b_over_M': (n_b / M) if (n_b is not None and M > 0) else None,
        'eigval_min': eigval_min,
        'eigval_max': eigval_max,
        # Curves for trajectory analysis
        'curves_A': curves.get('A'),
        'curves_beta': curves.get('beta'),
        'curves_train_r': curves.get('train_r'),
        'curves_train_loss': curves.get('train_loss'),
        'curves_train_log_lik': curves.get('train_log_lik'),
        # Metadata
        'status': result.get('status', 'success'),
        'timestamp': datetime.now().isoformat(timespec='seconds'),
    }


# =============================================================================
# Main
# =============================================================================

def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    if not os.path.exists(os.path.join(PROJ, DATA_64)):
        raise FileNotFoundError(f"Dataset not found: {DATA_64}")

    runs = [(cell, M, seed) for cell in CELLS for M in M_VALUES for seed in SEEDS]
    completed = load_completed(RESULTS_FILE)
    total = len(runs)
    n_skip = sum(1 for r in runs if r in completed)

    print(f"Corrected M sweep (ES config: A_init=1e-4, n_estep=50, n_mstep=20, n_iter=80)")
    print(f"{'='*70}")
    print(f"  {total} runs total, {n_skip} already done, {total - n_skip} to run")
    print(f"  Cells: {CELLS}")
    print(f"  M: {M_VALUES}")
    print(f"  Seeds: {SEEDS}")
    print(f"  Results: {RESULTS_FILE}")
    print(f"{'='*70}", flush=True)

    n_run = 0
    for cell, M, seed in runs:
        key = (cell, M, seed)
        if key in completed:
            continue

        n_run += 1
        print(f"\n  [{n_run}/{total - n_skip}] cell={cell} M={M} seed={seed} ...",
              flush=True)

        config = build_config_from_defaults(
            mode='vargp_direct',
            M=M,
            n_train=N_TRAIN,
            seed=seed,
            cell=cell,
            data_path=DATA_64,
            interleave_fstep=True,
            fix_Amp=True,
            **ES_CONFIG,
        )

        try:
            result = run_single_config(config)
        except Exception as e:
            print(f"  ERROR: {e}", flush=True)
            record = {
                'cell': cell, 'M': M, 'seed': seed,
                'config': 'es_intl_fixAmp',
                'n_train': N_TRAIN,
                'status': 'error', 'error': str(e),
                'timestamp': datetime.now().isoformat(timespec='seconds'),
            }
            with open(RESULTS_FILE, 'a') as f:
                f.write(json.dumps(record) + '\n')
            completed.add(key)
            continue

        record = extract_record(result, config)

        if record.get('test_r') is not None:
            print(
                f"  test_r={record['test_r']:.4f}  train_r={record['train_r']:.4f}"
                f"  n_b={record['n_b']}  n_b/M={record['n_b_over_M']:.3f}"
                f"  iters={record['n_iterations_run']}"
                f"  A={record['final_A']:.5f}  beta={record['final_beta']:.4f}"
                f"  ES={record['stopped_early']}  time={record['train_time']:.0f}s",
                flush=True
            )
        else:
            print(f"  FAILED (no test_r)  time={record.get('train_time', 0):.0f}s",
                  flush=True)

        with open(RESULTS_FILE, 'a') as f:
            f.write(json.dumps(record) + '\n')
        completed.add(key)

        del result
        gc.collect()
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass

    print(f"\nDone. Results in: {RESULTS_FILE}", flush=True)


if __name__ == '__main__':
    main()
