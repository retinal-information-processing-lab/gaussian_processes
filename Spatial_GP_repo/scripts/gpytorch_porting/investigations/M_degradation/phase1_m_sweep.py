"""
Phase 1: Characterize the M-degradation curve.

Phase 1a: Dense M sweep
  - Cells: 1, 3, 5, 8, 9 (2 strong / 2 medium / 1 weak at M=250)
  - M = [50, 100, 200, 300, 500, 750, 1000, 1500]
  - seed = 0, n_train = 3160
  - 5 cells x 8 M values = 40 runs

Phase 1b: Seed stability
  - Same 5 cells, M = [50, 250, 1000], seeds = [0, 1, 2]
  - 5 cells x 3 M values x 3 seeds = 45 runs (seed=0 entries reused from 1a)

All runs: vargp_direct, interleave_fstep=True, fix_Amp=True, ELBO early stopping.
Results saved to: investigations/M_degradation/results/phase1_results.jsonl

Run from gpytorch_porting/:
  python investigations/M_degradation/phase1_m_sweep.py

Resume-safe: skips (cell, M, seed) tuples already in the results file.
"""
import sys
import os
import json
import time
import gc
from pathlib import Path
from datetime import datetime

# Add gpytorch_porting/ to path for imports
PROJ = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJ)

from run_single_mode import build_config_from_defaults, run_single_config

# =============================================================================
# Experiment constants (all values commented with their source or rationale)
# =============================================================================

# 5 representative cells spanning performance range at M=250 intl+fixAmp
# Sources: experiments/2026-04-06_es_sweeps_64x64/sweep_64x64_elbo_es_results.jsonl
#   Cell 1: 0.981 (strong), Cell 3: 0.943 (strong)
#   Cell 9: 0.890 (medium), Cell 8: 0.867 (medium)
#   Cell 5: 0.634 (weak)
CELLS = [1, 3, 5, 8, 9]

# Phase 1a: dense sweep to characterize the degradation shape
M_VALUES_1A = [50, 100, 200, 300, 500, 750, 1000, 1500]
SEED_1A = 0  # single seed for 1a

# Phase 1b: seed stability at three representative M values
M_VALUES_1B = [50, 250, 1000]
SEEDS_1B = [0, 1, 2]

# Training setup: best known configuration (INVESTIGATION_PROMPT requirement)
N_TRAIN = 3160    # full dataset (same as ceiling fits)
INTERLEAVE_FSTEP = True   # damped Newton A/lambda0 inside E-step
FIX_AMP = True            # freeze Amp=1 (matches paper)

# Dataset: 64x64 center-cropped PNAS (relative to PROJ=gpytorch_porting/).
# The worktree datasets/ directory has symlinks to the original .npz files
# (NPZ files are git-ignored; symlinks created once per worktree setup).
DATA_64 = 'datasets/PNAS_64x64_center_crop_no_renorm.npz'

# Results file (single file for both phases; resume-safe via (cell,M,seed) key)
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
RESULTS_FILE = os.path.join(RESULTS_DIR, 'phase1_results.jsonl')


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


def extract_record(result, config, phase):
    """Build serializable record from run_single_config output."""
    if result is None:
        return {
            'cell': config['cell'], 'M': config['M'], 'seed': config['seed'],
            'phase': phase, 'status': 'failed',
            'timestamp': datetime.now().isoformat(timespec='seconds'),
        }

    # Extract eigenspace info from model state
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

    # Final train_log_lik from curves (last value)
    ll_curve = curves.get('train_log_lik') or []
    final_train_log_lik = float(ll_curve[-1]) if ll_curve else None

    record = {
        # Identification
        'cell': config['cell'],
        'M': M,
        'seed': config['seed'],
        'phase': phase,
        'n_train': result.get('n_train'),
        'interleave_fstep': config.get('interleave_fstep'),
        'fix_Amp': config.get('fix_Amp'),
        # Performance metrics
        'test_r': result.get('test_r'),
        'train_r': result.get('train_r'),
        'adjusted_r2': result.get('adjusted_r2'),
        'final_loss': result.get('final_loss'),       # ELBO (negative)
        'final_train_log_lik': final_train_log_lik,   # ELL part of ELBO
        # Optimization trajectory
        'n_iterations_run': result.get('n_iterations_run'),
        'stopped_early': result.get('stopped_early'),
        'best_iteration': result.get('best_iteration'),
        'train_time': result.get('train_time'),
        # Final hyperparameters (for H4: hyperparameter drift)
        'final_A': result.get('final_A'),
        'final_lambda0': result.get('final_lambda0'),
        'final_beta': result.get('final_beta'),
        'final_rho': result.get('final_rho'),
        'final_sigma_0': result.get('final_sigma_0'),
        'final_eps_0x': result.get('final_eps_0x'),
        'final_eps_0y': result.get('final_eps_0y'),
        # Eigenspace (for H2: rank saturation)
        'n_b': n_b,
        'n_b_over_M': (n_b / M) if (n_b is not None and M > 0) else None,
        'eigval_min': eigval_min,
        'eigval_max': eigval_max,
        # Curves for trajectory analysis (H3: optimization, H4: drift)
        # Full per-iteration lists; used by analysis scripts
        'curves_A': curves.get('A'),
        'curves_beta': curves.get('beta'),
        'curves_train_r': curves.get('train_r'),
        'curves_train_loss': curves.get('train_loss'),
        'curves_train_log_lik': curves.get('train_log_lik'),
        # Metadata
        'status': result.get('status', 'success'),
        'timestamp': datetime.now().isoformat(timespec='seconds'),
    }
    return record


def run_batch(phase_name, runs, results_file):
    """
    Run a list of (cell, M, seed) configurations, saving to results_file.

    Skips already-completed runs (resume-safe).
    """
    os.makedirs(RESULTS_DIR, exist_ok=True)
    completed = load_completed(results_file)
    total = len(runs)
    n_skip = sum(1 for (c, m, s) in runs if (c, m, s) in completed)

    print(f"\n{'='*65}", flush=True)
    print(f"{phase_name}", flush=True)
    print(f"  {total} runs total, {n_skip} already done, {total - n_skip} to run", flush=True)
    print(f"  Results: {results_file}", flush=True)
    print(f"{'='*65}", flush=True)

    n_run = 0
    for i, (cell, M, seed) in enumerate(runs):
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
            interleave_fstep=INTERLEAVE_FSTEP,
            fix_Amp=FIX_AMP,
        )

        try:
            result = run_single_config(config)
        except Exception as e:
            print(f"  ERROR: {e}", flush=True)
            record = {
                'cell': cell, 'M': M, 'seed': seed, 'phase': phase_name,
                'n_train': N_TRAIN,
                'interleave_fstep': INTERLEAVE_FSTEP, 'fix_Amp': FIX_AMP,
                'status': 'error', 'error': str(e),
                'timestamp': datetime.now().isoformat(timespec='seconds'),
            }
            with open(results_file, 'a') as f:
                f.write(json.dumps(record) + '\n')
            completed.add(key)
            continue

        record = extract_record(result, config, phase_name)

        if record.get('test_r') is not None:
            print(
                f"  test_r={record['test_r']:.4f}  train_r={record['train_r']:.4f}"
                f"  n_b={record['n_b']}  n_b/M={record['n_b_over_M']:.3f}"
                f"  iters={record['n_iterations_run']}"
                f"  A={record['final_A']:.5f}  beta={record['final_beta']:.4f}"
                f"  ES={record['stopped_early']}",
                flush=True
            )
        else:
            print(f"  FAILED (no test_r)", flush=True)

        with open(results_file, 'a') as f:
            f.write(json.dumps(record) + '\n')
        completed.add(key)

        # Free GPU memory between runs
        del result
        gc.collect()
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass


# =============================================================================
# Main
# =============================================================================

def main():
    print(f"Dataset: {DATA_64}", flush=True)
    if not os.path.exists(DATA_64):
        raise FileNotFoundError(f"Dataset not found: {DATA_64}")

    # Phase 1a: dense M sweep, seed=0
    runs_1a = [(cell, M, SEED_1A) for cell in CELLS for M in M_VALUES_1A]
    run_batch("Phase 1a", runs_1a, RESULTS_FILE)

    # Phase 1b: seed stability (seed=0 entries reuse Phase 1a results)
    runs_1b = [
        (cell, M, seed)
        for cell in CELLS
        for M in M_VALUES_1B
        for seed in SEEDS_1B
    ]
    run_batch("Phase 1b", runs_1b, RESULTS_FILE)

    print(f"\nDone. Results in: {RESULTS_FILE}", flush=True)


if __name__ == '__main__':
    main()
