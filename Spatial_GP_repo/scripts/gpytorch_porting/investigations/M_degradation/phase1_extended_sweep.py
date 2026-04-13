"""
Phase 1 extended: M sweep for the three cells that failed at M=1500 in the
ceiling fit plus 5 randomly selected additional cells.

Motivation: Phase 1a (5 representative cells) showed no clear general M
degradation. The known failing cells (0, 12, 30) are the most likely source
of the observed mean test_r drop (0.838 at M=250 → 0.829 at M=1500). This
sweep characterises their behaviour across M to identify whether their failure
is an instability, an optimisation failure, or something else.

Cells:
  - Failing at M=1500 (all 3 seeds): 0, 12, 30
    (from checkpoints/64x64_ceiling_M1500_intl_fixAmp/ceiling_results.json)
  - Randomly selected (seed=7, from remaining 33 cells): 25, 26, 28, 35, 36

M sweep: [50, 100, 200, 300, 500, 750, 1000, 1500], seed=0
8 x 8 = 64 runs total.

Results saved to: investigations/M_degradation/results/phase1_results.jsonl
(same file as Phase 1 — resume-safe, no duplicate keys).

Run from gpytorch_porting/:
  python investigations/M_degradation/phase1_extended_sweep.py
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
# Experiment constants
# =============================================================================

# Cells that failed on all 3 seeds at M=1500 intl+fixAmp
# Source: checkpoints/64x64_ceiling_M1500_intl_fixAmp/ceiling_results.json
CELLS_FAILING = [0, 12, 30]

# 5 randomly chosen cells from the remaining 33 not in Phase 1
# (numpy.random.default_rng(seed=7).choice(available, size=5) where
#  available = range(41) minus {0,1,3,5,8,9,12,30})
CELLS_RANDOM = [25, 26, 28, 35, 36]

CELLS = CELLS_FAILING + CELLS_RANDOM  # [0, 12, 25, 26, 28, 30, 35, 36]

# Same M sweep as Phase 1a
M_VALUES = [50, 100, 200, 300, 500, 750, 1000, 1500]
SEED = 0

# Training setup (must match Phase 1 for comparability)
N_TRAIN = 3160
INTERLEAVE_FSTEP = True
FIX_AMP = True

# Dataset (worktree symlinks to original gpytorch_porting/datasets/)
DATA_64 = 'datasets/PNAS_64x64_center_crop_no_renorm.npz'

# Append to the same results file used by Phase 1
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
RESULTS_FILE = os.path.join(RESULTS_DIR, 'phase1_results.jsonl')


# =============================================================================
# Helpers (copied from phase1_m_sweep.py)
# =============================================================================

def load_completed(results_file):
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
    if result is None:
        return {
            'cell': config['cell'], 'M': config['M'], 'seed': config['seed'],
            'phase': phase, 'status': 'failed',
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
        'phase': phase,
        'n_train': result.get('n_train'),
        'interleave_fstep': config.get('interleave_fstep'),
        'fix_Amp': config.get('fix_Amp'),
        'test_r': result.get('test_r'),
        'train_r': result.get('train_r'),
        'adjusted_r2': result.get('adjusted_r2'),
        'final_loss': result.get('final_loss'),
        'final_train_log_lik': final_train_log_lik,
        'n_iterations_run': result.get('n_iterations_run'),
        'stopped_early': result.get('stopped_early'),
        'best_iteration': result.get('best_iteration'),
        'train_time': result.get('train_time'),
        'final_A': result.get('final_A'),
        'final_lambda0': result.get('final_lambda0'),
        'final_beta': result.get('final_beta'),
        'final_rho': result.get('final_rho'),
        'final_sigma_0': result.get('final_sigma_0'),
        'final_eps_0x': result.get('final_eps_0x'),
        'final_eps_0y': result.get('final_eps_0y'),
        'n_b': n_b,
        'n_b_over_M': (n_b / M) if (n_b is not None and M > 0) else None,
        'eigval_min': eigval_min,
        'eigval_max': eigval_max,
        'curves_A': curves.get('A'),
        'curves_beta': curves.get('beta'),
        'curves_train_r': curves.get('train_r'),
        'curves_train_loss': curves.get('train_loss'),
        'curves_train_log_lik': curves.get('train_log_lik'),
        'status': result.get('status', 'success'),
        'timestamp': datetime.now().isoformat(timespec='seconds'),
    }


def run_batch(phase_name, runs, results_file):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    completed = load_completed(results_file)
    total = len(runs)
    n_skip = sum(1 for (c, m, s) in runs if (c, m, s) in completed)

    print(f"\n{'='*65}", flush=True)
    print(f"{phase_name}", flush=True)
    print(f"  {total} runs, {n_skip} already done, {total - n_skip} to run", flush=True)
    print(f"  Results: {results_file}", flush=True)
    print(f"{'='*65}", flush=True)

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

        del result
        gc.collect()
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass


def main():
    print(f"Extended Phase 1 sweep: cells {CELLS}", flush=True)
    print(f"  Failing cells: {CELLS_FAILING}", flush=True)
    print(f"  Random cells:  {CELLS_RANDOM}", flush=True)
    print(f"  M values: {M_VALUES}", flush=True)

    runs = [(cell, M, SEED) for cell in CELLS for M in M_VALUES]
    run_batch("Phase 1 extended", runs, RESULTS_FILE)

    print(f"\nDone. Results appended to: {RESULTS_FILE}", flush=True)


if __name__ == '__main__':
    main()
