"""
Phase 2: Determine whether A-collapse in cells 0, 12, 30 is M-dependent.

Critical question
-----------------
Seeds 1, 2, 3 work for cells 0, 12, 30 at M=250 (confirmed in ES sweep).
The M=1500 ceiling fit (seeds 0, 1, 2) failed for all three cells.
Seed 0 always fails regardless of M (confirmed in Phase 1 extended).

Therefore: seeds 1 and 2 must fail at M=1500 but succeed at M=250.
Phase 2a directly characterises the M-dependent failure threshold.

Phase 2a: Failing cells — full seed × M matrix
  Cells: [0, 12, 30]
  Seeds: [1, 2, 3]  (seed 0 already confirmed to fail at all M)
  M:     [50, 100, 250, 500, 1000, 1500]
  Runs:  3 × 3 × 6 = 54

Phase 2b: Non-failing cells — robustness check with extra seeds
  Cells: [1, 3, 5, 8, 9, 25, 26, 28, 35, 36]
  Seeds: [1, 2]  (seed 0 done in Phase 1; cells 1,3,5,8,9 have partial 1b data)
  M:     [50, 250, 1000, 1500]
  Runs:  10 × 2 × 4 = 80  (resume-safe: skips already-done Phase 1b runs)

Results appended to: investigations/M_degradation/results/phase1_results.jsonl

Run from gpytorch_porting/:
  python investigations/M_degradation/phase2_seed_sweep.py
"""
import sys
import os
import json
import gc
from pathlib import Path
from datetime import datetime

PROJ = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJ)

from run_single_mode import build_config_from_defaults, run_single_config

# =============================================================================
# Experiment constants
# =============================================================================

# Phase 2a ─ failing cells, full seed range, full M range
# Goal: find the M at which seeds 1/2/3 start to fail (A-collapse)
CELLS_2A  = [0, 12, 30]
SEEDS_2A  = [1, 2, 3]    # seed 0 always fails (confirmed), skip it
M_2A      = [50, 100, 250, 500, 1000, 1500]

# Phase 2b ─ non-failing cells, extra seeds, sparse M
# Goal: confirm no M-dependent degradation across seeds
CELLS_2B  = [1, 3, 5, 8, 9, 25, 26, 28, 35, 36]
SEEDS_2B  = [1, 2]       # seed 0 done in Phase 1; Phase 1b has partial seed 1 data
M_2B      = [50, 250, 1000, 1500]

N_TRAIN         = 3160
INTERLEAVE_FSTEP = True
FIX_AMP         = True
DATA_64         = 'datasets/PNAS_64x64_center_crop_no_renorm.npz'

RESULTS_DIR  = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
RESULTS_FILE = os.path.join(RESULTS_DIR, 'phase1_results.jsonl')  # same file as Phase 1


# =============================================================================
# Helpers (identical to phase1_m_sweep.py)
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
    n_b = eigval_min = eigval_max = None
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
    n_new = sum(1 for (c, m, s) in runs if (c, m, s) not in completed)

    print(f"\n{'='*65}", flush=True)
    print(f"{phase_name}", flush=True)
    print(f"  {len(runs)} runs total, {len(runs) - n_new} already done, {n_new} to run",
          flush=True)
    print(f"  Results: {results_file}", flush=True)
    print(f"{'='*65}", flush=True)

    done = 0
    for cell, M, seed in runs:
        key = (cell, M, seed)
        if key in completed:
            continue

        done += 1
        print(f"\n  [{done}/{n_new}] cell={cell} M={M} seed={seed} ...", flush=True)

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
                f"  test_r={record['test_r']:.4f}  A={record['final_A']:.2e}"
                f"  n_b={record['n_b']}  iters={record['n_iterations_run']}"
                f"  ES={record['stopped_early']}",
                flush=True
            )
        else:
            print(
                f"  FAIL  A={record.get('final_A')}  iters={record.get('n_iterations_run')}",
                flush=True
            )

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


# =============================================================================
# Main
# =============================================================================

def main():
    # Phase 2a: failing cells — does failure threshold shift with M?
    runs_2a = [
        (cell, M, seed)
        for cell in CELLS_2A
        for M    in M_2A
        for seed in SEEDS_2A
    ]
    run_batch("Phase 2a (failing cells, seeds 1-3, M sweep)", runs_2a, RESULTS_FILE)

    # Phase 2b: non-failing cells — extra seeds to confirm no degradation
    runs_2b = [
        (cell, M, seed)
        for cell in CELLS_2B
        for M    in M_2B
        for seed in SEEDS_2B
    ]
    run_batch("Phase 2b (non-failing cells, seeds 1-2, sparse M)", runs_2b, RESULTS_FILE)

    print(f"\nAll done. Results in: {RESULTS_FILE}", flush=True)


if __name__ == '__main__':
    main()
