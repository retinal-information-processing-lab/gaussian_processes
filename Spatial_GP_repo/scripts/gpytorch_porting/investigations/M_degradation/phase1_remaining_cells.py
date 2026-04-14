"""
Sweep the 28 remaining cells (of 41 total) with the correct ES config.

The first sweep (phase1_correct_config.py) covered 13 cells.
This adds the other 28 to get full population coverage.

Same config, same M grid, same seeds. Results appended to the SAME JSONL
so all 41 cells live in one file. Resume-safe.

Run from gpytorch_porting/:
  nohup python investigations/M_degradation/phase1_remaining_cells.py \
    > investigations/M_degradation/phase1_remaining_cells.log 2>&1 &
"""
import sys
import os
import json
import time
import gc
from datetime import datetime

PROJ = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJ)

from run_single_mode import build_config_from_defaults, run_single_config

# ES sweep config (same as phase1_correct_config.py)
ES_CONFIG = dict(
    A_init=1e-4,
    lambda0_init=-1.0,
    n_estep=50,
    n_mstep=20,
    n_iterations=80,
)

# The 28 cells not covered by the first sweep
CELLS = [2, 4, 6, 7, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20,
         21, 22, 23, 24, 27, 29, 31, 32, 33, 34, 37, 38, 39, 40]

M_VALUES = [50, 100, 200, 300, 500, 750, 1000, 1500]
SEEDS = [0, 1, 2]
N_TRAIN = 3160
DATA_64 = 'datasets/PNAS_64x64_center_crop_no_renorm.npz'

# SAME results file as the first sweep — all 41 cells in one place
RESULTS_FILE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    'results', 'phase1_correct_config.jsonl'
)


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


def extract_record(result, config):
    if result is None:
        return {
            'cell': config['cell'], 'M': config['M'], 'seed': config['seed'],
            'status': 'failed',
            'timestamp': datetime.now().isoformat(timespec='seconds'),
        }

    model = result.get('_model')
    n_b, eigval_min, eigval_max = None, None, None
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

    return {
        'cell': config['cell'], 'M': M, 'seed': config['seed'],
        'n_train': result.get('n_train'), 'config': 'es_intl_fixAmp',
        'test_r': result.get('test_r'), 'train_r': result.get('train_r'),
        'adjusted_r2': result.get('adjusted_r2'),
        'final_loss': result.get('final_loss'),
        'final_train_log_lik': float(ll_curve[-1]) if ll_curve else None,
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
        'n_b': n_b, 'n_b_over_M': (n_b / M) if n_b and M else None,
        'eigval_min': eigval_min, 'eigval_max': eigval_max,
        'curves_A': curves.get('A'), 'curves_beta': curves.get('beta'),
        'curves_train_r': curves.get('train_r'),
        'curves_train_loss': curves.get('train_loss'),
        'curves_train_log_lik': curves.get('train_log_lik'),
        'status': result.get('status', 'success'),
        'timestamp': datetime.now().isoformat(timespec='seconds'),
    }


def main():
    os.makedirs(os.path.dirname(RESULTS_FILE), exist_ok=True)
    runs = [(c, M, s) for c in CELLS for M in M_VALUES for s in SEEDS]
    completed = load_completed(RESULTS_FILE)
    n_skip = sum(1 for r in runs if r in completed)

    print(f"Remaining cells sweep (ES config)")
    print(f"{'='*60}")
    print(f"  {len(runs)} runs, {n_skip} already done, {len(runs)-n_skip} to run")
    print(f"  Cells: {CELLS}")
    print(f"  Results: {RESULTS_FILE}")
    print(f"{'='*60}", flush=True)

    n_run = 0
    for cell, M, seed in runs:
        if (cell, M, seed) in completed:
            continue
        n_run += 1
        print(f"\n  [{n_run}/{len(runs)-n_skip}] cell={cell} M={M} seed={seed} ...",
              flush=True)

        config = build_config_from_defaults(
            mode='vargp_direct', M=M, n_train=N_TRAIN, seed=seed, cell=cell,
            data_path=DATA_64, interleave_fstep=True, fix_Amp=True, **ES_CONFIG,
        )
        try:
            result = run_single_config(config)
        except Exception as e:
            print(f"  ERROR: {e}", flush=True)
            record = {
                'cell': cell, 'M': M, 'seed': seed, 'config': 'es_intl_fixAmp',
                'status': 'error', 'error': str(e),
                'timestamp': datetime.now().isoformat(timespec='seconds'),
            }
            with open(RESULTS_FILE, 'a') as f:
                f.write(json.dumps(record) + '\n')
            completed.add((cell, M, seed))
            continue

        record = extract_record(result, config)
        if record.get('test_r') is not None:
            print(f"  test_r={record['test_r']:.4f}  train_r={record['train_r']:.4f}"
                  f"  A={record['final_A']:.5f}  beta={record['final_beta']:.4f}"
                  f"  n_b={record['n_b']}  iters={record['n_iterations_run']}"
                  f"  time={record['train_time']:.0f}s", flush=True)
        else:
            print(f"  FAILED  time={record.get('train_time', 0):.0f}s", flush=True)

        with open(RESULTS_FILE, 'a') as f:
            f.write(json.dumps(record) + '\n')
        completed.add((cell, M, seed))

        del result; gc.collect()
        try:
            import torch; torch.cuda.empty_cache()
        except Exception:
            pass

    print(f"\nDone. Results: {RESULTS_FILE}", flush=True)


if __name__ == '__main__':
    main()
