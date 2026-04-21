"""
Phase 1+3+4 Baseline Comparison: default_gpy vs vargp_direct

Runs both modes on the same config (identical data, M, seed) and compares
test_r, final ELBO, and convergence behavior. Optionally also tests
default_gpy with UnwhitenedVariationalStrategy.

Outputs:
  results.jsonl   -- one JSON record per run (appends)
  Prints a comparison table to stdout.

Usage:
  cd Spatial_GP_repo/scripts/gpytorch_porting
  python investigations/default_gpy_gap/run_baseline.py
  python investigations/default_gpy_gap/run_baseline.py --cells 8 1 --seeds 42 123 789
  python investigations/default_gpy_gap/run_baseline.py --include-unwhitened
  python investigations/default_gpy_gap/run_baseline.py --M 100 --n-iterations 100
"""

import sys
import os
import json
import argparse
import datetime
from pathlib import Path

# Run from gpytorch_porting root
root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(root))

from run_single_mode import build_config_from_defaults, run_single_config  # noqa: E402

RESULTS_FILE = Path(__file__).parent / 'results.jsonl'
DATA_64x64 = 'datasets/PNAS_64x64_center_crop_no_renorm.npz'


def run_one(mode, cell, seed, M, n_iterations, unwhitened=False, alternating=False, extra_tag=None):
    """Run one mode/cell/seed combo. Returns the result dict (or None on failure).

    mode can be 'vargp_direct', 'default_gpy', or 'default_gpy_alt' (shorthand
    for default_gpy with alternating_fstep=True).
    """
    actual_mode = mode
    if mode == 'default_gpy_alt':
        actual_mode = 'default_gpy'
        alternating = True

    config = build_config_from_defaults(
        mode=actual_mode,
        data_path=DATA_64x64,
        M=M,
        n_train=500,
        n_iterations=n_iterations,
        cell=cell,
        seed=seed,
        ip_selection='random',          # prevent vargp_old confound
        unwhitened_variational_dist=unwhitened,
        alternating_fstep=alternating,
    )

    label = mode
    if unwhitened:
        label = 'default_gpy_unwhitened'

    print(f"\n{'='*60}")
    print(f"Running: mode={label}, cell={cell}, seed={seed}, M={M}, n_iter={n_iterations}")
    print(f"{'='*60}")

    try:
        result = run_single_config(config)
    except Exception as e:
        print(f"ERROR: {e}")
        result = None

    if result is None:
        print(f"  FAILED: run_single_config returned None")
        record = {
            'mode': label,
            'cell': cell,
            'seed': seed,
            'M': M,
            'n_iterations': n_iterations,
            'status': 'failed',
            'timestamp': datetime.datetime.now().isoformat(timespec='seconds'),
        }
    else:
        record = {
            'mode': label,
            'cell': cell,
            'seed': seed,
            'M': M,
            'n_iterations': n_iterations,
            'test_r': result.get('test_r'),
            'train_r': result.get('train_r'),
            'explained_var': result.get('explained_var'),
            'final_loss': result.get('final_loss'),
            'train_time': result.get('train_time'),
            'n_iterations_run': result.get('n_iterations_run'),
            'stopped_early': result.get('stopped_early'),
            'best_iteration': result.get('best_iteration'),
            'final_A': result.get('final_A'),
            'final_lambda0': result.get('final_lambda0'),
            'final_beta': result.get('final_beta'),
            'final_rho': result.get('final_rho'),
            'final_eps_0x': result.get('final_eps_0x'),
            'final_eps_0y': result.get('final_eps_0y'),
            'status': result.get('status', 'unknown'),
            'timestamp': datetime.datetime.now().isoformat(timespec='seconds'),
        }
        # Save ELBO curve (train_loss) for convergence analysis
        if result.get('curves') and result['curves'].get('train_loss'):
            record['train_loss_curve'] = result['curves']['train_loss']
            record['A_curve'] = result['curves'].get('A', [])

    with open(RESULTS_FILE, 'a') as f:
        f.write(json.dumps(record) + '\n')

    return record


def print_table(records):
    """Print a compact comparison table from list of records."""
    print(f"\n{'='*70}")
    print(f"RESULTS SUMMARY")
    print(f"{'='*70}")
    print(f"{'mode':<28} {'cell':>4} {'seed':>5} {'M':>4} {'test_r':>8} {'train_r':>8} {'final_loss':>12} {'iters_run':>10}")
    print(f"{'-'*70}")
    for r in records:
        if r.get('status') == 'failed':
            print(f"{'  '+r['mode']:<28} {r['cell']:>4} {r['seed']:>5} {r['M']:>4}  FAILED")
            continue
        test_r = r.get('test_r')
        train_r = r.get('train_r')
        final_loss = r.get('final_loss')
        iters_run = r.get('n_iterations_run')
        print(f"{'  '+r['mode']:<28} {r['cell']:>4} {r['seed']:>5} {r['M']:>4} "
              f"{test_r:>8.4f} {train_r:>8.4f} {final_loss:>12.2f} {iters_run:>10}")
    print(f"{'='*70}\n")

    # Gap summary
    direct_records = [r for r in records if r.get('mode') == 'vargp_direct' and r.get('test_r') is not None]
    gpy_records = [r for r in records if r.get('mode') == 'default_gpy' and r.get('test_r') is not None]

    if direct_records and gpy_records:
        print("GAP ANALYSIS (vargp_direct - default_gpy):")
        for d in direct_records:
            matches = [g for g in gpy_records
                       if g['cell'] == d['cell'] and g['seed'] == d['seed'] and g['M'] == d['M']]
            for g in matches:
                gap = d['test_r'] - g['test_r']
                print(f"  cell={d['cell']} seed={d['seed']} M={d['M']}: "
                      f"direct={d['test_r']:.4f}  gpy={g['test_r']:.4f}  gap={gap:+.4f}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cells', nargs='+', type=int, default=[8, 1],
                        help='Cell IDs to test (default: 8 1)')
    parser.add_argument('--seeds', nargs='+', type=int, default=[42, 123, 789],
                        help='Seeds to test (default: 42 123 789)')
    parser.add_argument('--M', type=int, default=50,
                        help='Number of inducing points (default: 50)')
    parser.add_argument('--n-iterations', type=int, default=50,
                        help='Number of training iterations (default: 50)')
    parser.add_argument('--include-unwhitened', action='store_true',
                        help='Also test default_gpy with UnwhitenedVariationalStrategy')
    parser.add_argument('--modes', nargs='+',
                        default=['vargp_direct', 'default_gpy'],
                        help='Modes to compare (default: vargp_direct default_gpy)')
    args = parser.parse_args()

    print(f"\nBaseline comparison: {args.modes}")
    print(f"Cells: {args.cells}, Seeds: {args.seeds}, M={args.M}, n_iter={args.n_iterations}")
    print(f"Results file: {RESULTS_FILE}")

    os.chdir(root)

    all_records = []

    for cell in args.cells:
        for seed in args.seeds:
            for mode in args.modes:
                record = run_one(mode, cell, seed, args.M, args.n_iterations)
                if record:
                    all_records.append(record)

            if args.include_unwhitened and ('default_gpy' in args.modes or 'default_gpy_alt' in args.modes):
                record = run_one('default_gpy', cell, seed, args.M, args.n_iterations,
                                 unwhitened=True)
                if record:
                    all_records.append(record)

    print_table(all_records)


if __name__ == '__main__':
    main()
