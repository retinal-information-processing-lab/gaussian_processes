#!/usr/bin/env python3
"""
Stuck-near-init diagnosis (H2): test whether more EM iterations or larger
initial set breaks cell 0 seed 1 out of the A-collapse trap.

Finding: A collapses to 0.0001 during Phase 1 for stuck seeds. Early stopping
freezes the model at iter 19. This script tests two interventions:

  (1) --more-iters: Phase 1 with n_iterations=200, early_stop=False
      Tests if the model can recover given unlimited EM budget.

  (2) --more-data: Phase 1 with phase1_M=100 (doubled from 50)
      Tests if more initial data prevents the A collapse.

Usage:
    python test_stuck_diagnosis.py --more-iters --output-dir /tmp/stuck_diag_iters
    python test_stuck_diagnosis.py --more-data --output-dir /tmp/stuck_diag_data
"""

import sys
import json
import argparse
from pathlib import Path

# Add script root to path
_script_dir = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_script_dir))

from run_single_mode import build_config_from_defaults, run_single_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--more-iters', action='store_true',
                        help='Phase 1 with 200 iterations, no early stopping')
    parser.add_argument('--more-data', action='store_true',
                        help='Phase 1 with M=100 instead of 50')
    parser.add_argument('--output-dir', type=str, required=True)
    parser.add_argument('--cell', type=int, default=0)
    parser.add_argument('--seed', type=int, default=1)
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    M = 100 if args.more_data else 50

    config = build_config_from_defaults(
        mode='vargp_direct',
        data_path='datasets/PNAS_64x64_center_crop_no_renorm.npz',
        M=M, n_train=M,
        seed=args.seed,
        cell=args.cell,
    )

    if args.more_iters:
        config['n_iterations'] = 200
        config['early_stop'] = False
        tag = f'cell{args.cell}_seed{args.seed}_M{M}_iters200_noES'
    elif args.more_data:
        tag = f'cell{args.cell}_seed{args.seed}_M{M}_iters50_ES'
    else:
        tag = f'cell{args.cell}_seed{args.seed}_M{M}_baseline'

    print(f'=== {tag} ===')
    print(f'M={M}, n_train={M}, n_iterations={config["n_iterations"]}, '
          f'early_stop={config["early_stop"]}')

    result = run_single_config(config)

    # Extract key metrics
    curves = result['curves']
    summary = {
        'tag': tag,
        'cell': args.cell,
        'seed': args.seed,
        'M': M,
        'n_iterations_config': config['n_iterations'],
        'early_stop': config['early_stop'],
        'n_iterations_run': result['n_iterations_run'],
        'stopped_early': result['stopped_early'],
        'best_iteration': result['best_iteration'],
        'test_r': result['test_r'],
        'final_loss': result['final_loss'],
        'A_trajectory': curves['A'],
        'beta_trajectory': curves['beta'],
        'lambda0_trajectory': curves['lambda0'],
        'final_A': curves['A'][-1],
        'final_beta': curves['beta'][-1],
        'final_lambda0': curves['lambda0'][-1],
        'final_Amp': curves['Amp'][-1],
    }

    # Save summary
    summary_path = out / f'{tag}.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f'\nResults:')
    print(f'  n_iterations_run: {summary["n_iterations_run"]}')
    print(f'  stopped_early: {summary["stopped_early"]}')
    print(f'  test_r: {summary["test_r"]:.4f}')
    print(f'  final_A: {summary["final_A"]:.6f}')
    print(f'  final_beta: {summary["final_beta"]:.4f}')
    print(f'  final_Amp: {summary["final_Amp"]:.4f}')
    print(f'  A range: {min(curves["A"]):.6f} to {max(curves["A"]):.6f}')
    print(f'\nSaved to {summary_path}')


if __name__ == '__main__':
    main()
