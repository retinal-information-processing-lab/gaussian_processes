#!/usr/bin/env python3
"""Run a few cells with val_from_train=True and generate training curve plots.

Purpose: test that plot_training.py works with corrected val splits.
Saves curves.jsonl and then calls plot_training.py on it.
"""
import sys
import os
import json
import math
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from run_single_mode import build_config_from_defaults, run_single_config

CELLS = [0, 8, 18, 28]
SEEDS = [1, 2, 3]
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'plot_test')
os.makedirs(OUTPUT_DIR, exist_ok=True)
curves_path = os.path.join(OUTPUT_DIR, 'curves.jsonl')

# Clear previous
if os.path.exists(curves_path):
    os.remove(curves_path)


def _sanitize(v):
    """Make values JSON-serializable."""
    if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
        return None
    if hasattr(v, 'item'):  # torch.Tensor scalar
        return v.item()
    return v


for cell in CELLS:
    for seed in SEEDS:
        print(f"\n{'='*60}")
        print(f"  Cell {cell}, Seed {seed}")
        print(f"{'='*60}")

        config = build_config_from_defaults(
            mode='vargp_direct',
            seed=seed, cell=cell, M=250, n_train=3160,
            interleave_fstep=True, fix_Amp=True, A_init=1e-4,
            n_iterations=30,
            val_from_train=True,
            n_val_split=250,
        )

        result = run_single_config(config)

        if result.get('curves'):
            record = {
                'mode': result['mode'],
                'M': result['M'],
                'n_train': result['n_train'],
                'n_val': result['n_val'],
                'seed': result['seed'],
                'cell': result['cell'],
                'best_iteration': result.get('best_iteration'),
                'stopped_early': result.get('stopped_early', False),
                'final_iteration': result.get('n_iterations_run'),
                'test_r': round(result['test_r'], 4) if result.get('test_r') is not None else None,
                'curves': {k: [_sanitize(x) for x in v] for k, v in result['curves'].items()},
            }
            with open(curves_path, 'a') as f:
                f.write(json.dumps(record) + '\n')

        print(f"  test_r={result.get('test_r', 'N/A'):.4f}")

print(f"\nCurves saved to: {curves_path}")
print(f"Now run: python plot_training.py --curves {curves_path} --output-dir {OUTPUT_DIR}/plots/")
