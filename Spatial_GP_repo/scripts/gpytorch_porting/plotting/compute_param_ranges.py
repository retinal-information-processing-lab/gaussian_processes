#!/usr/bin/env python3
"""
compute_param_ranges.py - Compute y-axis ranges for training curve plots.

Reads a sweep JSONL and computes min/max per parameter across all cells/seeds,
grouped by config_name. Adds 5% padding for visual margin.

Two modes:
  - Records WITH 'curves': uses full trajectory (all iterations) per param
  - Records WITHOUT 'curves': uses 'final_<param>' fields

Usage:
    python compute_param_ranges.py \
        --sweep path/to/sweep_results.jsonl \
        --output param_ranges.json
"""

import argparse
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path


# Parameters to extract ranges for.
# Maps from range key -> (curves key, final key)
PARAM_MAP = {
    'beta':     ('beta',         'final_beta'),
    'rho':      ('rho',          'final_rho'),
    'sigma_0':  ('sigma_0',      'final_sigma_0'),
    'eps_0x':   ('eps_0x',       'final_eps_0x'),
    'eps_0y':   ('eps_0y',       'final_eps_0y'),
    'A':        ('A',            'final_A'),
    'lambda0':  ('lambda0',      'final_lambda0'),
    'Amp':      ('Amp',          'final_Amp'),
    # Log-lik / loss only available when curves exist
    'train_log_lik': ('train_log_lik', None),
    'val_log_lik':   ('val_log_lik',   None),
    'train_loss':    ('train_loss',    None),
}


def compute_ranges(sweep_path):
    """Read JSONL, return {config_name: {param: {raw_min, raw_max, padded_min, padded_max}}}."""
    # Collect all values per config per param
    values = defaultdict(lambda: defaultdict(list))

    with open(sweep_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            config = record.get('config_name', 'default')
            curves = record.get('curves')

            for param_key, (curve_key, final_key) in PARAM_MAP.items():
                if curves and curve_key in curves:
                    # Use full trajectory
                    vals = [v for v in curves[curve_key] if v is not None]
                    values[config][param_key].extend(vals)
                elif final_key and final_key in record and record[final_key] is not None:
                    values[config][param_key].append(record[final_key])

    # Compute ranges with 5% padding
    ranges = {}
    for config, params in sorted(values.items()):
        ranges[config] = {}
        for param_key, vals in sorted(params.items()):
            if not vals:
                continue
            raw_min = min(vals)
            raw_max = max(vals)
            span = raw_max - raw_min
            if span == 0:
                # All identical — use 5% of absolute value as padding
                pad = abs(raw_min) * 0.05 if raw_min != 0 else 0.1
            else:
                pad = span * 0.05
            ranges[config][param_key] = {
                'raw_min': raw_min,
                'raw_max': raw_max,
                'padded_min': raw_min - pad,
                'padded_max': raw_max + pad,
                'n_values': len(vals),
            }

    return ranges


def main():
    parser = argparse.ArgumentParser(
        description='Compute parameter ranges from sweep JSONL for fixed y-axis plotting.')
    parser.add_argument('--sweep', required=True,
                        help='Path to sweep results JSONL')
    parser.add_argument('--output', required=True,
                        help='Path to output JSON file')
    args = parser.parse_args()

    sweep_path = Path(args.sweep)
    if not sweep_path.exists():
        print(f"Error: {sweep_path} not found")
        return 1

    ranges = compute_ranges(sweep_path)

    output = {
        '_metadata': {
            'source_script': 'compute_param_ranges.py',
            'source_file': str(sweep_path),
            'date': datetime.now().isoformat(timespec='seconds'),
            'method': 'min/max across all cells/seeds with 5% padding',
        },
        **ranges,
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"Wrote ranges to {out_path}")
    for config, params in sorted(ranges.items()):
        print(f"  {config}: {len(params)} params")
        for p, r in sorted(params.items()):
            print(f"    {p}: [{r['raw_min']:.4f}, {r['raw_max']:.4f}] "
                  f"-> [{r['padded_min']:.4f}, {r['padded_max']:.4f}] "
                  f"({r['n_values']} values)")

    return 0


if __name__ == '__main__':
    exit(main() or 0)
