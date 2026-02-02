#!/usr/bin/env python3
"""
Systematic test to investigate performance loss with increased ntrain and M.

Created by Claude (2026-02-01)

This script wraps run_single_mode.py to test multiple configurations systematically.

Tests:
- Cells: 8, 10, 15
- ntrain: 500, 2000
- M: 50, 100, 200
- Modes: vargp_old, vargp_direct, default_gpy (adam)
- Dtype: float32 for all
- Seed: 123 (fixed)

Usage:
    python run_systematic_test.py
    python run_systematic_test.py --quick  # Just cell 8
"""

import subprocess
import csv
import argparse
import json
from pathlib import Path
from datetime import datetime


def run_single_test(mode, cell, ntrain, M, seed=123):
    """Run a single test by calling run_single_mode.py.

    Returns:
        dict with results or None if failed
    """
    # Determine number of iterations based on mode
    if mode == 'default_gpy':
        n_iterations = 500  # 10x for adam
    else:
        n_iterations = 50

    # Build command
    script_path = Path(__file__).parent.parent.parent / 'run_single_mode.py'
    cmd = [
        'python', str(script_path),
        '--mode', mode,
        '--cell', str(cell),
        '--ntilde', str(M),
        '--n-train', str(ntrain),
        '--n-iterations', str(n_iterations),
        '--seed', str(seed),
        '--float32',
    ]

    # Add json output to capture results
    json_output = Path(__file__).parent / f'temp_{mode}_{cell}_{ntrain}_{M}.json'
    cmd.extend(['--json-append', str(json_output)])

    try:
        # Run the command
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600,  # 10 minute timeout
        )

        # Read results from JSON
        if json_output.exists():
            with open(json_output, 'r') as f:
                # Read last line (most recent result)
                lines = f.readlines()
                if lines:
                    result_data = json.loads(lines[-1])

                    # Clean up temp file
                    json_output.unlink()

                    return {
                        'test_r': result_data.get('test_r', float('nan')),
                        'explained_var': result_data.get('explained_var', float('nan')),
                        'time_s': result_data.get('time_total_s', float('nan')),
                        'final_A': result_data.get('final_A', float('nan')),
                        'final_lambda0': result_data.get('final_lambda0', float('nan')),
                    }

        # If we get here, something went wrong
        print(f"    Output: {result.stdout[-200:]}")  # Last 200 chars
        if result.stderr:
            print(f"    Error: {result.stderr[-200:]}")
        return None

    except subprocess.TimeoutExpired:
        print(f"    TIMEOUT after 10 minutes")
        if json_output.exists():
            json_output.unlink()
        return None
    except Exception as e:
        print(f"    EXCEPTION: {e}")
        if json_output.exists():
            json_output.unlink()
        return None


def main():
    parser = argparse.ArgumentParser(description='Systematic performance loss investigation')
    parser.add_argument('--quick', action='store_true', help='Only test cell 8')
    parser.add_argument('--output', type=str, default='results.csv', help='Output CSV file')
    args = parser.parse_args()

    # Configuration
    seed = 123
    cells = [8] if args.quick else [8, 10, 15]
    ntrains = [500, 2000]
    Ms = [50, 100, 200]
    modes = ['vargp_old', 'vargp_direct', 'default_gpy']

    print(f"Systematic Performance Loss Investigation")
    print(f"=========================================")
    print(f"Cells: {cells}")
    print(f"ntrains: {ntrains}")
    print(f"Ms: {Ms}")
    print(f"Modes: {modes}")
    print(f"Seed: {seed}")
    print(f"Dtype: float32")
    print()

    # Prepare output CSV
    output_path = Path(__file__).parent / args.output
    fieldnames = ['cell', 'mode', 'ntrain', 'M', 'test_r', 'explained_var', 'time_s', 'final_A', 'final_lambda0']

    with open(output_path, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        # Run all configurations
        total_runs = len(cells) * len(ntrains) * len(Ms) * len(modes)
        run_idx = 0

        for cell_id in cells:
            for ntrain in ntrains:
                for M in Ms:
                    for mode in modes:
                        run_idx += 1
                        print(f"[{run_idx}/{total_runs}] Cell {cell_id}, ntrain={ntrain}, M={M}, mode={mode}...")

                        result = run_single_test(mode, cell_id, ntrain, M, seed)

                        if result is not None:
                            # Write result
                            row = {
                                'cell': cell_id,
                                'mode': mode,
                                'ntrain': ntrain,
                                'M': M,
                                **result
                            }
                            writer.writerow(row)
                            csvfile.flush()

                            print(f"    ✓ test_r={result['test_r']:.4f}, time={result['time_s']:.1f}s")
                        else:
                            # Write failed result
                            print(f"    ✗ FAILED")
                            row = {
                                'cell': cell_id,
                                'mode': mode,
                                'ntrain': ntrain,
                                'M': M,
                                'test_r': float('nan'),
                                'explained_var': float('nan'),
                                'time_s': float('nan'),
                                'final_A': float('nan'),
                                'final_lambda0': float('nan'),
                            }
                            writer.writerow(row)
                            csvfile.flush()

    print()
    print(f"Results saved to: {output_path}")
    print()
    print("To analyze results:")
    print(f"  python analyze_results.py {args.output}")


if __name__ == '__main__':
    main()
