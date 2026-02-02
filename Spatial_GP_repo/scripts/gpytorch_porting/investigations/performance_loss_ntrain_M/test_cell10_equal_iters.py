#!/usr/bin/env python3
"""
Test cell 10 with EQUAL iterations for both modes.

Compare:
- vargp_direct (50 iters)
- default_gpy (50 iters) - NOT 500!

To see if default_gpy failure is due to too many iterations.
"""

import subprocess
import json
from pathlib import Path


def run_test(mode, M, n_iterations, seed=123):
    """Run a single test."""
    ntrain = 2000
    cell = 10

    script_path = Path(__file__).parent.parent.parent / 'run_single_mode.py'
    json_output = Path(__file__).parent / f'temp_{mode}_{M}_{n_iterations}.json'

    cmd = [
        'python', str(script_path),
        '--mode', mode,
        '--cell', str(cell),
        '--ntilde', str(M),
        '--n-train', str(ntrain),
        '--n-iterations', str(n_iterations),
        '--seed', str(seed),
        '--float32',
        '--json-append', str(json_output),
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)

        if json_output.exists():
            with open(json_output, 'r') as f:
                lines = f.readlines()
                if lines:
                    data = json.loads(lines[-1])
                    json_output.unlink()
                    return {
                        'test_r': data['test_r'],
                        'time_s': data['time_total_s'],
                    }
        return None
    except Exception:
        if json_output.exists():
            json_output.unlink()
        return None


def main():
    print("=" * 80)
    print("Cell 10: default_gpy with EQUAL iterations (50 vs 500)")
    print("=" * 80)
    print()

    Ms = [50, 100, 200, 500]

    print(f"{'M':>4} | {'vargp_direct':>12} | {'gpy_50iter':>11} | {'gpy_500iter':>12} | Notes")
    print("-" * 80)

    for M in Ms:
        print(f"{M:4d} | ", end='', flush=True)

        # vargp_direct (50 iters)
        r_direct = run_test('vargp_direct', M, 50)
        if r_direct:
            print(f"{r_direct['test_r']:6.4f} ({r_direct['time_s']:3.1f}s) | ", end='', flush=True)
        else:
            print("FAILED        | ", end='', flush=True)

        # default_gpy (50 iters - SAME as vargp_direct)
        r_gpy50 = run_test('default_gpy', M, 50)
        if r_gpy50:
            print(f"{r_gpy50['test_r']:6.4f} ({r_gpy50['time_s']:3.1f}s) | ", end='', flush=True)
        else:
            print("FAILED       | ", end='', flush=True)

        # default_gpy (500 iters - 10x)
        r_gpy500 = run_test('default_gpy', M, 500)
        if r_gpy500:
            print(f"{r_gpy500['test_r']:6.4f} ({r_gpy500['time_s']:3.1f}s) | ", end='', flush=True)
        else:
            print("FAILED        | ", end='', flush=True)

        # Analysis
        if r_direct and r_gpy50 and r_gpy500:
            if r_gpy50['test_r'] > r_gpy500['test_r']:
                print("50iter better")
            elif r_gpy500['test_r'] > r_gpy50['test_r']:
                print("500iter better")
            else:
                print("equal")
        else:
            print("")

    print()
    print("=" * 80)


if __name__ == '__main__':
    main()
