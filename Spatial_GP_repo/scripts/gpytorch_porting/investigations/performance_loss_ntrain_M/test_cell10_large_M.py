#!/usr/bin/env python3
"""
Test cell 10 with large ntrain and varying M.

Tests:
- Cell: 10
- ntrain: 2000 (fixed)
- M: 50, 100, 200, 500
- Modes: vargp_direct (50 iters), default_gpy (500 iters)
- Seed: 123
- Dtype: float32
"""

import subprocess
import csv
import json
from pathlib import Path


def run_test(mode, M, seed=123):
    """Run a single test."""
    ntrain = 2000
    cell = 10

    # default_gpy uses 10x iterations
    n_iterations = 500 if mode == 'default_gpy' else 50

    script_path = Path(__file__).parent.parent.parent / 'run_single_mode.py'
    json_output = Path(__file__).parent / f'temp_{mode}_{M}.json'

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
        print(f"  Running {mode}, M={M}, n_iter={n_iterations}...", end=' ', flush=True)

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600,
        )

        # Read results
        if json_output.exists():
            with open(json_output, 'r') as f:
                lines = f.readlines()
                if lines:
                    data = json.loads(lines[-1])
                    json_output.unlink()

                    print(f"✓ test_r={data['test_r']:.4f}, time={data['time_total_s']:.1f}s")

                    return {
                        'test_r': data['test_r'],
                        'explained_var': data['explained_var'],
                        'time_s': data['time_total_s'],
                        'final_A': data.get('final_A', float('nan')),
                        'final_lambda0': data.get('final_lambda0', float('nan')),
                    }

        print(f"✗ FAILED")
        return None

    except subprocess.TimeoutExpired:
        print(f"✗ TIMEOUT")
        if json_output.exists():
            json_output.unlink()
        return None
    except Exception as e:
        print(f"✗ ERROR: {e}")
        if json_output.exists():
            json_output.unlink()
        return None


def main():
    print("=" * 80)
    print("Cell 10: Performance vs M (ntrain=2000)")
    print("=" * 80)
    print(f"Config: ntrain=2000, seed=123, float32")
    print(f"Modes: vargp_direct (50 iters), default_gpy (500 iters)")
    print()

    Ms = [50, 100, 200, 500]
    modes = ['vargp_direct', 'default_gpy']

    results = []

    for M in Ms:
        print(f"\nTesting M={M}:")
        for mode in modes:
            result = run_test(mode, M)
            if result:
                results.append({
                    'M': M,
                    'mode': mode,
                    **result
                })

    # Save CSV
    output_path = Path(__file__).parent / 'cell10_large_M_results.csv'
    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['M', 'mode', 'test_r', 'explained_var', 'time_s', 'final_A', 'final_lambda0'])
        writer.writeheader()
        writer.writerows(results)

    print()
    print("=" * 80)
    print("RESULTS TABLE")
    print("=" * 80)
    print()

    # Print table
    print(f"{'M':>4} | {'Mode':>12} | {'test_r':>7} | {'expl_var':>8} | {'time':>8} | {'A':>10} | {'lambda0':>10}")
    print("-" * 80)

    for r in results:
        print(f"{r['M']:4d} | {r['mode']:>12s} | {r['test_r']:7.4f} | {r['explained_var']:8.4f} | {r['time_s']:7.1f}s | {r['final_A']:10.6f} | {r['final_lambda0']:10.4f}")

    print()
    print("=" * 80)
    print("PERFORMANCE COMPARISON")
    print("=" * 80)

    # Group by M
    for M in Ms:
        direct = [r for r in results if r['M'] == M and r['mode'] == 'vargp_direct']
        gpy = [r for r in results if r['M'] == M and r['mode'] == 'default_gpy']

        if direct and gpy:
            d = direct[0]
            g = gpy[0]
            delta = d['test_r'] - g['test_r']
            pct = 100 * delta / g['test_r']
            winner = "vargp_direct" if delta > 0 else "default_gpy"

            print(f"\nM={M:3d}:")
            print(f"  vargp_direct: {d['test_r']:.4f} ({d['time_s']:.1f}s)")
            print(f"  default_gpy:  {g['test_r']:.4f} ({g['time_s']:.1f}s, 10x iters)")
            print(f"  Winner: {winner} (Δ={delta:+.4f}, {pct:+.1f}%)")

    print()
    print(f"Results saved to: {output_path}")
    print("=" * 80)


if __name__ == '__main__':
    main()
