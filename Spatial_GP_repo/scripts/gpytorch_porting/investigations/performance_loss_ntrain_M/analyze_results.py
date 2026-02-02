#!/usr/bin/env python3
"""
Analyze performance loss results (no pandas required).

Usage:
    python analyze_results.py results.csv
"""

import sys
import csv
from pathlib import Path
from collections import defaultdict


def load_results(csv_path):
    """Load results from CSV into a dict."""
    results = []
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            results.append({
                'cell': int(row['cell']),
                'mode': row['mode'],
                'ntrain': int(row['ntrain']),
                'M': int(row['M']),
                'test_r': float(row['test_r']),
                'explained_var': float(row['explained_var']),
                'time_s': float(row['time_s']),
            })
    return results


def get_result(results, mode, ntrain, M):
    """Get a specific result."""
    for r in results:
        if r['mode'] == mode and r['ntrain'] == ntrain and r['M'] == M:
            return r
    return None


def analyze_results(csv_path):
    """Analyze the results and print summary."""
    results = load_results(csv_path)

    print("=" * 80)
    print("PERFORMANCE LOSS INVESTIGATION - CELL 8 RESULTS")
    print("=" * 80)
    print()

    # Check for performance loss patterns
    print("## 1. PERFORMANCE vs NTRAIN (fixing M)")
    print("-" * 80)
    for mode in ['vargp_old', 'vargp_direct', 'default_gpy']:
        print(f"\n{mode}:")
        for M in [50, 100, 200]:
            r500 = get_result(results, mode, 500, M)
            r2000 = get_result(results, mode, 2000, M)

            if r500 and r2000:
                test_r_500 = r500['test_r']
                test_r_2000 = r2000['test_r']
                delta = test_r_2000 - test_r_500
                pct_change = 100 * delta / test_r_500
                indicator = "⬇️ LOSS" if delta < -0.01 else ("⬆️ GAIN" if delta > 0.01 else "→ STABLE")

                print(f"  M={M:3d}: ntrain=500 ({test_r_500:.4f}) → ntrain=2000 ({test_r_2000:.4f}) "
                      f"| Δ={delta:+.4f} ({pct_change:+.1f}%) {indicator}")

    print()
    print("## 2. PERFORMANCE vs M (fixing ntrain)")
    print("-" * 80)
    for mode in ['vargp_old', 'vargp_direct', 'default_gpy']:
        print(f"\n{mode}:")
        for ntrain in [500, 2000]:
            r50 = get_result(results, mode, ntrain, 50)
            r100 = get_result(results, mode, ntrain, 100)
            r200 = get_result(results, mode, ntrain, 200)

            if r50 and r100 and r200:
                test_r_vals = [r50['test_r'], r100['test_r'], r200['test_r']]
                print(f"  ntrain={ntrain}: M=50 ({test_r_vals[0]:.4f}) → M=100 ({test_r_vals[1]:.4f}) → M=200 ({test_r_vals[2]:.4f})")

                # Check for monotonic degradation
                if test_r_vals[0] > test_r_vals[1] > test_r_vals[2]:
                    print(f"    ⚠️  MONOTONIC DEGRADATION with increasing M")
                elif test_r_vals[2] < test_r_vals[0]:
                    loss = test_r_vals[2] - test_r_vals[0]
                    pct = 100 * loss / test_r_vals[0]
                    print(f"    ⚠️  NET LOSS at M=200 vs M=50: {loss:.4f} ({pct:.1f}%)")

    print()
    print("## 3. MODE COMPARISON (best performance)")
    print("-" * 80)
    for ntrain in [500, 2000]:
        for M in [50, 100, 200]:
            subset = [r for r in results if r['ntrain'] == ntrain and r['M'] == M]
            subset_sorted = sorted(subset, key=lambda x: x['test_r'], reverse=True)

            if subset_sorted:
                best = subset_sorted[0]
                worst = subset_sorted[-1]
                print(f"ntrain={ntrain}, M={M:3d}: Best={best['mode']:12s} ({best['test_r']:.4f}) | "
                      f"Worst={worst['mode']:12s} ({worst['test_r']:.4f}) | Gap={best['test_r']-worst['test_r']:.4f}")

    print()
    print("## 4. TIMING COMPARISON")
    print("-" * 80)
    for ntrain in [500, 2000]:
        print(f"\nntrain={ntrain}:")
        for M in [50, 100, 200]:
            old = get_result(results, 'vargp_old', ntrain, M)
            direct = get_result(results, 'vargp_direct', ntrain, M)
            gpy = get_result(results, 'default_gpy', ntrain, M)

            print(f"  M={M:3d}: vargp_old={old['time_s'] if old else 0:.1f}s | "
                  f"vargp_direct={direct['time_s'] if direct else 0:.1f}s | "
                  f"default_gpy={gpy['time_s'] if gpy else 0:.1f}s")

    print()
    print("## 5. KEY FINDINGS")
    print("-" * 80)

    # Find worst degradation cases
    print("\nWorst performance degradations (ntrain 500→2000):")
    for mode in ['vargp_old', 'vargp_direct', 'default_gpy']:
        max_loss = 0
        max_loss_config = None

        for M in [50, 100, 200]:
            r500 = get_result(results, mode, 500, M)
            r2000 = get_result(results, mode, 2000, M)

            if r500 and r2000:
                delta = r2000['test_r'] - r500['test_r']
                if delta < max_loss:
                    max_loss = delta
                    max_loss_config = (M, r500['test_r'], r2000['test_r'])

        if max_loss_config:
            M, r500_val, r2000_val = max_loss_config
            pct = 100 * max_loss / r500_val
            print(f"  {mode:12s}: M={M}, {r500_val:.4f} → {r2000_val:.4f} (loss={max_loss:.4f}, {pct:.1f}%)")

    # Find best configurations
    print("\nBest overall performance:")
    best = max(results, key=lambda x: x['test_r'])
    print(f"  {best['mode']}, ntrain={best['ntrain']}, M={best['M']}: test_r={best['test_r']:.4f}")

    print()
    print("=" * 80)


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python analyze_results.py results.csv")
        sys.exit(1)

    csv_path = Path(sys.argv[1])
    if not csv_path.exists():
        print(f"Error: {csv_path} not found")
        sys.exit(1)

    analyze_results(csv_path)
