#!/usr/bin/env python3
"""
compare_whitening_strategies.py - Compare whitened vs unwhitened variational strategies.

Runs all combinations:
- M = 50, 100
- Modes: vargp_style, adam, efm
- Whitening: ON (standard VariationalStrategy) vs OFF (UnwhitenedVariationalStrategy)
- Caching: ON (for vargp_style)

Total: 2 M values × 3 modes × 2 whitening = 12 configurations

Created by Claude for investigating whitening strategy performance.

Usage:
    python investigations/compare_whitening_strategies.py
"""

import sys
import time
import subprocess
from pathlib import Path

# Results storage
results = []

# Configurations to test
M_VALUES = [50, 100]
MODES = ['vargp_style', 'adam', 'efm']
WHITENING = [True, False]  # True = standard, False = unwhitened

def run_single_mode(mode, ntilde, unwhitened=False):
    """Run run_single_mode.py and capture results."""
    script_path = Path(__file__).parent.parent / 'run_single_mode.py'

    cmd = [
        'python', str(script_path),
        '--mode', mode,
        '--ntilde', str(ntilde),
        '--save-plot', 'none',  # Don't save plots
    ]

    if unwhitened:
        cmd.append('--unwhitened')

    print(f"\n{'='*70}")
    strategy = 'unwhitened' if unwhitened else 'whitened'
    print(f"Running: mode={mode}, M={ntilde}, strategy={strategy}")
    print(f"Command: {' '.join(cmd)}")
    print('='*70)

    start = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = time.time() - start

    # Print output
    print(result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr)

    # Parse results from output
    output = result.stdout

    # Extract metrics
    metrics = {
        'mode': mode,
        'M': ntilde,
        'whitened': not unwhitened,
        'elapsed': elapsed,
    }

    # Parse lines like "  Test Pearson r:  0.8234"
    for line in output.split('\n'):
        if 'Test Pearson r:' in line:
            try:
                metrics['test_r'] = float(line.split(':')[-1].strip())
            except:
                metrics['test_r'] = None
        elif 'Explained var:' in line:
            try:
                metrics['explained_var'] = float(line.split(':')[-1].strip())
            except:
                metrics['explained_var'] = None
        elif 'Training time:' in line:
            try:
                metrics['train_time'] = float(line.split(':')[-1].replace('s', '').strip())
            except:
                metrics['train_time'] = None
        elif 'Reliability:' in line:
            try:
                metrics['reliability'] = float(line.split(':')[-1].strip())
            except:
                metrics['reliability'] = None

    return metrics


def print_results_table(results):
    """Print formatted comparison table."""
    print("\n" + "="*90)
    print("COMPARISON: Whitened vs Unwhitened Variational Strategy")
    print("="*90)

    # Group by M value
    for M in M_VALUES:
        print(f"\n--- M = {M} ---")
        print("┌────────────────┬───────────┬───────────┬────────────┬──────────┬──────────┐")
        print("│ Mode           │ Whitening │ Test r    │ Expl. Var  │ Time (s) │ Reliab.  │")
        print("├────────────────┼───────────┼───────────┼────────────┼──────────┼──────────┤")

        M_results = [r for r in results if r['M'] == M]

        # Sort: by mode, then whitened first
        M_results.sort(key=lambda x: (x['mode'], not x['whitened']))

        for r in M_results:
            mode = r['mode'][:14].ljust(14)
            whiten = 'ON'.ljust(9) if r['whitened'] else 'OFF'.ljust(9)
            test_r = f"{r.get('test_r', 'N/A'):.4f}".ljust(9) if r.get('test_r') else 'N/A'.ljust(9)
            expl_var = f"{r.get('explained_var', 'N/A'):.4f}".ljust(10) if r.get('explained_var') else 'N/A'.ljust(10)
            train_time = f"{r.get('train_time', 'N/A'):.1f}".ljust(8) if r.get('train_time') else 'N/A'.ljust(8)
            reliab = f"{r.get('reliability', 'N/A'):.4f}".ljust(8) if r.get('reliability') else 'N/A'.ljust(8)
            print(f"| {mode} | {whiten} | {test_r} | {expl_var} | {train_time} | {reliab} |")

        print("└────────────────┴───────────┴───────────┴────────────┴──────────┴──────────┘")

    # Print summary of differences
    print("\n" + "="*70)
    print("SUMMARY: Whitened - Unwhitened (positive = whitened is better)")
    print("="*70)

    for M in M_VALUES:
        print(f"\nM = {M}:")
        for mode in MODES:
            whitened = next((r for r in results if r['M'] == M and r['mode'] == mode and r['whitened']), None)
            unwhitened = next((r for r in results if r['M'] == M and r['mode'] == mode and not r['whitened']), None)

            if whitened and unwhitened and whitened.get('explained_var') and unwhitened.get('explained_var'):
                diff = whitened['explained_var'] - unwhitened['explained_var']
                time_ratio = whitened.get('train_time', 1) / unwhitened.get('train_time', 1) if unwhitened.get('train_time') else None
                time_str = f"{time_ratio:.2f}x" if time_ratio else "N/A"
                print(f"  {mode:14s}: expl_var diff = {diff:+.4f}, time ratio = {time_str}")


def main():
    print("Comparing whitened vs unwhitened variational strategies")
    print(f"M values: {M_VALUES}")
    print(f"Modes: {MODES}")
    print(f"Total configurations: {len(M_VALUES) * len(MODES) * len(WHITENING)}")

    all_results = []

    for M in M_VALUES:
        for mode in MODES:
            for whitened in WHITENING:
                result = run_single_mode(mode, M, unwhitened=not whitened)
                all_results.append(result)

    # Print final comparison table
    print_results_table(all_results)

    return all_results


if __name__ == '__main__':
    main()
