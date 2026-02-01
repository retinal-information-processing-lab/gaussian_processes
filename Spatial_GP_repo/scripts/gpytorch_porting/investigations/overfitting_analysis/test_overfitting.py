"""
Overfitting Analysis for vargp_direct

Created by Claude to investigate whether vargp_direct overfits as training progresses.
Tests performance on held-out test set at iterations: 25, 50, 75, 100, 125, 150

Usage:
    python test_overfitting.py --cell 10 --ntilde 200 --n-train 2000
    python test_overfitting.py --cell 8 --ntilde 50 --n-train 500
"""

import subprocess
import re
import argparse
import os

# Get path to run_single_mode.py
script_dir = os.path.dirname(os.path.abspath(__file__))
gpytorch_porting_dir = os.path.dirname(os.path.dirname(script_dir))
run_script = os.path.join(gpytorch_porting_dir, 'run_single_mode.py')


def run_single_test(cell, ntilde, n_train, n_iterations, seed):
    """Run run_single_mode.py and parse results."""

    cmd = [
        'python', run_script,
        '--mode', 'vargp_direct',
        '--float32',
        '--cell', str(cell),
        '--ntilde', str(ntilde),
        '--n-train', str(n_train),
        '--n-iterations', str(n_iterations),
        '--seed', str(seed),
        '--no-early-stop',
    ]

    result = subprocess.run(cmd, capture_output=True, text=True, cwd=gpytorch_porting_dir)
    output = result.stdout + result.stderr

    # Parse results
    train_r = None
    test_r = None
    loss = None

    for line in output.split('\n'):
        if 'Train Pearson r:' in line:
            train_r = float(line.split(':')[1].strip())
        elif 'Test Pearson r:' in line:
            test_r = float(line.split(':')[1].strip())
        elif 'Final loss:' in line:
            loss = float(line.split(':')[1].strip())

    return {
        'iteration': n_iterations,
        'train_r': train_r,
        'test_r': test_r,
        'loss': loss,
    }


def main():
    parser = argparse.ArgumentParser(description='Overfitting analysis for vargp_direct')
    parser.add_argument('--cell', type=int, default=10, help='Cell ID (default: 10)')
    parser.add_argument('--ntilde', type=int, default=200, help='Number of inducing points M (default: 200)')
    parser.add_argument('--n-train', type=int, default=2000, help='Number of training samples (default: 2000)')
    parser.add_argument('--seed', type=int, default=123, help='Random seed (default: 123)')
    parser.add_argument('--checkpoints', type=str, default='25,50,75,100,125,150',
                        help='Comma-separated iteration checkpoints (default: 25,50,75,100,125,150)')

    args = parser.parse_args()

    # Parse checkpoints
    checkpoints = [int(x.strip()) for x in args.checkpoints.split(',')]

    print("=" * 70)
    print("Overfitting Analysis: vargp_direct")
    print("=" * 70)
    print(f"Cell: {args.cell}")
    print(f"M (inducing points): {args.ntilde}")
    print(f"N_train: {args.n_train}")
    print(f"Checkpoints: {checkpoints}")
    print(f"Seed: {args.seed}")
    print("=" * 70)

    results = []
    for checkpoint in sorted(checkpoints):
        print(f"\nRunning {checkpoint} iterations...", end=" ", flush=True)

        result = run_single_test(
            cell=args.cell,
            ntilde=args.ntilde,
            n_train=args.n_train,
            n_iterations=checkpoint,
            seed=args.seed,
        )
        results.append(result)

        if result['test_r'] is not None:
            print(f"train_r={result['train_r']:.4f}, test_r={result['test_r']:.4f}, loss={result['loss']:.2f}")
        else:
            print("FAILED")

    # Print summary table
    print("\n" + "=" * 70)
    print("SUMMARY TABLE")
    print("=" * 70)
    print(f"{'Iter':>6} | {'Train r':>8} | {'Test r':>8} | {'Loss':>10}")
    print("-" * 70)

    valid_results = [r for r in results if r['test_r'] is not None]
    if not valid_results:
        print("No valid results!")
        return

    best_test_r = max(r['test_r'] for r in valid_results)
    best_test_iter = [r['iteration'] for r in valid_results if r['test_r'] == best_test_r][0]

    for r in results:
        if r['test_r'] is not None:
            marker = " <-- BEST" if r['iteration'] == best_test_iter else ""
            print(f"{r['iteration']:>6} | {r['train_r']:>8.4f} | {r['test_r']:>8.4f} | {r['loss']:>10.2f}{marker}")
        else:
            print(f"{r['iteration']:>6} | {'FAILED':>8} | {'FAILED':>8} | {'FAILED':>10}")

    print("-" * 70)
    print(f"Best test_r: {best_test_r:.4f} at iteration {best_test_iter}")

    # Check for overfitting
    test_rs = [r['test_r'] for r in valid_results]
    if test_rs[-1] < best_test_r - 0.01:  # Allow 0.01 tolerance
        decline = best_test_r - test_rs[-1]
        print(f"OVERFITTING DETECTED: test_r declined by {decline:.4f} after iter {best_test_iter}")
    else:
        print("No significant overfitting: test_r stable or improving")

    print("=" * 70)


if __name__ == '__main__':
    main()
