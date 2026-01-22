#!/usr/bin/env python3
"""
run_canonical_tests.py - Run the standard benchmark test matrix.

Runs all canonical configurations for regression testing and performance tracking.
Results are appended to results/benchmark_results.jsonl.

Test Matrix (12 configurations per seed):
  - ntrain=500:  M=50, 100, 200 × modes={vargp_old, vargp_style, default_gpy}
  - ntrain=2000: M=200 × modes={vargp_old, vargp_style, default_gpy}

Constraints (not configurable):
  - Whitened mode only (default)
  - Cached kernels only (default)
  - cell_id=8, niter=50, nestep=10, nmstep=10, nfstep=10

Usage:
    python run_canonical_tests.py --seed 123
    python run_canonical_tests.py --seed 456
    python run_canonical_tests.py --seed 123 --dry-run  # Preview without running
    python run_canonical_tests.py --seed 123 --output results/custom.jsonl

Note: Each seed produces a separate set of results. Run with multiple seeds
to assess seed sensitivity. Results are appended, not overwritten.
"""

import subprocess
import sys
import argparse
from pathlib import Path
from datetime import datetime


# Test matrix definition
# Each entry: (mode, M, ntrain)
TEST_MATRIX = [
    # ntrain=500: M=50, 100, 200
    ('vargp_old', 50, 500),
    ('vargp_style', 50, 500),
    ('default_gpy', 50, 500),
    ('vargp_old', 100, 500),
    ('vargp_style', 100, 500),
    ('default_gpy', 100, 500),
    ('vargp_old', 200, 500),
    ('vargp_style', 200, 500),
    ('default_gpy', 200, 500),
    # ntrain=2000: M=200 only
    ('vargp_old', 200, 2000),
    ('vargp_style', 200, 2000),
    ('default_gpy', 200, 2000),
]


def run_single_test(mode, M, ntrain, seed, output_file, dry_run=False):
    """Run a single test configuration."""
    cmd = [
        sys.executable, 'run_single_mode.py',
        '--mode', mode,
        '--ntilde', str(M),
        '--n-train', str(ntrain),
        '--seed', str(seed),
        '--json-append', str(output_file),
        '--save-plot', 'none',  # Disable plot saving for batch runs
    ]

    desc = f"{mode:12} M={M:3} ntrain={ntrain:4}"

    if dry_run:
        print(f"  [DRY RUN] {desc}")
        print(f"            {' '.join(cmd)}")
        return True

    print(f"  Running: {desc}", end='', flush=True)

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent,
        )

        if result.returncode != 0:
            print(f" FAILED")
            print(f"    stderr: {result.stderr[:500]}")
            return False

        # Extract key metrics from stdout for quick feedback
        lines = result.stdout.split('\n')
        test_r = None
        expl_var = None
        time_s = None
        for line in lines:
            if 'Test Pearson r:' in line:
                test_r = line.split(':')[1].strip()
            if 'Explained var:' in line:
                expl_var = line.split(':')[1].strip()
            if 'Training time:' in line:
                time_s = line.split(':')[1].strip()

        print(f" OK  (r={test_r}, expl={expl_var}, time={time_s})")
        return True

    except Exception as e:
        print(f" ERROR: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description='Run canonical benchmark test matrix',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument('--seed', type=int, required=True,
                        help='Random seed (required)')
    parser.add_argument('--output', type=str, default='results/benchmark_results.jsonl',
                        help='Output JSONL file (default: results/benchmark_results.jsonl)')
    parser.add_argument('--dry-run', action='store_true',
                        help='Print commands without running')
    parser.add_argument('--modes', type=str, nargs='+',
                        default=['vargp_old', 'vargp_style', 'default_gpy'],
                        choices=['vargp_old', 'vargp_style', 'default_gpy'],
                        help='Modes to test (default: all)')
    args = parser.parse_args()

    output_path = Path(__file__).parent / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Filter matrix by requested modes
    matrix = [(m, M, n) for m, M, n in TEST_MATRIX if m in args.modes]

    print(f"Canonical Benchmark Test Run")
    print(f"=" * 60)
    print(f"  Seed:   {args.seed}")
    print(f"  Output: {output_path}")
    print(f"  Tests:  {len(matrix)} configurations")
    print(f"  Modes:  {', '.join(args.modes)}")
    print(f"  Time:   {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"=" * 60)
    print()

    if args.dry_run:
        print("[DRY RUN MODE - No tests will be executed]\n")

    # Run all tests
    passed = 0
    failed = 0

    for mode, M, ntrain in matrix:
        success = run_single_test(
            mode=mode,
            M=M,
            ntrain=ntrain,
            seed=args.seed,
            output_file=output_path,
            dry_run=args.dry_run
        )
        if success:
            passed += 1
        else:
            failed += 1

    print()
    print(f"=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    if not args.dry_run:
        print(f"Output:  {output_path}")
    print(f"=" * 60)

    return 0 if failed == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
