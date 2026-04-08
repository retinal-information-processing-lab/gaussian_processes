#!/usr/bin/env python3
"""
Batch runner for run_active_loop.py across multiple cells and seeds.

Runs each (cell, seed, strategy) combination sequentially on the GPU.
Supports resume: skips complete runs (results.jsonl with exactly n_active+1 rows).
Partial runs (interrupted mid-loop) are detected and re-run.

Usage:
    # Paper-gap 5-cell subset, 3 seeds, both strategies
    python run_active_loop_batch.py \\
        --cells 9 14 18 28 39 \\
        --n-seeds 3 \\
        --output-dir results/active_loop/2026-04-08_paper_gap_5cells

    # Single cell smoke test
    python run_active_loop_batch.py \\
        --cells 8 --n-seeds 1 --strategies argmax \\
        --output-dir /tmp/batch_smoke --n-active 3

Monitor a running experiment:
    tail -f results/active_loop/.../cell_09/seed_0/argmax/run.log

Output structure:
    <output-dir>/
        cell_09/
            seed_0/
                argmax/   (results.jsonl, curves.jsonl, config.json, checkpoints/, run.log)
                random/
            seed_1/
                ...
        cell_14/
            ...
"""

import sys
import json
import time
import argparse
import subprocess
from pathlib import Path


_SCRIPT_DIR = Path(__file__).resolve().parent
_ACTIVE_LOOP_SCRIPT = _SCRIPT_DIR / 'run_active_loop.py'


def _load_n_active_default():
    """Read n_active_iterations default from default_params.json."""
    defaults_path = _SCRIPT_DIR / 'default_params.json'
    with open(defaults_path) as f:
        defaults = json.load(f)
    return defaults['active_learning']['n_active_iterations']


def _count_lines(path):
    """Count lines in a file. Returns 0 if file does not exist."""
    try:
        with open(path) as f:
            return sum(1 for _ in f)
    except FileNotFoundError:
        return 0


def _is_complete(run_dir, n_active):
    """True iff results.jsonl has exactly n_active+1 rows (iter 0..n_active)."""
    return _count_lines(run_dir / 'results.jsonl') == n_active + 1


def _read_final_test_r(run_dir):
    """Read test_r from the last line of results.jsonl. Returns None if file is missing or empty."""
    results_path = run_dir / 'results.jsonl'
    try:
        with open(results_path) as f:
            last_line = None
            for last_line in f:
                pass
        if last_line is None:
            return None
        return json.loads(last_line)['test_r']
    except (FileNotFoundError, json.JSONDecodeError) as e:
        return None


def _fmt_time(seconds):
    """Format elapsed seconds as Xh Ym Zs, Ym Zs, or Zs."""
    s = int(seconds)
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    if h:
        return f"{h}h {m}m {sec}s"
    if m:
        return f"{m}m {sec}s"
    return f"{sec}s"


def run_batch(cells, n_seeds, strategies, output_dir, n_active, passthrough_args):
    total_runs = len(cells) * n_seeds * len(strategies)
    succeeded = []
    failed = []
    skipped = []
    run_idx = 0

    overall_start = time.time()

    print(f"Batch: {len(cells)} cells x {n_seeds} seeds x {len(strategies)} strategies "
          f"= {total_runs} runs")
    print(f"  Cells:      {cells}")
    print(f"  Seeds:      {list(range(n_seeds))}")
    print(f"  Strategies: {strategies}")
    print(f"  n_active:   {n_active}")
    print(f"  Output:     {output_dir}")
    print()

    for cell in cells:
        for seed in range(n_seeds):
            for strategy in strategies:
                run_idx += 1
                run_dir = output_dir / f"cell_{cell:02d}" / f"seed_{seed}" / strategy
                tag = f"[{run_idx}/{total_runs}] Cell {cell:2d}  seed {seed}  {strategy:<7}"

                # --- Skip complete runs ---
                if _is_complete(run_dir, n_active):
                    test_r = _read_final_test_r(run_dir)
                    r_str = f"test_r={test_r:.4f}" if test_r is not None else "test_r=N/A"
                    print(f"{tag}  [SKIP] already complete  {r_str}")
                    skipped.append((cell, seed, strategy))
                    continue

                # --- Launch subprocess ---
                run_dir.mkdir(parents=True, exist_ok=True)
                log_path = run_dir / 'run.log'
                cmd = [
                    sys.executable, str(_ACTIVE_LOOP_SCRIPT),
                    '--cell', str(cell),
                    '--seed', str(seed),
                    '--strategy', strategy,
                    '--output-dir', str(run_dir),
                ] + passthrough_args

                print(f"{tag}  ->  {run_dir}")
                print(f"       monitor: tail -f {log_path}", flush=True)

                t_start = time.time()
                try:
                    with open(log_path, 'w') as log_f:
                        proc = subprocess.run(cmd, stdout=log_f, stderr=subprocess.STDOUT)
                    elapsed = time.time() - t_start

                    if proc.returncode == 0:
                        test_r = _read_final_test_r(run_dir)
                        r_str = f"test_r={test_r:.4f}" if test_r is not None else "test_r=N/A"
                        print(f"       -> DONE    {r_str}   time={_fmt_time(elapsed)}")
                        succeeded.append((cell, seed, strategy))
                    else:
                        print(f"       -> FAILED  returncode={proc.returncode}   "
                              f"time={_fmt_time(elapsed)}   see {log_path}")
                        failed.append((cell, seed, strategy, proc.returncode))

                except OSError as e:
                    elapsed = time.time() - t_start
                    print(f"       -> ERROR   {_fmt_time(elapsed)}: {e}")
                    failed.append((cell, seed, strategy, str(e)))

                print()

    total_time = time.time() - overall_start

    print(f"{'='*60}")
    print(f"BATCH COMPLETE")
    print(f"{'='*60}")
    print(f"Total: {total_runs}  |  "
          f"Success: {len(succeeded)}  |  "
          f"Failed: {len(failed)}  |  "
          f"Skipped: {len(skipped)}")
    print(f"Total time: {_fmt_time(total_time)}")

    if failed:
        print(f"\nFailed runs ({len(failed)}):")
        for cell, seed, strategy, reason in failed:
            log = output_dir / f"cell_{cell:02d}" / f"seed_{seed}" / strategy / "run.log"
            print(f"  cell_{cell:02d}/seed_{seed}/{strategy}  ({reason})  ->  {log}")

    return failed


def main():
    n_active_default = _load_n_active_default()

    parser = argparse.ArgumentParser(
        description='Batch runner for run_active_loop.py across cells and seeds'
    )
    parser.add_argument('--cells', type=int, nargs='+', required=True,
                        help='Cell IDs to run (e.g. --cells 9 14 18 28 39)')
    parser.add_argument('--n-seeds', type=int, default=1,
                        help='Number of seeds: uses seeds 0, 1, ..., n-1 (default: 1)')
    parser.add_argument('--strategies', type=str, nargs='+',
                        default=['argmax', 'random'],
                        choices=['argmax', 'random'],
                        help='Strategies to run (default: argmax random)')
    parser.add_argument('--output-dir', type=str, required=True,
                        help='Root output directory')

    # Pass-through args forwarded to run_active_loop.py if provided
    parser.add_argument('--n-active', type=int, default=None,
                        help=f'Pass-through: number of active iterations '
                             f'(default: {n_active_default} from default_params.json)')
    parser.add_argument('--phase1-M', type=int, default=None,
                        help='Pass-through: phase 1 inducing points')

    args = parser.parse_args()

    # n_active: used both for the skip check and optionally forwarded to subprocess
    n_active = args.n_active if args.n_active is not None else n_active_default

    # Resolve output dir to absolute path
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = _SCRIPT_DIR / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build pass-through arg list (only args explicitly provided on CLI)
    passthrough_args = []
    if args.n_active is not None:
        passthrough_args += ['--n-active', str(args.n_active)]
    if args.phase1_M is not None:
        passthrough_args += ['--phase1-M', str(args.phase1_M)]

    run_batch(
        cells=args.cells,
        n_seeds=args.n_seeds,
        strategies=args.strategies,
        output_dir=output_dir,
        n_active=n_active,
        passthrough_args=passthrough_args,
    )


if __name__ == '__main__':
    main()
