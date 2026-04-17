#!/usr/bin/env python3
"""
Phase 0 Analysis: Characterize M-step LBFGS behavior.

Reads results.jsonl produced by run_phase0.py and produces:

1. Histogram of actual LBFGS iterations run per M-step (vs max_iter=20)
2. Per-outer-iteration ELBO trace inside the M-step
3. Wall-time breakdown (M-step closure time vs total training)
4. Cell-to-cell variability
5. Summary statistics (text)

Usage:
    python analyze_phase0.py [--results results.jsonl] [--out-dir plots/]
"""

import argparse
import json
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def load_results(path):
    results = []
    with open(path) as f:
        for line in f:
            try:
                results.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return results


def extract_diagnostics(results):
    """Extract diagnostics into flat arrays for analysis."""
    records = []
    for r in results:
        cell = r.get('cell')
        seed = r.get('seed')
        diag = r.get('mstep_diagnostics')
        if diag is None:
            continue
        n_outer = len(diag)
        for entry in diag:
            oi = entry['outer_iter']
            records.append({
                'cell': cell,
                'seed': seed,
                'outer_iter': oi,
                'outer_iter_frac': oi / max(n_outer, 1),
                'n_lbfgs_iters': entry['n_lbfgs_iters'],
                'termination': entry['termination'],
                'total_time_s': entry['total_time_s'],
                'n_closure_calls': len(entry['closure_calls']),
                'closure_calls': entry['closure_calls'],
            })
    return records


def plot_lbfgs_iter_histogram(records, out_dir):
    """Histogram of actual LBFGS iterations run per M-step call."""
    n_iters = [r['n_lbfgs_iters'] for r in records if r['n_lbfgs_iters'] >= 0]
    terminations = [r['termination'] for r in records]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: histogram
    ax = axes[0]
    ax.hist(n_iters, bins=range(0, 22), align='left', edgecolor='black',
            alpha=0.7)
    ax.set_xlabel('LBFGS iterations actually run')
    ax.set_ylabel('Count (across all outer EM iters, all cells/seeds)')
    ax.set_title('Distribution of LBFGS iterations per M-step')
    ax.axvline(x=20, color='red', linestyle='--', label='max_iter=20')
    median_val = np.median(n_iters)
    ax.axvline(x=median_val, color='orange', linestyle='--',
               label=f'median={median_val:.0f}')
    ax.legend()

    # Right: termination reasons
    ax = axes[1]
    from collections import Counter
    counts = Counter(terminations)
    labels = list(counts.keys())
    values = [counts[l] for l in labels]
    ax.bar(labels, values, edgecolor='black', alpha=0.7)
    ax.set_xlabel('Termination reason')
    ax.set_ylabel('Count')
    ax.set_title('LBFGS termination reasons')
    for i, (l, v) in enumerate(zip(labels, values)):
        ax.text(i, v + 0.5, str(v), ha='center', fontsize=10)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'lbfgs_iter_histogram.png'), dpi=150)
    plt.close()
    print(f"  Saved lbfgs_iter_histogram.png")


def plot_elbo_trace_per_mstep(records, out_dir):
    """Per-outer-iteration ELBO trace inside the M-step.

    Shows how ELBO evolves across closure calls within a single M-step,
    for early vs mid vs late outer EM iterations.
    """
    # Group by cell and seed
    from collections import defaultdict
    by_run = defaultdict(list)
    for r in records:
        by_run[(r['cell'], r['seed'])].append(r)

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # Pick 3 phases of training: early (iter 1-5), mid (iter 15-25), late (iter 40-60)
    phases = [
        ('Early (iter 1-5)', lambda oi: 1 <= oi <= 5),
        ('Mid (iter 15-25)', lambda oi: 15 <= oi <= 25),
        ('Late (iter 40-60)', lambda oi: 40 <= oi <= 60),
    ]

    for col, (phase_name, phase_filter) in enumerate(phases):
        # Top row: ELBO values across closure calls
        ax_top = axes[0, col]
        # Bottom row: ELBO improvement (relative to first valid call)
        ax_bot = axes[1, col]

        for (cell, seed), entries in sorted(by_run.items()):
            phase_entries = [e for e in entries if phase_filter(e['outer_iter'])]
            for entry in phase_entries:
                calls = entry['closure_calls']
                valid_losses = [c['loss'] for c in calls
                                if not c['rejected'] and c['loss'] is not None]
                if len(valid_losses) < 2:
                    continue
                # ELBO = -loss
                elbos = [-v for v in valid_losses]
                xs = list(range(len(elbos)))
                ax_top.plot(xs, elbos, alpha=0.15, color='steelblue',
                            linewidth=0.8)
                # Relative improvement from first closure call
                baseline = elbos[0]
                if abs(baseline) > 1e-10:
                    rel_imp = [(e - baseline) / abs(baseline) for e in elbos]
                else:
                    rel_imp = [e - baseline for e in elbos]
                ax_bot.plot(xs, rel_imp, alpha=0.15, color='steelblue',
                            linewidth=0.8)

        ax_top.set_title(phase_name)
        ax_top.set_xlabel('Closure call index')
        if col == 0:
            ax_top.set_ylabel('ELBO (= -loss)')
            ax_bot.set_ylabel('Relative ELBO improvement')
        ax_bot.set_xlabel('Closure call index')

    fig.suptitle('M-step ELBO convergence across closure calls\n'
                 '(each line = one M-step call from one run)',
                 fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'elbo_trace_per_mstep.png'), dpi=150)
    plt.close()
    print(f"  Saved elbo_trace_per_mstep.png")


def plot_lbfgs_iters_over_training(records, out_dir):
    """How the number of LBFGS iterations changes over outer EM iterations."""
    from collections import defaultdict
    by_run = defaultdict(list)
    for r in records:
        by_run[(r['cell'], r['seed'])].append(r)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: n_lbfgs_iters vs outer_iter
    ax = axes[0]
    for (cell, seed), entries in sorted(by_run.items()):
        ois = [e['outer_iter'] for e in entries]
        nli = [e['n_lbfgs_iters'] for e in entries]
        ax.plot(ois, nli, alpha=0.3, linewidth=1, label=f'c{cell}s{seed}')
    ax.set_xlabel('Outer EM iteration')
    ax.set_ylabel('LBFGS iterations run')
    ax.set_title('LBFGS iterations vs training progress')
    ax.axhline(y=20, color='red', linestyle='--', alpha=0.5, label='max_iter=20')

    # Right: M-step wall time vs outer_iter
    ax = axes[1]
    for (cell, seed), entries in sorted(by_run.items()):
        ois = [e['outer_iter'] for e in entries]
        times = [e['total_time_s'] for e in entries]
        ax.plot(ois, times, alpha=0.3, linewidth=1)
    ax.set_xlabel('Outer EM iteration')
    ax.set_ylabel('M-step wall time (s)')
    ax.set_title('M-step wall time vs training progress')

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'lbfgs_iters_over_training.png'), dpi=150)
    plt.close()
    print(f"  Saved lbfgs_iters_over_training.png")


def plot_cell_comparison(records, out_dir):
    """Box plot of LBFGS iterations per cell."""
    from collections import defaultdict
    by_cell = defaultdict(list)
    for r in records:
        by_cell[r['cell']].append(r['n_lbfgs_iters'])

    cells_sorted = sorted(by_cell.keys())
    data = [by_cell[c] for c in cells_sorted]

    fig, ax = plt.subplots(figsize=(10, 5))
    bp = ax.boxplot(data, labels=[f'Cell {c}' for c in cells_sorted],
                    patch_artist=True)
    for patch in bp['boxes']:
        patch.set_facecolor('steelblue')
        patch.set_alpha(0.5)

    ax.set_ylabel('LBFGS iterations per M-step')
    ax.set_title('M-step LBFGS iterations by cell (3 seeds pooled)')
    ax.axhline(y=20, color='red', linestyle='--', alpha=0.5, label='max_iter=20')
    ax.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'cell_comparison.png'), dpi=150)
    plt.close()
    print(f"  Saved cell_comparison.png")


def print_summary(results, records):
    """Print text summary of key findings."""
    print("\n" + "=" * 70)
    print("PHASE 0 SUMMARY")
    print("=" * 70)

    # Per-run summary
    print("\nPer-run results:")
    print(f"  {'Cell':>6} {'Seed':>4} {'test_r':>7} {'ES?':>4} "
          f"{'Iters':>5} {'Mstep_s':>8} {'Total_s':>8}")
    for r in sorted(results, key=lambda x: (x.get('cell', 0), x.get('seed', 0))):
        cell = r.get('cell', '?')
        seed = r.get('seed', '?')
        tr = r.get('test_r', 0)
        se = 'Y' if r.get('stopped_early', False) else 'N'
        fi = r.get('n_iterations_run', '?')
        ms = r.get('time_mstep_s', 0)
        tt = r.get('time_total_s', 0)
        print(f"  {cell:>6} {seed:>4} {tr:>7.4f} {se:>4} {fi:>5} "
              f"{ms:>8.1f} {tt:>8.1f}")

    # LBFGS iteration statistics
    n_iters = [r['n_lbfgs_iters'] for r in records if r['n_lbfgs_iters'] >= 0]
    if n_iters:
        print(f"\nLBFGS iterations per M-step (N={len(n_iters)} M-step calls):")
        print(f"  Mean:   {np.mean(n_iters):.1f}")
        print(f"  Median: {np.median(n_iters):.0f}")
        print(f"  Min:    {np.min(n_iters)}")
        print(f"  Max:    {np.max(n_iters)}")
        print(f"  P25:    {np.percentile(n_iters, 25):.0f}")
        print(f"  P75:    {np.percentile(n_iters, 75):.0f}")
        print(f"  P90:    {np.percentile(n_iters, 90):.0f}")
        pct_maxed = 100 * sum(1 for x in n_iters if x >= 20) / len(n_iters)
        print(f"  Hit max_iter=20: {pct_maxed:.1f}% of M-steps")

    # Termination reasons
    from collections import Counter
    terms = Counter(r['termination'] for r in records)
    print(f"\nTermination reasons:")
    for reason, count in terms.most_common():
        print(f"  {reason}: {count} ({100*count/len(records):.1f}%)")

    # Closure calls per M-step
    n_calls = [r['n_closure_calls'] for r in records]
    if n_calls:
        print(f"\nClosure calls per M-step:")
        print(f"  Mean:   {np.mean(n_calls):.1f}")
        print(f"  Median: {np.median(n_calls):.0f}")

    # Wall time breakdown
    mstep_times = [r['total_time_s'] for r in records]
    if mstep_times:
        print(f"\nM-step wall time per call:")
        print(f"  Mean:   {np.mean(mstep_times):.3f}s")
        print(f"  Median: {np.median(mstep_times):.3f}s")

    # Per-cell median
    from collections import defaultdict
    by_cell = defaultdict(list)
    for r in records:
        by_cell[r['cell']].append(r['n_lbfgs_iters'])
    print(f"\nMedian LBFGS iters by cell:")
    for cell in sorted(by_cell.keys()):
        vals = by_cell[cell]
        print(f"  Cell {cell:>2}: median={np.median(vals):.0f}, "
              f"mean={np.mean(vals):.1f}, "
              f"max={np.max(vals)}")

    print("\n" + "=" * 70)


def main():
    parser = argparse.ArgumentParser(description='Analyze Phase 0 results')
    parser.add_argument('--results', default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), 'results.jsonl'),
        help='Path to results.jsonl')
    parser.add_argument('--out-dir', default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), 'plots'),
        help='Output directory for plots')
    args = parser.parse_args()

    if not os.path.exists(args.results):
        print(f"ERROR: {args.results} not found. Run run_phase0.py first.")
        sys.exit(1)

    results = load_results(args.results)
    print(f"Loaded {len(results)} runs from {args.results}")

    records = extract_diagnostics(results)
    print(f"Extracted {len(records)} M-step diagnostic entries")

    if not records:
        print("ERROR: No M-step diagnostics found in results.")
        sys.exit(1)

    os.makedirs(args.out_dir, exist_ok=True)
    print(f"\nGenerating plots in {args.out_dir}/")

    plot_lbfgs_iter_histogram(records, args.out_dir)
    plot_elbo_trace_per_mstep(records, args.out_dir)
    plot_lbfgs_iters_over_training(records, args.out_dir)
    plot_cell_comparison(records, args.out_dir)

    print_summary(results, records)


if __name__ == '__main__':
    main()
