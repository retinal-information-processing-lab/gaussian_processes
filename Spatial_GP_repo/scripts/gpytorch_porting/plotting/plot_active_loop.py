#!/usr/bin/env python3
"""
Plot active learning comparison curves: argmax vs random, for a single cell.

Reads results.jsonl files produced by run_active_loop.py and produces a
2-panel figure comparing the two selection strategies as the training set grows:

  Row 0: test_r       -- Pearson r on the 30-image held-out test set (primary metric)
  Row 1: log-lik      -- log-likelihood (train or held-out, see --heldout flag)

X-axis is n_training (number of images seen so far), starting at phase1_M (default 50)
and growing to phase1_M + n_active_iterations (default 300).

Multiple seeds are shown as faded individual lines (alpha=0.25) plus a bold mean line.
Single seed: one line at full opacity.

Input directory structure
-------------------------
Each --argmax / --random argument is a run directory produced by run_active_loop.py,
i.e. a folder containing results.jsonl (plus config.json, curves.jsonl, checkpoints/).

Single seed (flat layout, e.g. the 2026-04-08 cell 8 run):
    results/active_loop/2026-04-08_cell8_seed42_M50_n250/
        argmax/results.jsonl
        random/results.jsonl

    python plotting/plot_active_loop.py \\
        --argmax results/active_loop/2026-04-08_cell8_seed42_M50_n250/argmax \\
        --random  results/active_loop/2026-04-08_cell8_seed42_M50_n250/random

Multi-seed (batch layout from run_active_loop_batch.py):
    results/active_loop/2026-04-08_paper_gap_5cells/
        cell_09/seed_0/argmax/results.jsonl
        cell_09/seed_0/random/results.jsonl
        cell_09/seed_1/argmax/results.jsonl
        ...

    python plotting/plot_active_loop.py \\
        --argmax results/.../cell_09/seed_0/argmax \\
                 results/.../cell_09/seed_1/argmax \\
                 results/.../cell_09/seed_2/argmax \\
        --random  results/.../cell_09/seed_0/random \\
                 results/.../cell_09/seed_1/random \\
                 results/.../cell_09/seed_2/random

Options:
    --ceiling <path>    JSON file with per-cell ceiling test_r
                        (e.g. checkpoints/64x64_ceiling_M1500/ceiling_results.json)
    --cell <id>         Cell ID to look up in the ceiling JSON
    --heldout           Use heldout_metrics.jsonl for the log-lik panel
                        (per-image avg on a fixed held-out set, fair comparison)

Output
------
Default: <common ancestor of all input dirs>/plots/comparison.png
Override with --output <path>.
"""

import os
import json
import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


# -- Style constants (consistent with plot_training.py) -----------------------

COLOR_ARGMAX = '#66BB6A'   # light green
COLOR_RANDOM = '#000000'   # black
COLOR_CEILING = '#999999'  # gray

ALPHA_SEED   = 0.25   # individual seed lines
LW_SEED      = 1.0
ALPHA_MEAN   = 1.0    # bold mean (or single seed)
LW_MEAN      = 2.0    # single-seed line width
LW_MEAN_MULTI = 2.5   # mean-of-many line width


# -- Data loading -------------------------------------------------------------

def load_results_jsonl(results_dir):
    """Load results.jsonl from a run directory. Returns list of row dicts."""
    path = Path(results_dir) / 'results.jsonl'
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_heldout_jsonl(results_dir):
    """Load heldout_metrics.jsonl from a run directory. Returns list of row dicts."""
    path = Path(results_dir) / 'heldout_metrics.jsonl'
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _extract_series(rows, key):
    """Extract (x, y) arrays for a metric key, skipping null values.

    x = n_training, y = metric value.
    Rows where the value is None/null are dropped.
    Returns (np.ndarray, np.ndarray).
    """
    xs, ys = [], []
    for row in rows:
        val = row.get(key)
        if val is not None:
            xs.append(row['n_training'])
            ys.append(val)
    return np.array(xs, dtype=float), np.array(ys, dtype=float)


# -- Axis styling (adapted from plot_training.py:95-101) ----------------------

def _style_axis(ax, fontsize=11):
    ax.tick_params(labelsize=fontsize)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    for spine in ax.spines.values():
        spine.set_linewidth(0.6)
    ax.grid(axis='y', alpha=0.3, linewidth=0.5, zorder=0)


# -- Per-row plotting ---------------------------------------------------------

def _plot_metric_row(ax, argmax_runs, random_runs, key, ylabel,
                     plot_random=True):
    """Plot one metric row for all seeds.

    argmax_runs / random_runs: list of list-of-dicts (one list per seed).
    """
    n_seeds = len(argmax_runs)
    multi = n_seeds > 1

    lw_line = LW_SEED if multi else LW_MEAN
    alpha_line = ALPHA_SEED if multi else ALPHA_MEAN

    # --- argmax ---
    argmax_ys_interp = []
    x_common = None
    for seed_rows in argmax_runs:
        x, y = _extract_series(seed_rows, key)
        if len(x) == 0:
            continue
        if x_common is None:
            x_common = x
        ax.plot(x, y, color=COLOR_ARGMAX, alpha=alpha_line,
                linewidth=lw_line, zorder=2)
        argmax_ys_interp.append((x, y))

    if multi and argmax_ys_interp:
        # Interpolate all seeds onto a common x grid for the mean
        x_ref = argmax_ys_interp[0][0]
        stacked = np.array([
            np.interp(x_ref, xi, yi)
            for xi, yi in argmax_ys_interp
        ])
        mean_y = stacked.mean(axis=0)
        ax.plot(x_ref, mean_y, color=COLOR_ARGMAX, alpha=ALPHA_MEAN,
                linewidth=LW_MEAN_MULTI, label='argmax', zorder=3)
    elif argmax_ys_interp:
        # Single seed -- re-label the already-drawn line
        ax.get_lines()[-1].set_label('argmax')

    # --- random ---
    if plot_random:
        random_ys_interp = []
        for seed_rows in random_runs:
            x, y = _extract_series(seed_rows, key)
            if len(x) == 0:
                continue
            ax.plot(x, y, color=COLOR_RANDOM, alpha=alpha_line,
                    linewidth=lw_line, zorder=2)
            random_ys_interp.append((x, y))

        if multi and random_ys_interp:
            x_ref = random_ys_interp[0][0]
            stacked = np.array([
                np.interp(x_ref, xi, yi)
                for xi, yi in random_ys_interp
            ])
            mean_y = stacked.mean(axis=0)
            ax.plot(x_ref, mean_y, color=COLOR_RANDOM, alpha=ALPHA_MEAN,
                    linewidth=LW_MEAN_MULTI, label='random', zorder=3)
        elif random_ys_interp:
            ax.get_lines()[-1].set_label('random')

    _style_axis(ax)
    ax.set_ylabel(ylabel, fontsize=11)


# -- Top-level ----------------------------------------------------------------

def plot_active_loop(argmax_dirs, random_dirs, output_path, title=None,
                     ceiling_test_r=None, use_heldout=False):
    """Build the 2-panel active learning comparison figure and save it.

    Args:
        argmax_dirs: list of paths to argmax run directories (one per seed)
        random_dirs: list of paths to random run directories (one per seed)
        output_path: full path for the output PNG
        title: optional suptitle string
        ceiling_test_r: if set, draw a horizontal dashed line at this test_r
        use_heldout: if True, use heldout_metrics.jsonl for the log-lik panel
    """
    if len(argmax_dirs) != len(random_dirs):
        raise ValueError(
            f"--argmax and --random must have the same number of entries "
            f"(got {len(argmax_dirs)} vs {len(random_dirs)})"
        )

    # Load all runs from results.jsonl (always needed for test_r)
    argmax_runs = [load_results_jsonl(d) for d in argmax_dirs]
    random_runs = [load_results_jsonl(d) for d in random_dirs]

    # For log-lik panel: either use heldout_metrics.jsonl or results.jsonl
    if use_heldout:
        argmax_ll_runs = [load_heldout_jsonl(d) for d in argmax_dirs]
        random_ll_runs = [load_heldout_jsonl(d) for d in random_dirs]
        ll_key = 'heldout_log_lik'
        ll_ylabel = 'held-out log-lik (per image)'
    else:
        argmax_ll_runs = argmax_runs
        random_ll_runs = random_runs
        ll_key = 'train_log_lik'
        ll_ylabel = 'train log-lik'

    fig, axes = plt.subplots(2, 1, figsize=(7, 6),
                             sharex=True,
                             gridspec_kw={'hspace': 0.08})

    # Row 0: test_r
    _plot_metric_row(axes[0], argmax_runs, random_runs,
                     key='test_r', ylabel='test r (Pearson)',
                     plot_random=True)

    # Ceiling line
    if ceiling_test_r is not None:
        axes[0].axhline(y=ceiling_test_r, color=COLOR_CEILING,
                        linestyle='--', linewidth=1.5, zorder=1,
                        label=f'ceiling ({ceiling_test_r:.3f})')

    axes[0].legend(fontsize=10, framealpha=0.7)

    # Row 1: log-likelihood
    _plot_metric_row(axes[1], argmax_ll_runs, random_ll_runs,
                     key=ll_key, ylabel=ll_ylabel,
                     plot_random=True)

    # Shared x-axis label on bottom panel only
    axes[1].set_xlabel('Training set size (n)', fontsize=11)

    if title:
        fig.suptitle(title, fontsize=12, y=1.01)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close(fig)
    print(f"Saved: {output_path}")


# -- CLI ----------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Plot active learning comparison: argmax vs random'
    )
    parser.add_argument('--argmax', nargs='+', required=True,
                        help='Argmax run directories (one per seed)')
    parser.add_argument('--random', nargs='+', required=True,
                        help='Random run directories (one per seed)')
    parser.add_argument('--output', type=str, default=None,
                        help='Output PNG path. Default: <common ancestor of all '
                             'input dirs>/plots/comparison.png')
    parser.add_argument('--title', type=str, default=None,
                        help='Optional figure suptitle')
    parser.add_argument('--ceiling', type=str, default=None,
                        help='Path to ceiling_results.json (per-cell ceiling test_r)')
    parser.add_argument('--cell', type=int, default=None,
                        help='Cell ID to look up in --ceiling JSON')
    parser.add_argument('--heldout', action='store_true',
                        help='Use heldout_metrics.jsonl for the log-lik panel')

    args = parser.parse_args()

    # Load ceiling if provided
    ceiling_test_r = None
    if args.ceiling is not None:
        if args.cell is None:
            parser.error("--ceiling requires --cell to specify which cell to look up")
        with open(args.ceiling) as f:
            ceiling_data = json.load(f)
        cell_key = str(args.cell)
        if cell_key not in ceiling_data:
            print(f"WARNING: cell {args.cell} not found in ceiling JSON, skipping ceiling line")
        else:
            ceiling_test_r = ceiling_data[cell_key]['test_r']

    output = args.output
    if output is None:
        all_dirs = [str(Path(d).resolve()) for d in args.argmax + args.random]
        experiment_dir = Path(os.path.commonpath(all_dirs))
        output = experiment_dir / 'plots' / 'comparison.png'

    plot_active_loop(args.argmax, args.random, output, title=args.title,
                     ceiling_test_r=ceiling_test_r, use_heldout=args.heldout)


if __name__ == '__main__':
    main()
