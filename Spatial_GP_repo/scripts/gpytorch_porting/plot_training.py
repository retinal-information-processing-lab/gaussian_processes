#!/usr/bin/env python3
"""
plot_training.py - Training curve visualization for variational GP experiments.

Plots per-iteration training diagnostics grouped by cell, with one column per seed.

Layout: 3 rows x N_seeds columns per cell figure.
  Row 1: Log-likelihood (train + val, normalized per sample) with ELBO on right axis
  Row 2: Likelihood params (A + lambda0 on colored twin axes)
  Row 3: Kernel params (beta, rho, sigma_0, eps_0x/eps_0y on offset colored axes)

Best iteration marked with vertical dashed line in all subplots.

Usage:
    python plot_training.py --curves path/to/curves.jsonl --output-dir path/to/plots/
    python plot_training.py --curves path/to/curves.jsonl --cell 8 --output-dir path/to/plots/
    python plot_training.py --exp experiment_name --output-dir path/to/plots/
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np


# -- Style constants ----------------------------------------------------------

# Log-lik / ELBO panel
COLOR_TRAIN_LL = '#2176AE'   # steel blue
COLOR_VAL_LL = '#E63946'     # crimson
COLOR_ELBO = '#7D8491'       # slate gray

# Likelihood params
COLOR_A = '#2176AE'          # steel blue
COLOR_LAMBDA0 = '#E63946'    # crimson

# Kernel params
COLOR_BETA = '#2CA02C'       # green
COLOR_RHO = '#FF7F0E'        # orange
COLOR_SIGMA0 = '#9467BD'     # purple
COLOR_EPS = '#8C564B'        # brown

# Best-iteration line
COLOR_BEST = '#555555'

# Fallback dataset sizes (PNAS). Used ONLY for old JSONL files that predate
# the n_train/n_val fields. A warning is printed when these are used.
_FALLBACK_N_TRAIN = 2910
_FALLBACK_N_VAL = 250
_FALLBACK_WARNED = False


def load_curves_jsonl(path):
    """Load a JSONL file containing training curves.

    Each line must have at minimum: 'cell', 'seed', 'curves' keys.
    Returns list of record dicts.
    """
    records = []
    with open(path, 'r') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if 'curves' not in record:
                print(f"  Warning: line {line_num} has no 'curves' key, skipping")
                continue
            records.append(record)
    return records


def group_by_cell(records):
    """Group records by cell ID. Returns {cell_id: {seed: record}}."""
    grouped = defaultdict(dict)
    for r in records:
        grouped[r['cell']][r['seed']] = r
    return dict(grouped)


def _style_axis(ax, fontsize=8):
    """Apply clean styling to an axis."""
    ax.tick_params(labelsize=fontsize)
    ax.spines['top'].set_visible(False)
    for spine in ax.spines.values():
        spine.set_linewidth(0.6)
    ax.grid(axis='y', alpha=0.15, linewidth=0.5)


def _add_best_line(ax, best_iteration):
    """Add vertical dashed line at best iteration."""
    if best_iteration is not None and best_iteration > 0:
        ax.axvline(best_iteration, color=COLOR_BEST, linestyle='--',
                   alpha=0.6, linewidth=0.9, zorder=1)


def _plot_loglik_and_elbo(ax, curves, best_iteration, n_train, n_val):
    """Row 1: Normalized log-lik (train + val) on left axis, ELBO on right."""
    train_ll = curves.get('train_log_lik', [])
    val_ll = curves.get('val_log_lik', [])
    train_loss = curves.get('train_loss', [])

    if not train_ll and not val_ll and not train_loss:
        ax.text(0.5, 0.5, 'No data', ha='center', va='center',
                transform=ax.transAxes, fontsize=9, color='gray')
        return

    # Normalized log-likelihood (per sample)
    if train_ll:
        iters = np.arange(1, len(train_ll) + 1)
        ll_norm = np.array(train_ll) / n_train
        ax.plot(iters, ll_norm, color=COLOR_TRAIN_LL, linewidth=1.3,
                label='train LL', zorder=3)

    if val_ll:
        iters_v = np.arange(1, len(val_ll) + 1)
        vll_norm = np.array(val_ll) / n_val
        ax.plot(iters_v, vll_norm, color=COLOR_VAL_LL, linewidth=1.3,
                linestyle='--', label='val LL', zorder=3)

    ax.set_ylabel('Log-lik / sample', fontsize=8)
    _style_axis(ax)

    # ELBO on right axis (also normalized per training sample)
    if train_loss:
        ax_elbo = ax.twinx()
        iters_e = np.arange(1, len(train_loss) + 1)
        elbo_norm = np.array([-l for l in train_loss]) / n_train
        ax_elbo.plot(iters_e, elbo_norm, color=COLOR_ELBO, linewidth=1.0,
                     alpha=0.7, label='ELBO', zorder=2)
        ax_elbo.set_ylabel('ELBO / sample', color=COLOR_ELBO, fontsize=8)
        ax_elbo.tick_params(axis='y', labelcolor=COLOR_ELBO, labelsize=7)
        ax_elbo.spines['top'].set_visible(False)
        ax_elbo.spines['right'].set_linewidth(0.6)

    # Legend combining both axes
    lines1, labels1 = ax.get_legend_handles_labels()
    if train_loss:
        lines2, labels2 = ax_elbo.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2,
                  fontsize=7, loc='lower right', framealpha=0.8)
    elif lines1:
        ax.legend(fontsize=7, loc='lower right', framealpha=0.8)

    _add_best_line(ax, best_iteration)


def _plot_likelihood_params(ax, curves, best_iteration):
    """Row 2: A (left) + lambda0 (right), colored twin axes."""
    A_curve = curves.get('A', [])
    lam0_curve = curves.get('lambda0', [])

    if not A_curve and not lam0_curve:
        ax.text(0.5, 0.5, 'No data', ha='center', va='center',
                transform=ax.transAxes, fontsize=9, color='gray')
        return

    _style_axis(ax)

    if A_curve:
        iters = np.arange(1, len(A_curve) + 1)
        ax.plot(iters, A_curve, color=COLOR_A, linewidth=1.3)
        ax.set_ylabel('A', color=COLOR_A, fontsize=9, fontweight='semibold')
        ax.tick_params(axis='y', labelcolor=COLOR_A, labelsize=7)

    if lam0_curve:
        ax2 = ax.twinx()
        iters = np.arange(1, len(lam0_curve) + 1)
        ax2.plot(iters, lam0_curve, color=COLOR_LAMBDA0, linewidth=1.3)
        ax2.set_ylabel('lambda0', color=COLOR_LAMBDA0, fontsize=9,
                        fontweight='semibold')
        ax2.tick_params(axis='y', labelcolor=COLOR_LAMBDA0, labelsize=7)
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_linewidth(0.6)

    _add_best_line(ax, best_iteration)


def _plot_kernel_params(ax, curves, best_iteration):
    """Row 3: Kernel params on offset colored axes."""
    beta_curve = curves.get('beta', [])
    rho_curve = curves.get('rho', [])
    sigma0_curve = curves.get('sigma_0', [])
    eps0x_curve = curves.get('eps_0x', [])
    eps0y_curve = curves.get('eps_0y', [])

    has_any = any([beta_curve, rho_curve, sigma0_curve, eps0x_curve, eps0y_curve])
    if not has_any:
        ax.text(0.5, 0.5, 'No data', ha='center', va='center',
                transform=ax.transAxes, fontsize=9, color='gray')
        return

    _style_axis(ax)
    spine_offset = 0
    lw = 1.2
    fs_label = 8
    fs_tick = 6.5

    # beta on primary left axis
    if beta_curve:
        iters = np.arange(1, len(beta_curve) + 1)
        ax.plot(iters, beta_curve, color=COLOR_BETA, linewidth=lw)
        ax.set_ylabel('beta', color=COLOR_BETA, fontsize=fs_label,
                       fontweight='semibold')
        ax.tick_params(axis='y', labelcolor=COLOR_BETA, labelsize=fs_tick)

    # rho on first right axis
    if rho_curve:
        ax_rho = ax.twinx()
        iters = np.arange(1, len(rho_curve) + 1)
        ax_rho.plot(iters, rho_curve, color=COLOR_RHO, linewidth=lw)
        ax_rho.set_ylabel('rho', color=COLOR_RHO, fontsize=fs_label,
                           fontweight='semibold')
        ax_rho.tick_params(axis='y', labelcolor=COLOR_RHO, labelsize=fs_tick)
        ax_rho.spines['top'].set_visible(False)
        ax_rho.spines['right'].set_linewidth(0.6)
        spine_offset += 1

    # sigma_0 on offset right axis
    if sigma0_curve:
        ax_sig = ax.twinx()
        ax_sig.spines['right'].set_position(('axes', 1.0 + 0.18 * spine_offset))
        iters = np.arange(1, len(sigma0_curve) + 1)
        ax_sig.plot(iters, sigma0_curve, color=COLOR_SIGMA0, linewidth=lw)
        ax_sig.set_ylabel('sigma_0', color=COLOR_SIGMA0, fontsize=fs_label,
                           fontweight='semibold')
        ax_sig.tick_params(axis='y', labelcolor=COLOR_SIGMA0, labelsize=fs_tick)
        ax_sig.spines['top'].set_visible(False)
        ax_sig.spines['right'].set_linewidth(0.6)
        spine_offset += 1

    # eps_0x / eps_0y sharing one offset axis
    if eps0x_curve or eps0y_curve:
        ax_eps = ax.twinx()
        ax_eps.spines['right'].set_position(('axes', 1.0 + 0.18 * spine_offset))
        if eps0x_curve:
            iters = np.arange(1, len(eps0x_curve) + 1)
            ax_eps.plot(iters, eps0x_curve, color=COLOR_EPS, linewidth=1.0,
                        linestyle='-', label='eps_x')
        if eps0y_curve:
            iters = np.arange(1, len(eps0y_curve) + 1)
            ax_eps.plot(iters, eps0y_curve, color=COLOR_EPS, linewidth=1.0,
                        linestyle='--', label='eps_y')
        ax_eps.set_ylabel('eps_0', color=COLOR_EPS, fontsize=fs_label,
                           fontweight='semibold')
        ax_eps.tick_params(axis='y', labelcolor=COLOR_EPS, labelsize=fs_tick)
        ax_eps.spines['top'].set_visible(False)
        ax_eps.spines['right'].set_linewidth(0.6)
        ax_eps.legend(fontsize=6, loc='center right', framealpha=0.7)

    _add_best_line(ax, best_iteration)


def plot_cell_training(cell_id, seed_data, output_path, title_extra=''):
    """Plot training curves for one cell across multiple seeds.

    Args:
        cell_id: Cell identifier (int).
        seed_data: dict {seed: {'curves': {...}, 'best_iteration': int, ...}}
        output_path: Path to save the figure.
        title_extra: Optional text appended to suptitle.
    """
    seeds = sorted(seed_data.keys())
    n_seeds = len(seeds)
    n_rows = 3

    fig_width = max(5.5 * n_seeds, 6.0) + 1.5
    fig_height = 3.2 * n_rows + 0.6
    fig, axes = plt.subplots(n_rows, n_seeds, figsize=(fig_width, fig_height),
                              squeeze=False)

    for col_idx, seed in enumerate(seeds):
        record = seed_data[seed]
        curves = record.get('curves', {})
        best_it = record.get('best_iteration')
        stopped = record.get('stopped_early', False)
        test_r = record.get('test_r')
        n_train = record.get('n_train')
        n_val = record.get('n_val')
        if n_train is None or n_val is None:
            global _FALLBACK_WARNED
            if not _FALLBACK_WARNED:
                print(f"  Warning: n_train/n_val missing from record "
                      f"(old JSONL?), using PNAS defaults "
                      f"({_FALLBACK_N_TRAIN}/{_FALLBACK_N_VAL})")
                _FALLBACK_WARNED = True
            n_train = n_train or _FALLBACK_N_TRAIN
            n_val = n_val or _FALLBACK_N_VAL

        # Column title
        parts = [f'Seed {seed}']
        if test_r is not None:
            parts.append(f'test_r={test_r:.4f}')
        if stopped and best_it is not None:
            parts.append(f'ES@{best_it}')
        axes[0, col_idx].set_title('  '.join(parts), fontsize=9,
                                    fontweight='medium', pad=8)

        # Row 1: log-lik + ELBO
        _plot_loglik_and_elbo(axes[0, col_idx], curves, best_it, n_train, n_val)

        # Row 2: likelihood params
        _plot_likelihood_params(axes[1, col_idx], curves, best_it)

        # Row 3: kernel params
        _plot_kernel_params(axes[2, col_idx], curves, best_it)

        # x-label only on bottom
        axes[n_rows - 1, col_idx].set_xlabel('Iteration', fontsize=8)

    # Suptitle
    suptitle = f'Cell {cell_id}'
    if title_extra:
        suptitle += f'  |  {title_extra}'
    fig.suptitle(suptitle, fontsize=12, fontweight='bold', y=1.02)

    fig.tight_layout(h_pad=1.0, w_pad=2.5)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close(fig)
    print(f"  Saved: {output_path}")


def plot_experiment_training(records_or_path, output_dir, cell_filter=None,
                              title_extra=''):
    """Plot training curves for all cells in a dataset.

    Args:
        records_or_path: Path to curves JSONL or list of record dicts.
        output_dir: Directory for per-cell PNGs.
        cell_filter: Optional list of cell IDs. None = all.
        title_extra: Optional text for figure titles.
    """
    if isinstance(records_or_path, (str, Path)):
        path = Path(records_or_path)
        print(f"Loading curves from: {path}")
        records = load_curves_jsonl(path)
        print(f"  Loaded {len(records)} records")
    else:
        records = records_or_path

    if not records:
        print("No records with curves data found.")
        return

    grouped = group_by_cell(records)
    cells = sorted(grouped.keys())
    if cell_filter is not None:
        cells = [c for c in cells if c in cell_filter]

    print(f"Plotting {len(cells)} cells...")
    output_dir = Path(output_dir)

    for cell_id in cells:
        seed_data = grouped[cell_id]
        out_path = output_dir / f'cell_{cell_id:02d}.png'
        plot_cell_training(cell_id, seed_data, out_path, title_extra=title_extra)

    print(f"Done. {len(cells)} figures saved to {output_dir}/")


def find_experiment_dir(exp_name):
    """Find experiment directory by name (partial match)."""
    exp_base = Path(__file__).parent / 'experiments'
    for d in sorted(exp_base.iterdir()):
        if d.is_dir() and exp_name in d.name:
            return d
    exp_explore = exp_base / 'exploratory'
    if exp_explore.exists():
        for d in sorted(exp_explore.iterdir()):
            if d.is_dir() and exp_name in d.name:
                return d
    return None


def main():
    parser = argparse.ArgumentParser(
        description='Plot training curves from JSONL results with curve data.')
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--curves', type=str,
                        help='Path to a JSONL file containing curves data')
    source.add_argument('--exp', type=str,
                        help='Experiment name (reads curves.jsonl in experiment folder)')

    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory (default: plots/training/ next to curves)')
    parser.add_argument('--cell', type=int, nargs='+', default=None,
                        help='Plot only these cell IDs (default: all)')
    parser.add_argument('--title', type=str, default='',
                        help='Extra text for figure titles')
    args = parser.parse_args()

    if args.curves:
        curves_path = Path(args.curves)
        if not curves_path.exists():
            print(f"Error: {curves_path} not found")
            return 1
        default_output = curves_path.parent / 'plots' / 'training'
    else:
        exp_dir = find_experiment_dir(args.exp)
        if exp_dir is None:
            print(f"Error: experiment '{args.exp}' not found")
            return 1
        curves_path = exp_dir / 'curves.jsonl'
        if not curves_path.exists():
            print(f"Error: {curves_path} not found (no curves data)")
            return 1
        default_output = exp_dir / 'plots' / 'training'

    output_dir = Path(args.output_dir) if args.output_dir else default_output

    plot_experiment_training(
        curves_path, output_dir,
        cell_filter=args.cell,
        title_extra=args.title,
    )
    return 0


if __name__ == '__main__':
    exit(main() or 0)
