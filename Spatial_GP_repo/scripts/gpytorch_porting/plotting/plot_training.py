#!/usr/bin/env python3
"""
plot_training.py - Training curve visualization for variational GP experiments.

Plots per-iteration training diagnostics grouped by cell, with one column per seed.

Layout: 3 rows x N_seeds columns per cell figure.
  Row 1: Log-likelihood (train + val, normalized per sample) with ELBO on right axis
  Row 2: Likelihood params (A + lambda0 on colored twin axes)
  Row 3: Kernel params (beta, rho, sigma_0, eps_0x/eps_0y on offset colored axes)

Best iteration marked with vertical dashed line in all subplots.

NOTE on validation curves: Since April 2026 the project default is
`n_val_split=0` (no validation carving — see configs/canonical.yaml and
the early-stopping section of CLAUDE.md). When that default is used,
the val_log_lik / val_r / val_rho curves are all None and the val
panels in Row 1 will be empty (the train curves still render). To see
val curves, set `n_val_split > 0` in the run that produced the JSONL.

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

# Row 1: NLL / ELBO (left axis) and Pearson r (right axis)
COLOR_LL = '#2176AE'         # steel blue — train & val NLL (solid vs dashed)
COLOR_ELBO = '#7D8491'       # slate gray
COLOR_R = '#2CA02C'          # green — train & val Pearson r (solid vs dashed)
COLOR_SPEARMAN = '#FF7F0E'   # orange — train & val Spearman rho (solid vs dashed)

# Row 2: Likelihood params
COLOR_A = '#2176AE'          # steel blue
COLOR_LAMBDA0 = '#E63946'    # crimson

# Row 3: Kernel params
COLOR_BETA = '#2CA02C'       # green
COLOR_RHO = '#FF7F0E'        # orange
COLOR_SIGMA0 = '#9467BD'     # purple
COLOR_EPS = '#8C564B'        # brown

# Best-iteration line
COLOR_BEST = '#444444'

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


def _style_axis(ax, fontsize=11):
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


def _apply_ylim(ax, ylim_ranges, key):
    """Set y-axis limits from ranges dict if available."""
    if ylim_ranges and key in ylim_ranges:
        ax.set_ylim(ylim_ranges[key]['padded_min'], ylim_ranges[key]['padded_max'])


def _plot_loglik_and_elbo(ax, curves, best_iteration, n_train, n_val,
                          ylim_ranges=None, is_first_col=True, is_last_col=True):
    """Row 1: Negated NLL + ELBO loss (left, decreasing) and Pearson r (right, increasing).

    Left axis (blue/gray, decreasing = better):
      - train NLL = -train_log_lik / n_train
      - val NLL   = -val_log_lik / n_val  (dashed)
      - ELBO loss = train_loss / n_train  (train_loss is already -ELBO)

    Right axis (green, increasing = better):
      - train r   = train_r   (solid)
      - val r     = val_r     (dashed)
    """
    train_ll = curves.get('train_log_lik', [])
    val_ll = curves.get('val_log_lik', [])
    train_loss = curves.get('train_loss', [])
    train_r = curves.get('train_r', [])
    val_r = curves.get('val_r', [])
    train_rho = curves.get('train_rho', [])
    val_rho = curves.get('val_rho', [])

    if not train_ll and not val_ll and not train_loss and not train_r:
        ax.text(0.5, 0.5, 'No data', ha='center', va='center',
                transform=ax.transAxes, fontsize=12, color="gray")
        return

    # Left axis: negated log-likelihood (NLL, per sample) — decreasing = better
    if train_ll:
        iters = np.arange(1, len(train_ll) + 1)
        nll_train = -np.array(train_ll) / n_train
        ax.plot(iters, nll_train, color=COLOR_LL, linewidth=1.5,
                label='train NLL', zorder=3)

    if val_ll:
        vll_arr = np.array([v if v is not None else np.nan for v in val_ll])
        iters_v = np.arange(1, len(vll_arr) + 1)
        nll_val = -vll_arr / n_val
        ax.plot(iters_v, nll_val, color=COLOR_LL, linewidth=1.5,
                linestyle='--', label='val NLL', zorder=3)

    # ELBO loss on left axis (train_loss is already -ELBO)
    if train_loss:
        iters_e = np.arange(1, len(train_loss) + 1)
        elbo_loss = np.array(train_loss) / n_train
        ax.plot(iters_e, elbo_loss, color=COLOR_ELBO, linewidth=1.0,
                alpha=0.7, label='ELBO loss', zorder=2)

    _style_axis(ax)
    if is_first_col:
        ax.set_ylabel('NLL / sample', fontsize=12)
    else:
        ax.set_ylabel('')
        ax.tick_params(axis='y', labelleft=False)

    # Right axis: Correlations — Pearson r (green) and Spearman rho (orange)
    ax_r = None
    has_corr = (train_r and any(v is not None for v in train_r)) or \
               (val_r and any(v is not None for v in val_r)) or \
               (train_rho and any(v is not None for v in train_rho)) or \
               (val_rho and any(v is not None for v in val_rho))
    if has_corr:
        ax_r = ax.twinx()
        # Pearson r (green)
        if train_r:
            tr_arr = np.array([v if v is not None else np.nan for v in train_r])
            iters_tr = np.arange(1, len(tr_arr) + 1)
            ax_r.plot(iters_tr, tr_arr, color=COLOR_R, linewidth=1.5,
                      label='train r', zorder=3)
        if val_r:
            vr_arr = np.array([v if v is not None else np.nan for v in val_r])
            iters_vr = np.arange(1, len(vr_arr) + 1)
            ax_r.plot(iters_vr, vr_arr, color=COLOR_R, linewidth=1.5,
                      linestyle='--', label='val r', zorder=3)
        # Spearman rho (orange)
        if train_rho:
            trho_arr = np.array([v if v is not None else np.nan for v in train_rho])
            iters_trho = np.arange(1, len(trho_arr) + 1)
            ax_r.plot(iters_trho, trho_arr, color=COLOR_SPEARMAN, linewidth=1.5,
                      label='train rho', zorder=3)
        if val_rho:
            vrho_arr = np.array([v if v is not None else np.nan for v in val_rho])
            iters_vrho = np.arange(1, len(vrho_arr) + 1)
            ax_r.plot(iters_vrho, vrho_arr, color=COLOR_SPEARMAN, linewidth=1.5,
                      linestyle='--', label='val rho', zorder=3)
        ax_r.spines['top'].set_visible(False)
        if is_last_col:
            ax_r.set_ylabel('Correlation', fontsize=12)
            ax_r.tick_params(axis='y', labelsize=9)
            ax_r.spines['right'].set_linewidth(0.6)
        else:
            ax_r.set_ylabel('')
            ax_r.tick_params(axis='y', labelright=False)
            ax_r.spines['right'].set_visible(False)

    # Legend only on first column
    if is_first_col:
        lines1, labels1 = ax.get_legend_handles_labels()
        if ax_r is not None:
            lines2, labels2 = ax_r.get_legend_handles_labels()
            ax.legend(lines1 + lines2, labels1 + labels2,
                      fontsize=8, loc='center right', framealpha=0.8)
        elif lines1:
            ax.legend(fontsize=8, loc='center right', framealpha=0.8)

    # Apply fixed y-limits if provided (negated: swap and negate)
    if ylim_ranges:
        ll_keys = [k for k in ('train_log_lik', 'val_log_lik') if k in ylim_ranges]
        if ll_keys:
            # Original ranges are for positive LL; negate and swap for NLL
            pmin = -max(ylim_ranges[k]['padded_max'] for k in ll_keys)
            pmax = -min(ylim_ranges[k]['padded_min'] for k in ll_keys)
            ax.set_ylim(pmin, pmax)
        if train_loss and 'train_loss' in ylim_ranges:
            _apply_ylim(ax, ylim_ranges, 'train_loss')

    _add_best_line(ax, best_iteration)


def _plot_likelihood_params(ax, curves, best_iteration, ylim_ranges=None,
                            is_first_col=True, is_last_col=True):
    """Row 2: A (left) + lambda0 (right), colored twin axes."""
    A_curve = curves.get('A', [])
    lam0_curve = curves.get('lambda0', [])

    if not A_curve and not lam0_curve:
        ax.text(0.5, 0.5, 'No data', ha='center', va='center',
                transform=ax.transAxes, fontsize=12, color="gray")
        return

    _style_axis(ax)

    if A_curve:
        iters = np.arange(1, len(A_curve) + 1)
        ax.plot(iters, A_curve, color=COLOR_A, linewidth=1.5)
        if is_first_col:
            ax.set_ylabel('A', color=COLOR_A, fontsize=13, fontweight='semibold')
            ax.tick_params(axis='y', labelcolor=COLOR_A, labelsize=10)
        else:
            ax.set_ylabel('')
            ax.tick_params(axis='y', labelleft=False)

    if lam0_curve:
        ax2 = ax.twinx()
        iters = np.arange(1, len(lam0_curve) + 1)
        ax2.plot(iters, lam0_curve, color=COLOR_LAMBDA0, linewidth=1.5)
        ax2.spines['top'].set_visible(False)
        if is_last_col:
            ax2.set_ylabel('lambda0', color=COLOR_LAMBDA0, fontsize=13,
                            fontweight='semibold')
            ax2.tick_params(axis='y', labelcolor=COLOR_LAMBDA0, labelsize=9)
            ax2.spines['right'].set_linewidth(0.6)
        else:
            ax2.set_ylabel('')
            ax2.tick_params(axis='y', labelright=False)
            ax2.spines['right'].set_visible(False)

    _apply_ylim(ax, ylim_ranges, 'A')
    if lam0_curve:
        _apply_ylim(ax2, ylim_ranges, 'lambda0')

    _add_best_line(ax, best_iteration)


def _plot_kernel_params(ax, curves, best_iteration, ylim_ranges=None,
                        is_first_col=True, is_last_col=True):
    """Row 3: Kernel params on offset colored axes."""
    beta_curve = curves.get('beta', [])
    rho_curve = curves.get('rho', [])
    sigma0_curve = curves.get('sigma_0', [])
    eps0x_curve = curves.get('eps_0x', [])
    eps0y_curve = curves.get('eps_0y', [])

    has_any = any([beta_curve, rho_curve, sigma0_curve, eps0x_curve, eps0y_curve])
    if not has_any:
        ax.text(0.5, 0.5, 'No data', ha='center', va='center',
                transform=ax.transAxes, fontsize=12, color="gray")
        return

    _style_axis(ax)
    spine_offset = 0
    lw = 1.4

    # beta on primary left axis
    if beta_curve:
        iters = np.arange(1, len(beta_curve) + 1)
        ax.plot(iters, beta_curve, color=COLOR_BETA, linewidth=lw)
        if is_first_col:
            ax.set_ylabel('beta', color=COLOR_BETA, fontsize=13,
                           fontweight='semibold')
            ax.tick_params(axis='y', labelcolor=COLOR_BETA, labelsize=10)
        else:
            ax.set_ylabel('')
            ax.tick_params(axis='y', labelleft=False)

    # rho on first right axis
    if rho_curve:
        ax_rho = ax.twinx()
        iters = np.arange(1, len(rho_curve) + 1)
        ax_rho.plot(iters, rho_curve, color=COLOR_RHO, linewidth=lw)
        ax_rho.spines['top'].set_visible(False)
        if is_last_col:
            ax_rho.set_ylabel('rho', color=COLOR_RHO, fontsize=13,
                               fontweight='semibold')
            ax_rho.tick_params(axis='y', labelcolor=COLOR_RHO, labelsize=9)
            ax_rho.spines['right'].set_linewidth(0.6)
        else:
            ax_rho.set_ylabel('')
            ax_rho.tick_params(axis='y', labelright=False)
            ax_rho.spines['right'].set_visible(False)
        spine_offset += 1

    # sigma_0 on offset right axis
    if sigma0_curve:
        ax_sig = ax.twinx()
        iters = np.arange(1, len(sigma0_curve) + 1)
        ax_sig.plot(iters, sigma0_curve, color=COLOR_SIGMA0, linewidth=lw)
        ax_sig.spines['top'].set_visible(False)
        if is_last_col:
            ax_sig.spines['right'].set_position(('axes', 1.0 + 0.20 * spine_offset))
            ax_sig.set_ylabel('sigma_0', color=COLOR_SIGMA0, fontsize=13,
                               fontweight='semibold')
            ax_sig.tick_params(axis='y', labelcolor=COLOR_SIGMA0, labelsize=9)
            ax_sig.spines['right'].set_linewidth(0.6)
        else:
            ax_sig.set_ylabel('')
            ax_sig.tick_params(axis='y', labelright=False)
            ax_sig.spines['right'].set_visible(False)
        spine_offset += 1

    # eps_0x / eps_0y sharing one offset axis
    if eps0x_curve or eps0y_curve:
        ax_eps = ax.twinx()
        if eps0x_curve:
            iters = np.arange(1, len(eps0x_curve) + 1)
            ax_eps.plot(iters, eps0x_curve, color=COLOR_EPS, linewidth=1.2,
                        linestyle='-', label='eps_x')
        if eps0y_curve:
            iters = np.arange(1, len(eps0y_curve) + 1)
            ax_eps.plot(iters, eps0y_curve, color=COLOR_EPS, linewidth=1.2,
                        linestyle='--', label='eps_y')
        ax_eps.spines['top'].set_visible(False)
        if is_last_col:
            ax_eps.spines['right'].set_position(('axes', 1.0 + 0.20 * spine_offset))
            ax_eps.set_ylabel('eps_0', color=COLOR_EPS, fontsize=13,
                               fontweight='semibold')
            ax_eps.tick_params(axis='y', labelcolor=COLOR_EPS, labelsize=9)
            ax_eps.spines['right'].set_linewidth(0.6)
            ax_eps.legend(fontsize=10, loc='center right', framealpha=0.7)
        else:
            ax_eps.set_ylabel('')
            ax_eps.tick_params(axis='y', labelright=False)
            ax_eps.spines['right'].set_visible(False)

    # Apply fixed y-limits per axis
    _apply_ylim(ax, ylim_ranges, 'beta')
    if rho_curve:
        _apply_ylim(ax_rho, ylim_ranges, 'rho')
    if sigma0_curve:
        _apply_ylim(ax_sig, ylim_ranges, 'sigma_0')
    if eps0x_curve or eps0y_curve:
        # eps_0x and eps_0y share one axis — use union of both ranges
        if ylim_ranges:
            eps_keys = [k for k in ('eps_0x', 'eps_0y') if k in ylim_ranges]
            if eps_keys:
                pmin = min(ylim_ranges[k]['padded_min'] for k in eps_keys)
                pmax = max(ylim_ranges[k]['padded_max'] for k in eps_keys)
                ax_eps.set_ylim(pmin, pmax)

    _add_best_line(ax, best_iteration)


def plot_cell_training(cell_id, seed_data, output_path, title_extra='',
                       ylim_ranges=None):
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

        is_first = (col_idx == 0)
        is_last = (col_idx == n_seeds - 1)

        # Column title
        parts = [f'Seed {seed}']
        if test_r is not None:
            parts.append(f'test_r={test_r:.4f}')
        if stopped and best_it is not None:
            parts.append(f'ES@{best_it}')
        axes[0, col_idx].set_title('  '.join(parts), fontsize=13,
                                    fontweight='medium', pad=10)

        # Row 1: log-lik + ELBO
        _plot_loglik_and_elbo(axes[0, col_idx], curves, best_it, n_train, n_val,
                              ylim_ranges=ylim_ranges,
                              is_first_col=is_first, is_last_col=is_last)

        # Row 2: likelihood params
        _plot_likelihood_params(axes[1, col_idx], curves, best_it,
                                ylim_ranges=ylim_ranges,
                                is_first_col=is_first, is_last_col=is_last)

        # Row 3: kernel params
        _plot_kernel_params(axes[2, col_idx], curves, best_it,
                            ylim_ranges=ylim_ranges,
                            is_first_col=is_first, is_last_col=is_last)

        # x-label only on bottom
        axes[n_rows - 1, col_idx].set_xlabel('Iteration', fontsize=12)

    # Suptitle
    suptitle = f'Cell {cell_id}'
    if title_extra:
        suptitle += f'  |  {title_extra}'
    fig.suptitle(suptitle, fontsize=16, fontweight='bold', y=1.02)

    fig.tight_layout(h_pad=1.2, w_pad=1.5)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close(fig)
    print(f"  Saved: {output_path}")


def plot_experiment_training(records_or_path, output_dir, cell_filter=None,
                              title_extra='', ylim_ranges=None):
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
        plot_cell_training(cell_id, seed_data, out_path, title_extra=title_extra,
                           ylim_ranges=ylim_ranges)

    print(f"Done. {len(cells)} figures saved to {output_dir}/")


def find_experiment_dir(exp_name):
    """Find experiment directory by name (partial match)."""
    exp_base = Path(__file__).parent.parent / 'experiments'
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
    parser.add_argument('--ylim-json', type=str, default=None,
                        help='Path to param ranges JSON (from compute_param_ranges.py)')
    parser.add_argument('--ylim-config', type=str, default='64_intl_fixAmp_n3160',
                        help='Config key in the ylim JSON (default: 64_intl_fixAmp_n3160)')
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

    # Load fixed y-axis ranges if provided
    ylim_ranges = None
    if args.ylim_json:
        ylim_path = Path(args.ylim_json)
        if not ylim_path.exists():
            print(f"Error: ylim JSON not found: {ylim_path}")
            return 1
        with open(ylim_path) as f:
            ylim_data = json.load(f)
        config_key = args.ylim_config
        if config_key not in ylim_data:
            available = [k for k in ylim_data if k != '_metadata']
            print(f"Error: config '{config_key}' not in ylim JSON. "
                  f"Available: {available}")
            return 1
        ylim_ranges = ylim_data[config_key]
        print(f"Using fixed y-limits from: {ylim_path} [{config_key}]")

    plot_experiment_training(
        curves_path, output_dir,
        cell_filter=args.cell,
        title_extra=args.title,
        ylim_ranges=ylim_ranges,
    )
    return 0


if __name__ == '__main__':
    exit(main() or 0)
