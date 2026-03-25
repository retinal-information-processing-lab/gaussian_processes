#!/usr/bin/env python3
"""
visualize_experiment.py - Summary visualizations for all-cells experiments.

Produces STA galleries, RF overlay grids, and performance comparison plots
from experiment results (results.jsonl) and raw datasets.

Usage:
    # Visualize both massive experiments
    python visualize_experiment.py --exp massive_allcells_108 massive_allcells_64

    # Custom RF overlay combo (default: vargp_direct M=300 seed=1)
    python visualize_experiment.py --exp massive_allcells_108 --mode default_gpy --M 2910

    # Average test_r across seeds instead of using seed=1
    python visualize_experiment.py --exp massive_allcells_108 --mean-seeds
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


N_CELLS = 41
GRID_ROWS = 6
GRID_COLS = 7


def load_results(results_path):
    """Load results.jsonl into a list of dicts."""
    records = []
    with open(results_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def load_dataset(data_path):
    """Load dataset, combine train+val, return (X_flat, R, n_px_side)."""
    data = np.load(data_path)
    X_train = data['images_train']
    X_val = data['images_val']
    R_train = data['responses_train']
    R_val = data['responses_val']

    X = np.concatenate([X_train, X_val], axis=0)
    R = np.concatenate([R_train, R_val], axis=0)

    n_px_side = X.shape[1]
    X_flat = X.reshape(X.shape[0], -1)  # (N, n_pixels)

    return X_flat, R, n_px_side


def compute_sta_2d(X_flat, r, n_px_side):
    """Compute spike-triggered average (z-scored pixels, weighted by response)."""
    X_mean = X_flat.mean(axis=0, keepdims=True)
    X_std = X_flat.std(axis=0, keepdims=True)
    X_norm = (X_flat - X_mean) / (X_std + 1e-8)
    STA = (r[:, None] * X_norm).sum(axis=0) / r.sum()
    return STA.reshape(n_px_side, n_px_side)


def draw_rf_circle(ax, eps_0x, eps_0y, beta, n_px_side, color='lime'):
    """Draw 1-sigma RF circle on an image axis."""
    cx = (eps_0x + 1) / 2 * (n_px_side - 1)
    cy = (eps_0y + 1) / 2 * (n_px_side - 1)
    sigma_rf = beta * np.sqrt(2)
    sigma_px = sigma_rf * (n_px_side - 1) / 2

    circle = plt.Circle((cx, cy), sigma_px, fill=False, color=color,
                         linewidth=1.5, linestyle='-')
    ax.add_patch(circle)
    ax.plot(cx, cy, '+', color=color, markersize=6, markeredgewidth=1.5)


def get_result(records, mode, M, cell, seed=None, mean_seeds=False):
    """Get a result record for a specific (mode, M, cell) combo.

    If mean_seeds=True, average test_r across all seeds.
    Otherwise use the specified seed.
    """
    matches = [r for r in records
                if r['mode'] == mode and r['M'] == M and r['cell'] == cell
                and r.get('status') == 'success']

    if mean_seeds and matches:
        test_rs = [r['test_r'] for r in matches if r.get('test_r') is not None]
        if test_rs:
            avg = dict(matches[0])
            avg['test_r'] = np.mean(test_rs)
            return avg
        return None

    if seed is not None:
        matches = [r for r in matches if r['seed'] == seed]

    return matches[0] if matches else None


def plot_sta_gallery(X_flat, R, n_px_side, output_path, dataset_label):
    """Plot 6x7 grid of STA images for all 41 cells."""
    fig, axes = plt.subplots(GRID_ROWS, GRID_COLS, figsize=(21, 18))
    fig.suptitle(f'STA Gallery - {dataset_label}', fontsize=16, fontweight='bold')

    for cell_id in range(N_CELLS):
        row = cell_id // GRID_COLS
        col = cell_id % GRID_COLS
        ax = axes[row, col]

        r = R[:, cell_id]
        sta = compute_sta_2d(X_flat, r, n_px_side)
        vmax = np.abs(sta).max()

        ax.imshow(sta, cmap='RdBu_r', origin='lower', vmin=-vmax, vmax=vmax)
        mean_rate = r.mean()
        ax.set_title(f'Cell {cell_id} (rate={mean_rate:.1f})', fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])

    # Hide blank cells
    for idx in range(N_CELLS, GRID_ROWS * GRID_COLS):
        row = idx // GRID_COLS
        col = idx % GRID_COLS
        axes[row, col].axis('off')

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {output_path}")


def plot_rf_overlay(X_flat, R, n_px_side, records, output_path,
                    dataset_label, mode, M, seed, mean_seeds):
    """Plot 6x7 grid of STA images with learned RF circle overlay."""
    fig, axes = plt.subplots(GRID_ROWS, GRID_COLS, figsize=(21, 18))
    fig.suptitle(f'STA + Learned RF - {dataset_label}\n'
                 f'mode={mode}, M={M}, {"mean seeds" if mean_seeds else f"seed={seed}"}',
                 fontsize=14, fontweight='bold')

    for cell_id in range(N_CELLS):
        row = cell_id // GRID_COLS
        col = cell_id % GRID_COLS
        ax = axes[row, col]

        r = R[:, cell_id]
        sta = compute_sta_2d(X_flat, r, n_px_side)
        vmax = np.abs(sta).max()

        ax.imshow(sta, cmap='RdBu_r', origin='lower', vmin=-vmax, vmax=vmax)

        result = get_result(records, mode, M, cell_id,
                            seed=seed, mean_seeds=mean_seeds)

        if result is not None:
            test_r = result['test_r']
            draw_rf_circle(ax, result['final_eps_0x'], result['final_eps_0y'],
                           result['final_beta'], n_px_side)

            # Color-code title by performance
            if test_r is not None:
                if test_r > 0.5:
                    color = 'green'
                elif test_r < 0.2:
                    color = 'red'
                else:
                    color = 'orange'
                ax.set_title(f'Cell {cell_id}  r={test_r:.2f}', fontsize=8,
                             color=color, fontweight='bold')
            else:
                ax.set_title(f'Cell {cell_id}  r=NaN', fontsize=8, color='red')
        else:
            ax.set_title(f'Cell {cell_id}  FAILED', fontsize=8, color='red',
                         fontweight='bold')

        ax.set_xticks([])
        ax.set_yticks([])

    for idx in range(N_CELLS, GRID_ROWS * GRID_COLS):
        row = idx // GRID_COLS
        col = idx % GRID_COLS
        axes[row, col].axis('off')

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {output_path}")


def plot_performance(records, output_path, dataset_label, seed, mean_seeds):
    """Plot test_r across all cells for all (mode, M) combos."""
    combos = [
        ('vargp_direct', 300, 'o', 'tab:blue'),
        ('vargp_direct', 2910, 's', 'tab:cyan'),
        ('default_gpy', 300, '^', 'tab:orange'),
        ('default_gpy', 2910, 'D', 'tab:red'),
    ]

    fig, ax = plt.subplots(figsize=(18, 6))
    cells = list(range(N_CELLS))

    for mode, M, marker, color in combos:
        test_rs = []
        failed_cells = []
        for cell_id in cells:
            result = get_result(records, mode, M, cell_id,
                                seed=seed, mean_seeds=mean_seeds)
            if result is not None and result['test_r'] is not None:
                test_rs.append(result['test_r'])
            else:
                test_rs.append(None)
                failed_cells.append(cell_id)

        valid_cells = [c for c, r in zip(cells, test_rs) if r is not None]
        valid_rs = [r for r in test_rs if r is not None]
        avg_r = np.mean(valid_rs) if valid_rs else float('nan')
        ax.plot(valid_cells, valid_rs, marker=marker, color=color,
                label=f'{mode} M={M} (avg={avg_r:.3f})', linewidth=1, markersize=5, alpha=0.8)

        if failed_cells:
            ax.scatter(failed_cells, [0] * len(failed_cells),
                       marker='x', color=color, s=60, linewidth=2, zorder=5)

    ax.axhline(y=0, color='gray', linewidth=0.5, linestyle='--')
    ax.set_xlabel('Cell ID', fontsize=12)
    ax.set_ylabel('Test Pearson r', fontsize=12)
    ax.set_title(f'Performance Comparison - {dataset_label}\n'
                 f'{"mean across seeds" if mean_seeds else f"seed={seed}"}',
                 fontsize=14, fontweight='bold')
    ax.set_xticks(cells)
    ax.set_xticklabels(cells, fontsize=7)
    ax.legend(fontsize=10, loc='lower right')
    ax.set_ylim(-0.5, 1.05)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {output_path}")


def plot_cross_dataset(dataset_entries, output_path,
                       mode, M, seed, mean_seeds):
    """Plot test_r across multiple datasets side by side.

    Args:
        dataset_entries: list of (records, ds_label, marker, color) tuples
    """
    fig, ax = plt.subplots(figsize=(18, 6))
    cells = list(range(N_CELLS))

    for records, ds_label, marker, color in dataset_entries:
        test_rs = []
        failed_cells = []
        for cell_id in cells:
            result = get_result(records, mode, M, cell_id,
                                seed=seed, mean_seeds=mean_seeds)
            if result is not None and result['test_r'] is not None:
                test_rs.append(result['test_r'])
            else:
                test_rs.append(None)
                failed_cells.append(cell_id)

        valid_cells = [c for c, r in zip(cells, test_rs) if r is not None]
        valid_rs = [r for r in test_rs if r is not None]
        avg_r = np.mean(valid_rs) if valid_rs else float('nan')
        ax.plot(valid_cells, valid_rs, marker=marker, color=color,
                label=f'{ds_label} (avg={avg_r:.3f})', linewidth=1.5, markersize=6, alpha=0.8)

        if failed_cells:
            ax.scatter(failed_cells, [0] * len(failed_cells),
                       marker='x', color=color, s=60, linewidth=2, zorder=5)

    ax.axhline(y=0, color='gray', linewidth=0.5, linestyle='--')
    ax.set_xlabel('Cell ID', fontsize=12)
    ax.set_ylabel('Test Pearson r', fontsize=12)
    ax.set_title(f'Cross-Dataset Comparison - {mode} M={M}\n'
                 f'{"mean across seeds" if mean_seeds else f"seed={seed}"}',
                 fontsize=14, fontweight='bold')
    ax.set_xticks(cells)
    ax.set_xticklabels(cells, fontsize=7)
    ax.legend(fontsize=12, loc='lower right')
    ax.set_ylim(-0.5, 1.05)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {output_path}")


def find_experiment(name, experiments_dir):
    """Find experiment folder by name (supports date prefix)."""
    exact = experiments_dir / name
    if exact.exists():
        return exact
    matches = sorted(experiments_dir.glob(f"*_{name}"))
    if len(matches) == 1:
        return matches[0]
    return None


def main():
    parser = argparse.ArgumentParser(
        description='Visualize experiment results: STA galleries, RF overlays, performance plots')
    parser.add_argument('--exp', nargs='+', required=True,
                        help='Experiment name(s) to visualize')
    parser.add_argument('--mode', type=str, default='vargp_direct',
                        help='Mode for RF overlay (default: vargp_direct)')
    parser.add_argument('--M', type=int, default=300,
                        help='M for RF overlay (default: 300)')
    parser.add_argument('--seed', type=int, default=1,
                        help='Seed for single-seed plots (default: 1)')
    parser.add_argument('--mean-seeds', action='store_true',
                        help='Average test_r across seeds instead of using --seed')

    args = parser.parse_args()

    script_dir = Path(__file__).parent
    experiments_dir = script_dir / 'experiments'

    # Dataset paths keyed by experiment name substring
    DATASET_MAP = {
        '108': 'datasets/PNAS_108x108_original.npz',
        '64': 'datasets/PNAS_64x64_center_crop_no_renorm.npz',
        '48': 'datasets/PNAS_48x48_center_crop_no_renorm.npz',
    }

    all_exp_data = {}

    for exp_name in args.exp:
        exp_dir = find_experiment(exp_name, experiments_dir)
        if exp_dir is None:
            print(f"ERROR: Experiment '{exp_name}' not found")
            continue

        results_path = exp_dir / 'results.jsonl'
        if not results_path.exists():
            print(f"ERROR: No results.jsonl in {exp_dir}")
            continue

        # Determine dataset from experiment name
        ds_key = None
        for key in DATASET_MAP:
            if key in exp_name:
                ds_key = key
                break
        if ds_key is None:
            print(f"ERROR: Cannot determine dataset for '{exp_name}' (need '108' or '64' in name)")
            continue

        data_path = script_dir / DATASET_MAP[ds_key]
        ds_label = f'{ds_key}x{ds_key}'

        print(f"\n{'='*60}")
        print(f"Experiment: {exp_dir.name} ({ds_label})")
        print(f"{'='*60}")

        records = load_results(results_path)
        success = [r for r in records if r['status'] == 'success']
        errors = [r for r in records if r['status'] != 'success']
        print(f"  {len(success)} success, {len(errors)} errors")

        print(f"  Loading dataset: {data_path.name}")
        X_flat, R, n_px_side = load_dataset(data_path)

        # STA gallery
        plot_sta_gallery(X_flat, R, n_px_side,
                         exp_dir / 'sta_gallery.png', ds_label)

        # RF overlay
        plot_rf_overlay(X_flat, R, n_px_side, records,
                        exp_dir / 'rf_overlay.png', ds_label,
                        args.mode, args.M, args.seed, args.mean_seeds)

        # Performance comparison
        plot_performance(records, exp_dir / 'performance.png',
                         ds_label, args.seed, args.mean_seeds)

        all_exp_data[ds_key] = {
            'records': records,
            'exp_dir': exp_dir,
            'ds_label': ds_label,
        }

    # Cross-dataset comparison (if 2+ datasets available)
    if len(all_exp_data) >= 2:
        print(f"\nCross-dataset comparison...")
        styles = {
            '108': ('o', 'tab:blue'),
            '64': ('s', 'tab:orange'),
            '48': ('^', 'tab:green'),
        }
        entries = []
        for ds_key in ['108', '64', '48']:
            if ds_key in all_exp_data:
                marker, color = styles[ds_key]
                entries.append((all_exp_data[ds_key]['records'],
                                all_exp_data[ds_key]['ds_label'],
                                marker, color))
        plot_cross_dataset(
            entries,
            experiments_dir / 'performance_cross_dataset.png',
            args.mode, args.M, args.seed, args.mean_seeds,
        )


if __name__ == '__main__':
    main()
