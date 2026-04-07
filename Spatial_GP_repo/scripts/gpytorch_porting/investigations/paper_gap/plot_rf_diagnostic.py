"""
RF diagnostic plot: STA + mask contour + learned RF for selected cells.

Shows three columns per cell:
  1. STA (z-scored) -- raw spike-triggered average
  2. Init RF: ground-truth center (green cross), init mask contour at 0.001
     threshold (green dashed), 1-sigma circle (green dotted)
  3. Learned RF: learned center (red cross), learned mask contour at 0.001
     (red solid), 1-sigma circle (red dotted), adj_r2 in title

The mask contour shows what the model actually "sees" -- pixels outside
this contour have alpha < 0.001 and are effectively ignored by the kernel.

Usage:
  python investigations/paper_gap/plot_rf_diagnostic.py                    # defaults: cells 0,5,39
  python investigations/paper_gap/plot_rf_diagnostic.py --cells 0 5 8 39   # custom cells
  python investigations/paper_gap/plot_rf_diagnostic.py --config paper+interleave  # different sweep config
  python investigations/paper_gap/plot_rf_diagnostic.py --seed 2           # specific seed

Reads from: investigations/paper_gap/sweep_results.jsonl
Saves to:   investigations/paper_gap/rf_diagnostic.png
"""
import argparse
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

PROJ = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RESULTS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'sweep_results.jsonl')
OUTPUT_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'rf_diagnostic.png')


def load_sweep_results(config_name, seed, cells):
    """Load results for specific config/seed/cells from sweep JSONL."""
    results = {}
    with open(RESULTS_FILE) as f:
        for line in f:
            r = json.loads(line)
            if (r['config_name'] == config_name
                    and r['seed'] == seed
                    and r['cell'] in cells):
                results[r['cell']] = r
    return results


def compute_mask_alpha(xcord, ycord, eps_x, eps_y, beta_nat):
    """Compute locality mask alpha values on the coordinate grid.

    alpha_i = exp(-beta_code * ((x_i - eps_x)^2 + (y_i - eps_y)^2))
    where beta_code = 1 / (4 * beta_nat^2)
    """
    beta_code = 1.0 / (4 * beta_nat**2)
    return np.exp(-beta_code * ((xcord - eps_x)**2 + (ycord - eps_y)**2))


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--cells', type=int, nargs='+', default=[0, 5, 39],
                        help='Cell IDs to plot (default: 0 5 39)')
    parser.add_argument('--config', type=str, default='broad+interleave',
                        help='Sweep config name (default: broad+interleave)')
    parser.add_argument('--seed', type=int, default=1,
                        help='Seed to show (default: 1)')
    parser.add_argument('--init-beta', type=float, default=0.1,
                        help='Initial beta_nat for comparison (default: 0.1)')
    parser.add_argument('--output', type=str, default=OUTPUT_FILE,
                        help='Output PNG path')
    args = parser.parse_args()

    cells = args.cells
    n_cells = len(cells)

    # Load data
    data = np.load(os.path.join(PROJ, 'datasets', 'PNAS_108x108_original.npz'))
    X_all = np.concatenate([data['images_train'], data['images_val']])
    R_all = np.concatenate([data['responses_train'], data['responses_val']])
    rf = np.load(os.path.join(PROJ, 'datasets', 'rf_centers_ground_truth.npz'))
    n_px = 108

    # Coordinate grid (matches kernel's [-1, 1] range)
    ycord, xcord = np.meshgrid(np.linspace(-1, 1, n_px),
                               np.linspace(-1, 1, n_px), indexing='ij')

    # Load sweep results
    results = load_sweep_results(args.config, args.seed, cells)

    fig, axes = plt.subplots(n_cells, 3, figsize=(15, 4.3 * n_cells))
    if n_cells == 1:
        axes = axes[np.newaxis, :]

    for row, cell_id in enumerate(cells):
        r_cell = R_all[:, cell_id]

        # STA: spike-weighted average image, z-scored
        sta = (X_all.reshape(-1, n_px * n_px).T @ r_cell) / r_cell.sum()
        sta_2d = sta.reshape(n_px, n_px)
        sta_z = (sta_2d - sta_2d.mean()) / (sta_2d.std() + 1e-10)

        # Ground-truth RF center (normalized -> pixel)
        gt_x, gt_y = rf['norm_108'][cell_id]
        gt_px_x = (gt_x + 1) / 2 * (n_px - 1)
        gt_px_y = (gt_y + 1) / 2 * (n_px - 1)

        # Init mask and sigma
        init_beta = args.init_beta
        init_sigma_px = init_beta * np.sqrt(2) * (n_px - 1) / 2
        init_alpha = compute_mask_alpha(xcord, ycord, gt_x, gt_y, init_beta)

        # --- Column 1: STA ---
        ax = axes[row, 0]
        im = ax.imshow(sta_z, cmap='RdBu_r', vmin=-3, vmax=3)
        ax.set_title(f'Cell {cell_id}: STA (z-scored)', fontsize=11)
        ax.set_ylabel(f'Cell {cell_id}', fontsize=12, fontweight='bold')
        plt.colorbar(im, ax=ax, shrink=0.8)

        # --- Column 2: Init RF ---
        ax = axes[row, 1]
        ax.imshow(sta_z, cmap='RdBu_r', vmin=-3, vmax=3)
        ax.plot(gt_px_x, gt_px_y, 'g+', markersize=15, markeredgewidth=3,
                label='GT center')
        ax.contour(init_alpha, levels=[0.001], colors='green',
                   linewidths=2, linestyles='--')
        circle_init = plt.Circle((gt_px_x, gt_px_y), init_sigma_px,
                                 fill=False, color='green', linestyle=':',
                                 linewidth=1.5)
        ax.add_patch(circle_init)
        ax.set_title(f'Init: beta={init_beta}, sigma={init_sigma_px:.1f}px\n'
                     f'mask contour (green dashed) = 0.001', fontsize=9)
        ax.legend(fontsize=7, loc='upper left')

        # --- Column 3: Learned RF ---
        ax = axes[row, 2]
        ax.imshow(sta_z, cmap='RdBu_r', vmin=-3, vmax=3)

        res = results.get(cell_id)
        if res:
            l_epsx = res['final_eps_0x']
            l_epsy = res['final_eps_0y']
            l_beta = res['final_beta']
            adj_r2 = res['adjusted_r2']
            l_px_x = (l_epsx + 1) / 2 * (n_px - 1)
            l_px_y = (l_epsy + 1) / 2 * (n_px - 1)
            l_sigma_px = l_beta * np.sqrt(2) * (n_px - 1) / 2

            l_alpha = compute_mask_alpha(xcord, ycord, l_epsx, l_epsy, l_beta)

            ax.plot(l_px_x, l_px_y, 'r+', markersize=15, markeredgewidth=3,
                    label='Learned center')
            ax.plot(gt_px_x, gt_px_y, 'g+', markersize=10,
                    markeredgewidth=2, alpha=0.4)
            ax.contour(l_alpha, levels=[0.001], colors='red', linewidths=2)
            circle_l = plt.Circle((l_px_x, l_px_y), l_sigma_px,
                                  fill=False, color='red', linestyle=':',
                                  linewidth=1.5)
            ax.add_patch(circle_l)
            title_color = 'red' if adj_r2 < 0.4 else 'black'
            ax.set_title(f'Learned: beta={l_beta:.4f}, sigma={l_sigma_px:.1f}px\n'
                         f'adj_r2={adj_r2:.3f}, mask contour (red) = 0.001',
                         fontsize=9, color=title_color)
            ax.legend(fontsize=7, loc='upper left')
        else:
            ax.set_title(f'No result for cell {cell_id}\n'
                         f'config={args.config}, seed={args.seed}', fontsize=9)

    plt.suptitle(f'RF diagnostic: {args.config}, seed={args.seed}\n'
                 f'mask contour = 0.001 threshold, dotted circle = 1-sigma',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(args.output, dpi=150, bbox_inches='tight')
    print(f'Saved: {args.output}')


if __name__ == '__main__':
    main()
