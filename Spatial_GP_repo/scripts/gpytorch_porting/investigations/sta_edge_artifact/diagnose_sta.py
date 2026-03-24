#!/usr/bin/env python3
"""
Diagnose STA edge artifact for cells that fail on 108x108 but not 64x64.

For each failing cell, produces a 1x4 diagnostic figure:
  A: Full 108x108 STA with smoothed argmax peak + RF bounds rectangle
  B: Center 64x64 crop of 108x108 STA (same colorscale)
  C: Full 64x64 STA with its smoothed argmax peak
  D: Full 108x108 STA with both peaks marked

Also produces a scatter plot of edge/center STA ratio vs test_r for all 41 cells.

Usage:
    python diagnose_sta.py
    python diagnose_sta.py --cells 6 15 22 39
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
from scipy.ndimage import gaussian_filter

# Project paths
SCRIPT_DIR = Path(__file__).parent.parent.parent  # gpytorch_porting/
DATASET_108 = SCRIPT_DIR / 'datasets' / 'PNAS_108x108_original.npz'
DATASET_64 = SCRIPT_DIR / 'datasets' / 'PNAS_64x64_center_crop_no_renorm.npz'
RESULTS_108 = SCRIPT_DIR / 'experiments' / '2026-03-20_massive_allcells_108' / 'results.jsonl'
RESULTS_64 = SCRIPT_DIR / 'experiments' / '2026-03-20_massive_allcells_64' / 'results.jsonl'
OUTPUT_DIR = Path(__file__).parent

CROP_OFFSET = 22  # (108 - 64) // 2
BLUR_SIGMA = 3.0
N_CELLS = 41


def load_dataset(path):
    """Load dataset, combine train+val, return (X_flat, R, n_px_side)."""
    data = np.load(path)
    X = np.concatenate([data['images_train'], data['images_val']], axis=0)
    R = np.concatenate([data['responses_train'], data['responses_val']], axis=0)
    n_px = X.shape[1]
    return X.reshape(X.shape[0], -1), R, n_px


def compute_sta_2d(X_flat, r, n_px_side, zscore=True):
    """Compute STA, optionally z-scored. Returns 2D array."""
    if zscore:
        X_mean = X_flat.mean(axis=0, keepdims=True)
        X_std = X_flat.std(axis=0, keepdims=True)
        X_norm = (X_flat - X_mean) / (X_std + 1e-8)
    else:
        X_norm = X_flat
    STA = (r[:, None] * X_norm).sum(axis=0) / r.sum()
    return STA.reshape(n_px_side, n_px_side)


def find_peak(sta_2d, sigma=BLUR_SIGMA, mask_region=None):
    """Find smoothed argmax of |STA|. Optionally restrict to a region.

    Args:
        sta_2d: 2D STA array
        sigma: Gaussian blur sigma
        mask_region: (row_start, row_end, col_start, col_end) to restrict search

    Returns:
        (row, col) pixel coords of peak, peak value
    """
    smooth = gaussian_filter(np.abs(sta_2d), sigma=sigma)
    if mask_region is not None:
        r0, r1, c0, c1 = mask_region
        masked = np.zeros_like(smooth)
        masked[r0:r1, c0:c1] = smooth[r0:r1, c0:c1]
        smooth = masked
    peak = np.unravel_index(smooth.argmax(), smooth.shape)
    return peak, smooth[peak]


def pixel_to_norm(px, py, n_px_side):
    """Convert pixel coords to normalized [-1, 1]."""
    ex = (px / (n_px_side - 1)) * 2 - 1
    ey = (py / (n_px_side - 1)) * 2 - 1
    return ex, ey


def map_64_to_108(px_64, py_64):
    """Map 64x64 pixel coords to 108x108 pixel coords."""
    return px_64 + CROP_OFFSET, py_64 + CROP_OFFSET


def draw_cross(ax, px, py, color, size=8, linewidth=2):
    """Draw a crosshair marker at pixel (px, py)."""
    ax.plot(px, py, '+', color=color, markersize=size, markeredgewidth=linewidth, zorder=10)


def draw_bounds_rect(ax, center_px, center_py, radius_norm, n_px_side, color='yellow'):
    """Draw the RF bounds rectangle in pixel coords."""
    cx_norm, cy_norm = pixel_to_norm(center_px, center_py, n_px_side)
    x_min = max(-1, cx_norm - radius_norm)
    x_max = min(1, cx_norm + radius_norm)
    y_min = max(-1, cy_norm - radius_norm)
    y_max = min(1, cy_norm + radius_norm)
    # Convert back to pixels
    px_min = (x_min + 1) / 2 * (n_px_side - 1)
    px_max = (x_max + 1) / 2 * (n_px_side - 1)
    py_min = (y_min + 1) / 2 * (n_px_side - 1)
    py_max = (y_max + 1) / 2 * (n_px_side - 1)
    rect = mpatches.Rectangle((px_min, py_min), px_max - px_min, py_max - py_min,
                                linewidth=1.5, edgecolor=color, facecolor='none',
                                linestyle='--', zorder=9)
    ax.add_patch(rect)


def plot_cell_diagnostic(cell_id, sta_108, sta_64, peak_108, peak_64,
                         peak_108_center, output_path):
    """Create 1x4 diagnostic figure for one cell."""
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    fig.suptitle(f'Cell {cell_id} — STA Edge Artifact Diagnostic', fontsize=14, fontweight='bold')

    peak_108_yx, val_108 = peak_108
    peak_64_yx, val_64 = peak_64
    peak_108c_yx, val_108c = peak_108_center

    # Common colorscale for 108x108 panels (A, B, D)
    vmax_108 = np.abs(sta_108).max()
    vmax_64 = np.abs(sta_64).max()

    # Panel A: Full 108x108 STA with peak + bounds
    ax = axes[0]
    ax.imshow(sta_108, cmap='RdBu_r', origin='lower', vmin=-vmax_108, vmax=vmax_108)
    draw_cross(ax, peak_108_yx[1], peak_108_yx[0], 'red', size=12)
    # RF bounds rectangle (radius = 3 * 0.1 * sqrt(2) = 0.424)
    draw_bounds_rect(ax, peak_108_yx[1], peak_108_yx[0], 0.424, 108, color='yellow')
    # Draw center crop region
    rect_crop = mpatches.Rectangle((CROP_OFFSET, CROP_OFFSET), 64, 64,
                                    linewidth=1, edgecolor='white', facecolor='none',
                                    linestyle=':', zorder=8)
    ax.add_patch(rect_crop)
    ax.set_title(f'A: 108x108 STA\npeak=({peak_108_yx[1]},{peak_108_yx[0]}) val={val_108:.4f}',
                 fontsize=9, color='red')
    ax.set_xticks([]); ax.set_yticks([])

    # Panel B: Center 64x64 crop of 108x108 STA
    ax = axes[1]
    sta_108_crop = sta_108[CROP_OFFSET:CROP_OFFSET+64, CROP_OFFSET:CROP_OFFSET+64]
    ax.imshow(sta_108_crop, cmap='RdBu_r', origin='lower', vmin=-vmax_108, vmax=vmax_108)
    # Mark where the center peak falls (relative to crop)
    cx_crop = peak_108c_yx[1] - CROP_OFFSET
    cy_crop = peak_108c_yx[0] - CROP_OFFSET
    if 0 <= cx_crop < 64 and 0 <= cy_crop < 64:
        draw_cross(ax, cx_crop, cy_crop, 'lime', size=10)
    ax.set_title(f'B: 108 center crop (64x64)\ncenter peak val={val_108c:.4f}',
                 fontsize=9, color='green')
    ax.set_xticks([]); ax.set_yticks([])

    # Panel C: Full 64x64 STA with peak
    ax = axes[2]
    ax.imshow(sta_64, cmap='RdBu_r', origin='lower', vmin=-vmax_64, vmax=vmax_64)
    draw_cross(ax, peak_64_yx[1], peak_64_yx[0], 'lime', size=12)
    ax.set_title(f'C: 64x64 STA\npeak=({peak_64_yx[1]},{peak_64_yx[0]}) val={val_64:.4f}',
                 fontsize=9, color='green')
    ax.set_xticks([]); ax.set_yticks([])

    # Panel D: Full 108x108 STA with both peaks
    ax = axes[3]
    ax.imshow(sta_108, cmap='RdBu_r', origin='lower', vmin=-vmax_108, vmax=vmax_108)
    draw_cross(ax, peak_108_yx[1], peak_108_yx[0], 'red', size=12)
    draw_cross(ax, peak_108c_yx[1], peak_108c_yx[0], 'lime', size=12)
    # Draw center crop region
    rect_crop2 = mpatches.Rectangle((CROP_OFFSET, CROP_OFFSET), 64, 64,
                                     linewidth=1, edgecolor='white', facecolor='none',
                                     linestyle=':', zorder=8)
    ax.add_patch(rect_crop2)
    ax.set_title(f'D: Both peaks on 108x108\nred=edge ({val_108:.4f}), green=center ({val_108c:.4f})\n'
                 f'ratio={val_108/val_108c:.2f}',
                 fontsize=9)
    ax.set_xticks([]); ax.set_yticks([])

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {output_path}")


def plot_ratio_vs_testr(cell_data, output_path):
    """Scatter plot: edge/center STA ratio vs test_r for all cells."""
    fig, ax = plt.subplots(figsize=(10, 6))

    cells = sorted(cell_data.keys())
    ratios = [cell_data[c]['ratio'] for c in cells]
    test_rs = [cell_data[c]['test_r'] for c in cells]
    in_centers = [cell_data[c]['in_center'] for c in cells]

    for c, ratio, tr, ic in zip(cells, ratios, test_rs, in_centers):
        color = 'tab:blue' if ic else 'red'
        marker = 'o' if ic else 'X'
        ax.scatter(ratio, tr, c=color, marker=marker, s=60, zorder=5)
        if not ic:
            ax.annotate(f'  cell {c}', (ratio, tr), fontsize=8, color='red')

    ax.axvline(x=1.0, color='gray', linestyle='--', linewidth=0.5)
    ax.axhline(y=0, color='gray', linestyle='--', linewidth=0.5)
    ax.set_xlabel('Edge/Center STA ratio (>1 = edge peak dominates)', fontsize=12)
    ax.set_ylabel('Test Pearson r (108x108, vargp_direct M=300 seed=1)', fontsize=12)
    ax.set_title('STA Peak Location vs Model Performance\n'
                 'Red X = STA peak outside center 64x64 crop region', fontsize=13)

    # Legend
    ax.scatter([], [], c='tab:blue', marker='o', label='Peak in center')
    ax.scatter([], [], c='red', marker='X', label='Peak at edge')
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Diagnose STA edge artifact')
    parser.add_argument('--cells', nargs='+', type=int, default=[6, 15, 22, 39],
                        help='Cell IDs to produce diagnostic figures for (default: 6 15 22 39)')
    args = parser.parse_args()

    print("Loading datasets...")
    X108, R108, npx108 = load_dataset(DATASET_108)
    X64, R64, npx64 = load_dataset(DATASET_64)

    # Load experiment results for test_r
    records_108 = [json.loads(l) for l in open(RESULTS_108) if l.strip()]

    # Center crop region in 108x108 pixel coords
    center_region = (CROP_OFFSET, CROP_OFFSET + 64, CROP_OFFSET, CROP_OFFSET + 64)

    # ---- Analyze all 41 cells ----
    print("\nAnalyzing all 41 cells...")
    print(f"{'Cell':>4} {'rate':>5} {'peak_108':>12} {'in_ctr':>6} {'edge_val':>9} {'ctr_val':>9} "
          f"{'ratio':>6} {'test_r':>7}")
    print("-" * 70)

    cell_data = {}
    for cell in range(N_CELLS):
        r108 = R108[:, cell]
        sta_108 = compute_sta_2d(X108, r108, npx108)

        peak_global, val_global = find_peak(sta_108)
        peak_center, val_center = find_peak(sta_108, mask_region=center_region)

        in_center = (center_region[0] <= peak_global[0] < center_region[1] and
                     center_region[2] <= peak_global[1] < center_region[3])
        ratio = val_global / val_center if val_center > 0 else float('inf')

        # Get test_r
        matches = [rec for rec in records_108
                   if rec['cell'] == cell and rec['mode'] == 'vargp_direct'
                   and rec['M'] == 300 and rec['seed'] == 1
                   and rec.get('status') == 'success']
        test_r = matches[0]['test_r'] if matches else float('nan')

        cell_data[cell] = {
            'ratio': ratio, 'test_r': test_r, 'in_center': in_center,
            'peak_global': peak_global, 'val_global': val_global,
            'peak_center': peak_center, 'val_center': val_center,
            'rate': r108.mean(),
        }

        flag = '  <-- EDGE' if not in_center else ''
        print(f"{cell:4d} {r108.mean():5.2f} ({peak_global[1]:3d},{peak_global[0]:3d}) "
              f"{'yes' if in_center else 'NO':>6} {val_global:9.4f} {val_center:9.4f} "
              f"{ratio:6.2f} {test_r:7.3f}{flag}")

    # ---- Scatter plot: ratio vs test_r ----
    plot_ratio_vs_testr(cell_data, OUTPUT_DIR / 'ratio_vs_testr.png')

    # ---- Per-cell diagnostic figures ----
    for cell in args.cells:
        print(f"\nDiagnostic for cell {cell}...")
        r108 = R108[:, cell]
        r64 = R64[:, cell]

        sta_108 = compute_sta_2d(X108, r108, npx108)
        sta_64 = compute_sta_2d(X64, r64, npx64)

        peak_108, val_108 = find_peak(sta_108)
        peak_64, val_64 = find_peak(sta_64)
        peak_108_center, val_108c = find_peak(sta_108, mask_region=center_region)

        # Print coordinate mapping
        eps_108 = pixel_to_norm(peak_108[1], peak_108[0], 108)
        eps_64 = pixel_to_norm(peak_64[1], peak_64[0], 64)
        px108_from64 = map_64_to_108(peak_64[1], peak_64[0])
        eps_108_from64 = pixel_to_norm(px108_from64[0], px108_from64[1], 108)
        eps_108_center = pixel_to_norm(peak_108_center[1], peak_108_center[0], 108)

        print(f"  108 STA peak: pixel ({peak_108[1]},{peak_108[0]}), "
              f"eps=({eps_108[0]:.4f}, {eps_108[1]:.4f}), val={val_108:.4f}")
        print(f"  64 STA peak:  pixel ({peak_64[1]},{peak_64[0]}), "
              f"eps=({eps_64[0]:.4f}, {eps_64[1]:.4f}), val={val_64:.4f}")
        print(f"  64→108 mapped: pixel ({px108_from64[0]},{px108_from64[1]}), "
              f"eps=({eps_108_from64[0]:.4f}, {eps_108_from64[1]:.4f})")
        print(f"  108 center peak: pixel ({peak_108_center[1]},{peak_108_center[0]}), "
              f"eps=({eps_108_center[0]:.4f}, {eps_108_center[1]:.4f}), val={val_108c:.4f}")
        print(f"  Edge/center ratio: {val_108/val_108c:.3f}")

        plot_cell_diagnostic(cell, sta_108, sta_64,
                             (peak_108, val_108), (peak_64, val_64),
                             (peak_108_center, val_108c),
                             OUTPUT_DIR / f'cell_{cell:02d}_diagnostic.png')


if __name__ == '__main__':
    main()
