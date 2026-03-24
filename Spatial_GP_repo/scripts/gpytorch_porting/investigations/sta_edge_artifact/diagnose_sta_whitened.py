#!/usr/bin/env python3
"""
Diagnose STA edge artifact using WHITENED STA instead of z-scored.

Same diagnostic layout as diagnose_sta.py but computes whitened STAs:
  STA_white = C^{-1} @ STA_raw
where C is the pixel-pixel covariance matrix.

For each failing cell, produces a 1x4 diagnostic figure:
  A: Full 108x108 whitened STA with smoothed argmax peak + RF bounds rectangle
  B: Center 64x64 crop of 108x108 whitened STA (same colorscale)
  C: Full 64x64 whitened STA with its smoothed argmax peak
  D: Full 108x108 whitened STA with both peaks marked

Also produces a scatter plot of edge/center STA ratio vs test_r for all 41 cells.

Usage:
    python diagnose_sta_whitened.py
    python diagnose_sta_whitened.py --cells 6 15 22 39
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
OUTPUT_DIR = Path(__file__).parent

CROP_OFFSET = 22  # (108 - 64) // 2
BLUR_SIGMA = 3.0
N_CELLS = 41

# Regularization for covariance matrix inversion
COV_REG = 1e-3


def load_dataset(path):
    """Load dataset, combine train+val, return (X_flat, R, n_px_side)."""
    data = np.load(path)
    X = np.concatenate([data['images_train'], data['images_val']], axis=0)
    R = np.concatenate([data['responses_train'], data['responses_val']], axis=0)
    n_px = X.shape[1]
    return X.reshape(X.shape[0], -1).astype(np.float64), R, n_px


def compute_whitened_sta(X_flat, r, n_px_side, C_inv):
    """Compute whitened STA: C^{-1} @ STA_raw.

    The raw STA is the spike-weighted mean of mean-subtracted stimuli.
    Whitening by C^{-1} removes stimulus correlations, giving the
    maximum-likelihood linear RF estimate.

    Args:
        X_flat: (N, n_pixels) stimulus matrix
        r: (N,) spike counts
        n_px_side: image side length
        C_inv: (n_pixels, n_pixels) inverse covariance matrix (precomputed)

    Returns:
        2D whitened STA array (n_px_side, n_px_side)
    """
    X_mean = X_flat.mean(axis=0, keepdims=True)
    X_centered = X_flat - X_mean
    STA_raw = (r[:, None] * X_centered).sum(axis=0) / r.sum()  # (n_pixels,)
    STA_white = C_inv @ STA_raw  # (n_pixels,)
    return STA_white.reshape(n_px_side, n_px_side)


def compute_cov_inv(X_flat, reg=COV_REG):
    """Compute regularized inverse covariance matrix.

    C = X_centered^T @ X_centered / N + reg * I
    Returns C^{-1}.
    """
    X_mean = X_flat.mean(axis=0, keepdims=True)
    X_centered = X_flat - X_mean
    N = X_centered.shape[0]
    C = (X_centered.T @ X_centered) / N
    C += reg * np.eye(C.shape[0])
    print(f"  Computing C^{{-1}} ({C.shape[0]}x{C.shape[0]}, reg={reg})...", end=" ", flush=True)
    C_inv = np.linalg.inv(C)
    print("done.")
    return C_inv


def find_peak(sta_2d, sigma=BLUR_SIGMA, mask_region=None):
    """Find smoothed argmax of |STA|. Optionally restrict to a region."""
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
    ax.plot(px, py, '+', color=color, markersize=size, markeredgewidth=linewidth, zorder=10)


def draw_bounds_rect(ax, center_px, center_py, radius_norm, n_px_side, color='yellow'):
    cx_norm, cy_norm = pixel_to_norm(center_px, center_py, n_px_side)
    x_min = max(-1, cx_norm - radius_norm)
    x_max = min(1, cx_norm + radius_norm)
    y_min = max(-1, cy_norm - radius_norm)
    y_max = min(1, cy_norm + radius_norm)
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
    fig.suptitle(f'Cell {cell_id} — WHITENED STA Diagnostic', fontsize=14, fontweight='bold')

    peak_108_yx, val_108 = peak_108
    peak_64_yx, val_64 = peak_64
    peak_108c_yx, val_108c = peak_108_center

    vmax_108 = np.abs(sta_108).max()
    vmax_64 = np.abs(sta_64).max()

    # Panel A: Full 108x108 whitened STA
    ax = axes[0]
    ax.imshow(sta_108, cmap='RdBu_r', origin='lower', vmin=-vmax_108, vmax=vmax_108)
    draw_cross(ax, peak_108_yx[1], peak_108_yx[0], 'red', size=12)
    draw_bounds_rect(ax, peak_108_yx[1], peak_108_yx[0], 0.424, 108, color='yellow')
    rect_crop = mpatches.Rectangle((CROP_OFFSET, CROP_OFFSET), 64, 64,
                                    linewidth=1, edgecolor='white', facecolor='none',
                                    linestyle=':', zorder=8)
    ax.add_patch(rect_crop)
    ax.set_title(f'A: 108x108 whitened STA\npeak=({peak_108_yx[1]},{peak_108_yx[0]}) val={val_108:.4f}',
                 fontsize=9, color='red')
    ax.set_xticks([]); ax.set_yticks([])

    # Panel B: Center 64x64 crop of 108x108 whitened STA
    ax = axes[1]
    sta_108_crop = sta_108[CROP_OFFSET:CROP_OFFSET+64, CROP_OFFSET:CROP_OFFSET+64]
    ax.imshow(sta_108_crop, cmap='RdBu_r', origin='lower', vmin=-vmax_108, vmax=vmax_108)
    cx_crop = peak_108c_yx[1] - CROP_OFFSET
    cy_crop = peak_108c_yx[0] - CROP_OFFSET
    if 0 <= cx_crop < 64 and 0 <= cy_crop < 64:
        draw_cross(ax, cx_crop, cy_crop, 'lime', size=10)
    ax.set_title(f'B: 108 center crop (64x64)\ncenter peak val={val_108c:.4f}',
                 fontsize=9, color='green')
    ax.set_xticks([]); ax.set_yticks([])

    # Panel C: Full 64x64 whitened STA
    ax = axes[2]
    ax.imshow(sta_64, cmap='RdBu_r', origin='lower', vmin=-vmax_64, vmax=vmax_64)
    draw_cross(ax, peak_64_yx[1], peak_64_yx[0], 'lime', size=12)
    ax.set_title(f'C: 64x64 whitened STA\npeak=({peak_64_yx[1]},{peak_64_yx[0]}) val={val_64:.4f}',
                 fontsize=9, color='green')
    ax.set_xticks([]); ax.set_yticks([])

    # Panel D: Full 108x108 whitened STA with both peaks
    ax = axes[3]
    ax.imshow(sta_108, cmap='RdBu_r', origin='lower', vmin=-vmax_108, vmax=vmax_108)
    draw_cross(ax, peak_108_yx[1], peak_108_yx[0], 'red', size=12)
    draw_cross(ax, peak_108c_yx[1], peak_108c_yx[0], 'lime', size=12)
    rect_crop2 = mpatches.Rectangle((CROP_OFFSET, CROP_OFFSET), 64, 64,
                                     linewidth=1, edgecolor='white', facecolor='none',
                                     linestyle=':', zorder=8)
    ax.add_patch(rect_crop2)
    ax.set_title(f'D: Both peaks on 108x108\nred=global ({val_108:.4f}), green=center ({val_108c:.4f})\n'
                 f'ratio={val_108/val_108c:.2f}' if val_108c > 0 else 'D: Both peaks',
                 fontsize=9)
    ax.set_xticks([]); ax.set_yticks([])

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {output_path}")


def plot_ratio_vs_testr(cell_data, output_path):
    """Scatter plot: edge/center whitened STA ratio vs test_r."""
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
    ax.set_xlabel('Edge/Center whitened STA ratio (>1 = edge peak dominates)', fontsize=12)
    ax.set_ylabel('Test Pearson r (108x108, vargp_direct M=300 seed=1)', fontsize=12)
    ax.set_title('WHITENED STA Peak Location vs Model Performance\n'
                 'Red X = STA peak outside center 64x64 crop region', fontsize=13)

    ax.scatter([], [], c='tab:blue', marker='o', label='Peak in center')
    ax.scatter([], [], c='red', marker='X', label='Peak at edge')
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Diagnose STA edge artifact with whitened STA')
    parser.add_argument('--cells', nargs='+', type=int, default=[6, 15, 22, 39],
                        help='Cell IDs to produce diagnostic figures for (default: 6 15 22 39)')
    parser.add_argument('--reg', type=float, default=COV_REG,
                        help=f'Covariance regularization (default: {COV_REG})')
    args = parser.parse_args()

    print("Loading datasets...")
    X108, R108, npx108 = load_dataset(DATASET_108)
    X64, R64, npx64 = load_dataset(DATASET_64)

    # Load experiment results for test_r
    records_108 = [json.loads(l) for l in open(RESULTS_108) if l.strip()]

    # Precompute inverse covariance matrices (expensive but done once)
    print("\n108x108 covariance:")
    C_inv_108 = compute_cov_inv(X108, reg=args.reg)
    print("64x64 covariance:")
    C_inv_64 = compute_cov_inv(X64, reg=args.reg)

    center_region = (CROP_OFFSET, CROP_OFFSET + 64, CROP_OFFSET, CROP_OFFSET + 64)

    # ---- Analyze all 41 cells ----
    print("\nAnalyzing all 41 cells with whitened STA...")
    print(f"{'Cell':>4} {'rate':>5} {'peak_108':>12} {'in_ctr':>6} {'edge_val':>9} {'ctr_val':>9} "
          f"{'ratio':>6} {'test_r':>7}")
    print("-" * 70)

    cell_data = {}
    for cell in range(N_CELLS):
        r108 = R108[:, cell]
        sta_108 = compute_whitened_sta(X108, r108, npx108, C_inv_108)

        peak_global, val_global = find_peak(sta_108)
        peak_center, val_center = find_peak(sta_108, mask_region=center_region)

        in_center = (center_region[0] <= peak_global[0] < center_region[1] and
                     center_region[2] <= peak_global[1] < center_region[3])
        ratio = val_global / val_center if val_center > 0 else float('inf')

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

    # ---- Scatter plot ----
    plot_ratio_vs_testr(cell_data, OUTPUT_DIR / 'whitened_ratio_vs_testr.png')

    # ---- Per-cell diagnostic figures ----
    for cell in args.cells:
        print(f"\nWhitened diagnostic for cell {cell}...")
        r108 = R108[:, cell]
        r64 = R64[:, cell]

        sta_108 = compute_whitened_sta(X108, r108, npx108, C_inv_108)
        sta_64 = compute_whitened_sta(X64, r64, npx64, C_inv_64)

        peak_108, val_108 = find_peak(sta_108)
        peak_64, val_64 = find_peak(sta_64)
        peak_108_center, val_108c = find_peak(sta_108, mask_region=center_region)

        eps_108 = pixel_to_norm(peak_108[1], peak_108[0], 108)
        eps_64 = pixel_to_norm(peak_64[1], peak_64[0], 64)
        eps_108_center = pixel_to_norm(peak_108_center[1], peak_108_center[0], 108)

        print(f"  108 whitened peak: pixel ({peak_108[1]},{peak_108[0]}), "
              f"eps=({eps_108[0]:.4f}, {eps_108[1]:.4f}), val={val_108:.4f}")
        print(f"  64 whitened peak:  pixel ({peak_64[1]},{peak_64[0]}), "
              f"eps=({eps_64[0]:.4f}, {eps_64[1]:.4f}), val={val_64:.4f}")
        print(f"  108 center peak:   pixel ({peak_108_center[1]},{peak_108_center[0]}), "
              f"eps=({eps_108_center[0]:.4f}, {eps_108_center[1]:.4f}), val={val_108c:.4f}")
        print(f"  Edge/center ratio: {val_108/val_108c:.3f}" if val_108c > 0 else "  Edge/center ratio: inf")

        plot_cell_diagnostic(cell, sta_108, sta_64,
                             (peak_108, val_108), (peak_64, val_64),
                             (peak_108_center, val_108c),
                             OUTPUT_DIR / f'whitened_cell_{cell:02d}_diagnostic.png')


if __name__ == '__main__':
    main()
