#!/usr/bin/env python3
"""
Full 41-cell grid showing 108x108 z-scored STA with crop regions and all
three STA peak locations (108 full, 64-crop, 32-crop).

4 columns, 11 rows (41 cells + 3 blank).

Usage:
    python compare_sta_crops_grid.py
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import torch

SCRIPT_DIR = Path(__file__).parent.parent.parent
sys.path.insert(0, str(SCRIPT_DIR))
_repo_root = next(p for p in Path(__file__).resolve().parents if (p / 'Spatial_GP_repo').is_dir())
sys.path.insert(0, str(_repo_root.parent))

from utils import compute_rf_center_from_sta
from scipy.ndimage import gaussian_filter

OUTPUT_DIR = Path(__file__).parent

N_CELLS = 41
NCOLS = 4
NROWS = (N_CELLS + NCOLS - 1) // NCOLS  # 11

CROP_64_OFFSET = (108 - 64) // 2   # 22
CROP_32_OFFSET = (108 - 32) // 2   # 38

# Cells with edge artifact (for red title highlighting)
EDGE_CELLS = {0, 5, 6, 15, 22, 39}


def load_combined(path):
    data = np.load(SCRIPT_DIR / path)
    X = np.concatenate([data['images_train'], data['images_val']], axis=0)
    R = np.concatenate([data['responses_train'], data['responses_val']], axis=0)
    return X, R


def sta_peak_on_crop(X_images, r, crop_offset, crop_size):
    """Compute STA on a center crop, return peak in full 108x108 pixel coords."""
    X_crop = X_images[:, crop_offset:crop_offset+crop_size,
                       crop_offset:crop_offset+crop_size, :]
    X_flat = torch.tensor(X_crop.reshape(X_crop.shape[0], -1), dtype=torch.float32)
    r_t = torch.tensor(r, dtype=torch.float32)
    eps_x, eps_y = compute_rf_center_from_sta(X_flat, r_t, crop_size, zscore=True)
    px_crop = (eps_x + 1) / 2 * (crop_size - 1)
    py_crop = (eps_y + 1) / 2 * (crop_size - 1)
    return px_crop + crop_offset, py_crop + crop_offset


def compute_zscore_sta_2d(X_flat, r, npx):
    X_mean = X_flat.mean(axis=0, keepdims=True)
    X_std = X_flat.std(axis=0, keepdims=True)
    X_norm = (X_flat - X_mean) / (X_std + 1e-8)
    STA = (r[:, None] * X_norm).sum(axis=0) / r.sum()
    return STA.reshape(npx, npx)


def main():
    print("Loading 108x108 dataset...")
    X108, R108 = load_combined('datasets/PNAS_108x108_original.npz')
    X108_flat = X108.reshape(X108.shape[0], -1).astype(np.float32)

    fig, axes = plt.subplots(NROWS, NCOLS, figsize=(NCOLS * 5, NROWS * 5))
    fig.suptitle('All 41 Cells — 108x108 z-scored STA with crop peak comparison',
                 fontsize=16, fontweight='bold', y=1.0)

    for cell in range(N_CELLS):
        row = cell // NCOLS
        col = cell % NCOLS
        ax = axes[row, col]

        r = R108[:, cell].astype(np.float32)

        # Full 108x108 z-scored STA
        sta_108 = compute_zscore_sta_2d(X108_flat, r, 108)
        vmax = np.abs(sta_108).max()
        ax.imshow(sta_108, cmap='RdBu_r', origin='lower', vmin=-vmax, vmax=vmax)

        # Full 108 STA peak (smoothed argmax)
        smooth = gaussian_filter(np.abs(sta_108), sigma=3.0)
        peak_yx = np.unravel_index(smooth.argmax(), smooth.shape)
        px_108, py_108 = peak_yx[1], peak_yx[0]

        # 64-crop and 32-crop peaks
        px_64, py_64 = sta_peak_on_crop(X108, r, CROP_64_OFFSET, 64)
        px_32, py_32 = sta_peak_on_crop(X108, r, CROP_32_OFFSET, 32)

        # Crop region rectangles
        rect_64 = mpatches.Rectangle(
            (CROP_64_OFFSET, CROP_64_OFFSET), 64, 64,
            linewidth=1.5, edgecolor='white', facecolor='none', linestyle='--', zorder=8)
        ax.add_patch(rect_64)
        rect_32 = mpatches.Rectangle(
            (CROP_32_OFFSET, CROP_32_OFFSET), 32, 32,
            linewidth=1.5, edgecolor='cyan', facecolor='none', linestyle='--', zorder=8)
        ax.add_patch(rect_32)

        # Crosses
        ms, lw = 10, 2
        ax.plot(px_108, py_108, '+', color='red', markersize=ms, markeredgewidth=lw, zorder=10)
        ax.plot(px_64, py_64, '+', color='lime', markersize=ms, markeredgewidth=lw, zorder=10)
        ax.plot(px_32, py_32, 'x', color='yellow', markersize=ms, markeredgewidth=lw, zorder=10)

        # Title
        title_color = 'red' if cell in EDGE_CELLS else 'black'
        ax.set_title(f'Cell {cell}  (rate={r.mean():.2f})', fontsize=9,
                     color=title_color, fontweight='bold' if cell in EDGE_CELLS else 'normal')
        ax.set_xticks([])
        ax.set_yticks([])

        print(f"Cell {cell}: 108=({px_108},{py_108}), 64=({px_64:.0f},{py_64:.0f}), "
              f"32=({px_32:.0f},{py_32:.0f})"
              f"{'  <-- EDGE' if cell in EDGE_CELLS else ''}")

    # Hide blank cells
    for idx in range(N_CELLS, NROWS * NCOLS):
        row = idx // NCOLS
        col = idx % NCOLS
        axes[row, col].axis('off')

    # Shared legend in last blank cell
    ax_legend = axes[NROWS - 1, NCOLS - 1]
    ax_legend.axis('off')
    legend_elements = [
        plt.Line2D([0], [0], marker='+', color='red', linestyle='None',
                   markersize=10, markeredgewidth=2, label='108 full STA peak'),
        plt.Line2D([0], [0], marker='+', color='lime', linestyle='None',
                   markersize=10, markeredgewidth=2, label='64-crop STA peak'),
        plt.Line2D([0], [0], marker='x', color='yellow', linestyle='None',
                   markersize=10, markeredgewidth=2, label='32-crop STA peak'),
        mpatches.Patch(edgecolor='white', facecolor='lightgray', linestyle='--',
                       label='64x64 crop region'),
        mpatches.Patch(edgecolor='cyan', facecolor='lightgray', linestyle='--',
                       label='32x32 crop region'),
    ]
    ax_legend.legend(handles=legend_elements, loc='center', fontsize=12,
                     frameon=True, facecolor='white', edgecolor='black')
    ax_legend.set_title('Red titles = edge artifact cells', fontsize=10, color='red')

    plt.tight_layout()
    out = OUTPUT_DIR / 'crop_comparison_all_cells.png'
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"\nSaved: {out}")


if __name__ == '__main__':
    main()
