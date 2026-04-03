#!/usr/bin/env python3
"""
For each failing cell, show the 108x108 z-scored STA with:
  - 64x64 crop region outlined (white dashed)
  - 32x32 crop region outlined (cyan dashed)
  - Original 108 STA peak (red cross) — the WRONG init
  - 64x64 STA peak mapped to 108 coords (green cross)
  - 32x32 STA peak mapped to 108 coords (yellow cross) — the FIX

One figure per cell, single panel showing all overlays on the full 108x108 STA.

Usage:
    python compare_sta_crops.py
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
CELLS = [0, 5, 6, 15, 22, 39]

CROP_64_OFFSET = (108 - 64) // 2   # 22
CROP_32_OFFSET = (108 - 32) // 2   # 38


def load_combined(path):
    data = np.load(SCRIPT_DIR / path)
    X = np.concatenate([data['images_train'], data['images_val']], axis=0)
    R = np.concatenate([data['responses_train'], data['responses_val']], axis=0)
    return X, R


def sta_peak_on_crop(X_images, r, crop_offset, crop_size, full_npx):
    """Compute STA on a center crop, return peak in full-image pixel coords."""
    X_crop = X_images[:, crop_offset:crop_offset+crop_size,
                       crop_offset:crop_offset+crop_size, :]
    X_flat = torch.tensor(X_crop.reshape(X_crop.shape[0], -1), dtype=torch.float32)
    r_t = torch.tensor(r, dtype=torch.float32)
    eps_x, eps_y = compute_rf_center_from_sta(X_flat, r_t, crop_size, zscore=True)
    # Map to full-image pixel coords
    px_crop = (eps_x + 1) / 2 * (crop_size - 1)
    py_crop = (eps_y + 1) / 2 * (crop_size - 1)
    px_full = px_crop + crop_offset
    py_full = py_crop + crop_offset
    return px_full, py_full


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

    for cell in CELLS:
        r = R108[:, cell].astype(np.float32)

        # Full 108x108 z-scored STA
        sta_108 = compute_zscore_sta_2d(X108_flat, r, 108)

        # Full 108 STA peak (smoothed argmax) — the original (wrong) init
        smooth = gaussian_filter(np.abs(sta_108), sigma=3.0)
        peak_108_yx = np.unravel_index(smooth.argmax(), smooth.shape)
        px_108 = peak_108_yx[1]
        py_108 = peak_108_yx[0]

        # 64x64 crop STA peak mapped to 108 coords
        px_64, py_64 = sta_peak_on_crop(X108, r, CROP_64_OFFSET, 64, 108)

        # 32x32 crop STA peak mapped to 108 coords
        px_32, py_32 = sta_peak_on_crop(X108, r, CROP_32_OFFSET, 32, 108)

        # --- Plot ---
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        vmax = np.abs(sta_108).max()
        ax.imshow(sta_108, cmap='RdBu_r', origin='lower', vmin=-vmax, vmax=vmax)

        # 64x64 crop region
        rect_64 = mpatches.Rectangle(
            (CROP_64_OFFSET, CROP_64_OFFSET), 64, 64,
            linewidth=2, edgecolor='white', facecolor='none', linestyle='--', zorder=8)
        ax.add_patch(rect_64)

        # 32x32 crop region
        rect_32 = mpatches.Rectangle(
            (CROP_32_OFFSET, CROP_32_OFFSET), 32, 32,
            linewidth=2, edgecolor='cyan', facecolor='none', linestyle='--', zorder=8)
        ax.add_patch(rect_32)

        # Crosses
        ms = 14
        lw = 2.5
        ax.plot(px_108, py_108, '+', color='red', markersize=ms, markeredgewidth=lw, zorder=10)
        ax.plot(px_64, py_64, '+', color='lime', markersize=ms, markeredgewidth=lw, zorder=10)
        ax.plot(px_32, py_32, 'x', color='yellow', markersize=ms, markeredgewidth=lw, zorder=10)

        # Legend
        legend_elements = [
            plt.Line2D([0], [0], marker='+', color='red', linestyle='None',
                       markersize=10, markeredgewidth=2, label=f'108 STA peak ({px_108:.0f}, {py_108:.0f})'),
            plt.Line2D([0], [0], marker='+', color='lime', linestyle='None',
                       markersize=10, markeredgewidth=2, label=f'64-crop STA peak ({px_64:.0f}, {py_64:.0f})'),
            plt.Line2D([0], [0], marker='x', color='yellow', linestyle='None',
                       markersize=10, markeredgewidth=2, label=f'32-crop STA peak ({px_32:.0f}, {py_32:.0f})'),
            mpatches.Patch(edgecolor='white', facecolor='none', linestyle='--',
                           label='64x64 crop region'),
            mpatches.Patch(edgecolor='cyan', facecolor='none', linestyle='--',
                           label='32x32 crop region'),
        ]
        ax.legend(handles=legend_elements, loc='upper right', fontsize=9,
                  facecolor='black', edgecolor='white', labelcolor='white')

        ax.set_title(f'Cell {cell} — 108x108 z-scored STA\n'
                     f'rate={r.mean():.2f} spikes/img',
                     fontsize=13, fontweight='bold')
        ax.set_xticks([])
        ax.set_yticks([])

        out = OUTPUT_DIR / f'crop_comparison_cell_{cell:02d}.png'
        plt.tight_layout()
        fig.savefig(out, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"Cell {cell}: 108 peak=({px_108},{py_108}), "
              f"64 peak=({px_64:.0f},{py_64:.0f}), "
              f"32 peak=({px_32:.0f},{py_32:.0f}) -> {out.name}")


if __name__ == '__main__':
    main()
