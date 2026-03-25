#!/usr/bin/env python3
"""Explore the LSTA reference data: plot LSTAs, ellipses, and reference images."""

import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SCRIPT_DIR = Path(__file__).parent.parent.parent
OUTPUT_DIR = Path(__file__).parent

# Load LSTA reference
d = np.load(SCRIPT_DIR / 'datasets' / 'samuele_data' / 'lsta_ref.npz')
lsta = d['lsta']           # (41, 8, 72, 72)
ellipses = d['ellipses']   # (41, 2, 360)
image_indices = d['image_indices']  # (8,)
cell_indices = d['cell_indices']    # (41,)

# Load the PNAS images to show the reference images
data108 = np.load(SCRIPT_DIR / 'datasets' / 'PNAS_108x108_original.npz')
X_train = data108['images_train']  # (2910, 108, 108, 1)

print(f"LSTA: {lsta.shape}, ellipses: {ellipses.shape}")
print(f"image_indices: {image_indices}")
print(f"cell_indices: {cell_indices}")

# ============================================================
# Figure 1: The 8 reference images (from training set)
# ============================================================
fig, axes = plt.subplots(2, 4, figsize=(16, 8))
fig.suptitle(f'The 8 LSTA reference images (from PNAS training set)\nimage_indices = {image_indices}',
             fontsize=14, fontweight='bold')

for i, idx in enumerate(image_indices):
    ax = axes[i // 4, i % 4]
    img = X_train[idx, :, :, 0]
    ax.imshow(img, cmap='gray', origin='lower')
    ax.set_title(f'Frame {i} (train[{idx}])', fontsize=10)
    ax.set_xticks([]); ax.set_yticks([])

plt.tight_layout()
fig.savefig(OUTPUT_DIR / 'lsta_reference_images.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved: lsta_reference_images.png")

# ============================================================
# Figure 2: All 41 cells — best LSTA frame + ellipse overlay
# ============================================================
NCOLS = 4
NROWS = (41 + NCOLS - 1) // NCOLS

fig, axes = plt.subplots(NROWS, NCOLS, figsize=(NCOLS * 5, NROWS * 5))
fig.suptitle('All 41 Cells — Best LSTA frame + RF ellipse\n(72x72 LSTA grid)',
             fontsize=14, fontweight='bold', y=1.0)

for cell in range(41):
    row = cell // NCOLS
    col = cell % NCOLS
    ax = axes[row, col]

    # Pick the frame with strongest signal
    strengths = [np.abs(lsta[cell, f]).max() for f in range(8)]
    best_frame = np.argmax(strengths)

    sta_img = lsta[cell, best_frame]
    vmax = np.abs(sta_img).max()

    ax.imshow(sta_img, cmap='RdBu_r', origin='lower',
              vmin=-vmax if vmax > 0 else -1, vmax=vmax if vmax > 0 else 1)

    # Overlay ellipse contour
    ex, ey = ellipses[cell, 0], ellipses[cell, 1]
    ax.plot(ex, ey, '-', color='lime', linewidth=1.5, zorder=10)

    # Mark center
    cx, cy = ex.mean(), ey.mean()
    ax.plot(cx, cy, '+', color='lime', markersize=10, markeredgewidth=2, zorder=11)

    ax.set_title(f'Cell {cell} (orig={cell_indices[cell]}, frame={best_frame})',
                 fontsize=9)
    ax.set_xlim(0, 71); ax.set_ylim(0, 71)
    ax.set_xticks([]); ax.set_yticks([])

for idx in range(41, NROWS * NCOLS):
    axes[idx // NCOLS, idx % NCOLS].axis('off')

plt.tight_layout()
fig.savefig(OUTPUT_DIR / 'lsta_all_cells_best_frame.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved: lsta_all_cells_best_frame.png")

# ============================================================
# Figure 3: Cell 6 — all 8 LSTA frames
# ============================================================
fig, axes = plt.subplots(2, 4, figsize=(16, 8))
fig.suptitle(f'Cell 6 — All 8 LSTA frames (72x72)\nellipse center = ({ellipses[6,0].mean():.1f}, {ellipses[6,1].mean():.1f})',
             fontsize=14, fontweight='bold')

for f in range(8):
    ax = axes[f // 4, f % 4]
    sta_img = lsta[6, f]
    vmax = np.abs(sta_img).max()
    ax.imshow(sta_img, cmap='RdBu_r', origin='lower',
              vmin=-vmax if vmax > 0 else -1, vmax=vmax if vmax > 0 else 1)
    ex, ey = ellipses[6, 0], ellipses[6, 1]
    ax.plot(ex, ey, '-', color='lime', linewidth=1.5)
    ax.plot(ex.mean(), ey.mean(), '+', color='lime', markersize=10, markeredgewidth=2)
    ax.set_title(f'Frame {f} (img[{image_indices[f]}], max={np.abs(sta_img).max():.3f})',
                 fontsize=9)
    ax.set_xticks([]); ax.set_yticks([])

plt.tight_layout()
fig.savefig(OUTPUT_DIR / 'lsta_cell6_all_frames.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved: lsta_cell6_all_frames.png")
