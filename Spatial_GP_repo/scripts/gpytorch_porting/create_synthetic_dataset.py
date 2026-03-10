#!/usr/bin/env python3
"""
Create a synthetic dataset: diffusion-generated images + GP-predicted responses.

Takes synthetic images (from the diffusion image generator package) and runs them
through trained GP models for all cells, producing predicted firing rates.

The output is a single NPZ file with images and predicted responses for all cells.

Usage:
    # Standard usage with 64x64 checkpoints
    python create_synthetic_dataset.py \
        --images synthetic_images.npz \
        --checkpoints checkpoints/64x64/

    # Specific cells only
    python create_synthetic_dataset.py \
        --images synthetic_images.npz \
        --checkpoints checkpoints/64x64/ \
        --cells 1 8 10

    # Custom output path
    python create_synthetic_dataset.py \
        --images synthetic_images.npz \
        --checkpoints checkpoints/64x64/ \
        --output my_synthetic_dataset.npz

Input images:
    NPZ file with key 'images', shape (N, 64, 64).
    - If dtype is uint8: assumed [0, 255], converted to PNAS-normalized range.
    - If dtype is float: assumed PNAS-normalized (roughly [-2.478, 2.478]).

Output NPZ keys:
    images:    (N, 64, 64) float32, PNAS-normalized pixel range
    responses: (N, n_cells) float32, predicted firing rates
    cell_ids:  (n_cells,) int, cell indices matching columns of responses
"""

import sys
import argparse
import time
from pathlib import Path

import numpy as np
import torch

from checkpoint import load_checkpoint
from gpy_training import predict

# Verified: max(abs(all_images.min()), abs(all_images.max())) across all 3190 PNAS images.
# Same constant as in the diffusion image generator package.
PNAS_ABS_MAX = 2.478047


def load_synthetic_images(images_path, dtype=torch.float32):
    """Load synthetic images from NPZ and convert to PNAS-normalized range.

    Auto-detects pixel range from dtype:
    - uint8 [0, 255] -> PNAS-normalized [-2.478, 2.478]
    - float -> assumed already PNAS-normalized (passthrough)

    Args:
        images_path: Path to NPZ with 'images' key
        dtype: Target torch dtype

    Returns:
        torch.Tensor of shape (N, n_pixels), flattened, PNAS-normalized
    """
    data = np.load(images_path)
    images = data['images']  # (N, 64, 64)

    if images.ndim == 4 and images.shape[-1] == 1:
        images = images[:, :, :, 0]  # (N, 64, 64, 1) -> (N, 64, 64)

    n_images, h, w = images.shape
    assert h == w, f"Expected square images, got {h}x{w}"

    if images.dtype == np.uint8:
        # uint8 [0, 255] -> [0, 1] -> [-1, 1] -> PNAS range
        images = images.astype(np.float32)
        images_01 = images / 255.0
        images_model = 2.0 * images_01 - 1.0
        images_pnas = images_model * PNAS_ABS_MAX
        print(f"  Converted uint8 [0,255] -> PNAS range "
              f"[{images_pnas.min():.3f}, {images_pnas.max():.3f}]")
    else:
        images_pnas = images.astype(np.float32)
        print(f"  Float input, assuming PNAS-normalized. "
              f"Range: [{images_pnas.min():.3f}, {images_pnas.max():.3f}]")

    # Flatten to (N, n_pixels)
    X = torch.tensor(images_pnas, dtype=dtype).reshape(n_images, -1)

    return X, images_pnas, h


def main():
    parser = argparse.ArgumentParser(
        description='Create synthetic dataset: diffusion images + GP-predicted responses'
    )
    parser.add_argument('--images', type=str, required=True,
                        help='Path to NPZ with synthetic images (key: "images")')
    parser.add_argument('--checkpoints', type=str, required=True,
                        help='Directory containing cell_XX.pt checkpoint files')
    parser.add_argument('--output', type=str, default='synthetic_dataset.npz',
                        help='Output NPZ path (default: synthetic_dataset.npz)')
    parser.add_argument('--cells', type=int, nargs='+', default=None,
                        help='Specific cell IDs to process (default: all)')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device for inference (default: cuda)')

    args = parser.parse_args()

    # GPU check
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("ERROR: CUDA GPU required but not available.")
        sys.exit(1)

    checkpoints_dir = Path(args.checkpoints)
    output_path = Path(args.output)

    # --- Find checkpoints ---
    available = sorted(checkpoints_dir.glob('cell_*.pt'))
    if not available:
        print(f"ERROR: No checkpoint files found in {checkpoints_dir}")
        sys.exit(1)

    available_cells = {}
    for p in available:
        cell_id = int(p.stem.split('_')[1])
        available_cells[cell_id] = p

    if args.cells is not None:
        missing = [c for c in args.cells if c not in available_cells]
        if missing:
            print(f"WARNING: No checkpoints for cells {missing}")
        cell_ids = sorted(c for c in args.cells if c in available_cells)
    else:
        cell_ids = sorted(available_cells.keys())

    # --- Load images ---
    print(f"Loading images from {args.images}")
    X, images_pnas, n_px_side = load_synthetic_images(args.images)
    n_images = X.shape[0]
    print(f"  {n_images} images, {n_px_side}x{n_px_side} pixels")

    print(f"Checkpoints: {checkpoints_dir}")
    print(f"Output: {output_path}")
    print(f"Device: {args.device}")
    print(f"Cells: {len(cell_ids)} ({cell_ids[0]}..{cell_ids[-1]})")
    print()

    # --- Run predictions ---
    responses = np.zeros((n_images, len(cell_ids)), dtype=np.float32)

    t_total = time.time()
    for i, cell_id in enumerate(cell_ids):
        t0 = time.time()
        checkpoint_path = available_cells[cell_id]

        loaded = load_checkpoint(checkpoint_path, device=args.device)
        model = loaded['model']
        likelihood = loaded['likelihood']
        config = loaded['config']

        # Verify image size matches checkpoint
        ckpt_px = loaded['metadata']['n_px_side']
        if ckpt_px != n_px_side:
            print(f"ERROR: Cell {cell_id} checkpoint expects {ckpt_px}x{ckpt_px} "
                  f"but images are {n_px_side}x{n_px_side}")
            sys.exit(1)

        X_device = X.to(args.device)
        pred = predict(model, likelihood, X_device,
                       device=args.device,
                       jitter=config['jitter'],
                       cholesky_max_tries=config['cholesky_max_tries'],
                       lambda_var_clamp=config['lambda_var_clamp'])

        responses[:, i] = pred['f_pred'].cpu().numpy()

        elapsed = time.time() - t0
        mean_rate = responses[:, i].mean()
        print(f"  [{i+1}/{len(cell_ids)}] Cell {cell_id:2d}: "
              f"mean rate={mean_rate:.2f}, ({elapsed:.1f}s)")

    t_total = time.time() - t_total

    # --- Save ---
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output_path,
             images=images_pnas,           # (N, 64, 64) float32, PNAS-normalized
             responses=responses,           # (N, n_cells) float32, predicted firing rates
             cell_ids=np.array(cell_ids))   # (n_cells,) int

    file_size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"\nDone. {n_images} images x {len(cell_ids)} cells in {t_total:.1f}s")
    print(f"Saved to {output_path} ({file_size_mb:.1f} MB)")
    print(f"  images:    ({n_images}, {n_px_side}, {n_px_side}) float32")
    print(f"  responses: ({n_images}, {len(cell_ids)}) float32")
    print(f"  cell_ids:  {cell_ids}")


if __name__ == '__main__':
    main()
