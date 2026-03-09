#!/usr/bin/env python3
"""
Run inference on PNAS neural data using pre-trained GP checkpoints.

This is the primary entry point for the inference package. It loads pre-trained
checkpoints, runs prediction on the test set, computes metrics, and generates
per-cell summary plots (STA, RF overlay, scatter plot).

Usage:
    # Run inference on 108x108 data
    python run_inference.py --data data/PNAS_108x108_original.npz \\
                            --checkpoints checkpoints/108x108/

    # Run inference on 64x64 data
    python run_inference.py --data data/PNAS_64x64_center_crop_no_renorm.npz \\
                            --checkpoints checkpoints/64x64/

    # Specific cells only
    python run_inference.py --data data/PNAS_108x108_original.npz \\
                            --checkpoints checkpoints/108x108/ --cells 1 8 10

    # Custom output directory
    python run_inference.py --data data/PNAS_108x108_original.npz \\
                            --checkpoints checkpoints/108x108/ --output results/my_run/

    # CPU-only inference (slower)
    python run_inference.py --data data/PNAS_108x108_original.npz \\
                            --checkpoints checkpoints/108x108/ --device cpu
"""

import sys
import json
import csv
import argparse
import time
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for saving figures
import matplotlib.pyplot as plt

from checkpoint import load_checkpoint
from gpy_training import predict
from metrics import compute_pearson_correlation, compute_explained_variance


def load_dataset(data_path, dtype=torch.float32):
    """Load PNAS dataset and return structured dict.

    Args:
        data_path: Path to the .npz file
        dtype: Torch dtype for tensors

    Returns:
        dict with keys: X (train+val flattened), R (train+val responses),
        X_test (flattened), R_test, n_px_side
    """
    data = np.load(data_path)

    X_train = torch.tensor(data['images_train'], dtype=dtype)
    X_val = torch.tensor(data['images_val'], dtype=dtype)
    X_test = torch.tensor(data['images_test'], dtype=dtype)
    R_train = torch.tensor(data['responses_train'], dtype=dtype)
    R_val = torch.tensor(data['responses_val'], dtype=dtype)
    R_test = torch.tensor(data['responses_test'], dtype=dtype)

    # Auto-detect image size
    n_px_side = X_train.shape[1]
    assert X_train.shape[1] == X_train.shape[2], \
        f"Expected square images, got {X_train.shape[1]}x{X_train.shape[2]}"

    # Combine train + val, flatten to (N, n_pixels)
    X = torch.cat([X_train, X_val], dim=0).reshape(-1, n_px_side * n_px_side)
    R = torch.cat([R_train, R_val], dim=0)
    X_test = X_test.reshape(X_test.shape[0], -1)

    return {
        'X': X,
        'R': R,
        'X_test': X_test,
        'R_test': R_test,
        'n_px_side': n_px_side,
    }


def compute_sta_2d(X, r, n_px_side):
    """Compute z-scored spike-triggered average as 2D numpy array.

    Args:
        X: Flattened images, shape (N, n_pixels)
        r: Spike counts, shape (N,)
        n_px_side: Image side length

    Returns:
        STA as (n_px_side, n_px_side) numpy array
    """
    X_mean = X.mean(dim=0, keepdim=True)
    X_std = X.std(dim=0, keepdim=True)
    X_norm = (X - X_mean) / (X_std + 1e-8)
    STA = (r[:, None] * X_norm).sum(dim=0) / r.sum()
    return STA.reshape(n_px_side, n_px_side).cpu().numpy()


def plot_cell_summary(cell_id, STA_2d, hyperparams, r_test_mean, f_pred,
                      test_r, explained_var, reliability, n_px_side, M,
                      output_path):
    """Generate a 3-panel summary figure for one cell.

    Panels:
        Left: STA image with trained RF overlay (center + sigma circles)
        Center: Scatter plot (actual vs predicted firing rate)
        Right: Sorted comparison (actual and predicted, sorted by actual rate)

    Args:
        cell_id: Cell index
        STA_2d: STA image (n_px_side, n_px_side) numpy array
        hyperparams: Dict with A, lambda0, beta, rho, eps_0x, eps_0y, sigma_0, Amp
        r_test_mean: Mean actual firing rates, shape (n_test,)
        f_pred: Predicted firing rates, shape (n_test,)
        test_r: Pearson correlation
        explained_var: Explained variance
        reliability: Cell reliability
        n_px_side: Image side length
        M: Number of inducing points
        output_path: Path to save figure
    """
    r_actual = r_test_mean.cpu().numpy()
    f_predicted = f_pred.cpu().numpy()

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 5))

    # --- Panel 1: STA + RF overlay ---
    vmax = max(abs(STA_2d.min()), abs(STA_2d.max()))
    ax1.imshow(STA_2d, cmap='RdBu_r', origin='lower', vmin=-vmax, vmax=vmax)
    ax1.set_title(f'STA + Trained RF')
    ax1.set_xlabel('x (pixels)')
    ax1.set_ylabel('y (pixels)')

    # Draw RF center and sigma circles
    eps_0x = hyperparams['eps_0x']
    eps_0y = hyperparams['eps_0y']
    beta = hyperparams['beta']

    # Convert normalized coords [-1, 1] to pixel coords
    cx = (eps_0x + 1) / 2 * (n_px_side - 1)
    cy = (eps_0y + 1) / 2 * (n_px_side - 1)
    # sigma from natural beta: sigma_rf = beta * sqrt(2) in normalized coords
    sigma_rf = beta * np.sqrt(2)
    sigma_px = sigma_rf * (n_px_side - 1) / 2

    ax1.plot(cx, cy, 'ko', markersize=5)
    circle_1s = plt.Circle((cx, cy), sigma_px, fill=False,
                            color='black', linewidth=1.5)
    circle_2s = plt.Circle((cx, cy), 2 * sigma_px, fill=False,
                            color='black', linewidth=1, linestyle='--')
    ax1.add_patch(circle_1s)
    ax1.add_patch(circle_2s)
    ax1.set_xlim(0, n_px_side - 1)
    ax1.set_ylim(0, n_px_side - 1)

    param_text = (f'beta={beta:.3f}\n'
                  f'rho={hyperparams["rho"]:.3f}\n'
                  f'eps=({eps_0x:.3f}, {eps_0y:.3f})\n'
                  f'1sig={sigma_px:.1f}px')
    ax1.text(0.02, 0.98, param_text, transform=ax1.transAxes, fontsize=8,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # --- Panel 2: Scatter plot ---
    ax2.scatter(r_actual, f_predicted, c='steelblue', alpha=0.7, edgecolors='k', linewidth=0.5)
    # Identity line
    lo = min(r_actual.min(), f_predicted.min())
    hi = max(r_actual.max(), f_predicted.max())
    margin = (hi - lo) * 0.05
    ax2.plot([lo - margin, hi + margin], [lo - margin, hi + margin], 'k--', alpha=0.5, linewidth=1)
    ax2.set_xlabel('Actual firing rate (mean of 30 reps)')
    ax2.set_ylabel('Predicted firing rate')
    ax2.set_title('Predicted vs Actual')
    ax2.set_aspect('equal')
    ax2.grid(True, alpha=0.3)

    metrics_text = (f'r = {test_r:.3f}\n'
                    f'EV = {explained_var:.3f}\n'
                    f'rel = {reliability:.3f}\n'
                    f'A = {hyperparams["A"]:.4f}\n'
                    f'lam0 = {hyperparams["lambda0"]:.3f}')
    ax2.text(0.02, 0.98, metrics_text, transform=ax2.transAxes, fontsize=8,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # --- Panel 3: Sorted comparison ---
    sort_idx = np.argsort(r_actual)
    r_sorted = r_actual[sort_idx]
    f_sorted = f_predicted[sort_idx]
    x = np.arange(len(r_actual))

    ax3.plot(x, r_sorted, 'k-', linewidth=1.5, label='Actual')
    ax3.plot(x, f_sorted, 'r-', linewidth=1.5, label='Predicted')
    ax3.set_xlabel('Test images (sorted by actual rate)')
    ax3.set_ylabel('Firing rate')
    ax3.set_title('Sorted Comparison')
    ax3.legend(loc='upper left', fontsize=9)
    ax3.grid(True, alpha=0.3)

    fig.suptitle(f'Cell {cell_id} — M={M}, r={test_r:.3f}', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def run_inference(data_path, checkpoints_dir, output_dir, cells, device):
    """Run inference on all specified cells.

    Args:
        data_path: Path to .npz dataset
        checkpoints_dir: Directory containing cell_XX.pt files
        output_dir: Directory for results
        cells: List of cell IDs to process (None = all available checkpoints)
        device: 'cuda' or 'cpu'
    """
    data_path = Path(data_path)
    checkpoints_dir = Path(checkpoints_dir)
    output_dir = Path(output_dir)
    plots_dir = output_dir / 'plots'
    plots_dir.mkdir(parents=True, exist_ok=True)

    print(f"Data:        {data_path}")
    print(f"Checkpoints: {checkpoints_dir}")
    print(f"Output:      {output_dir}")
    print(f"Device:      {device}")
    print()

    # Find available checkpoints
    available = sorted(checkpoints_dir.glob('cell_*.pt'))
    if not available:
        print(f"ERROR: No checkpoint files found in {checkpoints_dir}")
        return

    # Parse cell IDs from filenames
    available_cells = {}
    for p in available:
        cell_id = int(p.stem.split('_')[1])
        available_cells[cell_id] = p

    if cells is not None:
        # Filter to requested cells
        missing = [c for c in cells if c not in available_cells]
        if missing:
            print(f"WARNING: No checkpoints for cells {missing}")
        cell_ids = [c for c in cells if c in available_cells]
    else:
        cell_ids = sorted(available_cells.keys())

    print(f"Running inference on {len(cell_ids)} cells: {cell_ids[0]}..{cell_ids[-1]}")
    print()

    # Load dataset
    print("Loading dataset...", end=" ", flush=True)
    dataset = load_dataset(data_path)
    n_px_side = dataset['n_px_side']
    print(f"done. Images: {n_px_side}x{n_px_side}, "
          f"{dataset['X'].shape[0]} train+val, "
          f"{dataset['X_test'].shape[0]} test")
    print()

    # Results accumulator
    all_results = []

    for i, cell_id in enumerate(cell_ids):
        checkpoint_path = available_cells[cell_id]
        print(f"[{i+1}/{len(cell_ids)}] Cell {cell_id:2d}...", end=" ", flush=True)

        start = time.time()

        # Load checkpoint
        loaded = load_checkpoint(checkpoint_path, device=device)
        model = loaded['model']
        likelihood = loaded['likelihood']
        config = loaded['config']
        hyperparams = loaded['hyperparams']
        metrics = loaded['metrics']
        metadata = loaded['metadata']
        M = metadata['M']

        # Get test data for this cell
        X_test = dataset['X_test'].to(device)
        R_test = dataset['R_test']  # (30 images, 30 repeats, 41 cells) or similar
        r_test = R_test[:, :, cell_id]  # (30 repeats, 30 images) or (30 images, 30 repeats)
        r_test_mean = r_test.mean(dim=0).to(device)

        # Run prediction
        pred = predict(model, likelihood, X_test, device=device,
                       jitter=config['jitter'],
                       cholesky_max_tries=config['cholesky_max_tries'],
                       lambda_var_clamp=config['lambda_var_clamp'])
        f_pred = pred['f_pred']

        # Compute metrics
        test_r = compute_pearson_correlation(r_test_mean, f_pred)
        explained_var, reliability = compute_explained_variance(r_test.to(device), f_pred)

        # Compute STA for this cell (from all available images)
        r_cell = dataset['R'][:, cell_id]
        STA_2d = compute_sta_2d(dataset['X'], r_cell, n_px_side)

        # Generate plot
        plot_path = plots_dir / f'cell_{cell_id:02d}.png'
        plot_cell_summary(
            cell_id, STA_2d, hyperparams,
            r_test_mean, f_pred,
            float(test_r), float(explained_var), float(reliability),
            n_px_side, M, plot_path
        )

        elapsed = time.time() - start

        # Collect results
        row = {
            'cell_id': cell_id,
            'test_r': float(test_r),
            'explained_var': float(explained_var),
            'reliability': float(reliability),
            'train_r': metrics.get('train_r'),
            'train_time_s': metrics.get('train_time'),
            'M': M,
            'n_px_side': n_px_side,
            **{k: v for k, v in hyperparams.items()},
        }
        all_results.append(row)

        print(f"r={float(test_r):.4f}  EV={float(explained_var):.3f}  "
              f"A={hyperparams['A']:.4f}  ({elapsed:.1f}s)")

    # =========================================================================
    # Save summary outputs
    # =========================================================================
    print(f"\n{'='*60}")
    print(f"SUMMARY ({len(all_results)} cells, {n_px_side}x{n_px_side})")
    print(f"{'='*60}")

    # Print table
    print(f"{'Cell':>4s}  {'test_r':>7s}  {'EV':>7s}  {'rel':>5s}  "
          f"{'A':>8s}  {'lam0':>8s}  {'beta':>6s}  {'rho':>6s}")
    print("-" * 65)
    for row in all_results:
        print(f"{row['cell_id']:4d}  {row['test_r']:7.4f}  {row['explained_var']:7.4f}  "
              f"{row['reliability']:5.3f}  {row['A']:8.4f}  {row['lambda0']:8.4f}  "
              f"{row['beta']:6.3f}  {row['rho']:6.3f}")

    # Summary stats
    test_rs = [r['test_r'] for r in all_results]
    print(f"\ntest_r: mean={np.mean(test_rs):.4f}, "
          f"median={np.median(test_rs):.4f}, "
          f"min={np.min(test_rs):.4f}, max={np.max(test_rs):.4f}")

    # Save CSV
    csv_path = output_dir / 'summary.csv'
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=all_results[0].keys())
        writer.writeheader()
        writer.writerows(all_results)
    print(f"\nSummary CSV: {csv_path}")

    # Save hyperparameters JSON
    json_path = output_dir / 'hyperparameters.json'
    with open(json_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"Hyperparameters JSON: {json_path}")

    print(f"Plots: {plots_dir}/cell_XX.png")

    return all_results


def main():
    parser = argparse.ArgumentParser(
        description='Run GP inference on PNAS data using pre-trained checkpoints'
    )
    parser.add_argument('--data', type=str, required=True,
                        help='Path to .npz dataset file')
    parser.add_argument('--checkpoints', type=str, required=True,
                        help='Directory containing cell_XX.pt checkpoint files')
    parser.add_argument('--output', type=str, default=None,
                        help='Output directory (default: results/<dataset_name>/)')
    parser.add_argument('--cells', type=int, nargs='+', default=None,
                        help='Specific cell IDs to process (default: all)')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device for inference (default: cuda). GPU required.')

    args = parser.parse_args()

    # GPU check
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("ERROR: CUDA GPU required but not available. "
              "Install PyTorch with CUDA support.")
        sys.exit(1)

    # Default output directory based on dataset name
    if args.output is None:
        ds_name = Path(args.data).stem  # e.g., PNAS_108x108_original
        args.output = f'results/{ds_name}'

    run_inference(args.data, args.checkpoints, args.output, args.cells, args.device)


if __name__ == '__main__':
    main()
