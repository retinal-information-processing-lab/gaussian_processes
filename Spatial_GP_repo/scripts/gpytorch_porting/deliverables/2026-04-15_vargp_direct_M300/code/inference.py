"""
Self-contained inference for vargp_direct M=300 checkpoints.

Loads each cell's eigenspace checkpoint, predicts firing rates on the held-out
test set, computes test_r / explained_var / adjusted_r2 / reliability, writes
a per-cell summary CSV, and produces a per-cell diagnostic plot.

Designed to run from the deliverable folder root:

    python code/inference.py --data data/PNAS_64x64_center_crop_no_renorm.npz \\
                             --checkpoints checkpoints \\
                             --output results

Optional --cells flag selects a subset of cells; default is all 41.
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Ensure the local code/ directory is on sys.path so the eigenspace modules
# can resolve their internal imports (kernels, likelihoods, _constants, ...).
HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from eigenspace_checkpoint import load_eigenspace_checkpoint
from eigenspace_training import predict_eigenspace
from metrics import (
    compute_pearson_correlation,
    compute_explained_variance,
    compute_adjusted_r_squared,
)


# Pool integrity tolerance used at training time (default_params.json ->
# active_learning.checkpoint_pool_sum_tolerance). Calibrated for float32 GPU
# reduction noise on a 3160 x 4096 pool. Do not change unless you know why.
POOL_SUM_TOLERANCE = 1e-3


def load_data(data_path, dtype=torch.float32):
    """Load the PNAS .npz dataset.

    Returns dict with X_pool (train+val concat, flattened), X_test (flattened),
    R_test (n_reps, n_test_images, n_cells), and n_px_side.
    """
    data = np.load(data_path)
    X_train = torch.tensor(data['images_train'], dtype=dtype)
    X_val = torch.tensor(data['images_val'], dtype=dtype)
    X_test = torch.tensor(data['images_test'], dtype=dtype)
    R_test = torch.tensor(data['responses_test'], dtype=dtype)

    X_pool = torch.cat([X_train, X_val], dim=0)
    n_pool = X_pool.shape[0]
    X_pool = X_pool.reshape(n_pool, -1)
    X_test = X_test.reshape(X_test.shape[0], -1)

    n_pixels = X_pool.shape[1]
    n_px_side = int(round(n_pixels ** 0.5))
    if n_px_side * n_px_side != n_pixels:
        raise ValueError(
            f"Non-square images: {n_pixels} pixels does not factor as a square."
        )

    return {
        'X_pool': X_pool,
        'X_test': X_test,
        'R_test': R_test,
        'n_px_side': n_px_side,
    }


def run_one_cell(cell_id, ckpt_path, X_pool, X_test, R_test, device,
                 plot_dir=None, n_px_side=None):
    """Load a checkpoint and evaluate on the test set for one cell."""
    bundle = load_eigenspace_checkpoint(
        checkpoint_path=str(ckpt_path),
        X_pool=X_pool.to(device),
        pool_sum_tolerance=POOL_SUM_TOLERANCE,
        device=device,
    )
    model = bundle['model']
    hyperparams = bundle['hyperparams']
    metadata = bundle['metadata']

    X_test_dev = X_test.to(device=device, dtype=X_pool.dtype)
    predictions = predict_eigenspace(model, X_test_dev)
    f_pred = predictions['f_pred']

    r_test = R_test[:, :, cell_id].to(device=device, dtype=torch.float32)
    r_test_mean = r_test.mean(dim=0)

    test_r = compute_pearson_correlation(r_test_mean, f_pred)
    explained_var, reliability = compute_explained_variance(r_test, f_pred)
    adjusted_r2 = compute_adjusted_r_squared(r_test, f_pred)

    if plot_dir is not None:
        plot_dir.mkdir(parents=True, exist_ok=True)
        _plot_cell(
            cell_id=cell_id,
            r_actual=r_test_mean.cpu().numpy(),
            f_pred=f_pred.cpu().numpy(),
            test_r=test_r,
            explained_var=explained_var,
            reliability=reliability,
            M=metadata.get('M'),
            output_path=plot_dir / f'cell_{cell_id:02d}.png',
        )

    return {
        'cell': cell_id,
        'test_r': test_r,
        'explained_var': explained_var,
        'adjusted_r2': adjusted_r2,
        'reliability': reliability,
        'M': metadata.get('M'),
        **{f'final_{k}': v for k, v in hyperparams.items()},
    }


def _plot_cell(cell_id, r_actual, f_pred, test_r, explained_var, reliability,
               M, output_path):
    """Two-panel plot: original-order and sorted-by-actual firing rate."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.5))

    n = len(r_actual)
    x = np.arange(n)

    ax1.plot(x, r_actual, 'k-', linewidth=1.5, label='Actual (mean of 30 reps)')
    ax1.plot(x, f_pred, 'r-', linewidth=1.5, label='Predicted')
    ax1.set_xlabel('Test image index')
    ax1.set_ylabel('Firing rate (spikes)')
    ax1.set_title(f'Cell {cell_id} - original order')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)

    sort_idx = np.argsort(r_actual)
    ax2.plot(x, r_actual[sort_idx], 'k-', linewidth=1.5, label='Actual')
    ax2.plot(x, f_pred[sort_idx], 'r-', linewidth=1.5, label='Predicted')
    ax2.set_xlabel('Test images (sorted by actual firing rate)')
    ax2.set_ylabel('Firing rate (spikes)')
    ax2.set_title(f'Cell {cell_id} - sorted')
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)

    metrics_text = (
        f'M = {M}\n'
        f'Pearson r = {test_r:.3f}\n'
        f'Reliability = {reliability:.3f}\n'
        f'Expl. var = {explained_var:.3f}'
    )
    ax2.text(
        0.02, 0.98, metrics_text, transform=ax2.transAxes, fontsize=10,
        verticalalignment='top', fontfamily='monospace',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
    )

    fig.tight_layout()
    fig.savefig(output_path, dpi=110)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True,
                        help='Path to the PNAS .npz dataset.')
    parser.add_argument('--checkpoints', type=Path, required=True,
                        help='Folder containing cell_XX.pt files.')
    parser.add_argument('--output', type=Path, default=Path('results'),
                        help='Output folder for summary.csv and plots/.')
    parser.add_argument('--cells', type=int, nargs='+', default=None,
                        help='Subset of cell ids to evaluate (default: all '
                             'cell_*.pt files in --checkpoints).')
    parser.add_argument('--device', type=str, default=None,
                        help='Device (cuda or cpu). Default: cuda if available.')
    parser.add_argument('--no-plots', action='store_true',
                        help='Skip per-cell plot generation.')
    args = parser.parse_args()

    device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    print(f"Loading data from {args.data}...")
    data = load_data(args.data)
    X_pool = data['X_pool']
    X_test = data['X_test']
    R_test = data['R_test']
    print(f"  X_pool: {tuple(X_pool.shape)}  X_test: {tuple(X_test.shape)}  "
          f"R_test: {tuple(R_test.shape)}")

    if args.cells is None:
        ckpt_files = sorted(args.checkpoints.glob('cell_*.pt'))
        cells = [int(p.stem.split('_')[1]) for p in ckpt_files]
    else:
        cells = list(args.cells)
        ckpt_files = [args.checkpoints / f'cell_{c:02d}.pt' for c in cells]

    if not cells:
        raise SystemExit(f"No checkpoints found in {args.checkpoints}.")

    args.output.mkdir(parents=True, exist_ok=True)
    plot_dir = None if args.no_plots else (args.output / 'plots')

    rows = []
    for cell_id, ckpt in zip(cells, ckpt_files):
        if not ckpt.exists():
            print(f"  Cell {cell_id}: missing checkpoint at {ckpt}, skipping.")
            continue
        try:
            row = run_one_cell(
                cell_id=cell_id,
                ckpt_path=ckpt,
                X_pool=X_pool,
                X_test=X_test,
                R_test=R_test,
                device=device,
                plot_dir=plot_dir,
                n_px_side=data['n_px_side'],
            )
        except Exception as e:
            print(f"  Cell {cell_id}: ERROR {type(e).__name__}: {e}")
            continue
        print(f"  Cell {cell_id:2d}: test_r={row['test_r']:.4f}  "
              f"adj_r2={row['adjusted_r2']:.4f}  "
              f"exp_var={row['explained_var']:.4f}")
        rows.append(row)

    if not rows:
        raise SystemExit("No cells evaluated successfully.")

    summary_path = args.output / 'summary.csv'
    fieldnames = list(rows[0].keys())
    with open(summary_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
    print(f"\nWrote summary: {summary_path}")

    test_rs = [r['test_r'] for r in rows]
    print(f"Mean test_r across {len(rows)} cells: {sum(test_rs) / len(test_rs):.4f}")


if __name__ == '__main__':
    main()
