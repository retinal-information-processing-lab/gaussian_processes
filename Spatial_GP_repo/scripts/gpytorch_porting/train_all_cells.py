#!/usr/bin/env python3
"""
Train default_gpy GP models for all cells and save checkpoints.

Trains all 41 cells in the PNAS dataset for each specified image size,
saving one .pt checkpoint per cell. All training parameters come from
default_params.json via build_config_from_defaults().

Usage:
    # Train all cells for both image sizes (default)
    python train_all_cells.py

    # Train only 108x108
    python train_all_cells.py --datasets 108

    # Train only 64x64
    python train_all_cells.py --datasets 64

    # Custom output directory
    python train_all_cells.py --output-dir /path/to/checkpoints

    # Override M (inducing points)
    python train_all_cells.py --M 50
"""

import sys
import time
import argparse
from pathlib import Path

# Add repo parent to sys.path for `from gaussian_processes.Spatial_GP_repo import ...`
_repo_root = next(p for p in Path(__file__).resolve().parents if (p / 'Spatial_GP_repo').is_dir())
sys.path.insert(0, str(_repo_root.parent))

import torch

from run_single_mode import build_config_from_defaults, run_single_config
from checkpoint import save_checkpoint


# Dataset paths relative to gpytorch_porting/
DATASETS = {
    '108': 'datasets/PNAS_108x108_original.npz',
    '64': 'datasets/PNAS_64x64_center_crop_no_renorm.npz',
}

N_CELLS = 41


def train_all_cells(datasets, output_dir, M_override=None):
    """Train all cells for specified datasets and save checkpoints.

    Args:
        datasets: List of dataset keys ('108', '64')
        output_dir: Base directory for checkpoints
        M_override: Override number of inducing points (None = use default)
    """
    output_dir = Path(output_dir)
    total_runs = len(datasets) * N_CELLS
    completed = 0
    failed = []

    print(f"Training {N_CELLS} cells x {len(datasets)} dataset(s) = {total_runs} total runs")
    print(f"Output: {output_dir}")
    print()

    overall_start = time.time()

    for ds_key in datasets:
        data_path = DATASETS[ds_key]
        ds_dir = output_dir / f"{ds_key}x{ds_key}"
        ds_dir.mkdir(parents=True, exist_ok=True)

        print(f"{'='*60}")
        print(f"Dataset: {ds_key}x{ds_key} ({data_path})")
        print(f"{'='*60}")

        for cell_id in range(N_CELLS):
            cell_start = time.time()
            completed += 1

            # Build config from defaults, override cell and data path
            overrides = dict(
                mode='default_gpy',
                cell=cell_id,
                data_path=data_path,
            )
            if M_override is not None:
                overrides['M'] = M_override

            config = build_config_from_defaults(**overrides)

            checkpoint_path = ds_dir / f"cell_{cell_id:02d}.pt"

            print(f"\n[{completed}/{total_runs}] Cell {cell_id:2d} ({ds_key}x{ds_key})...", end=" ", flush=True)

            try:
                result = run_single_config(config)

                if result is None:
                    print(f"FAILED (training returned None)")
                    failed.append((ds_key, cell_id, "training returned None"))
                    continue

                # Extract metrics for checkpoint
                metrics = {
                    'test_r': result['test_r'],
                    'train_r': result['train_r'],
                    'explained_var': result['explained_var'],
                    'reliability': result['reliability'],
                    'train_time': result['train_time'],
                    'final_loss': result['final_loss'],
                    'stopped_early': result['stopped_early'],
                    'n_iterations_run': result['n_iterations_run'],
                }

                save_checkpoint(result['_model'], result['_likelihood'], config, metrics, checkpoint_path)

                cell_time = time.time() - cell_start
                print(f"test_r={result['test_r']:.4f}  time={cell_time:.1f}s  -> {checkpoint_path.name}")

            except Exception as e:
                cell_time = time.time() - cell_start
                print(f"ERROR ({cell_time:.1f}s): {e}")
                failed.append((ds_key, cell_id, str(e)))

    total_time = time.time() - overall_start

    # Summary
    print(f"\n{'='*60}")
    print(f"TRAINING COMPLETE")
    print(f"{'='*60}")
    print(f"Total time: {total_time:.1f}s ({total_time/60:.1f} min)")
    print(f"Successful: {total_runs - len(failed)}/{total_runs}")

    if failed:
        print(f"\nFailed runs ({len(failed)}):")
        for ds_key, cell_id, reason in failed:
            print(f"  {ds_key}x{ds_key} cell {cell_id}: {reason}")

    return failed


def main():
    parser = argparse.ArgumentParser(
        description='Train GP models for all cells and save checkpoints'
    )
    parser.add_argument('--datasets', nargs='+', default=['108', '64'],
                        choices=['108', '64'],
                        help='Which dataset sizes to train (default: both)')
    parser.add_argument('--output-dir', type=str, default='checkpoints',
                        help='Base directory for checkpoint output (default: checkpoints/)')
    parser.add_argument('--M', type=int, default=None,
                        help='Override number of inducing points (default: from default_params.json)')

    args = parser.parse_args()
    train_all_cells(args.datasets, args.output_dir, M_override=args.M)


if __name__ == '__main__':
    main()
