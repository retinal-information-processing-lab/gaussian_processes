#!/usr/bin/env python3
"""
Train ceiling models: vargp_direct on full pool (3160 images) with large M.

These represent the best achievable performance per cell when all available
training data is used. Intended as a reference ceiling for active learning
comparison plots.

Usage:
    python train_ceiling_models.py --M 1500
    python train_ceiling_models.py --M 1500 --interleave-fstep --fix-Amp
    python train_ceiling_models.py --M 1500 --seeds 0 1 2 --cells 0 8 10

Output:
    checkpoints/<dir_name>/
        cell_XX_seedY.pt          # eigenspace checkpoints
        ceiling_results.json      # per-cell best test_r (across seeds)
        sweep_results.jsonl       # one row per (cell, seed) for analysis
"""

import sys
import time
import json
import argparse
from pathlib import Path

_repo_root = next(p for p in Path(__file__).resolve().parents if (p / 'Spatial_GP_repo').is_dir())
sys.path.insert(0, str(_repo_root.parent))

import torch
import numpy as np

from run_single_mode import (
    build_config_from_defaults,
    run_single_config,
    load_pnas_data,
)
from eigenspace_checkpoint import save_eigenspace_checkpoint


SCRIPT_DIR = Path(__file__).resolve().parent
N_CELLS = 41
DATA_PATH = 'datasets/PNAS_64x64_center_crop_no_renorm.npz'


def _fmt_time(seconds):
    s = int(seconds)
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    if h:
        return f"{h}h {m}m {sec}s"
    if m:
        return f"{m}m {sec}s"
    return f"{sec}s"


def main():
    parser = argparse.ArgumentParser(
        description='Train ceiling models (vargp_direct, full pool)')
    parser.add_argument('--M', type=int, required=True,
                        help='Number of inducing points (e.g. 1500)')
    parser.add_argument('--cells', type=int, nargs='+', default=list(range(N_CELLS)),
                        help=f'Cell IDs to train (default: all {N_CELLS})')
    parser.add_argument('--seeds', type=int, nargs='+', default=[0],
                        help='Random seeds (default: 0)')
    parser.add_argument('--interleave-fstep', action='store_true',
                        help='Enable interleaved F-step (damped Newton A/lambda0 inside E-step)')
    parser.add_argument('--fix-Amp', action='store_true',
                        help='Fix Amp=1 (do not optimize amplitude)')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory (default: auto-generated from config)')
    args = parser.parse_args()

    # Auto-generate output dir name from config
    if args.output_dir is None:
        parts = [f'64x64_ceiling_M{args.M}']
        if args.interleave_fstep:
            parts.append('intl')
        if args.fix_Amp:
            parts.append('fixAmp')
        output_dir = SCRIPT_DIR / 'checkpoints' / '_'.join(parts)
    else:
        output_dir = Path(args.output_dir)
        if not output_dir.is_absolute():
            output_dir = SCRIPT_DIR / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load pool once (needed for save_eigenspace_checkpoint integrity tags)
    data = load_pnas_data(DATA_PATH, dtype=torch.float32)
    X_pool = torch.cat([data['X_train'], data['X_val']], dim=0)
    X_pool = X_pool.reshape(X_pool.shape[0], -1).to('cuda')
    n_pool = X_pool.shape[0]

    cells = args.cells
    seeds = args.seeds
    total = len(cells) * len(seeds)
    failed = []
    sweep_rows = []

    print(f"Ceiling model training: {len(cells)} cells x {len(seeds)} seeds = {total} runs",
          flush=True)
    print(f"  M={args.M}, n_train={n_pool}, ip_selection=random", flush=True)
    print(f"  interleave_fstep={args.interleave_fstep}, fix_Amp={args.fix_Amp}", flush=True)
    print(f"  Output: {output_dir}", flush=True)
    print(flush=True)

    overall_start = time.time()
    run_idx = 0

    for cell_id in cells:
        for seed in seeds:
            run_idx += 1
            t0 = time.time()
            tag = f"[{run_idx}/{total}] Cell {cell_id:2d}  seed {seed}"
            print(f"{tag}...", end=" ", flush=True)

            config = build_config_from_defaults(
                mode='vargp_direct',
                M=args.M,
                n_train=n_pool,
                seed=seed,
                cell=cell_id,
                data_path=DATA_PATH,
                ip_selection='random',
                interleave_fstep=args.interleave_fstep,
                fix_Amp=args.fix_Amp,
            )

            try:
                result = run_single_config(config)

                if result is None:
                    print("FAILED (returned None)", flush=True)
                    failed.append((cell_id, seed, "returned None"))
                    continue

                metrics = {
                    'test_r': result['test_r'],
                    'adjusted_r2': result['adjusted_r2'],
                    'explained_var': result['explained_var'],
                    'reliability': result['reliability'],
                    'train_time': result['train_time'],
                    'final_loss': result['final_loss'],
                    'stopped_early': result['stopped_early'],
                    'n_iterations_run': result['n_iterations_run'],
                }

                ckpt_path = output_dir / f'cell_{cell_id:02d}_seed{seed}.pt'
                save_eigenspace_checkpoint(
                    model=result['_model'],
                    config=config,
                    metrics=metrics,
                    pool_indices=result['_indices_train'],
                    X_pool=X_pool,
                    checkpoint_path=ckpt_path,
                )

                elapsed = time.time() - t0
                print(f"test_r={result['test_r']:.4f}  adj_r2={result['adjusted_r2']:.4f}  "
                      f"time={_fmt_time(elapsed)}", flush=True)

                sweep_rows.append({
                    'cell': cell_id,
                    'seed': seed,
                    'M': args.M,
                    'n_train': n_pool,
                    'interleave_fstep': args.interleave_fstep,
                    'fix_Amp': args.fix_Amp,
                    'test_r': result['test_r'],
                    'adjusted_r2': result['adjusted_r2'],
                    'explained_var': result['explained_var'],
                    'reliability': result['reliability'],
                    'train_time': result['train_time'],
                    'stopped_early': result['stopped_early'],
                    'n_iterations_run': result['n_iterations_run'],
                })

            except Exception as e:
                elapsed = time.time() - t0
                print(f"ERROR ({_fmt_time(elapsed)}): {e}", flush=True)
                failed.append((cell_id, seed, str(e)))

    total_time = time.time() - overall_start

    # Write sweep JSONL (one row per run)
    jsonl_path = output_dir / 'sweep_results.jsonl'
    with open(jsonl_path, 'w') as f:
        for row in sweep_rows:
            f.write(json.dumps(row) + '\n')
    print(f"\nSweep JSONL: {jsonl_path}", flush=True)

    # Write ceiling_results.json (per-cell mean across seeds)
    results_summary = {}
    for cell_id in cells:
        cell_rows = [r for r in sweep_rows if r['cell'] == cell_id]
        if not cell_rows:
            continue
        results_summary[str(cell_id)] = {
            'test_r': float(np.mean([r['test_r'] for r in cell_rows])),
            'adjusted_r2': float(np.mean([r['adjusted_r2'] for r in cell_rows])),
            'explained_var': float(np.mean([r['explained_var'] for r in cell_rows])),
            'reliability': cell_rows[0]['reliability'],
            'M': args.M,
            'n_train': n_pool,
            'n_seeds': len(cell_rows),
            'seeds': seeds,
            'interleave_fstep': args.interleave_fstep,
            'fix_Amp': args.fix_Amp,
        }

    summary_path = output_dir / 'ceiling_results.json'
    with open(summary_path, 'w') as f:
        json.dump(results_summary, f, indent=2)
    print(f"Ceiling JSON: {summary_path}", flush=True)

    print(f"\n{'='*60}", flush=True)
    print(f"CEILING TRAINING COMPLETE", flush=True)
    print(f"{'='*60}", flush=True)
    print(f"Total time: {_fmt_time(total_time)}", flush=True)
    print(f"Successful: {total - len(failed)}/{total}", flush=True)

    if failed:
        print(f"\nFailed ({len(failed)}):", flush=True)
        for cell_id, seed, reason in failed:
            print(f"  cell {cell_id} seed {seed}: {reason}", flush=True)


if __name__ == '__main__':
    main()
