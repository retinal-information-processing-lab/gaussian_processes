#!/usr/bin/env python3
"""
Train all 41 cells at M=300 with the correct ES config and save .pt checkpoints.

Produces a standalone set of trained models ready for inference via
run_inference.py. Single seed (seed=0), 64x64 dataset, vargp_direct mode.

This exists because train_ceiling_models.py does not forward the ES config
overrides (A_init=1e-4, n_estep=50, n_mstep=20, n_iterations=80, lambda0_init=-1).
Using the defaults produces the "wrong config" fits that we replaced.

The fits produced here should match the test_r numbers in
M_sweep_results.jsonl at (cell=*, M=300, seed=0) exactly — same code, same
config, same seed. The only difference is this script SAVES the .pt files
via save_eigenspace_checkpoint (the main sweep did not).

Output: checkpoints/64x64_M300_es_intl_fixAmp/ with:
  cell_00.pt ... cell_40.pt  (eigenspace checkpoints)
  ceiling_results.json        (per-cell summary)
  sweep_results.jsonl         (one line per run)

Run from gpytorch_porting/:
  nohup python experiments/2026-04-13_M_sweep_64x64/train_and_save_M300.py \
    > experiments/2026-04-13_M_sweep_64x64/train_M300.log 2>&1 &
"""
import os
import sys
import json
import time
from pathlib import Path

PROJ = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJ)

import torch

from run_single_mode import build_config_from_defaults, run_single_config, load_pnas_data
from eigenspace_checkpoint import save_eigenspace_checkpoint

# ES sweep config overrides (same as experiments/2026-04-13_M_sweep_64x64/)
ES_CONFIG = dict(
    A_init=1e-4,
    lambda0_init=-1.0,
    n_estep=50,
    n_mstep=20,
    n_iterations=80,
)

# Fixed choices
MODE = 'vargp_direct'
M = 300
SEED = 0
N_CELLS = 41
DATA_PATH = 'datasets/PNAS_64x64_center_crop_no_renorm.npz'

# Output location
PROJ_PATH = Path(PROJ)
OUTPUT_DIR = PROJ_PATH / 'checkpoints' / '64x64_M300_es_intl_fixAmp'


def _fmt_time(seconds):
    s = int(seconds)
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    if h:
        return f"{h}h{m}m{sec}s"
    if m:
        return f"{m}m{sec}s"
    return f"{sec}s"


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results_jsonl = OUTPUT_DIR / 'sweep_results.jsonl'

    # Load pool once for integrity tags in save_eigenspace_checkpoint
    data = load_pnas_data(PROJ_PATH / DATA_PATH, dtype=torch.float32)
    X_pool = torch.cat([data['X_train'], data['X_val']], dim=0)
    X_pool = X_pool.reshape(X_pool.shape[0], -1).to('cuda')
    n_pool = X_pool.shape[0]

    # Resume-safe: skip cells already in sweep_results.jsonl
    completed = set()
    if results_jsonl.exists():
        with open(results_jsonl) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    completed.add(json.loads(line)['cell'])
                except (json.JSONDecodeError, KeyError):
                    pass

    cells = list(range(N_CELLS))
    total = len(cells)

    print(f"M={M} ES-config all-cells training + checkpoint save")
    print(f"  mode={MODE}  seed={SEED}  n_train={n_pool}")
    print(f"  ES overrides: {ES_CONFIG}")
    print(f"  interleave_fstep=True, fix_Amp=True, ip_selection='random'")
    print(f"  Output: {OUTPUT_DIR}")
    print(f"  Already done: {sorted(completed) if completed else 'none'}")
    print(f"  To run: {total - len(completed)} cells")
    print(flush=True)

    overall_start = time.time()
    sweep_rows = []
    failed = []

    for idx, cell_id in enumerate(cells):
        if cell_id in completed:
            continue

        t0 = time.time()
        tag = f"[{idx + 1}/{total}] Cell {cell_id:2d}"
        print(f"{tag}...", end=" ", flush=True)

        config = build_config_from_defaults(
            mode=MODE,
            M=M,
            n_train=n_pool,
            seed=SEED,
            cell=cell_id,
            data_path=DATA_PATH,
            ip_selection='random',
            interleave_fstep=True,
            fix_Amp=True,
            **ES_CONFIG,
        )

        try:
            result = run_single_config(config)
        except Exception as e:
            elapsed = time.time() - t0
            print(f"ERROR ({_fmt_time(elapsed)}): {type(e).__name__}: {e}", flush=True)
            failed.append((cell_id, str(e)))
            continue

        if result is None:
            elapsed = time.time() - t0
            print(f"FAILED (returned None) ({_fmt_time(elapsed)})", flush=True)
            failed.append((cell_id, 'returned None'))
            continue

        # Save full eigenspace checkpoint (pool_indices + integrity tags baked in)
        ckpt_path = OUTPUT_DIR / f'cell_{cell_id:02d}.pt'
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
        save_eigenspace_checkpoint(
            model=result['_model'],
            config=config,
            metrics=metrics,
            pool_indices=result['_indices_train'],
            X_pool=X_pool,
            checkpoint_path=ckpt_path,
        )

        elapsed = time.time() - t0
        print(
            f"test_r={result['test_r']:.4f}  "
            f"adj_r2={result['adjusted_r2']:.4f}  "
            f"exp_var={result['explained_var']:.4f}  "
            f"A={result['final_A']:.4f}  "
            f"time={_fmt_time(elapsed)}",
            flush=True,
        )

        row = {
            'cell': cell_id,
            'seed': SEED,
            'M': M,
            'n_train': n_pool,
            'test_r': result['test_r'],
            'adjusted_r2': result['adjusted_r2'],
            'explained_var': result['explained_var'],
            'reliability': result['reliability'],
            'train_time': result['train_time'],
            'final_loss': result['final_loss'],
            'n_iterations_run': result['n_iterations_run'],
            'stopped_early': result['stopped_early'],
            'final_A': result['final_A'],
            'final_lambda0': result['final_lambda0'],
            'final_beta': result['final_beta'],
            'final_rho': result['final_rho'],
            'final_sigma_0': result['final_sigma_0'],
            'final_eps_0x': result['final_eps_0x'],
            'final_eps_0y': result['final_eps_0y'],
            'ckpt_path': str(ckpt_path.relative_to(PROJ_PATH)),
        }
        sweep_rows.append(row)
        with open(results_jsonl, 'a') as f:
            f.write(json.dumps(row) + '\n')

        # Free model from GPU between runs
        del result
        import gc
        gc.collect()
        torch.cuda.empty_cache()

    total_elapsed = time.time() - overall_start
    print(flush=True)
    print(f"=== Summary ===", flush=True)
    print(f"  Total time: {_fmt_time(total_elapsed)}", flush=True)
    print(f"  Succeeded:  {len(sweep_rows)}", flush=True)
    print(f"  Failed:     {len(failed)}", flush=True)
    for cell_id, reason in failed:
        print(f"    Cell {cell_id}: {reason}", flush=True)

    # Aggregate summary
    if sweep_rows or results_jsonl.exists():
        all_rows = []
        if results_jsonl.exists():
            with open(results_jsonl) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        all_rows.append(json.loads(line))
        if all_rows:
            tests = [r['test_r'] for r in all_rows if r.get('test_r') is not None]
            print(f"  Mean test_r across {len(tests)} cells: {sum(tests) / len(tests):.4f}",
                  flush=True)

            # Per-cell summary for downstream use
            summary = {
                str(r['cell']): {
                    'test_r': r['test_r'],
                    'adjusted_r2': r['adjusted_r2'],
                    'explained_var': r['explained_var'],
                    'reliability': r['reliability'],
                    'ckpt_path': r['ckpt_path'],
                    'M': r['M'],
                    'seed': r['seed'],
                    'final_A': r['final_A'],
                    'final_lambda0': r['final_lambda0'],
                    'final_beta': r['final_beta'],
                    'final_rho': r['final_rho'],
                    'final_sigma_0': r['final_sigma_0'],
                    'final_eps_0x': r['final_eps_0x'],
                    'final_eps_0y': r['final_eps_0y'],
                }
                for r in all_rows if r.get('test_r') is not None
            }
            summary_path = OUTPUT_DIR / 'ceiling_results.json'
            with open(summary_path, 'w') as f:
                json.dump(summary, f, indent=2)
            print(f"  Summary written to: {summary_path}", flush=True)


if __name__ == '__main__':
    main()
