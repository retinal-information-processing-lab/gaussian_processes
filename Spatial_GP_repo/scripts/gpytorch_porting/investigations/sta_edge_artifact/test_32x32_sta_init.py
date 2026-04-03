#!/usr/bin/env python3
"""
Test whether using a 32x32 center crop for STA RF initialization fixes
the edge artifact for cells 0, 5, 6, 15, 22, 39 on both 108x108 and 64x64.

Approach:
  1. Crop center 32x32 from each dataset (same responses)
  2. Compute STA on cropped pixels → find RF center
  3. Map 32x32 coords back to full image coords
  4. Run fits with the corrected RF initialization
  5. Compare with baseline (original STA init)

Usage:
    python test_32x32_sta_init.py
"""

import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

# Project paths
SCRIPT_DIR = Path(__file__).parent.parent.parent  # gpytorch_porting/
PYTHON = sys.executable

sys.path.insert(0, str(SCRIPT_DIR))
_repo_root = next(p for p in Path(__file__).resolve().parents if (p / 'Spatial_GP_repo').is_dir())
sys.path.insert(0, str(_repo_root.parent))

from utils import compute_rf_center_from_sta

# Config
CELLS = [0, 5, 6, 15, 22, 39]
CROP_SIZE = 32
DATASETS = {
    '108': {
        'path': 'datasets/PNAS_108x108_original.npz',
        'n_px': 108,
        'crop_offset': (108 - CROP_SIZE) // 2,  # 38
    },
    '64': {
        'path': 'datasets/PNAS_64x64_center_crop_no_renorm.npz',
        'n_px': 64,
        'crop_offset': (64 - CROP_SIZE) // 2,  # 16
    },
}
RESULTS_108 = SCRIPT_DIR / 'experiments' / '2026-03-20_massive_allcells_108' / 'results.jsonl'
RESULTS_64 = SCRIPT_DIR / 'experiments' / '2026-03-20_massive_allcells_64' / 'results.jsonl'


def load_dataset(path):
    """Load dataset, combine train+val."""
    data = np.load(SCRIPT_DIR / path)
    X = np.concatenate([data['images_train'], data['images_val']], axis=0)
    R = np.concatenate([data['responses_train'], data['responses_val']], axis=0)
    return X, R


def compute_32x32_rf_center(X_images, r, n_px_full, crop_offset):
    """Compute RF center from 32x32 center crop, return in full-image normalized coords.

    Args:
        X_images: (N, n_px, n_px, 1) images
        r: (N,) responses
        n_px_full: full image side length (108 or 64)
        crop_offset: pixel offset for center crop

    Returns:
        eps_0x, eps_0y in normalized [-1, 1] coords of the FULL image
        peak_32x, peak_32y in 32x32 pixel coords (for reporting)
    """
    # Crop center 32x32
    X_crop = X_images[:, crop_offset:crop_offset+CROP_SIZE,
                       crop_offset:crop_offset+CROP_SIZE, :]
    X_crop_flat = torch.tensor(X_crop.reshape(X_crop.shape[0], -1), dtype=torch.float32)
    r_torch = torch.tensor(r, dtype=torch.float32)

    # Compute STA on cropped images
    eps_32x, eps_32y = compute_rf_center_from_sta(X_crop_flat, r_torch, CROP_SIZE, zscore=True)

    # Map from 32x32 normalized coords to 32x32 pixel coords
    peak_32x = (eps_32x + 1) / 2 * (CROP_SIZE - 1)
    peak_32y = (eps_32y + 1) / 2 * (CROP_SIZE - 1)

    # Map to full image pixel coords
    peak_full_x = peak_32x + crop_offset
    peak_full_y = peak_32y + crop_offset

    # Convert to full image normalized coords
    eps_full_x = (peak_full_x / (n_px_full - 1)) * 2 - 1
    eps_full_y = (peak_full_y / (n_px_full - 1)) * 2 - 1

    return eps_full_x, eps_full_y, peak_32x, peak_32y


def get_baseline_testr(results_path, cell, mode='vargp_direct', M=300, seed=1):
    """Get baseline test_r from experiment results."""
    if not results_path.exists():
        return float('nan')
    records = [json.loads(l) for l in open(results_path) if l.strip()]
    matches = [r for r in records if r['cell'] == cell and r['mode'] == mode
               and r['M'] == M and r['seed'] == seed and r.get('status') == 'success']
    return matches[0]['test_r'] if matches else float('nan')


def run_fit(data_path, cell, eps_0x, eps_0y, mode='vargp_direct', M=300,
            n_train=2910, seed=1):
    """Run a single fit with specified RF center, return test_r."""
    cmd = [
        PYTHON, str(SCRIPT_DIR / 'run_single_mode.py'),
        '--data-path', data_path,
        '--mode', mode,
        '--ntilde', str(M),
        '--n-train', str(n_train),
        '--seed', str(seed),
        '--cell', str(cell),
        '--eps-0x', f'{eps_0x:.6f}',
        '--eps-0y', f'{eps_0y:.6f}',
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=SCRIPT_DIR, timeout=300)

    # Extract test_r from output
    for line in proc.stdout.splitlines():
        if 'Test Pearson r:' in line:
            try:
                return float(line.split(':')[-1].strip())
            except ValueError:
                pass
    return float('nan')


def main():
    print("=" * 70)
    print("Test: 32x32 center crop STA initialization")
    print("=" * 70)

    # Load datasets
    datasets_loaded = {}
    for ds_key, ds_info in DATASETS.items():
        X, R = load_dataset(ds_info['path'])
        datasets_loaded[ds_key] = (X, R)
        print(f"Loaded {ds_key}x{ds_key}: {X.shape[0]} images, {X.shape[1]}x{X.shape[2]} pixels")

    # Step 1: Compute 32x32 STA centers for all cells
    print(f"\n--- 32x32 STA RF centers ---")
    print(f"{'Cell':>4} {'DS':>4} {'32x32 peak':>14} {'full eps':>20} {'crop_offset':>11}")
    print("-" * 60)

    rf_centers = {}  # (ds_key, cell) -> (eps_x, eps_y)
    for ds_key, ds_info in DATASETS.items():
        X, R = datasets_loaded[ds_key]
        for cell in CELLS:
            r = R[:, cell].astype(np.float32)
            eps_x, eps_y, p32x, p32y = compute_32x32_rf_center(
                X, r, ds_info['n_px'], ds_info['crop_offset'])
            rf_centers[(ds_key, cell)] = (eps_x, eps_y)
            print(f"{cell:4d} {ds_key:>4} ({p32x:5.1f}, {p32y:5.1f}) "
                  f"eps=({eps_x:+.4f}, {eps_y:+.4f}) offset={ds_info['crop_offset']}")

    # Step 2: Run fits with 32x32-derived RF centers
    print(f"\n--- Running fits ---")
    results = []

    for ds_key, ds_info in DATASETS.items():
        results_path = RESULTS_108 if ds_key == '108' else RESULTS_64
        for cell in CELLS:
            eps_x, eps_y = rf_centers[(ds_key, cell)]
            baseline_r = get_baseline_testr(results_path, cell)

            print(f"\n  {ds_key}x{ds_key} cell {cell}: baseline_r={baseline_r:.3f}, "
                  f"32x32 init eps=({eps_x:+.4f}, {eps_y:+.4f})...", end=" ", flush=True)

            t0 = time.time()
            new_r = run_fit(ds_info['path'], cell, eps_x, eps_y)
            dt = time.time() - t0

            delta = new_r - baseline_r
            print(f"test_r={new_r:.3f} (delta={delta:+.3f}) [{dt:.1f}s]")

            results.append({
                'dataset': ds_key,
                'cell': cell,
                'baseline_r': round(baseline_r, 4),
                'r_32x32_init': round(new_r, 4),
                'delta': round(delta, 4),
                'eps_0x': round(eps_x, 4),
                'eps_0y': round(eps_y, 4),
            })

    # Step 3: Summary table
    print(f"\n{'=' * 70}")
    print(f"RESULTS SUMMARY")
    print(f"{'=' * 70}")
    print(f"{'DS':>4} {'Cell':>4} {'Baseline':>9} {'32x32 init':>11} {'Delta':>7}")
    print("-" * 40)
    for r in results:
        flag = ' ***' if r['delta'] > 0.1 else ''
        print(f"{r['dataset']:>4} {r['cell']:4d} {r['baseline_r']:9.4f} "
              f"{r['r_32x32_init']:11.4f} {r['delta']:+7.4f}{flag}")

    # Save results
    output_path = Path(__file__).parent / 'results_32x32_init.json'
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_path}")


if __name__ == '__main__':
    main()
