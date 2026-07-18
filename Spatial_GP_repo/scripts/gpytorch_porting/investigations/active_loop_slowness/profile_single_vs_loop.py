#!/usr/bin/env python3
"""
Memory comparison: single fit at M=419 vs active loop at M=419.

Trains once with the same 419 indices the active loop used, measures
GPU memory after gc+empty_cache. Compares to the 19.9 GB the active
loop reported at the same M.

Usage:
    python investigations/active_loop_slowness/profile_single_vs_loop.py
"""
import gc
import sys
import torch
from pathlib import Path

# Add gpytorch_porting root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from run_single_mode import build_config_from_defaults, run_single_config

device = torch.device('cuda')

# Baseline
gc.collect()
torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats(device)
print(f"GPU before fit: {torch.cuda.memory_allocated(device)/1024**3:.3f} GB")

config = build_config_from_defaults(
    mode='vargp_direct',
    data_path='datasets/PNAS_64x64_center_crop_no_renorm.npz',
    M=419, n_train=419, seed=42, cell=8,
)
config['train_indices_from'] = str(
    Path('/tmp/task1_full_new/checkpoints/iter_369.pt')
)

result = run_single_config(config)

# Clean measurement
gc.collect()
torch.cuda.empty_cache()
mem_after = torch.cuda.memory_allocated(device) / 1024**3
mem_peak = torch.cuda.max_memory_allocated(device) / 1024**3

print()
print("=== MEMORY COMPARISON ===")
print(f"Single fit M=419 — after gc:  {mem_after:.3f} GB")
print(f"Single fit M=419 — peak:      {mem_peak:.3f} GB")
print(f"Active loop M=419 — reported: 19.923 GB")
print(f"test_r: {result['test_r']:.4f}")
