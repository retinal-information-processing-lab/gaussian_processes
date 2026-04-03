#!/usr/bin/env python3
"""
Test F: set_reproducible_seed(42, device='cuda') - WITH device parameter

Uses the actual function from test_utils.py with explicit device.
Run this in a fresh terminal.
"""

import sys
from pathlib import Path
_gpy_dir = next(p for p in Path(__file__).resolve().parents if p.name == 'gpytorch_porting')
sys.path.insert(0, str(_gpy_dir))

import torch
import numpy as np

print("=" * 60)
print("Test F: set_reproducible_seed(42, device='cuda') - WITH device param")
print("=" * 60)

# === Import and call WITH device param ===
from tests.test_utils import set_reproducible_seed
set_reproducible_seed(42, device='cuda')
print("Called: set_reproducible_seed(42, device='cuda')")

# === Generate random numbers ===
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device: {device}")

randn = torch.randn(5, device=device)
randperm = torch.randperm(100, device=device)[:5]
numpy_rand = np.random.rand(5)

print(f"\nrandn:    {randn.tolist()}")
print(f"randperm: {randperm.tolist()}")
print(f"numpy:    {numpy_rand.tolist()}")

checksum = sum(randn.tolist()) + sum(numpy_rand.tolist())
print(f"\nChecksum: {checksum:.10f}")
