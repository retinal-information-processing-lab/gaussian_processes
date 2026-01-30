#!/usr/bin/env python3
"""
Test E: set_reproducible_seed(42) - WITHOUT device parameter

Uses the actual function from test_utils.py without passing device.
Run this in a fresh terminal.
"""

import sys
sys.path.insert(0, '/home/idv-eqs8-pza/IDV_code/ClosedLoopProject/gaussian_processes/Spatial_GP_repo/scripts/gpytorch_porting')

import torch
import numpy as np

print("=" * 60)
print("Test E: set_reproducible_seed(42) - NO device param")
print("=" * 60)

# === Import and call WITHOUT device param ===
from tests.test_utils import set_reproducible_seed
set_reproducible_seed(42)
print("Called: set_reproducible_seed(42)")

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
