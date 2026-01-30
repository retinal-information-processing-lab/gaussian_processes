#!/usr/bin/env python3
"""
Test G: What happens when we import utils.py BEFORE seeding?

utils.py (line 2) has: torch.set_grad_enabled(False)
utils.py (line 51) has: torch.pi = torch.acos(torch.zeros(1)).item() * 2

This tests if the import order/side effects from utils.py matter.
"""

import sys
sys.path.insert(0, '/home/idv-eqs8-pza/IDV_code/ClosedLoopProject')
sys.path.insert(0, '/home/idv-eqs8-pza/IDV_code/ClosedLoopProject/gaussian_processes/torchlambertw')

print("=" * 60)
print("Test G: Import utils.py BEFORE seeding")
print("=" * 60)

# === IMPORT UTILS FIRST (this has side effects!) ===
print("Importing utils.py (has torch.set_grad_enabled(False) and torch.pi line)...")
from gaussian_processes.Spatial_GP_repo import utils
print("utils.py imported")
print(f"torch.is_grad_enabled(): {utils.torch.is_grad_enabled()}")

import torch
import numpy as np

# === NOW set seed ===
torch.manual_seed(42)
np.random.seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)
    torch.cuda.manual_seed_all(42)

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
