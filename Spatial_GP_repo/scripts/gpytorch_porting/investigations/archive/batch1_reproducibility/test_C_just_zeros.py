#!/usr/bin/env python3
"""
Test C: Just torch.zeros(1) - is tensor creation enough?

Tests if simply creating a tensor (without the acos/assignment) changes random state.
Run this in a fresh terminal.
"""

import torch
import numpy as np

print("=" * 60)
print("Test C: Just torch.zeros(1)")
print("=" * 60)

# === JUST CREATE A TENSOR ===
_ = torch.zeros(1)
print("Executed: _ = torch.zeros(1)")

# === Set seed ===
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
