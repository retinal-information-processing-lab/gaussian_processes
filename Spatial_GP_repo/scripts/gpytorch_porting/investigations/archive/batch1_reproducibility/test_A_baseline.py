#!/usr/bin/env python3
"""
Test A: BASELINE - No torch.pi line, no cuda.init

This establishes what random numbers we get with NOTHING special.
Run this in a fresh terminal.

Expected output: A sequence of random numbers that we'll compare to other tests.
"""

import torch
import numpy as np

print("=" * 60)
print("Test A: BASELINE (nothing special)")
print("=" * 60)

# === NOTHING SPECIAL - just set seed ===
torch.manual_seed(42)
np.random.seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)
    torch.cuda.manual_seed_all(42)

# === Generate random numbers ===
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device: {device}")
print(f"PyTorch version: {torch.__version__}")

randn = torch.randn(5, device=device)
randperm = torch.randperm(100, device=device)[:5]
numpy_rand = np.random.rand(5)

print(f"\nrandn:    {randn.tolist()}")
print(f"randperm: {randperm.tolist()}")
print(f"numpy:    {numpy_rand.tolist()}")

# Checksum for easy comparison
checksum = sum(randn.tolist()) + sum(numpy_rand.tolist())
print(f"\nChecksum: {checksum:.10f}")
