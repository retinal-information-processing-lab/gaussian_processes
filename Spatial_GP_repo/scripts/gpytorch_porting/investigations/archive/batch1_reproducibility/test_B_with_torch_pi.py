#!/usr/bin/env python3
"""
Test B: WITH torch.pi line (the mystery line)

Compare output with test_A_baseline.py to see if the torch.pi line changes anything.
Run this in a fresh terminal.
"""

import torch
import numpy as np

print("=" * 60)
print("Test B: WITH torch.pi line")
print("=" * 60)

# === THE MYSTERY LINE ===
torch.pi = torch.acos(torch.zeros(1)).item() * 2
print(f"Executed: torch.pi = torch.acos(torch.zeros(1)).item() * 2")
print(f"torch.pi is now: {torch.pi}")

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
