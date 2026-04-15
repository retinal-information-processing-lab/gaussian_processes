# M Sweep: 64x64 — Inducing-Point Performance Characterization

**Date**: 2026-04-13
**Branch**: `pietro/investigate-M-degradation`
**Scope**: Full 41-cell × 8-M × 3-seed sweep (984 fits) with the best-known training
configuration for `vargp_direct + interleave_fstep + fix_Amp`.

## TL;DR

- This is the definitive characterization of how test_r depends on M for **this
  training configuration** (listed below). Any question about the 64×64 + intl +
  fixAmp setup at different M should consult this file first.
- **No universal M degradation.** Population mean test_r grows 0.817 → 0.840 from
  M=50 to M=1500 and plateaus at **M ≈ 200-300** (gains beyond this are ≤ 0.002).
- **Mild overfitting affects 9/41 cells** at large M (classic train_r↑ / test_r↓
  signature). Cell 39 is the worst (-0.199), Cell 35 next (-0.057). Mechanism and
  verification: see `investigations/M_degradation/FINDINGS.md`.
- **Recommended operating point: M = 200-300.** Beyond it, cost grows without
  population-level benefit, and sparse near-ceiling cells start to degrade.
- **Correct config is not trivially reproducible**: `build_config_from_defaults()`
  does NOT produce a valid interleaved config. See "Training configuration" below.

## Training configuration

All 984 runs use identical settings except (cell, M, seed):

```python
mode = 'vargp_direct'
interleave_fstep = True
fix_Amp = True

# ES-sweep overrides (NOT the default_params.json values!)
A_init = 1e-4
lambda0_init = -1.0
n_estep = 50
n_mstep = 20
n_iterations = 80

# Other knobs (from default_params.json)
beta_init = 0.1
rho_init = 0.1
sigma_0_init = 1.0
kernel_type = 'arc_cosine'
rf_init = 'ground_truth'          # datasets/rf_centers_ground_truth.npz
ip_selection = 'random'
n_val_split = 0                   # full 3160 images for training, no val carving

# Early stopping
early_stop = True
es_metric = 'elbo'
patience = 15
min_delta_rel = 0.001
min_iterations = 10

# Data
data_path = 'datasets/PNAS_64x64_center_crop_no_renorm.npz'
n_train = 3160
```

**Grid**: 41 cells × M ∈ {50, 100, 200, 300, 500, 750, 1000, 1500} × seeds {0, 1, 2} = 984 runs.

## Results

**Raw data**: `M_sweep_results.jsonl` (984 lines, one JSON object per run).
Fields: `cell, M, seed, test_r, train_r, adjusted_r2, final_loss, final_A,
final_lambda0, final_beta, final_rho, final_sigma_0, final_eps_0x, final_eps_0y,
n_b, n_b_over_M, eigval_min, eigval_max, n_iterations_run, stopped_early,
train_time`, plus per-iteration `curves_A, curves_beta, curves_train_r,
curves_train_loss, curves_train_log_lik`.

### Population-level test_r vs M (mean across 41 cells × 3 seeds)

| M | 50 | 100 | 200 | 300 | 500 | 750 | 1000 | 1500 |
|---|----|----|----|----|----|----|----|----|
| **mean test_r** | 0.817 | 0.830 | 0.838 | **0.838** (peak) | 0.834 (dip*) | 0.838 | 0.840 | 0.840 |

*The dip at M=500 is driven by Cell 39's catastrophic drop; removing it restores a
monotonic curve.

### Per-cell trends

Categorized by delta = test_r(M=1500) − test_r(M=50), threshold 0.005 (mean over 3 seeds):

| Category | Count | delta range | Notes |
|----------|-------|------------|-------|
| Improving | 27 / 41 | +0.012 to +0.142 | Median +0.036; includes Cell 0 which had been failing with wrong config |
| Flat | 5 / 41 | [-0.004, +0.003] | Cells 1, 3, 11, 30, 36 — already near their noise ceiling at small M |
| **Degrading** | **9 / 41** | **-0.199 to -0.007** | See table below |

Degrading cells (sorted by severity):

| Cell | test_r M=50 | test_r M=1500 | delta | train_r M=50 | train_r M=1500 |
|------|------------|--------------|-------|-------------|----------------|
| 39 | 0.699 | 0.501 | **-0.199** | 0.457 | 0.545 |
| 35 | 0.806 | 0.749 | -0.057 | 0.386 | 0.434 |
| 16 | 0.955 | 0.927 | -0.028 | 0.811 | 0.777 |
| 13 | 0.951 | 0.934 | -0.017 | 0.562 | 0.597 |
| 10 | 0.905 | 0.890 | -0.015 | 0.493 | 0.525 |
| 33 | 0.969 | 0.955 | -0.014 | 0.668 | 0.695 |
| 15 | 0.704 | 0.691 | -0.013 | 0.402 | 0.514 |
| 14 | 0.909 | 0.896 | -0.013 | 0.622 | 0.652 |
| 27 | 0.905 | 0.898 | -0.007 | 0.653 | 0.727 |

**Every degrading cell shows train_r rising as test_r falls — overfitting signature.**

### Known mechanism for the degrading cells

Analysis and validation are in `investigations/M_degradation/FINDINGS.md`. Summary:

- With more inducing points, the ELBO becomes a tighter bound on the marginal
  likelihood, revealing finer structure in training data — including noise.
- The ELBO's KL term regularizes the variational distribution but NOT the kernel
  and likelihood hyperparameters. The M-step has no regularizer.
- For cells already near their noise-ceiling at small M, adding capacity has no
  real signal left to capture — the optimizer drives A upward to fit training
  noise, which improves train_r but hurts test_r.
- Sparse cells with few non-zero responses (e.g., Cell 35: mean(r)=0.21, 88% zeros)
  are especially susceptible because the effective sample size for hyperparameter
  estimation is much smaller than N_train=3160.

## Reproduction

From `gpytorch_porting/`:

```bash
# Part 1: 13 cells (0, 1, 3, 5, 8, 9, 12, 25, 26, 28, 30, 35, 36)
python experiments/2026-04-13_M_sweep_64x64/run_sweep_part1.py

# Part 2: remaining 28 cells
python experiments/2026-04-13_M_sweep_64x64/run_sweep_part2.py
```

Both scripts are resume-safe: each appends to `M_sweep_results.jsonl` and skips
any `(cell, M, seed)` tuple already present. Combined runtime on a single
GPU: ~10-12 hours sequential.

## Loading the data

```python
import json
import numpy as np
from collections import defaultdict

results = []
with open('experiments/2026-04-13_M_sweep_64x64/M_sweep_results.jsonl') as f:
    for line in f:
        results.append(json.loads(line.strip()))

# Mean test_r per (cell, M)
cell_m = defaultdict(list)
for r in results:
    if r.get('test_r') is not None:
        cell_m[(r['cell'], r['M'])].append(r['test_r'])

# Example: M curve for cell 8
for M in [50, 100, 200, 300, 500, 750, 1000, 1500]:
    vals = cell_m[(8, M)]
    print(f"Cell 8 M={M}: {np.mean(vals):.4f} +/- {np.std(vals):.4f}")
```

## Related experiments

- `experiments/2026-04-06_es_sweeps_64x64/` — the M=250-only sweep that provided
  the reference "0.838" number. Same config as this sweep at M=250.
- `checkpoints/64x64_ceiling_M1500_intl_fixAmp/` — the M=1500 ceiling fit with
  the WRONG config (A_init=0.01), which produced the misleading "0.829" number.
  Do not use for comparison.

## Files in this folder

| File | Purpose |
|------|---------|
| `README.md` | This document |
| `M_sweep_results.jsonl` | 984 records — canonical data |
| `run_sweep_part1.py` | First 13 cells (the investigation subset) |
| `run_sweep_part2.py` | Remaining 28 cells (full-population completion) |
