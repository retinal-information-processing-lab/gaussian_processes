# Multi-Cell Validation Results

**Date**: January 2025
**Test script**: `tests/test_reference_comparison.py`
**Settings**: ntilde=500, n_train=2000, Reference maxiter=400 (30 E/M steps), GPyTorch 4000 iterations

---

## Summary Table: All Tested Cells

| Cell | Ref r | GPyTorch r | Diff | Assessment | Ref Time | GPyTorch Time | Notes |
|------|-------|------------|------|------------|----------|---------------|-------|
| 1 | 0.9763 | 0.9813 | 0.0050 | EXCELLENT | - | - | GPyTorch better |
| 2 | 0.9245 | 0.8803 | 0.0442 | EXCELLENT | - | - | |
| 3 | 0.9438 | 0.8294 | 0.1144 | WARNING | - | - | Larger gap |
| 4 | (failed) | 0.9144 | - | - | - | - | Ref stopped early (loss stabilization) |
| 5 | 0.6112 | 0.6590 | 0.0478 | EXCELLENT | - | - | GPyTorch better |
| 6 | 0.8285 | 0.9099 | 0.0814 | GOOD | - | - | GPyTorch better |
| 7 | 0.6282 | 0.7051 | 0.0769 | GOOD | - | - | GPyTorch better |
| 8 | 0.8794 | 0.8639 | 0.0155 | EXCELLENT | - | - | |
| 9 | 0.9419 | 0.9649 | 0.0230 | EXCELLENT | 72.6s | 534.0s | GPyTorch better |
| 10 | 0.8911 | 0.8936 | 0.0025 | EXCELLENT | 77.8s | 428.6s | |
| 11 | 0.9247 | 0.9498 | 0.0251 | EXCELLENT | 72.5s | 345.8s | GPyTorch better |
| 12 | 0.9563 | 0.9590 | 0.0027 | EXCELLENT | 143.0s | 1240.8s | GPyTorch better |
| 13 | 0.9335 | 0.9436 | 0.0101 | EXCELLENT | 71.6s | 592.7s | GPyTorch better |
| 14 | 0.9097 | 0.9222 | 0.0125 | EXCELLENT | 70.4s | 483.2s | GPyTorch better |
| 15 | 0.7194 | 0.6962 | 0.0232 | EXCELLENT | 90.4s | 657.6s | Ref better |

---

## Assessment Criteria

- **EXCELLENT**: Pearson r difference < 0.05
- **GOOD**: Pearson r difference 0.05 - 0.10
- **WARNING**: Pearson r difference >= 0.10

---

## Summary Statistics

| Metric | Value |
|--------|-------|
| **Total cells tested** | 15 |
| **EXCELLENT (diff < 0.05)** | 11 cells |
| **GOOD (0.05 ≤ diff < 0.1)** | 2 cells (6, 7) |
| **WARNING (diff ≥ 0.1)** | 1 cell (3) |
| **Reference failed** | 1 cell (4) |
| **GPyTorch outperformed reference** | 10 cells (1, 5, 6, 7, 9, 11, 12, 13, 14) |
| **Reference outperformed GPyTorch** | 4 cells (2, 3, 8, 15) |

---

## Timing Analysis (Cells 9-15)

| Metric | Reference | GPyTorch |
|--------|-----------|----------|
| **Average time** | ~85s | ~610s |
| **Speed ratio** | 1x | ~7x slower |
| **Reason** | Closed-form E-step | Iterative Adam (4000 iters) |

---

## Key Observations

1. **Overall validation success**: 13/14 comparable cells achieved GOOD or better match (≥93%)

2. **GPyTorch often outperforms**: On 10/14 cells, GPyTorch achieved higher Pearson r than reference

3. **Cell 3 is an outlier**: Only cell with WARNING status (diff=0.1144). May warrant investigation.

4. **Cell 4 reference failure**: Reference varGP stopped early due to "loss stabilization" detection

5. **Timing trade-off**: Reference is ~7x faster due to closed-form E-step vs iterative optimization

6. **Harder cells** (lower r for both):
   - Cell 5: r ≈ 0.6
   - Cell 7: r ≈ 0.6-0.7
   - Cell 15: r ≈ 0.7

---

## Configuration Details

```python
CONFIG = {
    'ntilde': 500,           # Inducing points
    'n_train': 2000,         # Training samples
    'n_px_side': 108,        # Image size (108x108 = 11664 pixels)

    # Initial hyperparameters (matched for both)
    'beta_init': 0.1,
    'rho_init': 0.1,
    'sigma_0_init': 1.0,
    'eps_0x_init': 0.0,
    'eps_0y_init': 0.0,
    'A_init': 0.01,
    'lambda0_init': 1.0,
    'Amp_init': 1.0,

    # Reference settings
    'ref_maxiter': 400,
    'ref_nEstep': 30,
    'ref_nMstep': 30,

    # GPyTorch settings
    'gpy_iterations': 4000,
    'gpy_lr': 0.01,
}
```

---

## Conclusion

The GPyTorch implementation is **validated** across 15 cells with generally excellent agreement with the reference implementation. The GPyTorch version often achieves comparable or better predictive performance, though at the cost of longer training time (~7x slower).
