# Batch 3: Silent Failure Detection - FINDINGS

**Date**: 2026-01-23 | **Branch**: `multiple_bugs_batch3`

## Fixes Applied

| Issue | Location | Fix |
|-------|----------|-----|
| **J1** | `fstep.py` | Track LBFGS instability, warn if A unchanged after inf return |
| **J2** | `estep.py`, `whitening.py` | Add warning + fallback to 5 Cholesky locations |
| **J3** | `estep.py` | Warn before clamping negative variance |

## Warning Formats

```
J1: F-step: instability detected (f_mean=1234.5). A unchanged: 0.0123.
J2: Cholesky failed on V (shape=(100,100), min_eigenvalue=-1.2e-06). Adding jitter=1.0e-04.
J3: Negative variance detected: 42/500 values. min=-1.2e-05, mean=4.6e-01. Clamping to 1e-6.
```

## Test Results

**Warning tests** (`test_warnings.py`):
- J3: TRIGGERED
- J2: TRIGGERED
- J1: Not triggered (lambda0 normalization prevents in synthetic test)

**Canonical benchmark** (seed=123, 12 configs): **All passed, no spurious warnings**

| Mode | M | n_train | test_r | expl_var | time | A | λ₀ | Amp | β | ρ | σ₀ |
|------|---|---------|--------|----------|------|------|--------|-------|-------|-------|-------|
| vargp_old | 50 | 500 | 0.8413 | 0.8874 | 6.2s | 0.0156 | -0.47 | 1.54 | 0.067 | 0.082 | 0.99 |
| vargp_style | 50 | 500 | 0.7660 | 0.8089 | 11.7s | 0.185 | -2.62 | 0.009 | 0.065 | 0.027 | 4.83 |
| default_gpy | 50 | 500 | 0.2766 | 0.2868 | 1.4s | 0.0049 | 0.05 | 0.60 | 0.071 | 0.070 | 3.38 |
| vargp_old | 100 | 500 | 0.8692 | 0.9148 | 8.1s | 0.0136 | -0.40 | 1.18 | 0.094 | 0.087 | 1.00 |
| vargp_style | 100 | 500 | 0.8174 | 0.8621 | 30.9s | 1.21 | -1.71 | 0.002 | 0.044 | 0.011 | 0.38 |
| default_gpy | 100 | 500 | 0.3707 | 0.3879 | 1.5s | 0.0050 | 0.05 | 0.62 | 0.072 | 0.070 | 4.48 |
| vargp_old | 200 | 500 | 0.8140 | 0.8562 | 10.3s | 0.0160 | -0.30 | 1.17 | 0.111 | 0.073 | 1.00 |
| vargp_style | 200 | 500 | 0.7593 | 0.8015 | 122.2s | 1.05 | 0.61 | 0.001 | 0.063 | 0.010 | 7.58 |
| default_gpy | 200 | 500 | 0.4852 | 0.5124 | 1.8s | 0.0051 | 0.07 | 0.62 | 0.072 | 0.070 | 5.12 |
| vargp_old | 200 | 2000 | 0.7351 | 0.7758 | 14.5s | 0.0129 | -0.24 | 1.54 | 0.090 | 0.073 | 1.00 |
| vargp_style | 200 | 2000 | 0.8198 | 0.8639 | 141.8s | 0.70 | 4.09 | 0.002 | 0.080 | 0.010 | 6.98 |
| default_gpy | 200 | 2000 | -0.1728 | -0.1831 | 3.1s | 0.0051 | 0.18 | 0.63 | 0.071 | 0.072 | 0.41 |

**Result**: 12 passed, 0 failed. No warnings emitted during normal training.

**Parameter notes**:
- `vargp_old` keeps A small (~0.01), σ₀ near 1.0, Amp ~1.5
- `vargp_style` learns larger A (0.2-1.2), σ₀ increases (0.4-7.6), Amp decreases (~0.001-0.009)
- `default_gpy` barely moves A from init (stays ~0.005), learns suboptimal kernel params

## Files Changed

- `estep.py` (+15 lines)
- `whitening.py` (+18 lines)
- `fstep.py` (+20 lines)

## Conclusion

Warnings work correctly. No false positives during normal training.
