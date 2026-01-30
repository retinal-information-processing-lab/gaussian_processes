# Batch 3: Silent Failure Detection Investigation

**Created**: 2026-01-23 by Claude
**Branch**: `multiple_bugs_batch3`

## Overview

This investigation adds warnings for silent failures that previously masked numerical issues.

## Issues Addressed

| Issue | Location | Fix |
|-------|----------|-----|
| **J1** | `fstep.py:f_step_lbfgs()` | Track instability, warn if A unchanged |
| **J2** | `estep.py`, `whitening.py` | Add warning when Cholesky fallback triggers |
| **J3** | `estep.py:compute_moments_from_kernel_cache()` | Warn when negative variance is clamped |

## Files Modified

- `estep.py` - J2 (Cholesky fallback) + J3 (variance warning)
- `whitening.py` - J2 (Cholesky warnings in 3 functions)
- `fstep.py` - J1 (LBFGS instability tracking)

## Files Created

- `investigations/batch3_silent_failures/README.md` - This file
- `investigations/batch3_silent_failures/test_warnings.py` - Test script

## Warning Messages

### J1 - LBFGS Instability
```
F-step: instability detected (f_mean=1234.5). A unchanged: 0.0123.
F-step: instability detected (f_mean=1234.5). A changed: 0.0100 -> 0.0123.
```

### J2 - Cholesky Fallback
```
Cholesky failed on K_tilde in compute_kernel_cache() (shape=(100, 100), min_eigenvalue=-1.23e-06). Adding extra jitter=1.0e-04 and retrying.
Cholesky failed on V in update_variational_covar() (shape=(100, 100), min_eigenvalue=-1.23e-06). Adding jitter=1.0e-04 and retrying.
```

### J3 - Negative Variance
```
Negative variance detected: 42/500 values. min=-1.23e-05, mean=4.56e-01. Clamping to 1e-6.
```

## Testing

Run the test script:
```bash
python investigations/batch3_silent_failures/test_warnings.py
```

Run benchmark to verify no regression:
```bash
python run_single_mode.py --mode vargp_style --explicit-unwhitening --seed 42
```

## Cleanup

To remove investigation files after merging:
```bash
rm -rf investigations/batch3_silent_failures/
```
