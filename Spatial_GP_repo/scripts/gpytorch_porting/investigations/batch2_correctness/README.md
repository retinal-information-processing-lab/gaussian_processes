# Batch 2 Correctness Issues Investigation

**Date**: 2026-01-23
**Branch**: `multiple_bugs_batch2`
**Issues addressed**: B1, C1, D1, F1 from `VARIOUS_POSSIBLE_BUGS.md`

---

## Overview

This investigation addresses medium-priority correctness issues identified in the GPyTorch variational GP implementation:

| Issue | Type | Status |
|-------|------|--------|
| **B1** | Jitter consistency | BUG - needs fix |
| **C1** | Threshold inconsistency | INCONSISTENCY - needs unification |
| **D1** | Zero spike handling | EDGE CASE - needs guard |
| **F1** | Diagonal kernel | NOT A BUG - document only |

---

## Issue Details

### B1: Jitter Consistency
- **Files**: `whitening.py:170, 252`
- **Problem**: Hardcoded `1e-6` in fallback paths vs `model.jitter=1e-4`
- **Fix**: Use `model.jitter` instead

### C1: Stability Threshold Inconsistency
- **Files**: `estep.py:629,686,726` and `fstep.py:184`
- **Problem**: E-step uses 1000, F-step uses 100; E-step lacks NaN check
- **Fix**: Unify to 1000, add NaN check to E-step

### D1: Zero Spike Count
- **File**: `fstep.py:52`
- **Problem**: `log(0) = -inf` when `sum(r)=0`
- **Fix**: Add guard clause with ValueError

### F1: Diagonal Kernel Assumption
- **File**: `kernels.py:480`
- **Status**: NOT A BUG - GPyTorch guarantees `x1==x2` when `diag=True`
- **Action**: Document in VARIOUS_POSSIBLE_BUGS.md

---

## Files Created

- `README.md` (this file) - Investigation overview
- `test_batch2_issues.py` - Reproduction tests
- `FINDINGS.md` - Results and conclusions (created after fixes)

---

## Verification Commands

```bash
# Run reproduction tests
python investigations/batch2_correctness/test_batch2_issues.py

# Regression test
python run_single_mode.py --mode vargp_style --explicit-unwhitening --seed 42

# Additional seed for stability check
python run_single_mode.py --mode vargp_style --explicit-unwhitening --seed 123
```
