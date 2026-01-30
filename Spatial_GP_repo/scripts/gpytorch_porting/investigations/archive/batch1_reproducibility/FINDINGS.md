# Findings: Batch 1 Reproducibility Investigation

**Date**: 2026-01-23
**Investigator**: Claude (Opus 4.5)
**PyTorch Version**: 2.5.1+cu121

---

## Executive Summary

**Surprising Result**: Neither the `torch.pi` line nor the device parameter affects random state in our isolated tests. All 7 test scripts produced **identical random sequences**.

This suggests the documented issues were either:
1. Fixed in PyTorch 2.5.1
2. Only manifest in specific edge cases we didn't reproduce
3. Related to training loop dynamics, not just seeding

---

## Test Results

| Test | Description | Checksum |
|------|-------------|----------|
| A | Baseline (nothing special) | 3.9199287802 |
| B | With `torch.pi` line | 3.9199287802 |
| C | Just `torch.zeros(1)` | 3.9199287802 |
| D | With `cuda.init()` | 3.9199287802 |
| E | `set_reproducible_seed(42)` no device | 3.9199287802 |
| F | `set_reproducible_seed(42, device='cuda')` | 3.9199287802 |
| G | Import utils.py before seeding | 3.9199287802 |

**All tests produce the same random sequence:**
```
randn:    [0.194, 2.161, -0.172, 0.849, -1.924]
randperm: [31, 11, 6, 91, 74]
numpy:    [0.374, 0.951, 0.732, 0.599, 0.156]
```

---

## Issue A1: torch.pi Mystery

**Hypothesis tested**: `torch.pi = torch.acos(torch.zeros(1)).item() * 2` affects random state

**Result**: **NOT CONFIRMED** in our tests

**Details**:
- Tests A vs B: Identical results (baseline vs with torch.pi line)
- Tests A vs C: Identical results (baseline vs just tensor creation)
- The torch.pi line does NOT change random state in PyTorch 2.5.1

**Original context**: The archive mentions this issue was observed when `test_kernel_cache.py` failed while `test_estep_pnas.py` succeeded. The difference was the training trajectory collapsing at iteration 30-40, not just different random numbers.

---

## Issue A2: Device Parameter Issue

**Hypothesis tested**: `set_reproducible_seed(42, device='cuda')` produces different results than `set_reproducible_seed(42)`

**Result**: **NOT CONFIRMED** in our tests

**Details**:
- Tests E vs F: Identical results
- Passing device parameter explicitly vs using default produces same random sequence

---

## Possible Explanations

### 1. Fixed in PyTorch 2.5.1

The original issues were documented in January 2026. PyTorch version 2.5.1 may have fixed subtle initialization bugs that caused random state sensitivity.

### 2. Edge Case Not Reproduced

The original failure involved:
- 50 training iterations with E-step Newton updates
- Numerical instability building up over iterations
- Collapse at iteration 30-40

Our tests only checked random sequences, not long training runs. The issue might be:
- Floating point accumulation
- GPU kernel caching differences
- Specific data patterns triggering instability

### 3. The Workaround Was Cargo Cult All Along

The `torch.pi` line in `utils.py` may have been unnecessary even when it was written. The issue might have been something else entirely (import order, unrelated side effects) and the `torch.pi` fix was a coincidental correlation.

---

## Recommendations

### Option 1: Keep the Workaround (Conservative)

Keep the `torch.pi` line and `cuda.init()` calls as-is. They don't hurt anything and provide a safety margin.

**Pros**: No risk of breaking anything
**Cons**: Keeps "cargo cult" code in the codebase

### Option 2: Remove the Workaround (Clean)

Since we can't reproduce the issue, remove the mysterious code and simplify `set_reproducible_seed()`.

**Proposed simplified version**:
```python
def set_reproducible_seed(seed: int = 42):
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
```

**Risk**: If the issue was real and edge-case specific, removing the workaround might cause subtle failures in specific scenarios.

### Option 3: Keep But Document (Compromise)

Keep the code but update documentation to note that:
- We could not reproduce the issue in 2026-01-23 testing
- The workaround may be unnecessary in PyTorch 2.5.1+
- Monitor for any reproducibility issues

---

## Files Created

Scripts for cleanup later:
1. `test_A_baseline.py`
2. `test_B_with_torch_pi.py`
3. `test_C_just_zeros.py`
4. `test_D_with_cuda_init.py`
5. `test_E_seed_no_device.py`
6. `test_F_seed_with_device.py`
7. `test_G_import_utils.py`
8. `run_all_tests.sh`
9. `investigate_torch_pi.py` (uses subprocess - may delete)
10. `investigate_device_param.py` (uses subprocess - may delete)
11. `README.md`
12. `FINDINGS.md` (this file)

---

## Next Steps

**Decision needed**: Which option to proceed with?

If we choose Option 2 (remove workaround), we should:
1. Run the full test suite to verify no regression
2. Run `run_single_mode.py` with multiple seeds
3. Run `test_kernel_cache.py` to check the original failure scenario
