# vargp_direct Unit Tests

**File**: `test_vargp_direct_match.py`

**Purpose**: Verify that the `vargp_direct` implementation (with analytical M-step) produces correct gradients and matches `vargp_old` results.

---

## Background

A bug was fixed in `direct_vargp.py` where stale eigenvalues were used in the M-step closure. The fix uses `torch.linalg.solve()` instead of eigendecomposition for computing `K_tilde_inv_b`. These tests verify the fix and confirm that previously suspected bugs (BUG #3, #4) are not actually bugs.

---

## Tests

### Test 1: Softplus Chain Rule (`test_softplus_chain_rule`)

**What it tests**: The chain rule transformation for softplus-parameterized variables.

**Why it matters**: `sigma_0` and `Amp` are stored as raw parameters that go through softplus transform. The analytical gradient `dL['sigma_0']` is computed w.r.t. the transformed value, then multiplied by `sigmoid(raw)` to get the gradient w.r.t. the raw parameter.

**Method**: Computes analytical gradient and compares to finite differences (ground truth).

**Acceptance**: rel_err < 1e-4

---

### Test 2: Eigenspace Projection (`test_eigenspace_projection_gradients`)

**What it tests**: That projecting gradients to eigenspace preserves correctness.

**Why it matters**: The M-step uses `dK_tilde_b = B.T @ dK_tilde @ B` to project gradient matrices to the reduced eigenspace. This must preserve the gradient structure.

**Method**: For each of 6 hyperparameters, computes analytical projected gradient and compares to finite differences.

**Acceptance**: rel_err < 1e-4 for all parameters

---

### Test 3: K_tilde_inv Methods (`test_ktilde_inv_methods`)

**What it tests**: That `solve()` gives correct inverse, and that K_tilde_b becomes non-diagonal after hyperparameter changes.

**Why it matters**: The original bug used stale eigenvalues to compute K_tilde_inv. The fix uses `solve()` which works for any symmetric positive definite matrix, not just diagonal ones.

**Method**:
1. At initialization: verify eigenvalue-based and solve-based inverses match
2. After changing hyperparameters: verify K_tilde_b is NOT diagonal (justifying need for solve)
3. Verify solve produces valid inverse (K @ K^-1 = I)

**Acceptance**: Initial methods match (rel_err < 1e-6), K_tilde_b becomes non-diagonal (off-diag ratio > 1%)

---

### Test 4: Gradient Magnitude Sanity (`test_gradient_magnitude_sanity`)

**What it tests**: That analytical gradients have reasonable magnitudes.

**Why it matters**: Catches bugs like missing scale factors or wrong signs that produce gradients orders of magnitude off.

**Method**: Computes analytical gradients for all hyperparameters, checks for NaN/Inf and reasonable magnitude range.

**Acceptance**: No NaN/Inf, magnitudes in reasonable range (not < 1e-20 or > 1e20)

---

### Test 5: Multi-cell Comparison (`test_multicell_match`)

**What it tests**: End-to-end match between `vargp_direct` and `vargp_old` on multiple cells.

**Why it matters**: The ultimate test - if test_r values match, the implementations are functionally equivalent.

**Method**: Runs both modes via subprocess on cells 6, 8, 15 and compares Test Pearson r.

**Acceptance**: test_r difference < 0.05 for each cell

**Note**: This test is slow (~3-5 minutes). Use `--skip-slow` to skip it.

---

### Tests 6-7: Not Implemented

Tests for initialization match and single E-step match are placeholders for future implementation if needed.

---

## Usage

```bash
# Run all tests
python tests/test_vargp_direct_match.py

# Run fast tests only (skip multi-cell)
python tests/test_vargp_direct_match.py --skip-slow

# Run individual test
python tests/test_vargp_direct_match.py --test 1

# Verbose output
python tests/test_vargp_direct_match.py --test 1 --verbose
```

---

## Results Summary (January 2025)

| Test | Status | Key Finding |
|------|--------|-------------|
| 1. Softplus chain rule | PASS | Chain rule correctly implemented |
| 2. Eigenspace projection | PASS | Gradients preserved correctly |
| 3. K_tilde_inv methods | PASS | solve() fix is correct |
| 4. Gradient magnitude | PASS | All gradients reasonable |
| 5. Multi-cell match | PASS | Matches vargp_old within 0.01 |

---

## Related Files

- `direct_vargp.py` - Implementation being tested
- `.claude/VARGP_DIRECT_INVESTIGATION.md` - Full investigation context
