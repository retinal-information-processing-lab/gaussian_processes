# Potential Issues Reference Document

**Created**: 2026-01-23
**Purpose**: Comprehensive catalog of potential bugs, edge cases, and concerns identified during codebase review. This document serves as a reference for systematic investigation.

**Status**: REFERENCE ONLY - Issues listed here have NOT been verified as actual bugs. Each requires investigation before any fix is attempted.

---

## Table of Contents

1. [Category A: Reproducibility & Random State Issues](#category-a-reproducibility--random-state-issues)
2. [Category B: Jitter Consistency Issues](#category-b-jitter-consistency-issues)
3. [Category C: Stability Threshold Inconsistencies](#category-c-stability-threshold-inconsistencies)
4. [Category D: Edge Cases Not Handled](#category-d-edge-cases-not-handled)
5. [Category E: Code Quality / Maintenance Issues](#category-e-code-quality--maintenance-issues)
6. [Category F: Algorithmic / Mathematical Concerns](#category-f-algorithmic--mathematical-concerns)
7. [Category G: Performance / Efficiency Concerns](#category-g-performance--efficiency-concerns)
8. [Category H: Documentation / API Concerns](#category-h-documentation--api-concerns)
9. [Category I: Bounds and Constraints](#category-i-bounds-and-constraints)
10. [Category J: Potential Silent Failures](#category-j-potential-silent-failures)
11. [Priority Matrix](#priority-matrix)

---

## Category A: Reproducibility & Random State Issues

These issues affect the ability to reproduce results across runs, sessions, or environments.

### A1. The `torch.pi` Mystery

**Location**: `tests/test_utils.py:set_reproducible_seed()`

**Code in question**:
```python
torch.pi = torch.acos(torch.zeros(1)).item() * 2  # WHY DOES THIS MATTER?!
```

**What it does**: Reassigns `torch.pi` (a built-in constant since PyTorch 1.8) to a computed value that should be identical to the existing value.

**The problem**: This line somehow affects the random state. Without it, `test_kernel_cache.py` fails while `run_single_mode.py` succeeds - same code, different random sequences.

**Why this is concerning**:
- This is explicitly documented as "cargo cult programming" - the developers do not understand why it works
- The effect is reproducible but unexplained
- If PyTorch internals change, this could silently break reproducibility
- It replicates a side effect from `GP_utils.py` line 49, suggesting the original code also had this mystery

**Impact if triggered**: Silent changes in results when PyTorch is updated or when the line is removed during refactoring.

**Severity**: HIGH

**Investigation needed**:
- Determine if the line actually changes `torch.pi` or has a side effect
- Check if it triggers some lazy initialization in PyTorch
- Test with different PyTorch versions

---

### A2. `set_reproducible_seed` Device Parameter Issue

**Location**: `tests/test_utils.py:set_reproducible_seed(seed, device=device)`

**The problem**: Calling `set_reproducible_seed(seed, device=device)` produces DIFFERENT random sequences than calling `set_reproducible_seed(seed)` (without the device parameter), even though the device parameter defaults to 'cuda' internally.

**Observed impact**:
- Different inducing point selection
- Dramatically different results: r=0.70 vs r=-0.11 (documented in CLAUDE.md)

**Why this is concerning**:
- `run_single_mode.py` uses the device parameter
- Inline tests may not use the device parameter
- This creates silent result mismatches when comparing outputs

**Suspected cause**: The device parameter may trigger `torch.cuda.init()` at a different point in the execution, affecting the random state. However, this is speculation.

**Severity**: HIGH

**Investigation needed**:
- Create minimal reproduction case
- Trace exactly what differs between the two call patterns
- Document the correct calling convention

---

### A3. Whitened Mode is Seed-Sensitive

**Location**: Documented in CLAUDE.md as known limitation

**The problem**: Results vary more with seed when using whitened variational strategy compared to unwhitened.

**Why this is concerning**:
- Makes debugging harder (can't tell if result change is from code change or seed sensitivity)
- Makes validation harder (need more seeds to establish baseline)
- Unclear if this is inherent to whitened parameterization or indicates a bug

**Severity**: MEDIUM

**Investigation needed**:
- Quantify the variance across seeds for whitened vs unwhitened
- Determine if this matches theoretical expectations
- Check if original varGP has similar seed sensitivity

---

## Category B: Jitter Consistency Issues

Jitter is added for numerical stability in Cholesky decompositions. The documentation explicitly states all jitter values MUST match `model.jitter`.

### B1. Hardcoded Jitter in Cholesky Fallback (whitening.py)

**Locations**:
- `whitening.py:170` in `update_variational_covar()`
- `whitening.py:252` in `update_variational_covar_with_L_K()`

**Code in question**:
```python
# In update_variational_covar (line 170):
L_new = torch.linalg.cholesky(V_new + 1e-6 * eye)

# In update_variational_covar_with_L_K (line 252):
L_whitened = torch.linalg.cholesky(V_whitened + 1e-6 * eye)
```

**The problem**: Both use hardcoded `1e-6` jitter, but `model.jitter` is typically `1e-4`. This is a 100x difference.

**Why this is concerning**:
- CLAUDE.md Section "Jitter Consistency" explicitly documents that mismatched jitter caused test_r=0.21 instead of expected ~0.77
- The documentation says: "All jitter values MUST match model.jitter"
- These fallback paths violate that rule

**When triggered**: Only when Cholesky fails on the first attempt (V is near-singular).

**Severity**: MEDIUM (only triggers on edge cases, but when it does, could cause incorrect results)

**Investigation needed**:
- Determine if these fallback paths are ever reached in practice
- If yes, change to use `model.jitter` instead of hardcoded `1e-6`

---

### B2. Jitter in update_variational_parameters

**Location**: `whitening.py:283`

**Code in question**:
```python
L_new = torch.linalg.cholesky(V_new + jitter * eye)
```

**The problem**: Uses the `jitter` parameter passed to the function. If caller passes a different value than `model.jitter`, inconsistency occurs.

**Mitigating factor**: Upstream code should use `_validate_jitter()` which enforces consistency.

**Why still concerning**: The function signature allows passing arbitrary jitter, creating an API that's easy to misuse.

**Severity**: LOW (mitigated by validation upstream)

---

## Category C: Stability Threshold Inconsistencies

The code uses threshold checks to detect numerical instability (exploding firing rates).

### C1. E-step vs F-step Stability Thresholds

**Locations**:
- `estep.py:629`: `if f_mean.mean() > 1000:`
- `fstep.py:184`: `if f_mean.mean() > 100 or torch.any(torch.isnan(f_mean)):`

**The problem**: 10x difference in thresholds (1000 vs 100) for the same instability concept.

**Why this is concerning**:
- F-step might return inf and halt optimization while E-step continues
- Or E-step might continue with unstable values that F-step would have caught
- The thresholds represent the same concept (firing rate explosion) but have different values

**Note**: F-step also checks for NaN, E-step does not.

**Severity**: MEDIUM

**Investigation needed**:
- Determine which threshold is appropriate
- Unify the thresholds
- Consider adding NaN check to E-step

---

### C2. Hardcoded Magic Numbers

**Locations**:
- `f_mean.mean() > 1000` (estep.py:629)
- `f_mean.mean() > 100` (fstep.py:184)
- `rel_change < 1e-5` convergence threshold (estep.py:640)
- `f_mean.mean() > 1000` stability check (estep.py:686, 726)

**The problem**: These thresholds are hardcoded without clear rationale or documentation of why these specific values were chosen.

**Why this is concerning**:
- May not generalize to datasets with different firing rate ranges
- A neuron with high baseline firing rate might legitimately have f_mean > 100

**Severity**: LOW (current dataset works fine)

**Investigation needed**:
- Document the rationale for these values
- Consider making them configurable or relative to data statistics

---

## Category D: Edge Cases Not Handled

### D1. Zero Spike Count in Training Data

**Location**: `fstep.py:49-52` in `lambda0_given_A()`

**Code in question**:
```python
sumr = r.sum()
expexpr = torch.exp(A * lambda_m + 0.5 * A * A * lambda_var)
sumexpr = expexpr.sum()
return torch.log(sumr) - torch.log(sumexpr)
```

**The problem**: If `sum(r) = 0` (no spikes in training batch), `torch.log(sumr)` returns `-inf`.

**When this could happen**:
- Very sparse neural data
- Small batch sizes
- Neurons with very low firing rates
- Specific time windows with no activity

**Impact**: `lambda0` becomes `-inf`, propagates through computations

**Severity**: MEDIUM

**Investigation needed**:
- Add check for `sumr > 0`
- Decide on appropriate behavior (error? default value? skip update?)

---

### D2. All Pixels Masked Out

**Location**: `kernels.py:compute_mask()` and `_compute_C_matrix()`

**The problem**: If RF center is far outside image bounds and beta is large, the mask could include zero pixels.

**Code flow**:
```python
mask = alpha >= self.MASK_THRESHOLD  # Could be all False
...
xcord = xcord[mask]  # Empty tensor
ycord = ycord[mask]  # Empty tensor
```

**When this could happen**:
- RF center initialized or learned to be outside [-1, 1] bounds
- Very small beta (large RF that's centered far away)

**Impact**: Empty tensors, dimension mismatch errors downstream

**Severity**: LOW (bounds clamping should prevent this)

**Mitigating factor**: `clamp_hyperparameters()` constrains eps_0x, eps_0y to [-1, 1]

**Investigation needed**:
- Verify clamp is always called before mask computation
- Consider adding explicit check for empty mask

---

### D3. Near-Zero Variance Clamping

**Location**: `estep.py:192`

**Code in question**:
```python
lambda_var = torch.clamp(lambda_var, min=1e-6)
```

**The problem**: Silently clamps variance to small positive value. Negative variance (which shouldn't happen mathematically) would be hidden.

**Why this is concerning**:
- Negative variance indicates serious numerical issues
- The clamp masks the problem rather than surfacing it
- `1e-6` is hardcoded without clear rationale

**Severity**: LOW (defensive programming, but could mask issues)

**Investigation needed**:
- Add warning when clamping is triggered
- Consider if `1e-6` is appropriate for all scales

---

## Category E: Code Quality / Maintenance Issues

### E1. Dead Import in likelihoods.py

**Location**: `likelihoods.py:59`

**Code**:
```python
from gpytorch.constraints import Interval
```

**The problem**: `Interval` is imported but never used.

**Why this matters**: Suggests incomplete refactoring. Someone intended to use Interval constraints (probably for A or lambda0) but didn't complete the implementation.

**Severity**: LOW (no runtime impact)

**Fix**: Remove the unused import

---

### E2. Test Function Signature Mismatch

**Location**: `model.py:143`

**Code in question**:
```python
model = VariationalGPModel(inducing_points, kernel)
```

**The problem**: The `__init__` signature requires `jitter` and `standard_variational_distribution`:
```python
def __init__(self, inducing_points, kernel, jitter, standard_variational_distribution, ...):
```

**Impact**: The test function `test_model()` would fail if run.

**Severity**: LOW (test is broken, but doesn't affect runtime)

**Fix**: Update test to pass required arguments

---

### E3. Inconsistent Default Parameter Handling

**Observation**: Some functions have required parameters with comments like "Required - no default to prevent silent bugs":
```python
def f_step_lbfgs(..., lr: float, ...):  # Required - no default
```

**The problem**: This good practice is not consistently applied. Some critical parameters might have implicit defaults.

**Severity**: LOW

**Investigation needed**: Audit all critical parameters for appropriate default handling

---

## Category F: Algorithmic / Mathematical Concerns

### F1. Diagonal Kernel Assumption

**Status**: **CLOSED - NOT A BUG** (2026-01-23, Batch 2 investigation)

**Location**: `kernels.py:480-484`

**Code in question**:
```python
if diag:
    # Diagonal case: K(x_i, x_i)
    # When x1 = x2, cos(theta) = 1, theta = 0, J(0) = pi
    # K = M * pi / pi = M = sqrt(v_x * v_x) = v_x
    return V1
```

**Original concern**: When `diag=True`, the code returns `V1` and completely ignores `x2`.

**Investigation findings**:
1. GPyTorch's documented contract explicitly states: "If `diag=True`, it must be the case that `x1 == x2`."
2. All calls in our codebase use `kernel(X, diag=True)` with single argument - GPyTorch auto-sets `x2=x1`
3. The implementation correctly assumes GPyTorch's contract
4. Test verification shows diagonal matches full matrix diagonal (diff ~7e-6 = numerical precision)

**Conclusion**: This is correct behavior following GPyTorch's API contract. No code changes needed.

---

### F2. Mask Computed with Detached Parameters

**Location**: `kernels.py:340-346`

**Code in question**:
```python
with torch.no_grad():
    beta_detached = torch.exp(self.raw_m2log2beta.detach())
    dist_sq = (xcord - self.eps_0x.detach())**2 + (ycord - self.eps_0y.detach())**2
    logalpha = -beta_detached * dist_sq
    alpha = torch.exp(logalpha)
    mask = alpha >= self.MASK_THRESHOLD
```

**Design decision**: Mask is computed with DETACHED parameters (no gradient flow).

**Why this was done**: Documented in Q22 of DECISION_LOG.md as "structural stability" - prevents mask from changing during gradient computation.

**The concern**: If beta or epsilon change significantly during M-step, the mask becomes stale until the next iteration. Pixels that should now be included (or excluded) continue with the old mask.

**Severity**: MEDIUM

**Mitigating factor**: Hyperparameters are bounded and typically change slowly.

**Investigation needed**:
- Check how much beta/epsilon typically change per iteration
- Determine if stale mask causes measurable error

---

### F3. Symmetrization in Newton Update

**Location**: `estep.py:239`

**Code**:
```python
V_new = (V_new + V_new.T) / 2
```

**Why this exists**: Ensures V is exactly symmetric after numerical operations.

**The concern**: If V_new is significantly asymmetric before this line, it indicates numerical problems upstream. The symmetrization masks the issue.

**Severity**: LOW (defensive programming)

**Investigation needed**: Add diagnostic to check asymmetry magnitude before symmetrization

---

### F4. M-step Skipped on Last Iteration

**Location**: `train.py:395`

**Code**:
```python
if n_mstep > 0 and iteration < n_iterations - 1:
```

**Design decision**: Kernel parameters are NOT updated on the final iteration.

**Rationale from original varGP**: "to avoid generating a new eigenspace that will not be used by V and m"

**The concern**: The final model's kernel doesn't reflect the final E-step's variational parameters. If someone extracts the kernel for other purposes, it's one iteration behind.

**Severity**: LOW (documented behavior matching original)

---

## Category G: Performance / Efficiency Concerns

### G1. Non-Cached Path Inefficiency

**Location**: `estep.py:670-703` (non-cached whitening path)

**The problem**: When `kernel_cache=None` and `explicit_unwhitening=True`, the code:
1. Calls `compute_L_K()` to get Cholesky factor
2. Calls `model(X)` multiple times per Newton iteration
3. Each `model(X)` recomputes kernel matrices internally

**Why this is inefficient**: The cached path computes K, K_tilde once and reuses. The non-cached path recomputes them repeatedly.

**When this path is used**: When `use_cache=False` is passed to `train_varGP_style()`.

**Severity**: LOW (cached path is default)

**Note**: Documented performance difference is 8.8x speedup with caching.

---

### G2. Repeated Kernel Computation in Default GPyTorch Mode

**Location**: `train.py:77`

**Code**:
```python
output = model(train_x)  # Called every iteration
```

**The problem**: `train_gpy_default()` doesn't use kernel caching. Every iteration recomputes kernels.

**Severity**: LOW (this mode is for comparison, not production use)

---

## Category H: Documentation / API Concerns

### H1. Cache Clearing Documentation Asymmetry

**Location**: `whitening.py:292-303`

**Documentation states**:
```python
"""Clear GPyTorch's memoized cache for variational covariance.

MUST be called after updating chol_variational_covar.
NOT needed after updating variational_mean (mean updates are immediate).
"""
```

**The concern**: When both m and V are updated together (common case), the interaction is implicit. The caller must know to call `clear_variational_cache()` even though only V "needs" it.

**Severity**: LOW (existing code handles this correctly)

---

### H2. Gradient Mode Validation Asymmetry

**Location**: `kernels.py:217-218`

**Code**:
```python
if gradient_mode in ('vjp', 'jacobian') and n_px_side is None:
    raise ValueError(f"gradient_mode='{gradient_mode}' requires n_px_side to be set")
```

**The problem**: `vjp` and `jacobian` modes require `n_px_side`, but `autograd` works without it.

**Why this could confuse users**: Switching from autograd to vjp might fail unexpectedly if n_px_side wasn't set.

**Severity**: LOW (error message is clear)

---

### H3. Unclear Amp vs ScaleKernel Distinction

**Location**: `kernels.py` docstrings

**The documentation says**: Amp "is different from ScaleKernel which scales output linearly"

**The distinction**:
- `Amp` multiplies C BEFORE sqrt/arccos operations (non-linear effect)
- `ScaleKernel` multiplies K AFTER computation (linear scaling)

**Why this could confuse users**: GPyTorch users expect `ScaleKernel` patterns. Using `Amp` instead is a departure from standard GPyTorch patterns.

**Severity**: LOW (documented, but subtle)

---

## Category I: Bounds and Constraints

### I1. Amp Max Bound Rationale

**Location**: `kernels.py:124`

**Code**:
```python
AMP_MAX = 1000.0
```

**The problem**: No documented rationale for why 1000.0 was chosen.

**Questions**:
- Is this based on empirical observation?
- Does it relate to numerical stability?
- Would a different dataset need a different bound?

**Severity**: LOW

---

### I2. RF Parameter Bounds

**Location**: `kernels.py:127-138`

**Code**:
```python
BETA_MIN = 0.01
BETA_MAX = 1.0
RHO_MIN = 0.01
RHO_MAX = 0.5
```

**The problem**: These bounds are hardcoded, presumably based on what works for the PNAS dataset.

**Concern**: May not generalize to:
- Different image sizes
- Different neuron types
- Different experimental conditions

**Severity**: LOW (current use case works)

---

### I3. Epsilon Bounds vs Image Coordinates

**Location**: `kernels.py:142-143`

**Code**:
```python
EPS_MIN = -1.0
EPS_MAX = 1.0
```

**The constraint**: RF center is clamped to within the image bounds.

**Potential issue**: Some neurons might have RF centers slightly outside the image. Clamping forces them to the edge.

**Severity**: LOW (edge case)

---

## Category J: Potential Silent Failures

### J1. LBFGS Handling of Inf Return

**Status**: **FIXED** (2026-01-23, Batch 3 investigation)

**Location**: `fstep.py:f_step_lbfgs()`

**Original problem**: Closure returns `inf` on instability, but no tracking/warning of failure.

**Fix**: Added instability tracking and warning:
- Track A_before and A_after optimization
- Track when inf is returned (f_mean > 1000 or NaN)
- Emit warning with observed values: f_mean value, A before/after

**Warning format**:
```
F-step: instability detected (f_mean=1234.5). A unchanged: 0.0123.
F-step: instability detected (f_mean=1234.5). A changed: 0.0100 -> 0.0123.
```

**Severity**: MEDIUM → RESOLVED

---

### J2. Cholesky Failure Recovery

**Status**: **FIXED** (2026-01-23, Batch 3 investigation)

**Locations** (5 total):
- `estep.py:compute_kernel_cache()` - K_tilde Cholesky
- `estep.py:compute_L_K()` - K_tilde Cholesky
- `whitening.py:update_variational_covar()` - V Cholesky
- `whitening.py:update_variational_covar_with_L_K()` - V_whitened Cholesky
- `whitening.py:update_variational_parameters()` - V Cholesky

**Original problem**: Cholesky failure was silently recovered by adding jitter. User didn't know V was modified.

**Fix**: Added warning with observed values to all 5 locations:
- Report matrix shape and minimum eigenvalue
- Report jitter value being added

**Warning format**:
```
Cholesky failed on V in {function}() (shape=(100, 100), min_eigenvalue=-1.23e-06). Adding jitter=1.0e-04 and retrying.
```

**Also fixed**: Added fallback to `estep.py` functions that previously would crash on Cholesky failure.

**Severity**: MEDIUM → RESOLVED

---

### J3. Variance Clamping

**Status**: **FIXED** (2026-01-23, Batch 3 investigation)

**Location**: `estep.py:compute_moments_from_kernel_cache()` line ~205

**Original problem**: Negative variance was silently clamped to `1e-6`.

**Fix**: Added warning with observed values before clamping:
- Count of negative values
- Minimum value
- Mean value

**Warning format**:
```
Negative variance detected: 42/500 values. min=-1.23e-05, mean=4.56e-01. Clamping to 1e-6.
```

**Root causes** (for user debugging):
- Ill-conditioned K_tilde matrix
- V not positive definite
- Jitter mismatch

**Severity**: MEDIUM → RESOLVED

---

## Priority Matrix

| ID | Category | Severity | Likelihood | Recommended Action |
|----|----------|----------|------------|-------------------|
| A1 | Reproducibility | HIGH | Unknown | Investigate urgently |
| A2 | Reproducibility | HIGH | Confirmed | Document and fix |
| A3 | Reproducibility | MEDIUM | Confirmed | Quantify and document |
| B1 | Jitter | MEDIUM | Low | Fix (simple change) |
| B2 | Jitter | LOW | Low | Document API risk |
| C1 | Stability | MEDIUM | Medium | Unify thresholds |
| C2 | Stability | LOW | Low | Document rationale |
| D1 | Edge Case | MEDIUM | Low | Add guard clause |
| D2 | Edge Case | LOW | Very Low | Verify bounds prevent |
| D3 | Edge Case | LOW | Low | Add warning |
| E1 | Code Quality | LOW | N/A | Remove dead code |
| E2 | Code Quality | LOW | N/A | Fix test |
| E3 | Code Quality | LOW | N/A | Audit parameters |
| F1 | Algorithm | MEDIUM | Unknown | Verify GPyTorch usage |
| F2 | Algorithm | MEDIUM | Low | Quantify staleness |
| F3 | Algorithm | LOW | Low | Add diagnostic |
| F4 | Algorithm | LOW | N/A | Document (intentional) |
| G1 | Performance | LOW | N/A | Document (known) |
| G2 | Performance | LOW | N/A | Document (known) |
| H1 | Documentation | LOW | Low | Clarify docs |
| H2 | Documentation | LOW | Low | Clarify docs |
| H3 | Documentation | LOW | Low | Clarify docs |
| I1 | Bounds | LOW | Low | Document rationale |
| I2 | Bounds | LOW | Low | Document rationale |
| I3 | Bounds | LOW | Very Low | Consider relaxing |
| J1 | Silent Failure | MEDIUM | Low | **FIXED** (2026-01-23 Batch 3) |
| J2 | Silent Failure | MEDIUM | Low | **FIXED** (2026-01-23 Batch 3) |
| J3 | Silent Failure | MEDIUM | Low | **FIXED** (2026-01-23 Batch 3) |

---

## Recommended Investigation Order

### Batch 1: High-Priority Reproducibility Issues
- A1: torch.pi mystery
- A2: set_reproducible_seed device parameter issue

### Batch 2: Medium-Priority Correctness Issues
- B1: Jitter consistency in whitening.py
- C1: Stability threshold inconsistency
- F1: Diagonal kernel assumption
- D1: Zero spike count handling

### Batch 3: Silent Failure Detection (**COMPLETE** 2026-01-23)
- J1: LBFGS inf handling → **FIXED** (instability tracking + warning)
- J2: Cholesky failure warnings → **FIXED** (5 locations with eigenvalue reporting)
- J3: Variance clamping warnings → **FIXED** (count/min/mean reporting)

### Batch 4: Code Quality Cleanup
- E1: Dead import removal
- E2: Fix broken test
- Document rationale for all hardcoded values

---

## Notes for Investigation

When investigating each issue:

1. **Create a minimal reproduction case** if possible
2. **Document current behavior** before any changes
3. **Verify the issue is real**, not just theoretical
4. **Consider backward compatibility** - some "bugs" might be relied upon
5. **Add tests** that would catch regression
6. **Update CLAUDE.md** with findings

---

*This document will be updated as issues are investigated and resolved.*
