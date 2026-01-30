# M-step Analytical Gradients Implementation - Handoff Prompt

**Created**: January 2025
**Purpose**: Comprehensive handoff for implementing analytical M-step gradients in vargp_direct mode

---

## CRITICAL: Read These Files First (In Order)

### 1. Working Guidelines (MUST READ)
```
/home/idv-eqs8-pza/IDV_code/ClosedLoopProject/gaussian_processes/Spatial_GP_repo/scripts/gpytorch_porting/.claude/WORKING_GUIDELINES.md
```
Contains development rules, policies, and process requirements you MUST follow.

### 2. Project Context (Main Documentation)
```
/home/idv-eqs8-pza/IDV_code/ClosedLoopProject/gaussian_processes/Spatial_GP_repo/scripts/gpytorch_porting/.claude/CLAUDE.md
```
Main project documentation with architecture, decisions, current state, and quick start.

### 3. vargp_direct Context (Implementation Details)
```
/home/idv-eqs8-pza/IDV_code/ClosedLoopProject/gaussian_processes/Spatial_GP_repo/scripts/gpytorch_porting/.claude/VARGP_COPY_CONTEXT.md
```
Contains exact formulas for dlambda, dKL, diagonal gradients, and eigenspace projection details.

### 4. Implementation Plan (YOUR TASK)
```
/home/idv-eqs8-pza/.claude/plans/mellow-finding-kitten.md
```
Detailed implementation plan with:
- TDD test structure (8 tests)
- Numerical guardrails (8 guardrails)
- Exact gradient formulas (verbatim from utils.py)
- Success criteria and validation commands

---

## Task Overview

**Goal**: Optimize the M-step in `vargp_direct` mode to match original varGP performance by implementing analytical gradients.

**Current Problem**:
- `vargp_direct` M-step takes 17.3s (uses autograd, recomputes kernels every LBFGS closure call)
- `vargp_old` M-step takes 4.8s (uses analytical gradients, computes dK matrices ONCE)

**Target**: M-step time ≤7s with test_r within 0.02 of vargp_old

**Approach**: Port the analytical gradient computation from `utils.py:acosker()` with `grad=True` to compute explicit dK/dθ matrices that can be cached and reused in LBFGS closure.

---

## Codebase Orientation

### Directory Structure
```
gpytorch_porting/
├── .claude/
│   ├── CLAUDE.md                    # Main project docs
│   ├── VARGP_COPY_CONTEXT.md        # vargp_direct details + exact formulas
│   ├── WORKING_GUIDELINES.md        # Development rules
│   └── MSTEP_ANALYTICAL_HANDOFF.md  # This file
├── kernels.py                       # ArcCosineKernel (use for reference, don't modify much)
├── direct_vargp.py                  # ← MAIN FILE TO MODIFY (add analytical M-step)
├── eigenspace.py                    # Eigenspace projection utilities
├── analytical_gradients.py          # Existing Jacobian-based gradients (REFERENCE)
├── analytical_gradients_vjp.py      # Existing VJP gradients (REFERENCE, but can't use for M-step)
├── run_single_mode.py               # CLI entry point (add --mstep-analytical flag)
├── tests/
│   └── test_mstep_analytical.py     # ← CREATE THIS (TDD tests)
└── results/
    └── benchmark_results.jsonl      # Benchmark output
```

### Key Reference Files (Original Implementation)
```
/home/idv-eqs8-pza/IDV_code/ClosedLoopProject/gaussian_processes/Spatial_GP_repo/utils.py
```
- Lines 3577-3631: `localker()` - C matrix and dC gradients
- Lines 3663-3813: `acosker()` - K matrix and dK gradients (CRITICAL)
- Lines 3906-3956: `lambda_moments()` - dlambda_m, dlambda_var
- Lines 4081-4119: `compute_loglikelihood()` - dloglikelihood
- Lines 4121-4152: `compute_KL_div()` - dKL
- Lines 5859-5963: M-step closure implementation

### Mathematical Reference
```
/home/idv-eqs8-pza/IDV_code/Papers/latex_summaries/acosker_kernel_def_and_gradients.tex
```
LaTeX derivations for kernel gradients (already verified against code).

---

## Critical Information NOT in Plan/Context Files

### 1. Why VJP Cannot Be Used (IMPORTANT)

The existing `analytical_gradients_vjp.py` does backward-pass chaining:
```python
# VJP: given dL/dK, computes dL/dθ in ONE backward pass
# Does NOT materialize dK/dθ matrices
```

For M-step with LBFGS, we need:
```python
# Compute dK/dθ matrices ONCE at start
K, dK = compute_kernel_with_grads(...)  # dK is dict of matrices

# LBFGS closure reuses cached dK:
for key in dK:
    grad[key] = (dL_dK * dK[key]).sum()  # Fast, no recomputation
```

VJP can't do this because it requires dL/dK at backward time, which changes every LBFGS line search evaluation.

### 2. Parameter Transform Gotchas

**GPyTorch uses softplus for sigma_0 and Amp:**
```python
# GPyTorch: param = softplus(raw_param)
# Gradient chain: d/d(raw) = d/d(param) * sigmoid(raw)
```

**Original varGP uses direct values:**
```python
# varGP: sigma_0 and Amp are direct positive values
# No transform needed
```

**Recommendation**: Work with kernel parameter VALUES (not raw), compute gradients w.r.t. values, then apply sigmoid correction when setting `.grad` on raw parameters.

### 3. Eigenspace Projection Order

The M-step works in eigenspace. Order matters:
```python
# 1. Compute full kernels K, K_tilde, Kvec
# 2. Compute full gradients dK, dK_tilde, dKvec
# 3. Project to eigenspace: dK_b = dK @ B, dK_tilde_b = B.T @ dK_tilde @ B
# 4. Compute dlambda using projected quantities
# 5. Compute dloglikelihood and dKL
```

### 4. Memory Consideration

Storing 6 dK matrices:
- dK: (N, M) × 6 params = 6 × N × M floats
- dK_tilde: (M, M) × 6 = 6 × M² floats
- dKvec: (N,) × 6 = 6 × N floats

For N=500, M=50: ~1.8MB total. Acceptable.

### 5. The @torch.no_grad() Pattern

The original M-step closure uses `@torch.no_grad()`:
```python
@torch.no_grad()
def closure():
    # No autograd graph construction
    # Gradients computed analytically and set directly
    # This is WHY it's faster
```

### 6. Current mstep_lbfgs_autograd Location

In `direct_vargp.py` lines 463-577. This is what you're replacing.

Key issue: It calls `kernel(X, X_tilde).evaluate()` inside the closure, which triggers autograd and recomputes everything each call.

### 7. Testing Environment

```bash
# ALWAYS activate this conda environment
conda activate pytorch_gpytorch

# GPU is REQUIRED (CPU is too slow)
# Scripts will error if CUDA unavailable

# Run baseline comparison
python run_single_mode.py --mode vargp_old --ntilde 50 --n-iterations 50 --seed 123
python run_single_mode.py --mode vargp_direct --ntilde 50 --n-iterations 50 --seed 123
```

### 8. Known Numerical Stability Issues

From vargp_old investigation:
- `cosdelta` must be clamped to [-1+1e-6, 1-1e-6] before arccos
- Division by X1X2 needs jitter: `X1X2 + 1e-7`
- K_tilde_b must be symmetrized: `(K + K.T) / 2`
- Eigenvalue threshold: `eigvals > max(eigvals.max() * 1e-4, 1e-4)`

### 9. The C Matrix is Masked

The C matrix in the implementation is masked (reduced from 11664×11664 to ~2480×2480):
```python
# kernel._cached_mask contains the pixel mask
# All x inputs must be masked: x_masked = x[:, mask]
# dC matrices are also in masked space
```

### 10. Float64 Required

All computations must use float64 (kernel values can reach ~10,000):
```python
kernel = kernel.double().to(device)
X = X.double()
```

---

## Implementation Order (TDD)

1. **Create test file** `tests/test_mstep_analytical.py`
   - Write test stubs that will FAIL initially
   - This validates the test infrastructure

2. **Implement `compute_C_and_gradients()`**
   - Port from utils.py:3577-3631
   - Run `test_dC_gradients()` to verify

3. **Implement `compute_kernel_and_gradients()`**
   - Port from utils.py:3663-3813
   - Handle both full matrix and diagonal cases
   - Run `test_dK_gradients()` to verify

4. **Implement `compute_dlambda_moments()`**
   - Port from utils.py:3938-3952
   - Run `test_dlambda_moments()` to verify

5. **Implement `compute_loss_gradients()`**
   - Combine dloglikelihood (utils.py:4111-4117) and dKL (utils.py:4143-4150)
   - Run `test_loss_gradients()` to verify

6. **Implement `mstep_lbfgs_analytical()`**
   - Integrate all components with @torch.no_grad() closure
   - Add all 8 numerical guardrails
   - Run `test_training_equivalence()` to verify

7. **Integration**
   - Add `--mstep-analytical` flag to run_single_mode.py
   - Run end-to-end comparison with vargp_old

---

## Validation Commands

```bash
# 1. Run unit tests
python tests/test_mstep_analytical.py

# 2. Run with analytical M-step
python run_single_mode.py --mode vargp_direct --mstep-analytical --ntilde 50 \
    --n-iterations 50 --n-estep 10 --n-fstep 10 --n-mstep 10 --seed 123

# 3. Compare with baseline
python run_single_mode.py --mode vargp_old --ntilde 50 --n-iterations 50 \
    --n-estep 10 --n-fstep 10 --n-mstep 10 --seed 123
```

---

## Success Criteria

| Metric | Target | Fail Threshold |
|--------|--------|----------------|
| M-step time | ≤7s (vs current 17s) | >10s |
| Test r difference | ≤0.02 vs vargp_old | >0.05 |
| Gradient match | rel_err < 1e-5 | >1e-3 |
| Numerical stability | 0 NaN/Inf | Any NaN/Inf |

---

## If You Get Stuck

1. **Gradient mismatch**: Test each component independently (dC → dK → dlambda → dloss)
2. **Numerical instability**: Check all 8 guardrails are implemented
3. **LBFGS not converging**: Check bounds, try smaller learning rate
4. **Performance not improving**: Profile to find bottleneck (is it dK computation or dlambda?)

**Keep the user informed**: If something is blocking or unclear, ask before spending too much time.

---

## Quick Reference: Exact Formulas

### dlambda_m, dlambda_var
```python
da[key] = (dK[key] - a @ dK_tilde[key]) @ K_tilde_inv
dlambda_m[key] = da[key] @ m
dlambda_var[key] = (dK_vec[key]
    + torch.einsum('ij,ji->i', 2*da[key], V @ a.T)
    - torch.einsum('ij,ij->i', dK[key], a)
    - torch.einsum('ij,ij->i', K, da[key]))
```

### dKL
```python
c = V @ K_tilde_inv
b = K_tilde_inv @ m
B = dK_tilde[key] @ K_tilde_inv
dKL[key] = 0.5*trace(B) - 0.5*trace(c@B) - 0.5*b.T@(B@m)
```

### Diagonal kernel gradients
```python
dKvec['sigma_0'] = 2 * sigma_0
dKvec[key] = torch.sum(x * torch.matmul(dC[key], x), dim=0)
```

---

*Good luck! This is a precision-critical implementation. Test incrementally and verify each step.*
