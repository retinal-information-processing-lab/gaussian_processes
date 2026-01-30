# vargp_direct Implementation Reference

**Purpose**: Comprehensive guide to the `vargp_direct` training mode - a GPyTorch-based implementation that matches the original `varGP()` algorithm structure.

**Status**: COMPLETE (January 2025)
**Last Updated**: January 2025

---

## Table of Contents

1. [Quick Start](#1-quick-start)
2. [Architecture Overview](#2-architecture-overview)
3. [File Structure](#3-file-structure)
4. [The Training Loop](#4-the-training-loop)
5. [Component Details](#5-component-details)
6. [Important Caveats](#6-important-caveats)
7. [Resolved Bugs](#7-resolved-bugs)
8. [Deferred Items](#8-deferred-items)
9. [Unit Tests](#9-unit-tests)
10. [Math Reference](#10-math-reference)
11. [Code-to-Math Mapping](#11-code-to-math-mapping)

---

## 1. Quick Start

### Run vargp_direct

```bash
# Recommended: use --float32 for best performance (matches vargp_old)
python run_single_mode.py --mode vargp_direct --float32 --ntilde 50 \
    --n-iterations 50 --n-estep 10 --n-fstep 10 --n-mstep 10 --seed 123

# With analytical M-step gradients (faster, requires --float32)
python run_single_mode.py --mode vargp_direct --mstep-analytical --float32 \
    --ntilde 50 --n-iterations 50 --seed 123
```

### Performance Comparison

| Mode | Dtype | M-step | Total | Test r |
|------|-------|--------|-------|--------|
| vargp_old | float32 | 4.9s | 6.3s | 0.84 |
| vargp_direct (autograd) | float32 | 4.6s | 5.6s | 0.84 |
| vargp_direct (analytical) | float32 | 4.2s | 5.2s | 0.85 |

**Key point**: Use `--float32` to match vargp_old speed. float64 is ~3x slower.

---

## 2. Architecture Overview

### Why vargp_direct Exists

There are three training modes in this codebase:

| Mode | Description | Limitation |
|------|-------------|------------|
| `vargp_old` | Original implementation in utils.py | Complex codebase, hard to modify |
| `default_gpy` | Standard GPyTorch variational inference | Different optimization dynamics |
| `vargp_style` | Hybrid (custom E-step + GPyTorch strategy) | No eigenspace projection, Adam M-step |

`vargp_direct` fills the gap: it uses GPyTorch as a **kernel calculator only**, while managing variational parameters directly with eigenspace projection - exactly like the original.

### Key Design Decisions

1. **Eigenspace projection**: Reduces dimensionality from M inducing points to ~10-50 dimensions
2. **K̃_b is DIAGONAL**: In eigenspace, the inducing kernel becomes `diag(eigenvalues)` - trivial inverse!
3. **LBFGS for M-step**: Matches original optimization dynamics (not Adam)
4. **GPyTorch for kernels only**: We bypass `VariationalStrategy` entirely

---

## 3. File Structure

```
gpytorch_porting/
├── direct_vargp.py          # Main implementation (~1300 lines)
│   ├── compute_C_and_gradients()           # C matrix and dC/dθ
│   ├── compute_kernel_and_gradients()      # K matrix and dK/dθ
│   ├── compute_lambda_moments_and_gradients()  # Posterior moments
│   ├── compute_loss_gradients()            # dL/dθ for M-step
│   ├── DirectVariationalState              # Dataclass for state
│   ├── estep_eigenspace()                  # E-step Newton update
│   ├── lambda_moments_eigenspace()         # Compute λ_m, λ_var
│   ├── mstep_lbfgs_autograd()             # M-step with autograd
│   ├── mstep_lbfgs_analytical()           # M-step with explicit gradients
│   ├── train_vargp_direct()               # Main training loop
│   └── predict_direct()                    # Prediction at test points
│
├── eigenspace.py            # Eigenspace utilities (~200 lines)
│   ├── EIGVAL_TOL = 1e-4                  # Eigenvalue threshold
│   ├── compute_eigenspace()               # Eigendecomposition
│   ├── project_to_eigenspace()            # m_b = B.T @ m, etc.
│   ├── reproject_variational_params()     # After M-step changes B
│   ├── compute_KKtilde_inv_b()            # K @ K̃⁻¹ (element-wise!)
│   └── compute_K_tilde_b_diagonal()       # diag(eigenvalues)
│
├── kernels.py               # ArcCosineKernel (used as calculator)
├── likelihoods.py           # PoissonLikelihood (A, λ₀ parameters)
├── fstep.py                 # F-step utilities (lambda0_given_A)
│
└── tests/
    ├── test_vargp_direct_match.py     # Unit tests for vargp_direct
    └── README_vargp_direct_tests.md   # Test documentation
```

---

## 4. The Training Loop

### High-Level Structure

```python
def train_vargp_direct(kernel, likelihood, X, X_tilde, r, ...):
    # INITIALIZATION
    # 1. Compute initial kernels: K_tilde, K, Kvec
    # 2. Eigendecomposition: B, eigvals_b from K_tilde
    # 3. Initialize m_b = 0, V_b = K_tilde_b (diagonal)

    for iteration in range(1, n_iterations + 1):

        # KERNEL RECOMPUTATION (after iteration 1)
        if n_mstep > 0 and iteration > 1:
            # M-step changed hyperparams → recompute kernels
            # New eigenspace → reproject m_b, V_b

        # E-STEP: Newton updates for m_b, V_b
        for _ in range(n_estep):
            # Compute posterior moments
            lambda_m, lambda_var = lambda_moments_eigenspace(state)
            f_mean = exp(A*lambda_m + 0.5*A²*lambda_var + λ₀)

            # Newton update (see Section 5.2)
            m_b, V_b = estep_eigenspace(state, r, A, f_mean)

        # F-STEP: Optimize A, compute λ₀ analytically
        lambda0 = lambda0_given_A(A, r, lambda_m, lambda_var)
        LBFGS([logA], closure=f_closure)

        # M-STEP: Optimize kernel hyperparameters (skip last iter)
        if iteration < n_iterations:
            if use_analytical_mstep:
                mstep_lbfgs_analytical(kernel, ...)  # Explicit dK/dθ
            else:
                mstep_lbfgs_autograd(kernel, ...)    # PyTorch autograd

    return state, losses, timing
```

### Detailed Step Explanations

#### Initialization

```python
# Compute kernel matrices
K_tilde = kernel(X_tilde, X_tilde).evaluate()  # (M, M)
K = kernel(X, X_tilde).evaluate()               # (N, M)
Kvec = kernel(X, diag=True)                     # (N,)

# Eigendecomposition
B, eigvals_b, _ = compute_eigenspace(K_tilde)  # B: (M, n_b)
# n_b is typically 10-50, much smaller than M

# Project to eigenspace
K_tilde_b = diag(eigvals_b)      # DIAGONAL!
K_b = K @ B                       # (N, n_b)
KKtilde_inv_b = K_b / eigvals_b  # Element-wise (trivial inverse!)

# Initialize variational params
m_b = zeros(n_b)                 # Start with prior mean
V_b = K_tilde_b.clone()          # Start with prior covariance
```

#### E-step Details

The E-step performs Newton updates on the variational parameters:

```python
def estep_eigenspace(state, r, A, f_mean):
    a = state.KKtilde_inv_b  # (N, n_b), precomputed K @ K̃⁻¹

    # Gradient and Hessian of log-likelihood w.r.t. natural params
    g_b = A * (a.T @ (r - f_mean))           # (n_b,)
    G_b = A² * (a.T @ (f_mean[:, None] * a)) # (n_b, n_b)

    # Newton update for V
    V_b_new = solve(I + K_tilde_b @ G_b, K_tilde_b)

    # Newton update for m (using OLD formula - see Section 8)
    m_b_new = V_b_new @ (G_b @ m_b + g_b)

    # Symmetrize for numerical stability
    V_b_new = (V_b_new + V_b_new.T) / 2

    return m_b_new, V_b_new
```

#### F-step Details

The F-step optimizes the firing rate parameters:

```python
# λ₀ is computed analytically given A
lambda0 = log(sum(r) / sum(exp(A*lambda_m + 0.5*A²*lambda_var)))

# A is optimized with LBFGS
def f_closure():
    log_lik = r @ (A*lambda_m + lambda0) - f_mean.sum()
    return -log_lik  # Minimize negative log-likelihood

LBFGS([logA], closure=f_closure, max_iter=n_fstep)
```

#### M-step Details

The M-step optimizes kernel hyperparameters (σ₀, Amp, β, ρ, ε₀).

**With autograd** (`mstep_lbfgs_autograd`):
```python
def closure():
    # Recompute kernels with current hyperparams
    K_tilde = kernel(X_tilde, X_tilde).evaluate()
    K = kernel(X, X_tilde).evaluate()
    # ... compute loss ...
    loss.backward()  # PyTorch autograd
    return loss

LBFGS(kernel.parameters(), closure=closure)
```

**With analytical gradients** (`mstep_lbfgs_analytical`):
```python
def closure():
    # Compute kernels AND all dK/dθ matrices
    C, mask, dC = compute_C_and_gradients(kernel)
    K_tilde, dK_tilde = compute_kernel_and_gradients(X_tilde, X_tilde, C, dC, ...)
    K, dK = compute_kernel_and_gradients(X, X_tilde, C, dC, ...)

    # Compute loss
    loss = -log_lik + KL

    # Compute gradients explicitly
    dL = compute_loss_gradients(r, f_mean, A, m_b, V_b, ...)

    # Set parameter gradients manually
    kernel.raw_sigma_0.grad = dL['sigma_0'] * sigmoid(raw_sigma_0)
    # ... etc ...

    return loss
```

---

## 5. Component Details

### 5.1 Eigenspace Projection

**Why eigenspace?** The inducing point kernel K̃ (M×M) has effective rank ~10-50. Eigenspace projection:
1. Reduces computation from O(M³) to O(n_b³)
2. Makes K̃⁻¹ trivial: `diag(1/eigenvalues)`
3. Provides implicit regularization (drops small eigenvalues)

**Key quantities in eigenspace:**

| Symbol | Name | Shape | Formula |
|--------|------|-------|---------|
| B | Eigenvector matrix | (M, n_b) | From `eigh(K_tilde)` |
| K̃_b | Inducing kernel | (n_b, n_b) | `diag(eigenvalues)` - DIAGONAL |
| K_b | Cross-kernel | (N, n_b) | `K @ B` |
| m_b | Variational mean | (n_b,) | `B.T @ m` |
| V_b | Variational covariance | (n_b, n_b) | `B.T @ V @ B` - NOT diagonal |
| a | Projection vector | (N, n_b) | `K_b / eigvals_b` (element-wise) |

**Critical insight**: V_b is NOT diagonal even though K̃_b is diagonal. Don't assume diagonal V_b!

### 5.2 Posterior Moments (λ_m, λ_var)

The GP posterior mean and variance at training points:

```python
# In eigenspace:
a = K_b / eigvals_b              # K @ K̃⁻¹, element-wise division
lambda_m = a @ m_b               # Posterior mean
lambda_var = Kvec + (a @ (V_b - K̃_b) @ a.T).diag()  # Posterior variance
```

### 5.3 C Matrix (Receptive Field Structure)

The C matrix encodes receptive field (RF) properties:

```
C = Amp * α[:, None] * C_smooth * α[None, :]

where:
  α[i] = exp(-β_factor * ||pixel_i - center||²)    # Locality weight
  C_smooth[i,j] = exp(-ρ_factor * ||pixel_i - pixel_j||²)  # Smoothness
  β_factor = exp(raw_m2log2beta) = 1/(4β²)
  ρ_factor = exp(raw_mlog2rho2) = 1/(2ρ²)
```

### 5.4 Parameter Transforms

| Parameter | Raw Name | Transform | Gradient Chain Rule |
|-----------|----------|-----------|-------------------|
| σ₀ | `raw_sigma_0` | `softplus(raw)` | `dL/d(raw) = dL/d(σ₀) * sigmoid(raw)` |
| Amp | `raw_Amp` | `softplus(raw)` | Same as above |
| β | `raw_m2log2beta` | `β = exp(-raw/2)/2` | Direct (no transform in gradient) |
| ρ | `raw_mlog2rho2` | `ρ = sqrt(exp(-raw)/2)` | Direct |
| ε₀x, ε₀y | `eps_0x`, `eps_0y` | Direct | Direct |

---

## 6. Important Caveats

### 6.1 K̃_b is Only Diagonal at Initialization

**CRITICAL**: K̃_b = diag(eigenvalues) is ONLY valid immediately after eigendecomposition.

During M-step optimization, when kernel hyperparameters change:
```python
K_tilde_new = kernel(X_tilde, X_tilde)  # New kernel
K_tilde_b = B.T @ K_tilde_new @ B        # NOT diagonal anymore!
```

The fix (see Section 7) uses `torch.linalg.solve()` instead of assuming diagonal structure.

### 6.2 Eigenspace Changes After M-step

After M-step changes kernel hyperparameters, the eigenspace changes. You MUST reproject variational parameters:

```python
# After M-step
K_tilde_new = kernel(X_tilde, X_tilde)
B_new, eigvals_new, _ = compute_eigenspace(K_tilde_new)

# Reproject m_b, V_b to new eigenspace
m_b_new, V_b_new = reproject_variational_params(B_old, B_new, m_b, V_b)
```

### 6.3 LBFGS Closure Called Multiple Times

LBFGS with `line_search_fn='strong_wolfe'` calls the closure multiple times per step. Don't use `loss.backward()` inside - it frees the graph. Instead, compute gradients manually or use `torch.autograd.grad()`.

### 6.4 Symmetrize V After Updates

Always symmetrize V after Newton updates for numerical stability:
```python
V_b = (V_b + V_b.T) / 2
```

### 6.5 Clamp Hyperparameters After M-step

Call `kernel.clamp_hyperparameters()` after M-step to ensure parameters stay in valid ranges.

### 6.6 Float32 vs Float64

- **Float32**: Matches vargp_old, fast, recommended for production
- **Float64**: More precise but ~3x slower, useful for debugging gradient issues

---

## 7. Resolved Bugs

### BUG #6: Stale Eigenvalues in M-step (FIXED January 2025)

**Problem**: The M-step closure used stale eigenvalues (`state.eigvals_b`) while computing fresh `K_tilde_b` from updated hyperparameters.

```python
# BUGGY CODE
K_tilde_b = B.T @ K_tilde_new @ B     # Fresh K_tilde_b
K_tilde_inv_b = diag(1/eigvals_b)     # STALE eigenvalues!
```

**Fix**: Use `torch.linalg.solve()` for the inverse:

```python
# FIXED CODE
K_tilde_b = B.T @ K_tilde_new @ B
K_tilde_inv_b = torch.linalg.solve(K_tilde_b, eye)  # Works for any SPD matrix
```

**Location**: `direct_vargp.py:mstep_lbfgs_analytical()` lines 1102-1107

### Verified Non-Bugs

Unit tests confirmed these are NOT bugs:

- **BUG #4 (Softplus chain rule)**: `dL['σ₀'] * sigmoid(raw)` is correct
- **BUG #3 (Eigenspace projection)**: `dK_tilde_b = B.T @ dK_tilde @ B` preserves gradient structure

---

## 8. Deferred Items

### 8.1 Correct E-step m_new Formula (NOT IMPLEMENTED)

**Background**: The original varGP uses a mathematically **incorrect** m_new formula that works empirically.

**Current implementation** (OLD formula, matching vargp_old):
```python
m_new = V_new @ (G @ m + g)
```

**Mathematically correct formula** (NOT implemented):
```python
m_new = m + K_tilde @ solve(K_tilde + G, g - m)
```

**Why deferred**: When we tried using the correct formula, the model collapsed. The issue is that `g_b` and `G_b` are **transformed** quantities (pre-multiplied by K_tilde_inv), and the old formulas work with these transformed quantities.

**See**: `.claude/ESTEP_MATH_ANALYSIS.md` for full derivation

### 8.2 Extended Testing

- Only tested M=50. Original benchmark included M=250.
- Only tested ntrain=500. Could test ntrain=2000.
- Only tested 3 cells (6, 8, 15). Could test all 10 benchmark cells.

---

## 9. Unit Tests

### Test File: `tests/test_vargp_direct_match.py`

| Test | Purpose | Acceptance |
|------|---------|------------|
| 1. `test_softplus_chain_rule` | Verify gradient transform for σ₀, Amp | rel_err < 1e-4 vs finite diff |
| 2. `test_eigenspace_projection_gradients` | Verify dK_tilde_b preserves gradients | rel_err < 1e-4 for all params |
| 3. `test_ktilde_inv_methods` | Verify solve() vs eigenvalue inverse | Match at init, non-diagonal after change |
| 4. `test_gradient_magnitude_sanity` | Check for NaN/Inf in gradients | No NaN/Inf, reasonable magnitudes |
| 5. `test_multicell_match` | Compare to vargp_old on cells 6, 8, 15 | test_r diff < 0.05 |

### Running Tests

```bash
# All tests
python tests/test_vargp_direct_match.py

# Skip slow tests
python tests/test_vargp_direct_match.py --skip-slow

# Individual test
python tests/test_vargp_direct_match.py --test 1 --verbose
```

### Test Results (January 2025)

All tests PASS:
- Softplus chain rule: rel_err < 1e-7
- Eigenspace projection: rel_err < 1e-6 for all 6 hyperparameters
- K_tilde_inv methods: solve() works for non-diagonal K_tilde_b
- Multi-cell: matches vargp_old within 0.01 on all tested cells

---

## 10. Math Reference

### LaTeX Source Documents

| File | Content |
|------|---------|
| `~/IDV_code/Papers/latex_summaries/Gaussian_process_theory.tex` | Full variational GP derivation (E-step, M-step) |
| `~/IDV_code/Papers/latex_summaries/Estep_corrected.tex` | Correct E-step derivation |
| `~/IDV_code/Papers/latex_summaries/Estep_corrected_mderivation.tex` | m_new formula derivation |
| `~/IDV_code/Papers/latex_summaries/acosker_kernel_def_and_gradients.tex` | Arc-cosine kernel math and gradients |

### Local Context Documents

| File | Content |
|------|---------|
| `.claude/ESTEP_MATH_ANALYSIS.md` | Analysis of E-step formula discrepancy |
| `.claude/MATH_REFERENCE.md` | Quick math reference for this codebase |
| `.claude/ANALYTICAL_GRADIENTS_MATH.md` | M-step gradient derivations |

---

## 11. Code-to-Math Mapping

### Variables

| Code | Math | Shape | Description |
|------|------|-------|-------------|
| `m_b` | m | (n_b,) | Variational mean in eigenspace |
| `V_b` | V | (n_b, n_b) | Variational covariance (NOT diagonal) |
| `K_tilde_b` | K̃ | (n_b, n_b) | Inducing kernel (diagonal in eigenspace) |
| `K_b` | K | (N, n_b) | Cross-kernel to inducing points |
| `Kvec` | k(x,x) | (N,) | Self-kernel (diagonal) |
| `eigvals_b` | λ_i | (n_b,) | Eigenvalues of K̃ |
| `B` | B | (M, n_b) | Eigenvector matrix |
| `a` / `KKtilde_inv_b` | K K̃⁻¹ | (N, n_b) | Projection vector |
| `lambda_m` | μ(x) | (N,) | Posterior mean |
| `lambda_var` | σ²(x) | (N,) | Posterior variance |
| `f_mean` | E[f] | (N,) | Expected firing rate |
| `A` | A | scalar | Gain parameter |
| `lambda0` | λ₀ | scalar | Bias parameter |

### Key Formulas

**Posterior moments**:
```
μ(x) = k(x)ᵀ K̃⁻¹ m
σ²(x) = k(x,x) + k(x)ᵀ K̃⁻¹ (V - K̃) K̃⁻¹ k(x)
```

**Expected log-likelihood**:
```
E[log p(r|λ)] = r(Aμ + λ₀) - exp(Aμ + ½A²σ² + λ₀)
```

**KL divergence**:
```
KL = ½[tr(K̃⁻¹V) + mᵀK̃⁻¹m - n_b + log|K̃| - log|V|]
```

**E-step Newton updates**:
```
g = A · aᵀ(r - f)           # Gradient
G = A² · aᵀ diag(f) a       # Hessian
V_new = solve(I + K̃G, K̃)
m_new = V_new(Gm + g)        # OLD formula (see Section 8.1)
```

---

## Appendix: Original Code References

| Component | utils.py Lines | Description |
|-----------|----------------|-------------|
| Main loop | 5293-5975 | `varGP()` function |
| C matrix | 3577-3631 | `localker()` |
| Kernel | 3663-3813 | `acosker()` |
| E-step | 4217-4277 | `Estep()` |
| λ moments | 3906-3956 | `lambda_moments()` |
| Log-likelihood | 4064-4119 | `compute_loglikelihood()` |
| KL divergence | 4121-4152 | `compute_KL_div()` |
| Eigenspace | 5435-5444 | Eigendecomposition |
| Reprojection | 5619-5627 | After M-step |

---

*Created: January 2025*
*Purpose: Comprehensive reference for vargp_direct implementation*
