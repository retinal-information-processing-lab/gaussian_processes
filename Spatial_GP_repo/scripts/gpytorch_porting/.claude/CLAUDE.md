# GPyTorch Porting Project - Development Tracking

This document tracks the porting effort from the custom variational GP implementation (`utils.py:varGP()`) to GPyTorch. It serves as both a mathematical reference and a decision log.

**Session started**: January 2025
**Goal**: Create a GPyTorch-based implementation that produces qualitatively similar results to the custom implementation, with cleaner code structure.

---

## Quick Start for New Sessions

| Item | Value |
|------|-------|
| **Conda environment** | `pytorch_gpytorch` - ALWAYS use this for running scripts |
| **Current status** | Stage 2 + Masking COMPLETE, E-step partially working (see Section 6.2) |
| **Key files** | `kernels.py`, `test_fit.py`, `likelihoods.py`, `model.py`, `train.py`, `estep.py`, `test_estep_pnas.py`, `tests/` |
| **Run test** | `conda run -n pytorch_gpytorch python test_estep_pnas.py` (modes: adam, efm, vargp_style) |
| **GPU REQUIRED** | Scripts default to CUDA. CPU is too slow. Will error if CUDA unavailable. |
| **Deferred** | E-step improvements for large M (Section 6.2), eigenspace projection (Section 6.4) |
| **Known limitations** | RF center needs reasonable init (Q20); E-step degrades for M>50 |
| **Read first** | WORKING_GUIDELINES.md (process), then this file |

**CRITICAL RULES:**
> - **NEVER use nMstep=0 or n_mstep=0 as default.** Disables kernel learning. Always use nMstep >= 10.
> - **Parameters in GPyTorch modes MUST match varGP.** See Section 6.2 for test script architecture.

---

## Table of Contents

1. [Mathematical Foundation](#1-mathematical-foundation)
2. [Current Custom Implementation](#2-current-custom-implementation)
3. [Porting Strategy](#3-porting-strategy)
4. [Decision Log](#4-decision-log)
5. [Implementation Stages](#5-implementation-stages)
6. [Deferred Items](#6-deferred-items)
7. [Key Files Reference](#7-key-files-reference)
8. [Codebase Structure](#8-codebase-structure)
9. [Data Format](#9-data-format)
10. [Preprocessing Steps](#10-preprocessing-steps)
11. [Working GPyTorch Patterns](#11-working-gpytorch-patterns)

---

## 1. Mathematical Foundation

### 1.1 The Model

**Observation model** (Poisson likelihood with exponential link):
```
r(x) ~ Poisson(f(x))
f(x) = exp(A·λ(x) + λ₀)
```
where:
- `r(x)` = observed spike count for stimulus x
- `f(x)` = firing rate (Poisson rate parameter)
- `λ(x)` = latent GP function
- `A` = gain parameter
- `λ₀` = bias parameter (baseline log-firing rate)

**Latent GP prior**:
```
λ ~ GP(0, K_θ)
```
where K_θ is the kernel with hyperparameters θ.

### 1.2 Variational Inference (Sparse GP)

Since the Poisson likelihood makes the posterior intractable, we use variational inference with inducing points.

**Inducing points**: A set of M << N pseudo-inputs {z̃₁, ..., z̃_M} with corresponding latent values λ̃ = {λ(z̃₁), ..., λ(z̃_M)}.

**Variational approximation**:
```
q(λ̃) = N(m, V)
```
where m ∈ ℝᴹ and V ∈ ℝᴹˣᴹ are variational parameters.

**Approximate posterior** for any point x:
```
q(λ(x)) = ∫ p(λ(x)|λ̃) q(λ̃) dλ̃

Mean:     μ(x) = k(x)ᵀ K̃⁻¹ m
Variance: σ²(x) = k(x,x) + k(x)ᵀ K̃⁻¹ (V - K̃) K̃⁻¹ k(x)
```
where:
- k(x) = [K(x, z̃₁), ..., K(x, z̃_M)]ᵀ  (cross-covariance vector)
- K̃ = K(Z̃, Z̃)  (inducing point kernel matrix)

### 1.3 Evidence Lower Bound (ELBO)

The objective to maximize:
```
L = E_q[log p(Y|λ)] - KL(q(λ̃) || p(λ̃))
```

**KL divergence term** (between two Gaussians):
```
-KL = ½ log|V| - ½ log|K̃| - ½ mᵀK̃⁻¹m - ½ Tr(K̃⁻¹V) + const
```

**Expected log-likelihood term** (for Poisson with exponential link):
```
E_q[log p(rᵢ|λᵢ)] = rᵢ(A·μᵢ + λ₀) - exp(A·μᵢ + ½A²σᵢ² + λ₀) + const
```

### 1.4 EM Algorithm

**E-step**: Update variational parameters (m, V) for fixed hyperparameters.

Newton update (closed-form when α=1):
```
g = A · (K·K̃⁻¹)ᵀ @ (r - f_mean)
G = A² · (K·K̃⁻¹)ᵀ @ diag(f_mean) @ (K·K̃⁻¹)

V_new = solve(I + K̃·G, K̃)
m_new = V_new @ (G·m + g)
```

**M-step**: Update kernel hyperparameters θ for fixed (m, V).
- Gradient-based optimization of ELBO w.r.t. θ

**F-step**: Update firing rate parameters (A, λ₀).
- Can be done with E-step (same loop) since no kernel recomputation needed

### 1.5 Arc-Cosine Kernel

Non-stationary kernel derived from infinite-width 2-layer ReLU network:
```
K(x, x') = (1/π) · M · J(θ)

where:
  v_x = xᵀCx + σ₀²
  v_x' = x'ᵀCx' + σ₀²
  M = √(v_x · v_x')
  cos(θ) = (xᵀCx' + σ₀²) / M
  J(θ) = sin(θ) + (π - θ)cos(θ)
```

**Structured covariance C** (encodes receptive field properties):
```
C_ij = α_i^local · α_j^local · C_ij^smooth

α_i^local = exp(-‖ξ_i - ξ₀‖² / 4β²)     [locality/RF size]
C_ij^smooth = exp(-‖ξ_i - ξ_j‖² / 2ρ²)  [smoothness]
```

Hyperparameters:
- ξ₀ = (eps_0x, eps_0y): RF center position
- β: RF size
- ρ: smoothness scale
- σ₀: bias variance
- Amp: amplitude

**Pixel coordinate grid** (from `localker_clean()`):
```python
# Normalized grid on [-1, 1] × [-1, 1]
ycord, xcord = torch.meshgrid(
    torch.linspace(-1, 1, n_px_side),  # n_px_side = 108 for PNAS
    torch.linspace(-1, 1, n_px_side),
    indexing='ij'
)
xcord = xcord.flatten()  # (n_px_side², ) = (11664,)
ycord = ycord.flatten()
```
- Center of image: (eps_0x, eps_0y) = (0, 0)
- Corners: (±1, ±1)

**Log-space parameterization** (for numerical stability):
```
Code parameter        →  Math symbol  →  Transform
-2log2beta            →  β            →  β = exp(-2log2beta) / 2
-log2rho2             →  ρ²           →  ρ² = exp(-log2rho2) / 2

Example: beta=0.1, rho=0.1
  -2log2beta = -2 * log(2 * 0.1) = -2 * log(0.2) ≈ 3.22
  -log2rho2  = -log(2 * 0.1²)    = -log(0.02)    ≈ 3.91
```

**Implementation details** (from `kernels/kernels.py:localker_clean()`):

1. **Mask computation with detached theta**:
   ```python
   # Mask computed with DETACHED hyperparameters
   # This keeps mask topology fixed during backprop (structural stability)
   dist_sq = (xcord - eps_0x.detach())**2 + (ycord - eps_0y.detach())**2
   alpha_for_mask = torch.exp(-beta.detach() * dist_sq)
   mask = alpha_for_mask >= 0.001  # Threshold: include if α ≥ 0.001
   ```
   Rationale: Prevents the set of active pixels from changing during optimization.

2. **C matrix assembly**:
   ```python
   # Locality weights (using NON-detached theta for gradients)
   dist_sq_center = (xcord - eps_0x)**2 + (ycord - eps_0y)**2
   logalpha = -beta * dist_sq_center
   alpha = torch.exp(logalpha)  # (n_px,)

   # Smoothness kernel (pairwise distances)
   dx = xcord[:, None] - xcord[None, :]
   dy = ycord[:, None] - ycord[None, :]
   dist_sq_pairwise = dx**2 + dy**2
   C_smooth = torch.exp(-rho2 * dist_sq_pairwise)  # (n_px, n_px)

   # Full C matrix
   C = alpha[:, None] * C_smooth * alpha[None, :]
   ```

3. **Symmetrization** (numerical stability):
   ```python
   C = (C + C.T) / 2  # Enforce exact symmetry
   ```

### 1.6 Eigenspace Projection (Numerical Stability)

The custom implementation projects all quantities into the eigenspace of K̃ for stability:
```
K̃ = B · Λ · Bᵀ  (eigendecomposition)
Keep only eigenvalues > threshold

Projected quantities:
  K̃_b = Λ_kept  (diagonal!)
  m_b = Bᵀ @ m
  V_b = Bᵀ @ V @ B
  K_b = K @ B
```

This avoids explicit K̃⁻¹ computation - uses element-wise division with diagonal K̃_b.

---

## 2. Current Custom Implementation

### 2.1 Main Function: `varGP()` (utils.py:5291)

**Inputs**:
- x: stimuli (nt, nx)
- r: spike counts (nt,)
- xtilde: inducing points (ntilde, nx)
- hyperparams_tuple: (theta, theta_lower_lims, theta_higher_lims)
- f_params: {'logA': ..., 'lambda0': ...}
- fit_parameters: {ntilde, maxiter, nEstep, nMstep, nFparamstep, kernfun, ...}

**Outputs**:
- fit_model dict containing: m_b, V_b, B, K_tilde_b, C, mask, hyperparams_tuple, f_params, etc.

### 2.2 Supporting Functions

| Function | Location | Purpose |
|----------|----------|---------|
| `Estep()` | utils.py:4215 | Newton update for (m, V) |
| `localker()` | utils.py | Compute C matrix with RF structure |
| `acosker()` | utils.py | Arc-cosine kernel computation |
| `lambda_moments()` | utils.py | Compute posterior mean/var at test points |

### 2.3 Test Script: `one_cell_fit.py`

Location: `scripts/one_cell_fit.py`

Uses PNAS dataset, fits single cell with:
- ntilde = 50 inducing points (standardized default)
- n_train = 500 training samples
- nEstep = 10, nMstep = 10, nFparamstep = 10, maxiter = 50
- acosker kernel

---

## 3. Porting Strategy

### 3.1 Chosen Approach: GPyTorch-Native (Staged)

Start with GPyTorch's built-in variational inference, then progressively add custom components.

**Why this approach**:
- Leverages GPyTorch's numerical stability and auto-differentiation
- Validates model structure before adding complexity
- Clean separation between model and optimization allows hybrid later

### 3.2 Alternative Approaches Considered

| Approach | Description | Why Not Chosen |
|----------|-------------|----------------|
| **Hybrid from start** | GPyTorch for kernels, custom E-step | More complex initial debugging |
| **Kernels only** | Replace kernels in existing varGP | Doesn't leverage GPyTorch's variational framework |
| **Full custom** | Rewrite everything with better structure | Loses GPyTorch benefits, more work |

### 3.3 Key Architectural Decision

GPyTorch separates **model architecture** from **optimization**:
- Model: `ApproximateGP` with `VariationalStrategy`
- Optimization: standard PyTorch optimizer loop

This means we can:
1. Start with GPyTorch's Adam-based training
2. Later replace with custom E-step (just change the optimizer loop, not the model)
3. Access variational parameters via `model.variational_strategy.variational_distribution`

---

## 4. Decision Log

### Session 1: Initial Planning (January 2025)

**Q1: How close do results need to match?**
> A: Qualitative match is sufficient. Exact numerical match not required.

**Q2: Should GPyTorch handle M-step (hyperparameter optimization)?**
> A: Yes, for initial implementation. Let GPyTorch's Adam optimize the ELBO w.r.t. hyperparameters. Custom analytical gradients deferred to later.

**Q3: What test data to use?**
> A: PNAS data from `one_cell_fit.py` - same dataset as current implementation.

**Q4: Should utility functions be ported?**
> A: NO. Utility functions (active learning, information gain) are explicitly OUT OF SCOPE for this effort.

**Q5: Should custom E-step be implemented now?**
> A: Deferred to later. Start with GPyTorch's variational inference.
>
> **Rationale**:
> - Custom E-step is a closed-form Newton update (faster than iterative)
> - But adding it later is straightforward (just replace optimizer loop)
> - Starting simple validates model structure first
> - GPyTorch's natural gradient optimizer is decent baseline

**Q6: Is hybrid approach practical to add later?**
> A: Yes, very practical. GPyTorch exposes `variational_mean` and `chol_variational_covar`. We can:
> 1. Read current parameters from GPyTorch model
> 2. Apply custom E-step update
> 3. Write back to GPyTorch model
>
> Only complexity: convert between Cholesky (GPyTorch) and full V (custom).

**Q7: Should we start with RBF kernel or arc-cosine with C=I?**
> A: **Arc-cosine with C=I** (identity covariance matrix, no RF structure).
>
> **Rationale**: RBF kernel is stationary (depends on distance), while arc-cosine is non-stationary (depends on actual input values via xᵀCx'). If RBF works but arc-cosine fails, we wouldn't know if the issue is:
> 1. The kernel implementation
> 2. The GPyTorch integration
> 3. Something inherent to arc-cosine with high-dimensional data
>
> Starting with arc-cosine (C=I) keeps the kernel math the same while removing RF complexity. This is a better stepping stone than RBF.

### Session 2: Stage 1 Validation and Debugging (January 2025)

**Q8: How to add amplitude scaling to the kernel?**
> A: Use GPyTorch's `ScaleKernel` wrapper rather than building amplitude into ArcCosineKernel.
>
> **Rationale**:
> - `ScaleKernel` is GPyTorch's standard pattern: `K_scaled = outputscale × K_base`
> - Equivalent to `Amp` parameter in original implementation
> - Keeps ArcCosineKernel simple and matching reference implementation
> - Can verify kernel correctness independent of scaling
>
> **Alternative considered**: Add `Amp` parameter directly to ArcCosineKernel
> - Rejected because: complicates kernel unit test (must match reference exactly)

**Q9: What precision (dtype) is required?**
> A: **float64 is required** for numerical stability.
>
> **Problem**: Arc-cosine kernel values are ~10,000 for PNAS data (11,664 pixels per image). With float32:
> - Cholesky decomposition fails with NaN
> - Precision loss in kernel matrix computations
>
> **Solution**: Use `model.double()` and load data with `dtype=torch.float64`
>
> **Note**: Order matters - call `.double()` BEFORE `.to(device)`

**Q10: Why does arc-cosine with C=I perform poorly on PNAS data?**
> A: **C=I is fundamentally unsuited** for image data without receptive field structure.
>
> **Experimental findings**:
> | Test | RBF Kernel | Arc-Cosine (C=I) |
> |------|-----------|------------------|
> | Synthetic (linear) | r = 0.94 | r = 0.43 |
> | PNAS real data | (not tested) | r ≈ 0.2 |
>
> **Root cause**: Arc-cosine with C=I captures only:
> - Input norms: `||x||²`
> - Angles between inputs: `x·x' / (||x|| ||x'||)`
>
> For random Gaussian inputs (or images with similar total energy), all points have similar norms → similar kernel values → constant predictions.
>
> **The original implementation works because C matrix encodes**:
> - Locality (β, eps_0x, eps_0y): which pixels matter
> - Smoothness (ρ): how nearby pixels correlate
>
> **Conclusion**: Stage 2 (structured C) is essential for PNAS data, not optional.

**Q11**: GPyTorch deprecation warnings? → **No action** - cosmetic, from `linear_operator` internals.

**Q12**: Rename `kernels.py` to avoid ambiguity? → **Deferred**. Two files share name but Python's search order works.

**Q13: Will C=I in Stage 2 reproduce Stage 1 results?**
> A: **Yes, exactly.** This is a key validation check.
>
> Mathematically:
> - C=None: `V = ||x||² + σ₀²`, cross = `x·x' + σ₀²`
> - C=I: `V = xᵀIx + σ₀² = ||x||² + σ₀²`, cross = `xᵀIx' + σ₀² = x·x' + σ₀²`
>
> Only difference: C=None avoids unnecessary `x @ I` matrix multiply (efficiency).
> Kernel values are identical.

### Session 3: Stage 2 Planning (January 2025)

**Q14**: Pixel masking in Stage 2? → **Deferred initially** (adds complexity). Later implemented - see Q22.

**Q15**: C matrix in separate class? → **No** - integrate into ArcCosineKernel. Keeps kernel logic in one place, avoids over-engineering.

**Q16**: Amplitude handling? → **Keep using ScaleKernel wrapper** (GPyTorch standard pattern, consistent with Stage 1). Alternative rejected: adding Amp directly to ArcCosineKernel would break Stage 1 compatibility.

**Q17**: Stage 2 validation approach? → **Two-step**: (1) C=I equivalence test (large β,ρ → C≈I), (2) Performance test (proper RF params → r > 0.5).

### Session 4: Validation Tests (January 2025)

**Q18: How does GPyTorch compare to reference implementation?**
> A: **Excellent match.** Test A ran both implementations on same data with proper hyperparameter learning.
>
> | Metric | Reference (varGP) | GPyTorch | Diff |
> |--------|-------------------|----------|------|
> | Pearson r | **0.8697** | **0.8433** | 0.0264 |
> | Final beta | 0.0606 | 0.1038 | 0.0432 |
> | Final rho | 0.0655 | 0.0797 | 0.0142 |
> | Final A | 0.0169 | 0.9489 | 0.9320 |
>
> **Settings**: ntilde=200, n_train=2000, maxiter=300 (ref) / iterations=500 (GPyTorch)
>
> **Success criterion**: Pearson r within 0.1 ✓

**Q19: Does A initialization matter?**
> A: **NO - model is robust to A initialization.**
>
> | A_init | Pearson r | Final beta | Final rho |
> |--------|-----------|------------|-----------|
> | 1.0 | 0.8714 | 0.1001 | 0.0851 |
> | 0.01 | 0.8715 | 0.0781 | 0.0696 |
>
> Both converge to same Pearson r (diff = 0.0001).
> Lower A_init (0.01) achieves better ELBO (1577 vs 1641) and learns smaller RF (beta closer to reference).

**Q20: Can RF center learn from bad initialization?**
> A: **NO - this is a limitation.** RF center barely moves from bad initialization.
>
> | Init eps_0 | Final eps_0 | Pearson r |
> |------------|-------------|-----------|
> | (0.0, 0.0) | (0.11, -0.04) | **0.87** |
> | (0.5, 0.5) | (0.53, 0.43) | **0.35** |
>
> **Root cause**: Adam struggles to move RF center far from initial position. The loss landscape may have local minima.
>
> **Practical implication**: Start with eps_0 near (0,0) or use prior knowledge about RF location.

### Session 5: Evaluation Metrics (January 2025)

**Q21: Which evaluation metric should we use?**
> A: **Explained variance** (Pearson r / reliability), matching `utils.py:explained_variance()`.
> Added `compute_explained_variance()` to `train.py` and `--plot`/`--save-plot` to `test_fit.py`.

### Session 7: Training Loop Comparison (January 2025)

**Q23: Training mode comparison?**
> A: `efm` mode (E-F-M loop) outperforms pure `adam` for M≤50.
>
> **Canonical test**: `python tests/test_estep_comparison.py`
>
> This script compares varGP (reference), GPyTorch efm, and GPyTorch adam with frozen parameters from `one_cell_fit.py`.

**Q24: Should we use softplus or exp/log for A parameterization?**
> A: **exp/log** (A = exp(raw_A), raw_A = logA).
>
> **Rationale**:
> - Matches original varGP's logA parameterization exactly
> - Simpler code (no inverse softplus computation in f_step_lbfgs)
> - Testing showed equivalent performance between transforms
> - Multiplicative gradient scaling (same relative change at any A value)

### Session 6: Pixel Masking (January 2025)

**Q22: How should pixel masking be implemented?**
> A: Match reference implementation in `kernels/kernels.py:localker_clean()`.
>
> **Key design choices:**
> - Mask computed with **detached** theta parameters (structural stability during backprop)
> - Mask applied internally in `forward()` - user passes full images, kernel handles masking
> - `use_mask=True` by default when `n_px_side` is set
>
> **Hard-coded values:**
> - `MASK_THRESHOLD = 0.001` - pixels with locality weight α >= 0.001 included (matches reference)
> - Typical mask size: ~2400-2500 pixels (out of 11664) for beta=0.1, eps_0=(0,0)
>
> **Validation test tolerances** (`tests/test_mask_validation.py`):
> - Mask equivalence: exact match required
> - C matrix equivalence: max diff < 1e-6 (absolute)
> - Kernel equivalence: relative diff < 1e-5
> - End-to-end fit: Pearson r difference < 0.05 between masked and full
>
> **Memory reduction**: 11664×11664 (~1GB) → ~2480×2480 (~50MB) = ~20x reduction

---

## 5. Implementation Stages

### Stage 1: Arc-Cosine Kernel with C=I (Identity Covariance)
**Status**: COMPLETE (January 2025)

**Goal**: GPyTorch model with arc-cosine kernel (C=I) that fits PNAS data. All tasks completed.

**Results**:
- Model trains successfully on PNAS data
- Loss decreases (40 → 23 with 50 inducing points, 100 iterations)
- Test Pearson r ≈ 0.2 (modest correlation - expected without RF structure)
- Kernel unit test passes (exact match with reference implementation)

**Key implementation notes**:
1. Must use `float64` for numerical stability (kernel values ~10000)
2. Wrap kernel with `ScaleKernel` to add amplitude (prevents exp() overflow)
3. Order matters: call `.double()` before `.to(device)`

**Files created**:
- `kernels.py` - ArcCosineKernel class (verified against reference)
- `likelihoods.py` - PoissonLikelihood class with A, λ₀ parameters
- `model.py` - VariationalGPModel wrapping GPyTorch's ApproximateGP
- `train.py` - Training and evaluation utilities
- `test_fit.py` - Main test script for PNAS data

### Stage 2: Structured Covariance Matrix C
**Status**: COMPLETE (January 2025)

**Goal**: Add locality and smoothness structure to kernel via RF parameters.

**Mathematical reference**: See Section 1.5 for C matrix formula, pixel grid, and implementation details.

**Design decisions**: See Q14-Q17 in Section 4 (Session 3).

All tasks completed. Validations passed: C=I equivalence, gradient flow, performance (r=0.75 vs 0.53), pixel masking.

**Results** (initial quick tests):

| Configuration | Training | Inducing | Pearson r | R² |
|--------------|----------|----------|-----------|-----|
| Stage 1 (C=I) | 500 | 100 | 0.53 | 0.22 |
| Stage 2 (RF) | 500 | 100 | **0.75** | **0.49** |

**Results** (full validation with proper settings):

| Configuration | Training | Inducing | Iterations | Pearson r |
|--------------|----------|----------|------------|-----------|
| Reference (varGP) | 2000 | 200 | 300 | **0.87** |
| GPyTorch (Stage 2) | 2000 | 200 | 500 | **0.84** |

**Final RF parameters** (cell 8, GPyTorch with n_train=2000, ntilde=200):
- beta: 0.10 → 0.10 (minimal change)
- rho: 0.10 → 0.08 (learned smaller)
- eps_0: (0.0, 0.0) → (0.11, -0.04) - RF center learned reasonable location

**Key implementation notes**:
1. C matrix is 11,664 × 11,664 (~1GB float64) - works but slow on CPU
2. Use `--device cuda` for reasonable performance
3. Must use `outputscale=1e-4` to prevent exp() overflow in likelihood
4. Log-space parameterization:
   - `raw = -2*log(2*beta)` → `beta = exp(-raw/2) / 2`
   - `raw = -log(2*rho²)` → `rho = sqrt(exp(-raw) / 2)`

**Files modified**:
- `kernels.py` - added RF parameters, `_setup_pixel_coords()`, `_compute_C_matrix()`
- `test_fit.py` - added `--use-rf`, `--n-train`, `--beta`, `--rho`, `--eps-0x`, `--eps-0y` flags

### Stage 3: Custom E-step (Deferred)
**Status**: DEFERRED

**Goal**: Replace GPyTorch's iterative optimization with closed-form Newton update.

**Mathematical reference**: See `.claude/ESTEP_MATH_ANALYSIS.md` for:
- Correct formulas (V and m updates)
- Comparison with old LaTeX and current code
- Key finding: V update in code is CORRECT, m update has minor discrepancy

**Tasks**:
- [ ] Extract variational parameters from GPyTorch model
- [ ] Implement E-step update using CORRECT formulas (not old LaTeX)
- [ ] Convert between Cholesky and full V representations
- [ ] Write updated parameters back to model
- [ ] Benchmark speed improvement

### Stage 4: Custom M-step Gradients (Deferred)
**Status**: DEFERRED

**Goal**: Implement analytical gradients for kernel hyperparameters.

**Tasks**:
- [ ] Port gradient formulas from LaTeX documents
- [ ] Create custom `backward()` methods
- [ ] Benchmark vs autograd

---

## 6. Deferred Items

### 6.1 Utility Functions
**Status**: EXPLICITLY OUT OF SCOPE

The utility/acquisition functions in `utility.py` are NOT part of this porting effort:
- `nd_utility_new()`
- `distribution_aware_utility_gpytorch()`
- `conditioned_utility_clean()`
- Any active learning functionality

### 6.2 Custom E-step
**Status**: vargp_style STABLE across M, outperforms varGP for M≥75 (see `results/BENCHMARK_LOG.md`)

#### Test Script Architecture

| Script | Role | Description |
|--------|------|-------------|
| `tests/test_estep_comparison.py` | **Canonical comparison** - runs all 4 implementations | varGP + 3 GPyTorch modes |
| `test_estep_pnas.py` | **Active development** - experiment with training modes | Single-mode testing |

**Run canonical test:** `python tests/test_estep_comparison.py` (M=50 default)

**Training modes:**
- `adam`: Pure Adam optimization (no E-step)
- `efm`: E-F-M loop (1 E-step, n F-steps, n M-steps)
- `vargp_style`: Matches original varGP structure (LBFGS F-step, analytical λ₀)

#### CRITICAL: Parameter Equivalence

**GPyTorch `vargp_style` mode MUST mirror original varGP parameters exactly:**

| Parameter | varGP | vargp_style | adam/efm |
|-----------|-------|-------------|----------|
| A_init | 0.01 | 0.01 | 1.0 |
| lambda0_init | 1.0 | 1.0 | 0.0 |
| lr_f (F-step) | 0.1 (LBFGS) | 0.1 (LBFGS) | 0.01 (Adam) |
| lr_m (M-step) | 0.1 | 0.1 | 0.01 |
| F-step optimizer | LBFGS | LBFGS | Adam |
| M-step optimizer | LBFGS | **Adam** | Adam |

**Before modifying training code, ALWAYS verify parameters match between implementations.**

**Known gaps in `vargp_style` vs original varGP:**
1. M-step uses Adam (not LBFGS with analytical gradients)
2. No eigenspace projection (works in full M-dimensional space)

Performance: see `results/BENCHMARK_LOG.md`.

#### Key Implementation Details

- `f_step_lbfgs()`: LBFGS with `logA` parameterization (matching varGP)
- `m_step()`: Uses Adam (NOT matching varGP's LBFGS)

**Note**: efm mode degrades at M>50, but vargp_style remains stable. Original varGP also degrades at M>50.

### 6.3 LBFGS M-step
**Status**: DEFERRED (January 2025) - investigated, Adam works better

**Investigation summary** (see `LBFGS_MSTEP_INVESTIGATION.md`):
- LBFGS with autograd underperforms Adam (0.61 vs 0.83 explained variance)
- Root cause: gradient scale imbalance (sigma_0 gradient ~1000x smaller than others)
- LBFGS uses single step size, follows large gradients, ignores sigma_0
- Grouped LBFGS experiment also failed (kernel becomes non-PD during line search)

**Conclusion**: Adam's adaptive per-parameter learning rates handle the gradient imbalance naturally. Keep Adam for M-step unless analytical gradients are implemented.

**Code preserved**: `m_step_lbfgs()` and `m_step_lbfgs_grouped()` in `estep.py` for future reference.

### 6.4 Custom M-step Gradients
**Reason for deferral**: Autograd works, optimization later.

**Gradient formulas available in**:
- `latex_summaries/acosker_kernel_def_and_gradients.tex`
- `kernels/kernels.py` (C_gradients_hyp, analytical dK/dX)

### 6.5 Eigenspace Projection
**Status**: DEFERRED (January 2025) - potential improvement for large M

**Context**: The original `utils.py:varGP()` stores variational parameters (m_b, V_b) permanently in a reduced eigenspace of K̃. The GPyTorch E-step currently works in full M-dimensional space.

#### K̃ Eigenvalue Analysis
```
M= 25: cond=3.0e+03, eigval=[4.3e-03, 1.3e+01]
M= 50: cond=1.3e+04, eigval=[1.9e-03, 2.6e+01]
M=100: cond=1.4e+05, eigval=[3.7e-04, 5.1e+01]
```
Condition numbers are acceptable for float64. Effective dimensionality is ~10-11 regardless of M.

#### What Full Eigenspace Projection Does (from `utils.py:Estep()`):
1. Projects K̃ = B Λ Bᵀ, keeps only eigenvalues > threshold
2. Stores m_b, V_b in reduced n_b-dimensional space **permanently**
3. All predictions use the reduced representation
4. K̃_b becomes diagonal, simplifying computations

#### Why It Might Help
- Constrains variational distribution to principal subspace
- Acts as implicit regularization
- Reduces noise from small-eigenvalue directions
- Matches the original implementation architecture

#### Current Status
E-step works without eigenspace projection (see Section 6.2), but performance degrades for M≥100. Eigenspace projection may improve this, but has not been implemented or tested.

**Implementation would require**: Storing variational parameters in reduced eigenspace throughout, not just during E-step updates.

### 6.6 Pixel Masking
**Status**: COMPLETE (January 2025). See Q22 for design choices and implementation details.

**Result**: C reduced from 11664×11664 to ~2480×2480 (~20x memory reduction).

### 6.7 Multi-Cell Validation (Test D)
**Reason for deferral**: Cell 8 validation sufficient for initial implementation.

**Plan**: Run on cells 0, 4, 8, 12 to verify robustness across different neurons.

**Success criteria**: All cells achieve r > 0.3, no NaN/crashes.

---

## 7. Key Files Reference

### Source LaTeX Documents
| File | Content |
|------|---------|
| `~/IDV_code/Papers/latex_summaries/Gaussian_process_theory.tex` | Full variational GP derivation (E-step Newton, M-step) |
| `~/IDV_code/Papers/latex_summaries/acosker_kernel_def_and_gradients.tex` | Arc-cosine kernel math |
| `~/IDV_code/Papers/latex_summaries/distribution_aware_utility_pietro.tex` | Utility functions (OUT OF SCOPE) |

### Existing Code to Reference
| File | Content |
|------|---------|
| `utils.py` | varGP(), Estep(), acosker(), localker() |
| `kernels/kernels.py` | Clean kernel implementations with autograd |
| `scripts/one_cell_fit.py` | Test script with PNAS data |
| `scripts/1D_playground/gp_utility_playground.py` | GPyTorch example with Poisson likelihood |

### Data
| File | Content |
|------|---------|
| `notebooks/PNAS_paper_sorted_data.npz` | Test dataset (images + neural responses) |

### GPyTorch Porting Files (this project)
| File | Content |
|------|---------|
| `kernels.py` | ArcCosineKernel with RF structure and masking |
| `likelihoods.py` | PoissonLikelihood with A, λ₀ |
| `model.py` | VariationalGPModel |
| `train.py` | Training utilities (Adam-based) |
| `estep.py` | Custom E-step Newton update + `train_efm()` |
| `test_fit.py` | Test script for Adam training |
| `test_estep_pnas.py` | Single-mode testing for development |
| `tests/test_estep_comparison.py` | **Canonical test** - compares varGP + 3 GPyTorch modes |
| `tests/test_mask_validation.py` | Pixel masking validation |
| `tests/test_reference_comparison.py` | GPyTorch vs varGP comparison |
| `results/BENCHMARK_LOG.md` | Performance tracking across milestones |

---

## 8. Codebase Structure

```
Spatial_GP_repo/
├── utils.py                 (301 KB) - Main GP: varGP(), Estep(), acosker(), localker()
├── utility.py               (290 KB) - Active learning (OUT OF SCOPE)
├── model.py                 - Backward compat shim for GPModel
│
├── GP_model/
│   └── model.py             - GPModel class (structured model storage)
│
├── kernels/
│   ├── kernels.py           (15.9 KB) - Clean kernel implementations
│   │   ├── localker_clean() - Locality/RF covariance C
│   │   ├── acosker_clean()  - Arc-cosine kernel (the one we're porting)
│   │   └── acosker_with_grad() - Autograd wrapper
│   └── __init__.py
│
├── notebooks/
│   ├── one_cell_fit.ipynb   - Basic GP fitting workflow (REFERENCE)
│   └── PNAS_paper_sorted_data.npz - Dataset
│
├── scripts/
│   ├── one_cell_fit.py      - Standalone fitting script (REFERENCE)
│   ├── 1D_playground/
│   │   └── gp_utility_playground.py - GPyTorch example (REFERENCE)
│   └── gpytorch_porting/    - THIS PROJECT
│       ├── .claude/CLAUDE.md - This file
│       └── (new files to create)
│
└── tests/
    └── test_acosker_clean.py - Kernel unit tests
```

**Key insight**: 90% of the GP logic is in `utils.py:varGP()`. The kernels are in `kernels/kernels.py`.

---

## 9. Data Format

**File**: `notebooks/PNAS_paper_sorted_data.npz`

| Key | Shape | Dtype | Range | Description |
|-----|-------|-------|-------|-------------|
| `images_train` | (2910, 108, 108, 1) | float32 | [-2.4, 2.5] | Training images (normalized) |
| `images_val` | (250, 108, 108, 1) | float32 | [-2.4, 2.5] | Validation images |
| `images_test` | (30, 108, 108, 1) | float32 | [-2.4, 2.5] | Test images |
| `responses_train` | (2910, 41) | float64 | [0, ~25] | Spike counts (41 neurons) |
| `responses_val` | (250, 41) | float64 | [0, ~26] | Validation responses |
| `responses_test` | **(30, 30, 41)** | float64 | [0, ~27] | **30 repeats × 30 images × 41 neurons** |

**Important notes**:
- Images are already normalized (mean ~0, std ~1)
- Responses are non-negative spike counts
- Test set has 30 repetitions per image for reliability estimates
- We fit **one neuron at a time** (cellid selects which)

---

## 10. Preprocessing Steps

From `scripts/one_cell_fit.py`:

```python
# 1. Load data
data = np.load('notebooks/PNAS_paper_sorted_data.npz')
X_train = torch.tensor(data['images_train'], dtype=torch.float32, device=device)
R_train = torch.tensor(data['responses_train'], dtype=torch.float32, device=device)

# 2. Flatten images: (n_samples, 108, 108, 1) → (n_samples, 11664)
X_train = X_train.reshape(X_train.shape[0], -1)  # (2910, 11664)

# 3. Select single neuron
cellid = 0
r = R_train[:, cellid]  # (2910,)

# 4. Select inducing points (random subset)
ntilde = 100  # or up to 2100
indices = torch.randperm(X_train.shape[0])[:ntilde]
xtilde = X_train[indices]  # (ntilde, 11664)

# 5. Initialize hyperparameters
theta = {
    'sigma_0': torch.tensor(1.0),
    'Amp': torch.tensor(1.0),
    'eps_0x': torch.tensor(0.0),
    'eps_0y': torch.tensor(0.0),
    '-2log2beta': torch.tensor(-2 * np.log(2 * 0.1)),  # beta=0.1
    '-log2rho2': torch.tensor(-np.log(2 * 0.1**2)),    # rho=0.1
}

# 6. Initialize firing rate parameters
f_params = {
    'logA': torch.log(torch.tensor(0.01)),  # A = 0.01
    'lambda0': torch.tensor(1.0),
}
```

---

## 11. Working GPyTorch Patterns

From `scripts/1D_playground/gp_utility_playground.py`:

### 11.1 Poisson Likelihood

```python
class PoissonLikelihood(gpytorch.likelihoods.Likelihood):
    """Poisson likelihood with log-link: r ~ Poisson(exp(f))

    For our model: f = A·λ + λ₀, so rate = exp(A·λ + λ₀)

    Expected log-likelihood under q(λ) = N(μ, σ²):
        E[r·f - exp(f)] = r·(A·μ + λ₀) - exp(A·μ + A²σ²/2 + λ₀)
    """

    def __init__(self, A_init=1.0, lambda0_init=0.0):
        super().__init__()
        # Learnable parameters
        self.register_parameter('raw_A', torch.nn.Parameter(torch.tensor(A_init)))
        self.register_parameter('lambda0', torch.nn.Parameter(torch.tensor(lambda0_init)))

    @property
    def A(self):
        # A = exp(raw_A), where raw_A = logA (matching varGP)
        return self.raw_A_constraint.transform(self.raw_A)

    def expected_log_prob(self, target, input):
        """
        Args:
            target: Observed spike counts (n_samples,)
            input: GP output MultivariateNormal with mean, variance
        Returns:
            Expected log probability (summed over samples)
        """
        mu = input.mean      # (n_samples,)
        var = input.variance # (n_samples,)
        A = self.A
        lambda0 = self.lambda0

        # E[r·(A·λ + λ₀) - exp(A·λ + λ₀)]
        # = r·(A·μ + λ₀) - exp(A·μ + A²·σ²/2 + λ₀)
        log_prob = target * (A * mu + lambda0) - torch.exp(A * mu + 0.5 * A**2 * var + lambda0)
        return log_prob.sum(-1)

    def forward(self, function_samples):
        """For sampling: return Poisson distribution given function samples."""
        A = self.A
        lambda0 = self.lambda0
        rate = torch.exp(A * function_samples + lambda0)
        return torch.distributions.Poisson(rate=rate)
```

### 11.2 Variational GP Model

```python
class VariationalGPModel(gpytorch.models.ApproximateGP):
    def __init__(self, inducing_points, kernel):
        # Variational distribution q(u) = N(m, LLᵀ)
        variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(
            inducing_points.size(0)
        )
        # Strategy for computing q(f) from q(u)
        variational_strategy = gpytorch.variational.VariationalStrategy(
            self,
            inducing_points,
            variational_distribution,
            learn_inducing_locations=False  # Keep inducing points fixed
        )
        super().__init__(variational_strategy)

        self.mean_module = gpytorch.means.ZeroMean()
        self.covar_module = kernel

    def forward(self, x):
        mean = self.mean_module(x)
        covar = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean, covar)
```

### 11.3 Training Loop

```python
def train_adam(model, likelihood, train_x, train_y, n_iterations=500, lr=0.1):
    model.train()

    # Optimize both model and likelihood parameters
    optimizer = torch.optim.Adam([
        {'params': model.parameters()},
        {'params': likelihood.parameters()}
    ], lr=lr)

    with torch.enable_grad():  # Important: utility.py disables grad globally
        for i in range(n_iterations):
            optimizer.zero_grad()

            # Forward pass
            output = model(train_x)

            # ELBO = E_q[log p(y|f)] - KL(q(u) || p(u))
            expected_log_lik = likelihood.expected_log_prob(train_y, output)
            kl_div = model.variational_strategy.kl_divergence()

            # Minimize negative ELBO
            loss = -expected_log_lik + kl_div

            loss.backward()
            optimizer.step()

            if (i + 1) % 100 == 0:
                print(f"Iter {i+1}/{n_iterations}, Loss: {loss.item():.2f}")

    return model, likelihood
```

### 11.4 Prediction

```python
def predict(model, likelihood, test_x):
    model.eval()
    with torch.no_grad():
        # Get posterior q(λ*) at test points
        posterior = model(test_x)
        mu = posterior.mean
        var = posterior.variance

        # Predicted firing rate: E[exp(A·λ + λ₀)] = exp(A·μ + A²σ²/2 + λ₀)
        A = likelihood.A
        lambda0 = likelihood.lambda0
        f_pred = torch.exp(A * mu + 0.5 * A**2 * var + lambda0)

    return f_pred, mu, var
```

---

## Appendix B: Mathematical Notation Reference

| Symbol | Meaning | Shape |
|--------|---------|-------|
| x | Input stimulus (image) | (n_pixels,) |
| r | Observed spike count | scalar |
| λ(x) | Latent GP function value | scalar |
| f(x) | Firing rate = exp(A·λ + λ₀) | scalar |
| K̃ | Inducing point kernel matrix | (M, M) |
| K | Cross-kernel (data to inducing) | (N, M) |
| m | Variational mean | (M,) |
| V | Variational covariance | (M, M) |
| C | Structured covariance (RF) | (n_px, n_px) |
| A | Gain parameter | scalar |
| λ₀ | Bias parameter | scalar |
| β | RF size parameter | scalar |
| ρ | Smoothness parameter | scalar |
| ξ₀ | RF center (eps_0x, eps_0y) | (2,) |
| σ₀ | Kernel bias variance | scalar |

---

*Last updated: January 2025 (Session 7 - E-step M scaling: M=50 works, M>50 degrades)*
