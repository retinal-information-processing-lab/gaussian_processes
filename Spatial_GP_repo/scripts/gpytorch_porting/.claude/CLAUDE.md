# GPyTorch Porting Project - Development Tracking

This document tracks the porting effort from the custom variational GP implementation (`utils.py:varGP()`) to GPyTorch. It serves as both a mathematical reference and a decision log.

**Session started**: January 2025
**Goal**: Create a GPyTorch-based implementation that produces qualitatively similar results to the custom implementation, with cleaner code structure.

---

## Quick Start for New Sessions

| Item | Value |
|------|-------|
| **Conda environment** | `pytorch_gpytorch` - ALWAYS use this for running scripts |
| **Current status** | Stage 2 + Masking + Analytical Gradients + E-step Kernel Caching + UnwhitenedVariationalStrategy COMPLETE |
| **Key files** | `kernels.py`, `estep.py`, `analytical_gradients_vjp.py`, `test_estep_pnas.py`, `tests/` |
| **Run test** | `conda run -n pytorch_gpytorch python test_estep_pnas.py` (modes: vargp_old, adam, efm, vargp_style) |
| **Gradient modes** | `--gradient-mode autograd` (default), `vjp` (fast analytical), `jacobian` (slow, reference) |
| **E-step caching** | Enabled by default (8.8x faster). Use `--no-cache` to disable for testing. |
| **GPU REQUIRED** | Scripts default to CUDA. CPU is too slow. Will error if CUDA unavailable. |
| **Deferred** | Eigenspace projection (Section 6.5), LBFGS M-step (Section 6.3) |
| **Known limitations** | RF center needs reasonable init (Q20); Hacky `torch.pi` workaround (see below); Jitter consistency (see below) |
| **Current focus** | Unspecified |
| **Read first** | WORKING_GUIDELINES.md (process), then this file |

**CRITICAL RULES:**
> - **NEVER use nMstep=0 or n_mstep=0 as default.** Disables kernel learning. Always use nMstep >= 10.
> - **Parameters in GPyTorch modes MUST match varGP.** See Section 6.2 for test script architecture.

**Gradient Mode Selection (January 2025):**
> Use `ArcCosineKernel(..., gradient_mode='MODE')` or `--gradient-mode MODE` in CLI:
> - `'autograd'`: PyTorch autograd (default) - automatic differentiation
> - `'vjp'`: VJP analytical - same speed as autograd, explicit formulas
> - `'jacobian'`: Old Jacobian materialization - slow but matches original varGP exactly
>
> Note: `use_analytical_grads` flag was removed (Jan 2025). Use `gradient_mode` instead.

**HACKY WORKAROUND - Random State Reproducibility (January 2025):**
> Test scripts use `tests/test_utils.py:set_reproducible_seed()` which contains a **hacky workaround**:
> ```python
> torch.pi = torch.acos(torch.zeros(1)).item() * 2  # WHY DOES THIS MATTER?!
> ```
> This replicates a side effect from `GP_utils.py` line 49. Without it, `test_kernel_cache.py` fails
> while `test_estep_pnas.py` succeeds - same code, different random sequences.
>
> **We don't understand why this works.** The assignment to `torch.pi` (a built-in constant since
> PyTorch 1.8) somehow affects random state. This is cargo cult programming.
>
> See `.claude/archive/ARCHIVE_2026-01-18_kernel_caching_and_whitening.md` Section 17 for full details and what we tried that didn't work.

**Jitter Consistency (Critical - January 2025):**
> All jitter values MUST match `model.jitter` (default 1e-4). Mismatched jitter causes whitening
> conversion failures in non-cached E-step path.
>
> **Problem discovered**: Config C (non-cached + whitening) failed at M=50 with test_r=0.21 instead
> of expected ~0.77. Root cause was jitter mismatch:
> - `VariationalGPModel` created with jitter=1e-4 (stored in `model.jitter`)
> - `estep.py` functions defaulted to jitter=1e-6
> - GPyTorch's internal `VariationalStrategy.jitter_val` = 1e-4
>
> When non-cached whitening path calls `model(X)`, GPyTorch uses its internal L_K (with 1e-4),
> but our whitening conversions used L_K computed with 1e-6. This 100x mismatch corrupted
> the variational parameters.
>
> **Fix**: All `estep.py` functions now default to `model.jitter` and warn if an explicit jitter
> parameter mismatches. The warning includes the corrective action (using model.jitter instead).
>
> **Affected functions**: `compute_kernel_cache()`, `compute_L_K()`, `e_step()`,
> `e_step_explicit()`, `e_step_loop()`.

**UnwhitenedVariationalStrategy Option (January 2025):**
> `VariationalGPModel` now supports a `whitening` parameter (default `True`):
> - `whitening=True`: Use `VariationalStrategy` (default, stores whitened params, faster)
> - `whitening=False`: Use `UnwhitenedVariationalStrategy` (stores natural params directly)
>
> **CLI usage**: `python test_estep_pnas.py --mode vargp_style --unwhitened`
>
> **When to use unwhitened**: Investigating EM-style optimization, debugging whitening issues.
>
> **Performance**: Unwhitened is ~4x slower and achieves lower test r (0.65 vs 0.80).
>
> **Details**: See Q26-Q28 in `DECISION_LOG.md` and `TECHNICAL_ANALYSIS_2026-01-20_whitening_LK_mismatch.md`.

---

## Table of Contents

| Section | Description | When to Read |
|---------|-------------|--------------|
| [Quick Start](#quick-start-for-new-sessions) | Status, commands, current focus | **Always read first** |
| [Mathematical Foundation](#1-mathematical-foundation) | Model, ELBO, kernel math | See **MATH_REFERENCE.md** for details |
| [Current Custom Implementation](#2-current-custom-implementation) | varGP() function reference | When comparing to original |
| [Porting Strategy](#3-porting-strategy) | Approach and architecture | For context on design choices |
| [Decision Log](#4-decision-log) | Key decisions summary | See **DECISION_LOG.md** for full Q1-Q25 |
| [Implementation Stages](#5-implementation-stages) | Stage 1-4 status | To check what's done/pending |
| [Deferred Items](#6-deferred-items) | Parked work items | Before starting new features |
| [Key Files Reference](#7-key-files-reference) | File locations and purposes | When looking for specific code |
| [Appendix](#8-codebase-structure) | Data format, preprocessing, patterns | Reference material |

**Related Documentation:**
- `MATH_REFERENCE.md` - Full mathematical derivations (read when working on math problems)
- `DECISION_LOG.md` - Design decisions Q1-Q25 (read when implementation seems confusing)
- `WORKING_GUIDELINES.md` - Process rules and policies (read at session start)
- `SESSION_LOG.md` - Recent session summaries (read for context on recent work)

---

## 1. Mathematical Foundation

> **Full derivations in `MATH_REFERENCE.md`** - Read when working on math-related problems.

**Quick Summary:**
- **Model**: Poisson likelihood with exponential link: `r ~ Poisson(exp(A·λ + λ₀))`
- **Inference**: Sparse variational GP with inducing points, ELBO objective
- **Kernel**: Arc-cosine kernel with structured covariance C (encodes RF properties)
- **Algorithm**: EM with closed-form E-step (Newton), gradient-based M-step

**Key equations** (see MATH_REFERENCE.md for derivations):
- Posterior mean: `μ(x) = k(x)ᵀ K̃⁻¹ m`
- Posterior var: `σ²(x) = k(x,x) + k(x)ᵀ K̃⁻¹ (V - K̃) K̃⁻¹ k(x)`
- Expected log-lik: `E[log p(r|λ)] = r(A·μ + λ₀) - exp(A·μ + ½A²σ² + λ₀)`

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

Note: Avaid using the .data parameter and if you need to, raise it to the user. prefer torch.no_grad() + copy()

---

## 4. Decision Log

> **Full decision history in `DECISION_LOG.md`** (Q1-Q25)
>
> Read this file when something about the implementation seems confusing or when facing similar design choices.

**Key decisions summary:**
- Q4: Utility functions OUT OF SCOPE
- Q9: float64 required (kernel values ~10,000)
- Q20: RF center needs reasonable init (won't learn from bad init)
- Q22: Pixel masking uses detached theta for structural stability
- Q25: Kernel caching gives 8.8x E-step speedup

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
- `test_stage1_cI.py` - Stage 1 (C=I) testing script, supports `--no-rf` flag

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
- `test_stage1_cI.py` - added `--use-rf`, `--n-train`, `--beta`, `--rho`, `--eps-0x`, `--eps-0y` flags

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

### Stage 4: Analytical M-step Gradients
**Status**: COMPLETE (January 2025)

**Goal**: Implement analytical gradients for kernel hyperparameters, matching varGP.

**Implementation**:
- New file `analytical_gradients.py` with:
  - `acosker_with_hyp_grad()`: Computes K and all dK/dθ matrices
  - `compute_C_and_gradients()`: Computes C and dC matrices
  - `ArcCosineJacobianGradients(torch.autograd.Function)`: Wrapper for clean integration
- Use `gradient_mode='jacobian'` in `ArcCosineKernel`
- When enabled, forward pass computes K and saves dK; backward uses analytical gradients

**Validation results** (from `tests/test_analytical_gradients.py`):
- Gradient correctness: relative error < 1e-7 vs autograd
- Numerical check: finite differences match within 1e-6
- Training equivalence: final loss differs by ~1% between modes

**Usage**:
```bash
python test_estep_pnas.py --mode vargp_style --gradient-mode jacobian
```

**Performance characteristics** (old Jacobian-materialization approach):

| M | Autograd | Old Analytical | Slowdown | Notes |
|---|----------|----------------|----------|-------|
| 50 | 16s | 30s | 1.9x | Equivalent results (0.83) |
| 75 | 18s | 46s | 2.5x | Equivalent results |
| 100 | 45s | 84s | 1.9x | **Analytical worse: 0.71 vs 0.87** |

**Why was old implementation 2x slower**: It materialized 5 dC matrices and 5 dK matrices (15 large matrix multiplies). Autograd uses VJPs without materializing intermediates.

### Stage 4b: VJP-Based Analytical Gradients (INTEGRATED)
**Status**: COMPLETE + INTEGRATED (January 2025)

**Files**:
- `analytical_gradients_vjp.py` - VJP implementation (fast, recommended)
- `analytical_gradients.py` - Jacobian implementation (slow, reference)

**Integration**: Both implementations are now accessible via `gradient_mode` parameter:

```python
# Use VJP (fast, same speed as autograd)
kernel = ArcCosineKernel(n_px_side=108, gradient_mode='vjp')

# Use Jacobian (slow, matches original varGP exactly)
kernel = ArcCosineKernel(n_px_side=108, gradient_mode='jacobian')

# Use autograd (default)
kernel = ArcCosineKernel(n_px_side=108, gradient_mode='autograd')
```

**CLI usage**:
```bash
python test_estep_pnas.py --gradient-mode vjp       # Fast analytical
python test_estep_pnas.py --gradient-mode jacobian  # Slow reference
python test_estep_pnas.py --gradient-mode autograd  # PyTorch autograd (default)
```

**Performance comparison**:

| Mode | Time per call | Notes |
|------|---------------|-------|
| `autograd` | 16.2 ms | PyTorch automatic differentiation |
| `vjp` | 16.2 ms | Same speed, explicit kernel formulas |
| `jacobian` | 73.1 ms | 4.5x slower, matches original varGP |

**When to use each mode**:
- `autograd`: Default, simplest, works well
- `vjp`: When you want explicit control with no speed penalty
- `jacobian`: When you need exact match with original varGP for debugging

**Key insight**: VJP computes dL/dC ONCE in backward, then chains to each hyperparameter via element-wise ops. Jacobian materializes 5 dK matrices explicitly.

**Math reference**: `.claude/VJP_ANALYTICAL_GRADIENTS.md`

**Note**: `use_analytical_grads` parameter was removed (Jan 2025). Use `gradient_mode='jacobian'` instead.

**estep.py optimization** (still relevant):
- Training loop toggles `requires_grad` on kernel parameters
- E-step and F-step: kernel gradients disabled
- M-step only: kernel gradients enabled

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
- `vargp_old`: Original varGP implementation (reference baseline)
- `adam`: Pure Adam optimization (no E-step)
- `efm`: E-F-M loop (1 E-step, n F-steps, n M-steps)
- `vargp_style`: Matches original varGP structure (LBFGS F-step, analytical λ₀)

**Metrics**: All modes report standardized metrics (test_corr, explained_var, reliability) computed identically.

**Timing Note**: GPU warmup within a Python session can affect timing (~0.5s). First runs are slower due to CUDA kernel compilation.

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

#### Kernel Caching Optimization (January 2025)

E-step now caches kernel matrices to avoid redundant computation. See Q25 in Decision Log.

**New functions** (in `estep.py`):
- `compute_kernel_cache()`: Compute K, K̃, k0 once
- `compute_moments_from_kernel_cache()`: Compute moments without calling `model(X)`
- `e_step_with_kernel_cache()`: Newton update using cached kernels

**Performance**: E-step reduced from 8.8s to 1.0s (8.8x faster, now faster than original varGP).

**Usage**: Enabled by default. Use `use_cache=False` or `--no-cache` to disable for testing.

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
**Status**: COMPLETE + INTEGRATED (January 2025) - see Implementation Stage 4b

**Implementation**: Three gradient modes available via `gradient_mode` parameter:
- `'autograd'`: PyTorch autograd (default)
- `'vjp'`: VJP-based analytical (fast, same speed as autograd)
- `'jacobian'`: Jacobian-based analytical (slow, matches original varGP)

**Usage**:
```python
kernel = ArcCosineKernel(n_px_side=108, gradient_mode='vjp')
```
Or CLI: `python test_estep_pnas.py --gradient-mode vjp`

**Reference formulas in**:
- `latex_summaries/acosker_kernel_def_and_gradients.tex`
- `kernels/kernels.py` (C_gradients_hyp, analytical dK/dX)
- `.claude/ANALYTICAL_GRADIENTS_MATH.md`
- `.claude/VJP_ANALYTICAL_GRADIENTS.md` (VJP-specific derivation)

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
| `kernels.py` | ArcCosineKernel with RF structure, masking, `gradient_mode` selection |
| `likelihoods.py` | PoissonLikelihood with A, λ₀ |
| `model.py` | VariationalGPModel |
| `train.py` | Training utilities (Adam-based) |
| `estep.py` | Custom E-step Newton update + `train_efm()` + kernel caching (`compute_kernel_cache`, `compute_moments_from_kernel_cache`, `e_step_with_kernel_cache`) |
| `analytical_gradients.py` | Jacobian-based analytical gradients (slow, reference) |
| `analytical_gradients_vjp.py` | VJP-based analytical gradients (fast, same speed as autograd) |
| `.claude/VJP_ANALYTICAL_GRADIENTS.md` | Mathematical derivation for VJP approach |
| `test_stage1_cI.py` | Stage 1 (C=I) testing with Adam, supports `--no-rf` for identity covariance |
| `test_estep_pnas.py` | **Main test script** - all training modes, supports `--gradient-mode`, `--no-cache`, `--no-whitening` |
| `tests/test_estep_comparison.py` | **Canonical test** - compares varGP + 3 GPyTorch modes |
| `tests/test_whitening_paths.py` | Whitening path validation (cached vs non-cached, whitening on/off) |
| `tests/test_kernel_cache.py` | Kernel caching validation |
| `tests/test_m_whitening.py` | Whitening conversion unit tests |
| `tests/test_mask_validation.py` | Pixel masking validation |
| `tests/test_reference_comparison.py` | GPyTorch vs varGP comparison |
| `tests/test_analytical_gradients.py` | Analytical gradient validation |
| `results/BENCHMARK_LOG.md` | Performance tracking across milestones |
| `results/PROFILING_2026-01-18.md` | E-step profiling results (kernel caching optimization) |
| `.claude/archive/ARCHIVE_2026-01-18_kernel_caching_and_whitening.md` | Kernel caching implementation details and whitening analysis (historical) |

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

*Last updated: January 2025 (Session 10 - E-step kernel caching optimization, 8.8x speedup)*
