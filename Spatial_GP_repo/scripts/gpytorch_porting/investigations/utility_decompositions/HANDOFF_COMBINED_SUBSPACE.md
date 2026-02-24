# Investigation: Combined PCA + C-Eigenspace Utility Optimization

**Branch**: `pietro/pca-utility-optimization`
**Date**: 2026-02-23
**Status**: Continuing
**Location**: `investigations/utility_decompositions/`

---

## Problem Statement

We want to optimize images to maximize distribution-aware (DA) utility for active learning in a variational GP model. We have two separate subspace parameterizations that each capture different information:

1. **PCA** (data manifold): x* = mu + V_K @ z. Constrains the optimized image to the space of natural images. V_K are top eigenvectors of dataset covariance. This is a **genuine constraint** — it excludes unnatural images. Lower variance thresholds = more constrained = more natural-looking.

2. **C-eigenspace** (kernel geometry): x* = U_K @ z. U_K are top eigenvectors of the kernel's C matrix. This is mathematically a **reparameterization** — it does NOT change the solution compared to unconstrained pixel optimization. Its value is purely in removing near-zero gradient directions (the C matrix acts as a spatial low-pass filter, so most high-frequency directions have negligible eigenvalues and contribute nothing to the gradient).

The user's goal for the next session: create a script that **combines both decompositions together** in a single optimization. The exact mathematical formulation of "combined" is an open design question that the next session must resolve.

## Current State of the Codebase

### Scripts in `investigations/utility_decompositions/`

| File | Lines | Description |
|------|-------|-------------|
| `subspace_optimization.py` | 853 | **Unified script (this session)**. Supports `--method {pca, c_eigen}` as separate alternatives. Has shared `gradient_ascent()` function, shared plotting, multi-conditioning support. Both methods tested and working. |
| `pca_optimization.py` | 673 | **Original PCA script** (previous session). Single-target only, PCA method only. |
| `c_eigen_optimization.py` | 1067 | **C-eigen script** (extracted from `pietro/c-eigen-utility-optimization` branch as reference). Has multi-conditioning, chunked gradients, early stopping. |
| `test_subspace_optimization.py` | 596 | **Tests** (written by user). 9 test groups: PCA math properties, C-eigen math properties, gradient flow, rf_pearson_r, norm constraint, utility improvement, subspace membership, cross-method consistency, early stopping/timing. |
| `HANDOFF_PCA.md` | 149 | Previous session's PCA handoff. |
| `HANDOFF_C_EIGEN.md` | ~350 | C-eigen handoff (outdated — see user's comprehensive summary below). |
| `HANDOFF_COMBINED_SUBSPACE.md` | This file. |

### LaTeX Documents (on `pietro/c-eigen-utility-optimization` branch only)

These are NOT on the current branch but contain critical mathematical foundations:

- `pca_vs_c_eigenspace.tex` (~1160 lines): Proves PCA is a genuine constraint while C-eigenspace is a reparameterization. Describes C matrix structure: `C = Amplitude * diag(alpha) @ C_smooth @ diag(alpha)` where alpha is Gaussian locality mask.

- `gradient_and_smoothing_bottleneck_nat_imgs_utility_optimization.tex` (~920 lines): Diagnoses why gradient-ascent optimization produces spatially smooth images. Key finding: **C matrix acts as a spatial low-pass filter on ALL utility gradients**. With typical kernel parameters (rho_nat ~ 0.1), only ~3 spatial frequency modes per axis survive. The smoothing bottleneck is **fundamental to the kernel**, not a numerical artifact.

### Branches

- `pietro/pca-utility-optimization` (current): has `subspace_optimization.py`, `pca_optimization.py`, `c_eigen_optimization.py` (reference), tests, handoffs.
- `pietro/c-eigen-utility-optimization`: has the standalone C-eigen script + LaTeX documents + 8 experiment PNGs. Only merge conflict with PCA branch is `SESSION_LOG.md`.

## What Was Done This Session

### Attempt: Unified Script with Two Separate Methods

- **What**: Created `subspace_optimization.py` that supports `--method {pca, c_eigen}` as two independent alternatives. Shared code: `z_to_image()`, `image_to_z()`, `rf_pearson_r()`, unified `gradient_ascent()`, shared plotting. Both decompositions return `(offset, basis, eigenvalues_all, K, meta)` — PCA offset=mu_rf, C-eigen offset=zeros.
- **Result**: Both methods tested and working.
  - PCA (RBF, var_threshold=0.80): K=197/2356 (8.4%), utility 0.2027->0.2033, Pearson r 0.32->0.38, 0% OOB, converged in 7 steps (early stop).
  - C-eigen (arc_cosine, thresh=1e-3): K=28/1725 (1.6%), utility 0.976->1.013, Pearson r 0.30->0.32, 49% OOB (expected norm explosion), converged in 7 steps.
- **Verdict**: Technically correct but **NOT what the user wants**. The user wants PCA and C-eigenspace combined in a single optimization, not as alternative methods in a single script.

### Brought C-eigen Script as Reference

Extracted `c_eigen_optimization.py` from `pietro/c-eigen-utility-optimization` into the PCA branch as a reference file. This makes all code available on one branch.

## Key Mathematical Context for Combined Approach

### What PCA Gives

PCA on RF-masked images yields basis V_K in R^{n_rf x K} where K = min(n_samples, n_rf) at full rank, or fewer with variance truncation. The offset mu_rf is the dataset mean.

- Eigenvalues = variances along each principal direction
- Truncation discards low-variance (high-frequency) directions
- K=197 captures 80% of variance in 2356 RF pixels
- Full rank (K=2356) is mathematically equivalent to unconstrained optimization

### What C-eigenspace Gives

C matrix eigendecomposition yields basis U_K in R^{n_rf x K} where K depends on eigenvalue threshold.

- C matrix structure: locality mask (alpha) x smooth spatial correlations (C_smooth) x locality mask
- Eigenvalue spectrum extremely skewed: top eigenvalue = ~44% of total mass
- K=28 at threshold 1e-3 captures 99.7% of eigenvalue mass
- At no_filter (K=n_rf), it's just a rotation — no information loss or gain
- **The gradient of utility through C is: nabla_x U = C @ nabla_pixel U.** The C matrix multiplies ALL gradients. Directions with low C eigenvalues get near-zero gradients regardless.

### How They Could Combine

The open question is what "combined PCA + C-eigenspace" means mathematically. Some possibilities:

**Option A: Two-stage**. First restrict to PCA subspace (naturalness), then within that subspace, use C-eigenspace to identify which PCA directions the kernel can actually "see" (non-zero gradient). Mathematically: project the C matrix into PCA space and eigendecompose the projected C matrix.

**Option B: Joint basis**. Find eigenvectors of C restricted to the PCA subspace. Concretely: compute `C_pca = V_K^T @ C @ V_K` (K x K matrix), eigendecompose it, optimize in the resulting basis. This gives directions that are BOTH natural (in data manifold) AND have non-zero kernel gradient.

**Option C: Weighted PCA**. Instead of standard PCA (covariance eigenvectors), do PCA weighted by C eigenvalues. Directions with high data variance AND high C eigenvalue are prioritized.

**Option D: Cascaded constraint**. Parameterize as `x_rf = mu + V_K @ U_proj @ z` where U_proj are the C-eigenspace directions projected into PCA space.

**Note**: Options A and B may be mathematically equivalent — the next session should verify this.

### Why Combining Makes Scientific Sense

- PCA alone at low thresholds (e.g., K=197 for 80% variance) discards high-frequency content, but the surviving directions may include some that the kernel cannot utilize (near-zero C eigenvalue in that direction). These directions waste optimization capacity.
- C-eigenspace alone doesn't constrain to natural images — the optimizer can (and does) produce extreme pixel values (49% OOB with arc-cosine).
- Combined: only optimize along directions that are (a) natural (high PCA variance) AND (b) visible to the kernel (high C eigenvalue). This should give a smaller, better-conditioned optimization space.

## Key Findings From Previous Sessions

1. **CONFIRMED**: LBFGS works with RBF kernel at M=50 (utility scale ~0.1-0.2). Failed at M=300 (only 72 RF pixels, utility ~1e-4 below tolerance).

2. **CONFIRMED**: Arc-cosine kernel causes norm-driven utility explosion (k(x,x) ~ ||x||^2). RBF (k(x,x)=1) is well-behaved for optimization.

3. **CONFIRMED**: PCA covariance should be computed on ALL available images (3160 = train+pool), not just training set (50). With 3160 > n_rf, PCA is full rank.

4. **CONFIRMED**: Norm constraint (95th percentile of training z-norms) needed for PCA with arc-cosine, not needed for RBF.

5. **CONFIRMED**: Early stopping (5-step convergence check) works — both methods converge quickly (5-7 steps) in tested configurations.

6. **HYPOTHESIS**: The optimal var_threshold (PCA) balances naturalness vs expressiveness. Lower threshold = more natural but lower achievable utility. Not yet systematically tested.

7. **CONFIRMED**: C matrix eigenvalue spectrum is extremely skewed — top eigenvalue ~44% of total mass. K=28 at threshold 1e-3 captures 99.7%.

## Why This Was Stopped

Context running out. The unified-but-separate script (`subspace_optimization.py`) was built and tested, but the user clarified they want the two approaches **combined together** in a single optimization, not as alternatives. This requires mathematical design work that the next session should tackle.

## Things Noticed But Not Acted Upon

1. The `subspace_optimization.py` script we built is still useful infrastructure — its shared `gradient_ascent()`, `z_to_image()`, and plotting functions can be reused for the combined approach. The combined script may want to import or build on it.

2. The test file `test_subspace_optimization.py` was written by the user and tests the current separate-method architecture. It will need updating once the combined approach is implemented.

3. The LaTeX documents on the C-eigen branch should probably be brought over to the PCA branch (or a combined branch) so all mathematical references are co-located with the code.

4. The user's comprehensive summary of the C-eigen branch (provided in conversation context, not in a file) contains more detail than `HANDOFF_C_EIGEN.md`. Key additions: multi-conditioning mode, chunked gradient accumulation, early stopping, per-step timing, dual plot functions, complete constant/CLI reference. This is captured in the conversation summary at the top of this session.

5. Model test_r is low (0.25 for RBF M=50 n=50, 0.36 for arc_cosine M=50 n=50). This is expected for M=50/n_train=50 — the model doesn't fit well, but the optimization landscape is richer.

## Uncommitted Changes

```
 D investigations/utility_decompositions/pca_optimization.png         (deleted old generic output)
?? investigations/utility_decompositions/c_eigen_optimization.py       (reference, from C-eigen branch)
?? investigations/utility_decompositions/subspace_optimization.py      (new unified script)
?? investigations/utility_decompositions/test_subspace_optimization.py (tests, written by user)
?? investigations/utility_decompositions/subspace_c_eigen_arc_cosine_thresh1e-03.png  (test output)
?? investigations/utility_decompositions/subspace_pca_rbf_vt0.80.png                  (test output)
?? investigations/utility_decompositions/HANDOFF_COMBINED_SUBSPACE.md  (this file)
?? investigations/utility_decompositions/pca_optimization_*.png        (previous session outputs)
```

## Files Created This Session

| File | Purpose | Keep/Delete |
|------|---------|-------------|
| `subspace_optimization.py` | Unified script (separate PCA/C-eigen methods) | Keep — reusable infrastructure |
| `c_eigen_optimization.py` | Reference copy from C-eigen branch | Keep as reference |
| `test_subspace_optimization.py` | Tests (user-written) | Keep |
| `subspace_pca_rbf_vt0.80.png` | Test output | Delete when no longer needed |
| `subspace_c_eigen_arc_cosine_thresh1e-03.png` | Test output | Delete when no longer needed |
| `HANDOFF_COMBINED_SUBSPACE.md` | This handoff | Keep |

## If Someone Revisits This

**The goal**: Create a script that combines PCA and C-eigenspace in a single optimization. The most promising approach (Option B above): eigendecompose `C_pca = V_K^T @ C @ V_K` and optimize in the resulting basis. This gives a basis that is both natural (PCA) and kernel-visible (C-eigenspace).

**What to try first**:
1. Compute PCA basis V_K (from dataset covariance, truncated by var_threshold)
2. Project C matrix into PCA space: `C_pca = V_K^T @ C @ V_K` (K x K)
3. Eigendecompose C_pca to get U_combined. The combined basis is `V_K @ U_combined`
4. Optimize z in this combined basis: `x_rf = mu + V_K @ U_combined @ z`
5. Compare with: PCA-only at same total dimension, C-eigen-only, unconstrained pixel

**What NOT to try**:
- LBFGS with M=300/n=300 and arc_cosine: only 72 RF pixels, dead end.
- Adam + hard norm projection with arc_cosine: momentum interaction causes utility descent.
- Avoid building a new script from scratch — the existing `subspace_optimization.py` has working gradient_ascent, plotting, and shared infrastructure.

**Key infrastructure already available**:
- `subspace_optimization.py`: `z_to_image()`, `image_to_z()`, `gradient_ascent()`, `compute_pca()`, `compute_c_eigenspace()`, plotting functions. All tested.
- `acquisition.py`: `distribution_aware_utility()` — fully differentiable, supports gradient flow through x*.
- `explore_utility.py`: `setup()` — trains model and returns everything needed.
- `test_subspace_optimization.py`: mathematical invariant tests, needs extension for combined method.

**Parameter defaults for testing**:
- Model: `default_gpy`, M=50, n_train=50, seed=42, cell=8 (from `default_params.json`)
- PCA: var_threshold=0.80, computed on all 3160 images
- C-eigen: eigen_rel_threshold=1e-3, relative to max eigenvalue
- Optimizer: LBFGS, lr=0.5, 50 steps, strong_wolfe
- Kernel: RBF for well-behaved optimization, arc_cosine to test norm sensitivity

---

## Continuation Prompt

```
I am continuing the subspace utility optimization investigation.
The goal: COMBINE PCA and C-eigenspace decompositions into a single
optimization — not as separate alternatives, but used together.

Read the handoff: investigations/utility_decompositions/HANDOFF_COMBINED_SUBSPACE.md

Branch: pietro/pca-utility-optimization

Key context:
- PCA gives data manifold directions (naturalness constraint)
- C-eigenspace gives kernel-visible directions (gradient conditioning)
- Combined: optimize only along directions that are BOTH natural AND kernel-visible
- Most promising approach: eigendecompose V_K^T @ C @ V_K (PCA-projected C matrix)
- Existing infrastructure in subspace_optimization.py: z_to_image, gradient_ascent, compute_pca, compute_c_eigenspace, plotting — all tested
- Tests in test_subspace_optimization.py (user-written, 9 test groups)
- LBFGS optimizer, RBF kernel, M=50, n_train=50, default_gpy mode

Check git status and git branch before starting.
Do NOT modify pietro/workingbranch.
```
