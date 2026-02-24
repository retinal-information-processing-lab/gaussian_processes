# Subspace Utility Optimization — Reference Document

**Branch**: `pietro/pca-utility-optimization`
**Last updated**: 2026-02-24
**Location**: `investigations/utility_decompositions/`

---

## What This Script Does

`subspace_optimization.py` (~1400 lines) optimizes images to maximize distribution-aware (DA) utility for active learning in a variational GP model. Instead of optimizing in full pixel space (11664 dimensions), it optimizes in a low-dimensional subspace parameterized by `x_rf = mu + basis @ z`, where z is the optimization variable.

Three methods select the basis differently, but all share the same optimizer, plotting, and diagnostics.

---

## The Three Methods

### PCA (`--method pca`)

**Math**: `x_rf = mu + V_K @ z`, where V_K = top K eigenvectors of dataset covariance.

- Offset mu is the dataset mean over RF pixels. This is **mathematically intrinsic** to PCA (centering before SVD).
- Truncation is a **genuine constraint** — it excludes directions with low data variance (unnatural images).
- K controlled by `--var-threshold` (fraction of variance retained) or `--n-components` (explicit count).
- At var_threshold=0.80: K=197 out of 2356 RF pixels.

### C-eigenspace (`--method c_eigen`)

**Math**: `x_rf = mu + U_K @ z`, where U_K = top K eigenvectors of the kernel's C matrix.

- Offset mu is the dataset mean. This is **NOT mathematically intrinsic** to C-eigenspace (the C eigendecomposition passes through the origin). We use mu as a practical choice to keep reconstructions within pixel bounds and give the kernel a realistic operating point. This changes the utility landscape — the kernel sees `C @ (mu + U_K @ z)` instead of `C @ (U_K @ z)`.
- Truncation is a **reparameterization** at full rank (no information loss), but at reduced rank it removes near-zero gradient directions.
- K controlled by `--eigen-threshold` (relative to max eigenvalue). Default 1e-3.
- At threshold=1e-3: K=28 out of 2356 RF pixels, capturing 99.7% of eigenvalue mass.
- **Float32 noise floor**: eigenvalues below ~1e-7 * max are numerical noise. Setting threshold below ~1e-6 degrades reconstruction quality because the corresponding eigenvectors are garbage. The default 1e-3 is well above this limit.

### Combined (`--method combined`)

**Math**: `x_rf = mu + (V_K @ W) @ z`, where W = eigenvectors of `C_pca = V_K^T @ C @ V_K`.

- First computes PCA basis V_K (truncated by var_threshold).
- Projects the C matrix into PCA space: `C_pca = V_K^T @ C @ V_K` (K_pca x K_pca).
- Eigendecomposes C_pca and keeps directions above eigen_threshold.
- Combined basis = `V_K @ W_retained` — directions that are BOTH natural (PCA) AND kernel-visible (C-eigenspace).
- Two independent thresholds: `--var-threshold` (PCA step) and `--eigen-threshold` (C_pca step).
- Offset mu from PCA step (mathematically intrinsic).
- At var_threshold=0.80, eigen_threshold=1e-3: n_rf=2356 -> K_pca=197 -> K_combined=28.

---

## Offset Convention

All three methods use `offset = mu_rf` (dataset mean over RF pixels). This makes the script consistent:
- `z_to_image(z, basis, offset, ...)` always computes `x_rf = mu + basis @ z`
- `image_to_z(x_full, basis, offset, ...)` always computes `z = basis^T @ (x_rf - mu)`
- z=0 maps to the dataset mean image for all methods.

For PCA and combined, centering is part of the math. For C-eigen, it is a practical choice (documented in function docstrings).

---

## Optimization Details

**Optimizer**: LBFGS with strong_wolfe line search.

| Constant | Value | Purpose |
|----------|-------|---------|
| N_STEPS | 50 | Max outer LBFGS steps |
| LR | 0.5 | LBFGS step size |
| LBFGS_MAX_ITER | 20 | Line search iterations per step |
| LBFGS_MAX_EVAL | 25 | Function evals per step |
| LBFGS_HISTORY_SIZE | 10 | Past gradients for Hessian approximation |
| GRAD_CHUNK_SIZE | 30 | Images per gradient accumulation chunk |
| TARGET_INDEX | 5 | Default pool image for single-target |
| M_OVERRIDE | 50 | Inducing points |
| N_TRAIN_OVERRIDE | 50 | Training points |

**Starting point**: Dataset mean projected into subspace (all methods, both single-target and multi-cond).

**No constraints**: No norm constraint, no sigmoid, no pixel clipping during optimization. The optimization is fully unconstrained within the subspace. OOB pixels are flagged visually in plots (red title) but not prevented.

**Early stopping**: Stops if utility doesn't change for 5 consecutive steps.

**f_max guard**: If predicted firing rate exceeds f_max (100), LBFGS step is rejected.

**Multi-conditioning** (`--n-cond N` for N > 1): Conditions DA utility on N randomly selected pool images instead of a single target. Uses chunked gradient accumulation for GPU memory.

---

## File Inventory

| File | Lines | Description |
|------|-------|-------------|
| `subspace_optimization.py` | ~1400 | Main script. Three methods, shared optimizer/plotting. |
| `test_subspace_optimization.py` | 596 | Tests (user-written). 9 test groups: PCA math, C-eigen math, gradient flow, rf_pearson_r, norm constraint, utility improvement, subspace membership, cross-method consistency, early stopping. **Note**: tests were written before the offset unification — C-eigen tests may need updating to expect mu_rf offset instead of zeros. |
| `HANDOFF_PCA.md` | 149 | Historical PCA-only handoff (superseded by this file). |
| `HANDOFF_C_EIGEN.md` | ~350 | Historical C-eigen handoff (outdated). |
| `HANDOFF_COMBINED_SUBSPACE.md` | This file. |
| `*.png` | various | Test output plots. Can be deleted when no longer needed. |

Old scripts `pca_optimization.py` and `c_eigen_optimization.py` have been deleted (recoverable from git history).

---

## Function Map

| Function | Purpose |
|----------|---------|
| `z_to_image(z, basis, offset, rf_mask, ...)` | Subspace -> full image: `x_rf = offset + basis @ z` |
| `image_to_z(x_full, basis, offset, rf_mask)` | Full image -> subspace: `z = basis^T @ (x_rf - offset)` |
| `rf_pearson_r(x, target, rf_mask)` | Pearson correlation within RF mask |
| `compute_pca(X_images, rf_mask, var_threshold, n_components)` | PCA decomposition. Returns `(offset, basis, eigenvalues_all, K, meta)` |
| `compute_c_eigenspace(kernel, rf_mask, eigen_rel_threshold, data_mean_rf, no_filter)` | C matrix eigendecomposition. Returns same tuple signature. |
| `compute_combined(X_images, kernel, rf_mask, var_threshold, eigen_rel_threshold, n_components)` | PCA + C_pca combined. Returns same tuple signature. |
| `gradient_ascent(model, likelihood, z_start, x_samples, basis, offset, ...)` | LBFGS optimization. Returns `(x_final, z_final, history)` |
| `_setup_plot_helpers(rf_mask, kernel, n_px_side, vmin, vmax)` | Creates closures: `masked_crop`, `check_oob`, `draw_rf_overlay` |
| `plot_results_single(...)` | 2x4 grid: 4 images + eigenvalue spectrum + convergence |
| `plot_results_multicond(...)` | 2x4 grid: start/final/diff/summary + spectrum + convergence |

All decomposition functions return the same 5-tuple `(offset, basis, eigenvalues_all, K, meta)`, enabling shared downstream code.

---

## CLI Reference

```
python subspace_optimization.py [OPTIONS]

Method selection:
  --method {pca,c_eigen,combined}   Subspace method (default: pca)

PCA controls (used by pca and combined):
  --var-threshold FLOAT             Variance retained (default: 0.8)
  --n-components INT                Explicit PCA rank (overrides var-threshold)

C-eigenspace controls (used by c_eigen and combined):
  --eigen-threshold FLOAT           Relative eigenvalue threshold (default: 1e-3)
  --no-filter                       C-eigen: keep ALL eigenvectors

Model/kernel:
  --kernel-type {arc_cosine,arc_sine,rbf}
  --M INT                           Inducing points (default: 50)
  --n-train INT                     Training points (default: 50)

Image selection:
  --target-index INT                Pool image for single-target (default: 5)
  --n-cond INT                      Conditioning images (1=single, >1=multi)
  --cond-seed INT                   Seed for conditioning selection (default: 42)
  --start-index INT                 Explicit start image (multi-cond)
  --start-seed INT                  Seed for random start (default: 123)
```

---

## Dependencies

| Dependency | Imported from |
|------------|---------------|
| `setup()` | `investigations/utility/explore_utility.py` |
| `distribution_aware_utility()` | `acquisition.py` (via importlib) |
| `get_gp_marginal_moments()`, `compute_H()` | `utils.py` (via importlib) |
| Model training, kernel, likelihood | `default_params.json` via `build_config_from_defaults()` |

---

## Key Findings

1. **RBF kernel is well-behaved** for optimization (k(x,x)=1, no norm explosion). Arc-cosine causes norm-driven utility explosion (k(x,x) ~ ||x||^2).

2. **Combined achieves near-identical utility to PCA** with far fewer dimensions (K=28 vs K=197 for same thresholds).

3. **C-eigen eigenvalue spectrum is extremely skewed**: top eigenvalue ~44% of total mass. K=28 at threshold 1e-3 captures 99.7%.

4. **Float32 noise floor**: C-eigen threshold below ~1e-6 degrades reconstruction quality. Eigenvectors for tiny eigenvalues are numerical garbage.

5. **Subspace projection can produce OOB pixels**: Linear projection doesn't preserve element-wise pixel bounds, only L2 norm. This is mathematically expected, not a bug. Using mu as offset (centering) significantly reduces the effect.

6. **Early stopping works**: All methods converge quickly (5-7 steps) in tested configurations.

7. **Model test_r is low** at M=50/n_train=50 (0.25 for RBF). Expected — small model, but optimization landscape is richer.

---

## LaTeX Documents

On branch `pietro/c-eigen-utility-optimization` only (not on current branch):

- `pca_vs_c_eigenspace.tex` (~1160 lines): Proves PCA = genuine constraint, C-eigen = reparameterization.
- `gradient_and_smoothing_bottleneck_nat_imgs_utility_optimization.tex` (~920 lines): C matrix as spatial low-pass filter. Only ~3 spatial frequency modes per axis survive with typical kernel params.

---

## Open Idea: Input Warping for Pixel Bound Enforcement

**Status**: Untested idea. Noted here for future investigation.

### The Problem

Subspace optimization (all three methods) can produce images with pixel values outside the physical display range [vmin, vmax]. The utility function does not penalize this because it operates through the kernel, which applies the C matrix (a spatial smoother) before evaluating. Individual pixel bound violations are invisible to C — a pixel at 2.5 (in bounds) and 2.6 (out of bounds) look the same after spatial smoothing. The utility landscape has no "walls" at pixel bounds.

With arc-cosine kernel, norm growth eventually triggers the f_max firing rate guard, which indirectly limits OOB. But with RBF kernel (k(x,x)=1 always), there is no such mechanism.

### The Idea: Kernel-Level Input Warping

Instead of constraining the optimizer (sigmoid reparameterization, clipping, norm projection), modify the kernel itself to be unaware of OOB pixel values. Apply a per-pixel nonlinear warping function w() **before** the C matrix:

```
k(x, y) = exp(-||C @ w(x) - C @ w(y)||^2 / (2l^2))
```

where w() applies a soft-clipping function to each pixel independently:
- Within [vmin, vmax]: w(x_i) ~ x_i (approximately linear, minimal distortion)
- Outside [vmin, vmax]: w(x_i) saturates (diminishing response)

Candidate functions: scaled tanh, sigmoid mapped to [vmin, vmax], Beta CDF.

### Why This Differs From Sigmoid Reparameterization

Sigmoid reparameterization during optimization (`x = sigmoid(z)`) is a constraint on the search — the model trains on raw pixels and may predict high utility just beyond the bound. The sigmoid forces the optimizer to stay in bounds but doesn't change what the model has learned. You are fighting the model's landscape.

Input warping changes the model itself. The kernel trains on w(x), so learned parameters (lengthscale, C matrix, inducing points) all adapt to a world where pixel values saturate at bounds. The model genuinely cannot distinguish "at the bound" from "past the bound." The utility landscape itself has no incentive to go OOB. No optimizer constraint needed.

### Analogy

The C matrix is a **spatial RF** — it determines WHERE the kernel looks. Input warping would be a **dynamic range RF** — it determines WHAT pixel values the kernel can respond to. Biological neurons have both: a spatial receptive field and a saturation/response curve that compresses extreme inputs. The physical display projector has this too (it clips at [0, 255]).

### Considerations

- This is a **modeling change**, not a post-hoc fix. The model must be retrained with warped inputs. Results will differ from the unwarped model.
- The warping goes **before C** in the pipeline: `x -> w(x) -> C @ w(x) -> kernel`. Order matters — we want per-pixel saturation before spatial smoothing.
- The warping function must be differentiable for gradient-based training and optimization.
- The warping adds a nonlinearity to what is currently a linear pipeline (`x -> C @ x`). This may interact with the eigenspace decompositions in nontrivial ways — the C eigenvectors are computed for the linear C, not for `C @ w()`.
- Input warping is well-established in the GP/Bayesian optimization literature (Snoek et al. 2014, "Input Warping for Bayesian Optimization of Non-stationary Functions"), but typically for low-dimensional inputs. Applying it per-pixel to 11664 dimensions is unusual and may have unexpected effects.
- It is unclear whether the warping would need to be differentiable through the training loop or only during utility optimization. If only during optimization, the model could train on raw pixels and the warping would be applied post-hoc — but then we lose the "model learns the bounds" property.
- An intermediate approach: add a multiplicative penalty term to the kernel rather than warping inputs. Something like `k_bounded(x, y) = k_base(Cx, Cy) * prod_i phi(x_i)` where phi(x_i) ~ 1 in bounds and drops toward 0 outside. This preserves the linear C pipeline but still encodes bounds at the kernel level. Untested.

### What To Try First

If investigating this, a minimal experiment would be:
1. Pick a simple warping function (e.g., scaled tanh that maps [vmin, vmax] to ~[vmin, vmax] with saturation outside)
2. Apply it in `kernels.py` before the C matrix multiplication
3. Retrain the model with warped inputs
4. Check: does test_r change? Do utility-optimized images stay in bounds without explicit constraints?

This should be a separate investigation, not mixed into the subspace optimization work.

---

## Continuation Prompt

```
I am working on the subspace utility optimization investigation.

Read: investigations/utility_decompositions/HANDOFF_COMBINED_SUBSPACE.md

Branch: pietro/pca-utility-optimization

The script subspace_optimization.py supports three methods:
  --method pca        (data manifold constraint)
  --method c_eigen    (kernel C matrix eigenvectors)
  --method combined   (PCA-filtered C-eigenspace)

All methods use offset = dataset mean, LBFGS optimizer, no constraints.

Check git status and git branch before starting.
Do NOT modify pietro/workingbranch.
```
