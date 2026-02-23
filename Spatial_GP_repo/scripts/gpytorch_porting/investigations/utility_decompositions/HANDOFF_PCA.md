# Investigation: PCA-Constrained Utility Optimization

**Branch**: `pietro/pca-utility-optimization`
**Date**: 2026-02-23
**Status**: Continuing
**Location**: `investigations/utility_decompositions/`

---

## Problem Statement

We want to optimize images to maximize distribution-aware (DA) utility for a variational GP model. Unconstrained pixel-space optimization produces images that exploit kernel properties (especially arc-cosine norm sensitivity) rather than finding informative natural-looking images. PCA-constrained optimization restricts the optimizer to the data manifold: x* = mu + V_K @ z, where V_K are the top K principal components of the natural image dataset. This adds genuine information about the data distribution that is absent from the kernel, and changes the optimization solution (unlike C-eigenspace optimization which is just a reparameterization).

The long-term goal is to use this for active learning — find the most informative image to show a neuron, constrained to look like a natural image.

## What Was Tried

### 1. Initial Implementation (Agent-Written)
- **What**: PCA agent created `pca_optimization.py` using Adam optimizer, sigmoid pixel bounds, PCA on training images only (n_train=300, M=300), START_NOISE for perturbed start
- **Result**: Working script. With M=300/n=300, only 72 RF pixels. Utility ~1e-4, gradients ~1e-5. LBFGS failed (strong_wolfe rejected all steps below tolerance). Adam worked but with norm explosion (z-norm 5 -> 54 without constraint).
- **Interpretation**: The tiny RF (72 pixels) made the utility landscape too flat for LBFGS. Adam always steps regardless of gradient magnitude.
- **Verdict**: Foundation works, but needed iteration on optimizer, constraints, and model config.

### 2. Switch to M=50, n_train=50 with RBF kernel
- **What**: Used smaller M to get larger RF (2356 pixels for RBF, 1725 for arc_cosine). Ran both kernel types.
- **Result**: RBF utility ~0.14-0.20 (well above LBFGS tolerance). Arc-cosine utility ~0.005-0.07. RBF convergence smooth; arc-cosine suffers from norm-driven utility explosion.
- **Interpretation**: M=50 gives a wider RF mask, making the optimization landscape much richer. RBF is better behaved because K(x,x)=1 (no norm sensitivity).
- **Verdict**: M=50 + RBF is the better test configuration. Arc-cosine needs norm constraints or different approach.

### 3. PCA Covariance on Full Dataset
- **What**: Changed PCA computation from training images only (50 images) to all available images (3160 = train + pool, excluding test). This ensures the covariance estimate is full-rank when n_samples > n_rf.
- **Result**: With 3160 images and 2356 RF pixels, PCA gives full-rank decomposition. At 95% variance threshold: K=773 components.
- **Interpretation**: Using all images eliminates the rank bottleneck (previously limited to min(n_train, n_rf) = 50 components). At var_threshold=1.0, optimization is mathematically identical to unconstrained pixel-space optimization.
- **Verdict**: Confirmed improvement. This is the correct approach.

### 4. Adam vs LBFGS
- **What**: Tested both optimizers with M=50, RBF, START_NOISE=5 (random start), norm constraint.
- **Result**:
  - Adam (100 steps, lr=0.1): utility 0.10 -> 0.20, aggressive movement, Pearson r near 0, momentum+projection interaction causes instability with arc-cosine
  - LBFGS (50 outer steps, lr=0.5): utility 0.19 -> 0.20, converges in ~5 steps then stops, less destructive to structure (Pearson r 0.69 for target 5), cosine similarity z_final/z_target = 0.72
- **Interpretation**: LBFGS converges fast and preserves more structure. Adam is more aggressive but fights the norm projection (momentum state becomes stale after projection). Arc-cosine + Adam + projection caused utility to DECREASE for target_index=5.
- **Verdict**: LBFGS is the current choice. Works with RBF + M=50 (utility scale ~0.1-0.2).

### 5. Starting Point: Mean Image
- **What**: Changed start from noisy z_target to z=0 (dataset mean in PCA space). Added firing rate sanity check before optimization.
- **Result**: Mean image has modest firing rate with RBF. Optimization proceeds normally from this neutral starting point.
- **Interpretation**: Starting from mean is cleaner experimentally — no bias toward the target, tests whether optimizer genuinely finds high-utility images.
- **Verdict**: Current default. Arc-cosine may fail from mean start (firing rate overflow) — error message explains this.

### 6. Variance Threshold Experiments
- **What**: User ran experiments at var_threshold = 0.60, 0.80, 0.95 with RBF, and 0.80 with arc_cosine. Output PNGs saved with threshold in filename.
- **Result**: PNGs exist at `pca_optimization_rbf_vt0.60.png`, `_vt0.80.png`, `_vt0.95.png`, and `pca_optimization_arc_cosine_vt0.80.png`. Script now shows U_DA on all four panels (original target, PCA-projected target, start, final) with independent OOB checks per panel.
- **Interpretation**: Lower thresholds = more aggressive constraint = more natural-looking but less room to optimize. The PCA-projected target panel shows how much the truncation itself costs in utility.
- **Verdict**: This is the key experimental axis to explore further.

## Key Findings

1. CONFIRMED: PCA-constrained optimization genuinely constrains the solution (unlike C-eigenspace which is a reparameterization). At var_threshold < 1.0, the optimizer cannot reach arbitrary pixel combinations — only directions with observed variance in the dataset.

2. CONFIRMED: var_threshold=1.0 with n_samples >= n_rf is mathematically equivalent to unconstrained pixel-space optimization. The PCA basis is complete, z <-> x_rf is a bijection.

3. CONFIRMED: The rank of the PCA decomposition is min(n_samples, n_rf). With n_train=50 and n_rf=1725, even var_threshold=1.0 gives only 50 components. Using all 3160 images gives full rank (2356 components for RBF).

4. CONFIRMED: LBFGS works with RBF kernel at M=50 (utility ~0.1-0.2 scale). It failed at M=300 with only 72 RF pixels (utility ~1e-4, below LBFGS tolerance).

5. CONFIRMED: Arc-cosine kernel causes norm-driven utility explosion (k(x,x) ~ ||x||^2). Projected gradient + Adam momentum interaction can cause utility to DECREASE (observed for target_index=5). RBF (k(x,x)=1) is well-behaved.

6. CONFIRMED: Per-panel OOB pixel checking catches silent clipping. At var_threshold=0.80/RBF, final images show 10-12% OOB pixels.

7. HYPOTHESIS: The optimal var_threshold balances naturalness constraint vs optimization expressiveness. Lower threshold = more natural but lower achievable utility. This tradeoff is the scientific question to answer next.

## Why This Was Stopped

Context running out. The investigation is producing useful results and the script is in good shape. Continuing in next session with focus on: systematic threshold sweep, interpreting what the optimizer actually does in PCA space, and whether the PCA constraint produces scientifically meaningful "optimal" images.

## Things Noticed But Not Acted Upon

1. The norm constraint (95th percentile of training z-norms) interacts badly with Adam's momentum. LBFGS doesn't have this problem because it doesn't maintain momentum state. If Adam is ever needed again, consider SGD without momentum or Lagrangian penalty instead of hard projection.

2. The C-eigenvalue branch (`pietro/c-eigen-utility-optimization`) exists with 3 commits and its own handoff (`HANDOFF_C_EIGEN.md`). It has a working script and results but was not the focus of this session. The only merge conflict with PCA branch is `SESSION_LOG.md`.

3. The `START_NOISE` parameter is still in the script (default 5) but is not used when starting from mean (z=0). It's dead code in the current flow. Could be cleaned up or repurposed for a "start from noisy target" mode.

4. The firing rate diagnostics section in the script prints lambda_m, mu_g, firing_rate, H_marg for target/start/final — useful diagnostic data that could be added to the plot.

5. `explore_utility.setup()` defaults (M, n_train) come from module-level constants in that file, overridable via CLI args. The PCA script passes `--M` and `--n-train` through to setup(). But `default_params.json` values for seed and cell are used throughout.

## Uncommitted Changes

```
 D investigations/utility_decompositions/pca_optimization.png   (deleted old generic output)
?? investigations/utility_decompositions/pca_optimization_arc_cosine_vt0.80.png  (user run)
?? investigations/utility_decompositions/pca_optimization_rbf_vt0.60.png         (user run)
?? investigations/utility_decompositions/pca_optimization_rbf_vt0.80.png         (user run)
?? investigations/utility_decompositions/pca_optimization_rbf_vt0.95.png         (user run)
?? investigations/utility_decompositions/HANDOFF_C_EIGEN.md     (from C-eigen agent, not our work)
```

The PNGs are user-generated experiment outputs. The script itself is fully committed (3e7f9eb).

## Files Created

| File | Purpose | Keep/Delete |
|------|---------|-------------|
| `investigations/utility_decompositions/pca_optimization.py` | Main PCA optimization script | Keep |
| `investigations/utility_decompositions/HANDOFF_PCA.md` | This handoff | Keep |
| `investigations/utility_decompositions/pca_optimization_rbf_vt*.png` | User experiment outputs | Keep (reference) |
| `investigations/utility_decompositions/pca_optimization_arc_cosine_vt0.80.png` | User experiment output | Keep (reference) |

## If Someone Revisits This

**Next steps (most promising first):**
1. Systematic var_threshold sweep: run 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 1.00 with RBF and compare utility achieved vs image naturalness. The 4-panel plot (original target, PCA target, start, final) with U_DA values makes comparison straightforward.
2. Interpret what the optimizer does in PCA space: which PCA components get amplified? Does the optimizer concentrate on low-frequency (large eigenvalue) or high-frequency (small eigenvalue) directions?
3. Consider whether Pearson r to target is the right structural similarity metric, or whether something else (SSIM, perceptual loss) is more informative.
4. Try multiple target images systematically (--target-index) to see if results are consistent.

**What NOT to try:**
- LBFGS with M=300/n=300 and arc_cosine: only 72 RF pixels, utility ~1e-4, LBFGS rejects all steps. Dead end.
- Adam + hard norm projection with arc_cosine: momentum/projection interaction causes utility descent. Use LBFGS or SGD without momentum.
- START_NOISE > 1 with norm constraint: z gets immediately clipped to boundary, losing all target structure. If you want a random start, just use z=0 (mean).

**Prerequisites that would help:**
- Understanding whether the neuron actually cares about high-frequency structure (if not, aggressive PCA truncation is free).
- A way to compare "naturalness" of optimized images beyond Pearson r.

---

## Continuation Prompt

```
I am continuing the PCA-constrained utility optimization investigation.

Read the handoff: investigations/utility_decompositions/HANDOFF_PCA.md

Branch: pietro/pca-utility-optimization
Script: investigations/utility_decompositions/pca_optimization.py

Key context:
- Optimizes x* = mu + V_K @ z to maximize DA utility, constrained to PCA subspace
- LBFGS optimizer, RBF kernel, M=50, n_train=50, default_gpy mode
- PCA computed on all 3160 images (not just training), full rank when n_samples > n_rf
- var_threshold controls truncation: lower = more natural, less expressive
- Script produces 4-panel plots with U_DA on each panel + convergence curves
- Start from dataset mean (z=0), norm constraint = 95th percentile of training z-norms

Check git status and git branch before starting.
Do NOT modify pietro/workingbranch.
```