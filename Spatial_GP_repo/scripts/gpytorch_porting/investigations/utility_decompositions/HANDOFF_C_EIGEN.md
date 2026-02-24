# Handoff: C-Eigenspace Utility Optimization Investigation

**Date**: 2026-02-21
**Branch**: `pietro/c-eigen-utility-optimization` (branched from `pietro/workingbranch`)
**Status**: First working version complete, ready for exploration and iteration

---

## Continuation Prompt

Copy-paste the block below into a new Claude Code session to continue this investigation.



```
You are continuing an investigation into C-eigenspace image optimization for maximizing
distribution-aware utility in a variational GP model. This is experimental scientific work.
Your job is to modify the working script, interpret results, and explore the mathematics.

======================================================================
BRANCH SAFETY
======================================================================
You are on branch: pietro/c-eigen-utility-optimization
NEVER touch, push to, or checkout pietro/workingbranch.
Verify with `git branch --show-current` before any commit.

======================================================================
WHAT EXISTS
======================================================================

Working script:
  investigations/utility_decompositions/c_eigen_optimization.py

Output plot:
  investigations/utility_decompositions/c_eigen_optimization.png

LaTeX references (READ THESE for full mathematical context):
  investigations/utility_decompositions/pca_vs_c_eigenspace.tex
  investigations/utility_decompositions/gradient_and_smoothing_bottleneck_nat_imgs_utility_optimization.tex

======================================================================
WHAT THE SCRIPT DOES (step by step)
======================================================================

1. TRAIN MODEL: Calls setup(M_override=50, n_train_override=50) from
   investigations/utility/explore_utility.py, which internally calls
   build_config_from_defaults(mode='default_gpy', M=50, n_train=50)
   and then run_single_config(config). Returns trained model, likelihood,
   pool images, training images, config dict.

2. EXTRACT C MATRIX: Gets the kernel's C matrix on RF-masked pixels:
     kernel._compute_C_matrix(apply_mask=True)
   C = Amp * alpha[:, None] * C_smooth * alpha[None, :]
   where:
     alpha_i = exp(-beta_code * ||pixel_i - RF_center||^2)    [locality mask]
     C_smooth_ij = exp(-rho2_code * ||pixel_i - pixel_j||^2)  [spatial smoothness]
     beta_code = exp(raw_m2log2beta) = 1/(4 * beta_nat^2)
     rho2_code = exp(raw_mlog2rho2) = 1/(2 * rho_nat^2)
   With M=50: C is ~1725 x 1725 on RF-masked pixels.

3. EIGENDECOMPOSE C: eigvals, eigvecs = torch.linalg.eigh(C)
   Flip to descending order. Keep eigenvalues > eigen_rel_threshold * max_eigenvalue.
   Default threshold: 1e-6. Typical result: K=91 out of 1725 (5.3%).

   The eigenvalue spectrum is EXTREMELY skewed:
     - 1st eigenvalue: ~48.7 (44% of total mass)
     - 2nd eigenvalue: ~16.5
     - Top 3: ~73% of mass
     - Mass-based thresholds (e.g., "keep 95% mass") give K=1, so they're useless.
       The relative threshold approach works much better.

4. PROJECT & OPTIMIZE: z_init = U_K^T @ x_rf_start, then LBFGS maximization:
     closure():
       x_rf = U_K @ z              # (K,) -> (n_rf,)
       x_full = reconstruct(x_rf)  # place into full image, zeros outside RF
       result = distribution_aware_utility(model, likelihood, x_full, x_target, ...)
       loss = -result['utility']    # NEGATE: LBFGS minimizes, we want to maximize
       loss.backward()
       return loss

   Gradient chain: z -> U_K @ z -> x_rf -> x_full -> kernel -> GP moments -> entropy -> utility

   Firing rate guard: if exp(mu_g) > f_max (100.0), return +inf to reject step.

5. PLOT: Target, start (smoothed target), final, diff images.
   Eigenvalue spectrum, cumulative mass, utility convergence, gradient norms.

======================================================================
RESULTS FROM FIRST RUN (M=50, n_train=50, arc_cosine kernel)
======================================================================

- C matrix: 1725 x 1725 (RF-masked pixels)
- Eigenspace: K=91 dimensions retained (5.3%)
- Optimization: Converges in ~5 LBFGS steps. U_DA: 0.977 -> 1.043 (then plateaus)
- Image norm explodes: ||x||_RF goes from 20.7 to 553.5 (proj_coeff = 9.88)
- The optimizer amplifies along dominant C-eigenvectors.
  This is EXPECTED with arc-cosine kernel: K(x,x) ~ ||x||^2, so larger norm =
  larger posterior variance = larger entropy difference = larger utility.
- The final image is qualitatively similar to what pixel-space gradient ascent
  would produce -- confirming the theoretical prediction that C-eigenspace
  optimization is a reformulation, not a new constraint.

======================================================================
MATHEMATICAL CONTEXT (from the LaTeX documents)
======================================================================

## C matrix structure

C = Amp * diag(alpha) @ C_smooth @ diag(alpha)

alpha is a Gaussian locality mask centered at the RF center (beta parameter).
C_smooth is a spatial smoothness kernel (rho parameter).
C acts as a spatial LOW-PASS FILTER on all utility gradients.

For the arc-cosine kernel:
  k(x, y) = (1/pi) * ||x||_C * ||y||_C * J(theta)
  where ||x||_C = sqrt(x^T C x), theta = arccos(x^T C y / (||x||_C ||y||_C))

Gradient of k w.r.t. x*:
  nabla_{x*} k(x*, z_m) involves C @ (x* - z_m) terms -- the C matrix
  filters the gradient, removing high-frequency spatial content.

## C-eigenspace optimization

C = U Gamma U^T where Gamma = diag(gamma_1, ..., gamma_d), gamma_1 >= gamma_2 >= ...

The kernel gradient component along eigenvector u_i is proportional to gamma_i.
Directions with small gamma_i get near-zero gradients regardless of utility.

By parameterizing x* = U_K @ z (keeping only large-gamma_i eigenvectors):
  - Remove directions where gradient is already near-zero
  - Improve optimization conditioning
  - Reduce dimensionality (1725 -> 91)
  - But: does NOT change the solution (unlike PCA, which constrains to data manifold)

## Why norm explodes (arc-cosine kernel)

K(x, x) = (1/pi) * ||x||_C^2 * J(0) = ||x||_C^2 / pi
So self-kernel grows quadratically with image norm.
Posterior variance sigma^2(x*) = K(x*,x*) - ...
The K(x*,x*) term dominates for large ||x*||, so variance grows -> entropy grows -> utility grows.
The optimizer exploits this by scaling up the image along dominant C-eigenvectors.

## Smoothing bottleneck

The C matrix acts as a discrete convolution with Gaussian kernel:
  sigma_rho = 1/sqrt(2 * rho_code) in normalized coordinates
For typical learned rho_nat ~ 0.1: sigma_rho ~ 0.1 normalized ~ 5.4 pixels.
Frequency cutoff: ~1/sigma_rho = 10 normalized.
For 30x30 RF region: only ~3 spatial frequency modes per axis survive.
This is fundamental to the kernel choice, not a numerical artifact.

======================================================================
KEY CODE LOCATIONS
======================================================================

Script being investigated:
  investigations/utility_decompositions/c_eigen_optimization.py

Functions it calls:
  investigations/utility/explore_utility.py  -> setup() trains the model
  acquisition.py                             -> distribution_aware_utility()
  utils.py (local gpytorch_porting/)         -> get_gp_marginal_moments(), compute_H()
  kernels.py                                 -> ArcCosineKernel, _compute_C_matrix()
  run_single_mode.py                         -> build_config_from_defaults(), run_single_config()

Config:
  default_params.json  (all model parameters, DO NOT hardcode values)

LaTeX math references:
  investigations/utility_decompositions/pca_vs_c_eigenspace.tex
  investigations/utility_decompositions/gradient_and_smoothing_bottleneck_nat_imgs_utility_optimization.tex

Utility investigation folder (related scripts):
  investigations/utility/gradient.py        -> pixel-space LBFGS optimization (for comparison)
  investigations/utility/explore_utility.py -> model training + exploration helpers
  investigations/utility/REFERENCE.md       -> utility definitions and findings

======================================================================
PARAMETER DISCIPLINE (CRITICAL)
======================================================================

- ALL model parameters come from default_params.json via build_config_from_defaults().
- Investigation-specific constants (N_STEPS, LR, LBFGS_MAX_ITER, SIGMA_SMOOTH,
  EIGEN_REL_THRESHOLD, M_OVERRIDE, N_TRAIN_OVERRIDE) are at the top of the script.
  These are tuning knobs for THIS investigation, not model parameters.
- Do NOT hardcode seed, cell, kernel params, etc. as literals.
- If you need a new parameter, add it as a CLI arg or top-of-script constant with a comment.

======================================================================
KNOWN ISSUES AND OPEN QUESTIONS
======================================================================

1. NORM EXPLOSION: The optimizer amplifies image norm to maximize utility.
   No pixel bounds are currently enforced. Need to decide:
   - Add sigmoid bounds (like gradient.py supports)?
   - Add norm constraint (like PCA script's projected gradient)?
   - Or accept this as correct behavior of the arc-cosine kernel?

2. FLAT PLATEAU: Utility plateaus after ~5 steps. Is this:
   - True convergence?
   - LBFGS getting stuck in a saddle?
   - Insufficient dimensions (K=91)?

3. EIGENVALUE SPECTRUM: Extremely skewed (1st eigenvalue = 44% of mass).
   The dominant eigenvector is essentially the locality mask alpha itself.
   This means the first optimization direction is "make pixels brighter near RF center."
   Is this scientifically meaningful or a trivial artifact?

4. COMPARISON WITH PIXEL-SPACE: The theoretical prediction says C-eigenspace
   should give the same solution as pixel-space optimization. This hasn't been
   verified empirically yet. Running gradient.py with same model params and
   comparing final images would be the test.

5. M=50 vs M=300: Results are very different:
   - M=300: tight RF (72 pixels), even more extreme eigenvalue skew (K=1 with mass threshold)
   - M=50: wider RF (1725 pixels), richer eigenstructure (K=91)
   Which is more representative for real use?

6. sample_lambda=False: Currently deterministic (no MC sampling in DA utility).
   Setting sample_lambda=True adds noise but gives the "true" expected conditional entropy.

======================================================================
RUNNING THE SCRIPT
======================================================================

The conda environment pytorch_gpytorch is ALREADY ACTIVE. Just:

  python investigations/utility_decompositions/c_eigen_optimization.py

  # With custom eigenvalue threshold:
  python investigations/utility_decompositions/c_eigen_optimization.py --eigen-threshold 1e-4

Output: c_eigen_optimization.png in the same folder.

GPU is required. Float32 is the default.

======================================================================
LOSS SIGN WARNING
======================================================================

We already had a bug where utility was accidentally MINIMIZED instead of MAXIMIZED.
The closure must return: loss = -utility (NEGATE for LBFGS which minimizes).
Always verify utility INCREASES from start to finish. If it decreases, the sign is WRONG.

======================================================================
SIMPLIFICATIONS IN CURRENT VERSION (documented decisions)
======================================================================

- Single target image (pool[0]), no pool-of-images conditioning
- Start from Gaussian-smoothed target projected into C-eigenspace
- Relative eigenvalue threshold (not mass-based -- mass-based gives K=1)
- No pixel bounds (unconstrained optimization)
- sample_lambda=False (deterministic DA utility, no MC noise)
- M=50, n_train=50 (smaller than default for wider RF)

======================================================================
SUBAGENTS
======================================================================

You are free to use subagents (Task tool) for mathematical exploration,
reading documentation, or any other purpose. Mathematical precision and
clarity are the priority in this investigation. Use them liberally.

======================================================================
REPORTING
======================================================================

This is experimental scientific work. When you modify the script and get results:
- Report what changed and why
- Show the key numbers (utility start/end, norm, structural metrics)
- If you make simplifying decisions, document them clearly
- If results are surprising (good or bad), flag them immediately
- Keep plots simple and human-readable
```

## Files on this branch

| File | Purpose |
|------|---------|
| `c_eigen_optimization.py` | Main investigation script (677 lines) |
| `c_eigen_optimization.png` | Output plot from first run |
| `pca_vs_c_eigenspace.tex` | LaTeX: PCA vs C-eigenspace math comparison |
| `gradient_and_smoothing_bottleneck_nat_imgs_utility_optimization.tex` | LaTeX: gradient smoothing bottleneck diagnosis |
| `HANDOFF_C_EIGEN.md` | This file |
