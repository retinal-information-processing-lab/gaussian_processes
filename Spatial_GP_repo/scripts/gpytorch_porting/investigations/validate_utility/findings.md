# Gradient Investigation Findings

**Location**: `investigations/validate_utility/`
**Branch**: `pietro/acquisition-functions`
**Script**: `gradient.py`

---

## Finding 1: Gradient flow works end-to-end (Session 1)

**Date**: 2026-02-07
**Status**: CONFIRMED

Both `standard_utility` and `distribution_aware_utility` produce correct, non-zero gradients w.r.t. input pixels. Gradients are confined to the RF mask region (1156/11664 pixels active, 0 non-zero gradients outside RF).

Key numbers (seed=42, cell=8, M=50, n_train=50, default_gpy):
- Standard utility: gradient norms [1.2e-3, 4.8e-2], 20 candidates, 0.2s
- DA utility: gradient norms [1.1e-4, 2.2e-3], 20 candidates + 100 MC samples, 0.7s
- Gradient norm correlation (std vs DA): 0.9779
- Cosine similarity: mean 0.82, range [0.55, 0.94]

Root cause of old gradient failure: `laplace_approximations_new` in `utility.py` used `torch.empty` + indexed assignment, breaking autograd silently. Fixed with `torch.where` in local differentiable pipeline.

---

## Finding 2: Unconstrained gradient ascent does NOT converge to target image (Session 2)

**Date**: 2026-02-07
**Status**: CONFIRMED

### Setup
- Target image: natural image from pool (index 0)
- Starting image: target + Gaussian noise (NOISE_SCALE=0.1, within RF only)
- DA utility conditioned on target only (n_mc=1, sample_lambda=False)
- Plain gradient ascent: LR=1.0, 100 steps
- Standard utility as control (no conditioning target)

### Results

| Metric | DA start | DA final | Std start | Std final |
|--------|----------|----------|-----------|-----------|
| utility | 0.009918 | 0.011031 | 0.005681 | 0.006142 |
| frac_dist | 1.0000 | 1.0067 | 1.0000 | 1.0032 |
| rf_pixel_dist | 2.1469 | 2.1613 | 2.1469 | 2.1538 |
| kernel_sim | 651.76 | 678.87 | 651.76 | 668.27 |

Reference at x_target: DA utility = 0.010148, Std utility = 0.005778, k(target, target) = 668.96

### Interpretation

1. **Utility increases monotonically** for both DA and Std. The gradient ascent is working correctly — it maximizes utility as intended.

2. **Pixel distance INCREASES** (frac_dist > 1.0). The optimizer is moving AWAY from the target in pixel space.

3. **Kernel similarity exceeds k(target, target)**. DA reaches 678.87 vs k(target, target) = 668.96. The optimizer creates an image with higher kernel value to the target than the target's self-similarity.

4. **DA utility exceeds the reference at target**. Final DA utility = 0.011031 > utility at target = 0.010148. The utility maximum is NOT at x_target.

### Why this happens

DA utility is `H_marg(x*) - H_cond(x* | x_target)`. The maximum is not necessarily at `x* = x_target` because:

- H_marg increases with the C-weighted norm `||x*||_C` (higher norm = more uncertainty in the GP posterior)
- H_cond also increases with norm, but the conditioning reduces it partially
- The gradient has a component that reduces noise (toward target) AND a component that increases the norm (away from target in an "amplified" direction)
- The norm-scaling component dominates

Evidence for the two-component structure: frac_dist initially dips to 0.9994 (steps 10-20) before increasing. The gradient initially pushes slightly toward the target before norm-scaling dominates.

The arc-cosine kernel is `K(x,y) = (1/pi) * ||x||_C * ||y||_C * J(angle)`. Increasing `||x||_C` while maintaining angle to target increases k(x, target) beyond k(target, target). This is the norm-scaling effect.

### Standard utility comparison

Standard utility shows qualitatively identical behavior — also diverges from target, also increases kernel similarity. This suggests the norm-scaling behavior is a property of the kernel/utility structure, not specific to the DA conditioning.

### Implications

- Unconstrained gradient ascent on utility will always find "louder" versions of images rather than converging to a natural image target
- For meaningful image optimization, constraints are needed (pixel bounds, norm constraint, or projection onto natural image manifold)
- The utility function provides correct gradient information about DIRECTION (cosine similarity between gradient at perturbed and unperturbed points is high), but the magnitude structure of the kernel creates a degenerate scaling direction

### Visualizations

- `gradient_ascent_images.png`: Side-by-side comparison of target, start, DA final, Std final (top row) and difference images (bottom row)
- `gradient_ascent_trajectories.png`: Utility, frac_dist, kernel_sim, and gradient norm over optimization steps

---

## Finding 3: Diagnostic dissection of extreme images (Session 2, continued)

**Date**: 2026-02-07
**Status**: CONFIRMED
**Script**: `diagnose_extreme.py` (loads artifacts from `gradient.py` with N_STEPS=5000)

### Setup

Extended gradient ascent from Finding 2 to 5000 steps (LR=1.0). Standard utility hit NaN at step 1479 (numerical overflow). DA utility ran all 5000 steps. Saved artifacts and ran full diagnostic.

### Key diagnostic: the entropy heatmap argument

The Poisson-GP entropy H(R | mu, sigma2) is only non-zero in a narrow band of mu (log-firing rate mean). Outside this band (|mu| > ~10), entropy is ~0 regardless of variance. The question: do the optimized images land inside or outside this band?

### Results — Table A: Marginal GP moments

| Image | raw mu | sigma2 | logf_mean | logf_var | exp(logf_mean) | in_band? |
|-------|--------|--------|-----------|----------|----------------|----------|
| x_target | 7.86 | 70.87 | -1.477 | 0.050 | 0.23 | YES |
| x_final_da | 100.96 | 3240.18 | 0.993 | 2.282 | 2.70 | YES |
| x_final_std | 68497 | 285299328 | 1815.9 | 200895 | inf | NO |
| natural_1 | -2.81 | 56.76 | -1.760 | 0.040 | 0.17 | YES |
| natural_2 | -5.67 | 267.28 | -1.836 | 0.188 | 0.16 | YES |
| natural_3 | 16.38 | 136.70 | -1.251 | 0.096 | 0.29 | YES |

Critical observation: **A = 0.0265 compresses raw GP moments enormously.** Raw mu=101 at x_final_da maps to logf_mean=0.99 — still in the useful band. The band boundary (logf_mean=10) requires raw mu > (10 + 1.69) / 0.0265 = 441. The DA optimizer at 5000 steps has only reached mu=101.

### Results — Table B: Conditional moments (conditioned on x_target)

| Image | sigma2 | sigma2_cond | var_ratio | conditioning |
|-------|--------|-------------|-----------|--------------|
| x_target | 70.87 | 0.0002 | 0.000003 | STRONG (near-perfect) |
| x_final_da | 3240.18 | 193.57 | 0.060 | STRONG (94% reduction) |
| x_final_std | 285299328 | 285277568 | 0.9999 | NONE |
| natural_1 | 56.76 | 53.55 | 0.944 | NONE |
| natural_2 | 267.28 | 218.35 | 0.817 | weak |
| natural_3 | 136.70 | 136.31 | 0.997 | NONE |

The DA-optimized image has angle 0.114 rad (6.5 degrees) from the target — nearly the same direction in kernel space, just 60x larger norm. This is why conditioning is so effective: 94% variance reduction. Random natural images are 37-41 degrees from the target and conditioning barely helps.

### Results — Table C: Laplace validity and entropy

| Image | sum(p_r) | H_marg | H_cond | U_DA | U_std |
|-------|----------|--------|--------|------|-------|
| x_target | 1.000 | 0.593 | 0.583 | 0.010 | 0.006 |
| x_final_da | 0.984 | 2.836 | 2.036 | 0.800 | 8.201 |
| x_final_std | 0.000007 | 0.000 | 0.000 | 0.000 | inf |
| natural_1 | 1.000 | 0.492 | 0.491 | 0.000 | 0.003 |
| natural_2 | 1.001 | 0.492 | 0.486 | 0.006 | 0.016 |
| natural_3 | 1.000 | 0.694 | 0.694 | 0.000 | 0.014 |

### Interpretation — Three regimes

**x_final_std: Numerical collapse (as predicted by entropy heatmap)**
- logf_mean=1816 → way outside the useful band → H_marg = 0.000090 ≈ 0
- sum(p_r) = 0.000007 → catastrophic truncation (probability mass at r ≈ exp(1816), way beyond r_max=100)
- U_std = inf → exp(mu_g + 0.5*sigma2_g) overflow in nd_utility_new
- Conditioning does nothing (var_ratio=0.9999)
- Standard utility optimization diverged to NaN at step 1479

**x_final_da: Genuine high utility (NOT a numerical artifact)**
- logf_mean=0.993 → in the useful band (A=0.0265 compresses raw mu=101)
- sum(p_r) = 0.984 → Laplace approximation working (slight truncation)
- H_marg=2.836 → genuinely high entropy (large variance + moderate mean = entropy sweet spot)
- 94% variance reduction from conditioning → H_cond=2.036
- U_DA=0.800 → the gap H_marg - H_cond is real

**Natural images: Low utility, weak conditioning**
- logf_mean ≈ [-1.84, -1.25] → well in the useful band but in a low-entropy region
- Conditioning barely helps (var_ratio 0.82-0.997)
- U_DA = [0.00007, 0.006] → orders of magnitude lower than x_final_da

### Why DA utility is genuinely high

The optimizer discovered that increasing ||x||_C while staying aligned with x_target (angle=0.114 rad) creates an image where:

1. **GP posterior variance is large** (sigma2=3240 vs ~70 for natural images) → high H_marg
2. **A=0.0265 compresses the mean** so logf_mean stays in the band (0.99 vs ~-1.5 for natural images) → entropy function is still non-zero
3. **Conditioning is very effective** (94% variance reduction because small angular distance to target) → H_cond is substantially lower than H_marg
4. The combination gives U_DA=0.800 — a factor of 80x above natural image utility

### When will DA utility also collapse?

logf_mean exceeds 10 (edge of useful band) when raw GP mu > (10 + 1.69) / 0.0265 = 441. Currently mu=101 after 5000 steps. Linear extrapolation: mu grows by ~0.019/step, so ~18,000 more steps to reach mu=441. At that point, DA utility would also collapse.

### Confirmed numerical bugs

1. **nd_utility_new line 486**: `exp(mu_g + 0.5*sigma2_g)` overflows for extreme inputs. The Std optimizer hit this at step 1479. This is a real bug — the function returns inf/NaN instead of gracefully handling extreme inputs.

2. **r_max=100 truncation**: For logf_mean > ~4.6 (exp > 100), the Laplace approximation misses probability mass beyond r=100. The x_final_da case already shows sum(p_r)=0.984 (1.6% mass missing). This will worsen as the optimizer continues.

---

## Finding 4: Verification of mathematical claims and fixed-angle scaling (Session 3)

**Date**: 2026-02-08
**Status**: CONFIRMED (with corrections)
**Script**: `verify_conclusions/verify_conclusions.py`

### Mathematical proofs verified

All theorems in `norm_scaling_analysis.tex` are mathematically correct. Numerical tests confirm:

- **C1 (Theorem 1)**: K(cx, z)/K(x, z) = c within 0.6% for c >= 2 (sigma_0 correction)
- **C2 (Corollary 2)**: mu(cx)/mu(x) ≈ c within 0.06%, var(cx)/var(x) ≈ c^2 within 0.65%
- **C3 (Corollary 1)**: rho^2 converges to 0.987 (not 1.0) due to sigma_0^2 > 0. Variance ratio converges to ~1.3%.
- **C4 (utility growth)**: Growth exponent alpha ≈ 1.91 (Laplace-valid range c=2..10). Consistent with c^2 prediction.

### Corrections made to LaTeX

1. **Proposition label/statement**: Changed "constant utility" → "logarithmic growth". Proof was correct, statement was wrong.
2. **sigma_0^2 remark**: Correction is O(sigma_0^2/v_t) ≈ 0.6% (constant in c), NOT O(sigma_0^2/(c^2 q)) (vanishing). Verified numerically in C7.
3. **Heuristic lower bound**: Removed incorrect claim that Poisson maximizes entropy under mean constraint (that's the geometric distribution; Poisson requires Var=mean). Bound H_marg >= H_Poisson(E[f]) flagged as unproven but empirically supported (alpha ≈ 1.9).
4. **Table 1**: Fixed x_target mu (1.68 → 7.86), sigma2 (11.3 → 70.9), x_DA sigma2 (~40k → 3240), x_Std angle (0.096 → 0.866), natural ranges.
5. **Table 2**: Fixed x_target H_marg (0.39 → 0.593), H_cond (0 → 0.583), U_DA (0.39 → 0.010). Old values were entirely wrong.
6. **Threshold**: c > 56 (not 262, which used wrong mu(x_t)=1.68 instead of 7.86).

### C8: Fixed-angle norm-scaling test

Key result: utility growth with norm is **strongly angle-dependent**.

| Base angle | Valid range | Monotonic? | Max U_DA | Conditioning |
|-----------|-------------|------------|----------|--------------|
| 0.00 rad (0 deg) | c <= 10 | YES | 0.965 | ~1% var ratio (strong) |
| 0.57 rad (32 deg) | c <= 5 | NO | 0.010 | ~96% var ratio (weak) |
| 1.00 rad (57 deg) | c <= 1 | YES | 0.009 | ~94% var ratio (weak) |
| 1.51 rad (86 deg) | c <= 5 | YES | 0.0005 | ~100% var ratio (none) |

Angles stay approximately constant across c values (confirmed Theorem 1 with sigma_0 > 0).
At small angles, utility grows aggressively. At moderate/large angles, conditioning is too weak.

### Laplace safety metric

Pre-check for Laplace validity (no Laplace computation needed):

    z_safe = (log(r_max) - logf_mean) / sqrt(logf_var)

Calibration: z_safe > 2.0 is safe, 1.5-2 borderline, < 1.5 broken.
Can be computed from GP marginal moments alone — O(1) cost.

### Minor corrections to Finding 2/3

- Std utility hit **inf** (not NaN) at step 1479. grad_norm was NaN. The `diverged` flag has a minor bug: checks `np.isnan` but not `np.isinf`.
- Std-optimized image angle is 0.866 rad (not 0.096 as in LaTeX). Std optimizer diverged in a DIFFERENT direction from DA — no alignment benefit without conditioning.

---

## Open Questions

1. **Would norm-constrained gradient ascent converge to target?** If we project onto the sphere `||x||_C = ||x_target||_C` after each step, the norm-scaling escape route is blocked. Only the directional component remains.

2. **Does the gradient direction point toward the target?** The initial dip in frac_dist suggests yes, but this needs quantification: compute the cosine similarity between the gradient and the (x_target - x_opt) direction over steps.

3. **Is this a property of the arc-cosine kernel specifically?** The K ~ ||x|| * ||y|| * J(angle) structure makes norm-scaling easy. A stationary kernel (e.g., RBF) would not have this property since K only depends on ||x - y||.

4. **Should the utility functions guard against extreme inputs?** The Laplace sum(p_r) and nd_utility_new exp overflow are real numerical hazards. The z_safe metric provides a cheap pre-check. Adding a guard (e.g., return 0 when z_safe < 1.5, or when sum(p_r) < 0.9) would prevent silent numerical garbage.

5. **What is the practical implication for active learning?** In real use, candidate images are from the natural pool (pixel values in [-2.4, 2.5]), so logf_mean stays in band. The extreme-input issue only arises with unconstrained gradient optimization — which is not the standard active learning workflow.
