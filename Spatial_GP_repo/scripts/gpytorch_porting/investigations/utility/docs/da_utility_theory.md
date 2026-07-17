# Utility Investigation Reference

Source of truth for DA utility behavior, kernel properties, and cross-kernel findings.
Scripts: `workbench.py`, `gradient_ascent.py`. Entropy analysis: `entropy_landscape.py`.

---

## The Two Utilities

**Standard utility** U_std(x*) = H_marg(x*) - E[H_noise(x*)]
- Measures: how much would observing a spike count at x* reduce GP uncertainty?
- Only depends on marginal GP moments at x*
- No awareness of other stimuli in the world

**Distribution-aware (DA) utility** U_DA(x*) = H_marg(x*) - E[H_cond(x* | x_sample)]
- Measures: how much would observing x* reduce uncertainty about responses to OTHER natural images?
- Depends on both marginal moments AND cross-covariance with conditioning images
- The conditioning step uses Gaussian conditioning on the GP posterior

---

## Arc-Cosine Kernel Structure

K(x, y) = (1/pi) * ||x||_C * ||y||_C * J(angle)

where:
- ||x||_C = sqrt(x^T C x + sigma_0^2) is the C-weighted norm (amplitude)
- angle = arccos( (x^T C y + sigma_0^2) / (||x||_C * ||y||_C) ) is the direction similarity
- J(theta) = sin(theta) + (pi - theta)*cos(theta) is the angular factor
- C encodes the receptive field structure

Self-kernel: K(x, x) = ||x||_C^2. Prior variance at any point equals its squared C-norm.

Two images can differ in TWO independent ways:
1. **Angle** (direction in C-space): changes J(angle), affects cross-covariance
2. **Amplitude** (norm in C-space): scales K linearly, affects variance

---

## Scaling Behavior (Key Theorems)

For x* = c * x_target (same direction, different amplitude):

| Quantity | Scaling | Why |
|----------|---------|-----|
| K(x*, z) | ~ c * K(x_t, z) | Kernel is 1-homogeneous in first arg (Theorem 1) |
| mu(x*) | ~ c * mu(x_t) | Mean is linear in cross-kernel (Corollary 2) |
| sigma2(x*) | ~ c^2 * sigma2(x_t) | Variance is quadratic in cross-kernel (Corollary 2) |
| angle(x*, x_t) | ~ constant | Scaling preserves direction |
| rho^2(x*, x_t) | = 1 (exact for sigma_0=0) | Perfect correlation (Corollary 1) |

**With sigma_0^2 > 0** (our trained model): fractional error converges to O(sigma_0^2/v_t) ~ 0.6%.
rho^2 converges to ~0.987 (not 1.0), giving residual variance ratio ~1.3%.
These are exact for sigma_0^2 = 0 and approximate for sigma_0^2 > 0.

**Full proofs**: `proof_divergence_theorems.tex` (Theorem 1, Corollaries 1-2, Remark 1)
**Derivations of GP moments and conditioning**: `proof_moments_and_conditioning.tex`

---

## DA Utility Formula (Deterministic Mode)

When lambda_t = mu(x_t) (sample_lambda=False):

    U_DA(x*) = H(mu_g, sigma2_g) - H(mu_g, sigma2_g * (1 - rho^2))

So DA utility depends on exactly three quantities:
1. mu_g = A*mu(x*) + lambda0 -- the operating point on the entropy landscape
2. sigma2_g = A^2 * sigma2(x*) -- the marginal variance (how much entropy to start with)
3. rho^2 -- how much of that variance conditioning removes

Conditioning moves the point from (mu_g, sigma2_g) to (mu_g, sigma2_g*(1-rho^2)):
same mu_g, reduced sigma2_g. The utility is the difference in H between these two points.

**Derivation**: `proof_moments_and_conditioning.tex` (Section 6)

---

## How Angle Affects Conditioning

sigma2_cond = sigma2_marg * (1 - rho^2), where rho^2 depends on the angle between x* and x_sample.

| Angle (rad) | Angle (deg) | rho^2 | Var reduction | Conditioning effect |
|-------------|-------------|-------|---------------|---------------------|
| 0.0 | 0 | ~1.0 | ~100% | Perfect -- sigma2_cond ~ 0 |
| 0.1 | 6 | ~0.99 | ~99% | Very strong |
| 0.6 | 34 | ~0.05 | ~5% | Weak |
| 1.0 | 57 | ~0.05 | ~5% | Weak |
| 1.5 | 86 | ~0.001 | ~0.1% | None |

---

## Why Utility Diverges Under Gradient Ascent

Scaling x* by c increases sigma2(x*) ~ c^2, which increases H_marg (more GP uncertainty).

For DA utility, H_cond depends on rho^2:
- Small angle (rho ~ 1): sigma2_cond ~ 0 regardless of c.
  H_cond stays low while H_marg grows. U_DA grows ~ c^2.
- Large angle (rho ~ 0): sigma2_cond ~ sigma2_marg.
  H_cond grows as fast as H_marg. U_DA stays near 0.

**Growth rate** (numerically confirmed): U_DA ~ c^1.9 in the Laplace-valid range (c <= 10).
This is faster than the Gaussian case where U ~ log(c) (Proposition 1 in `proof_divergence_theorems.tex`).

**Root cause**: the arc-cosine kernel is positively homogeneous: K(cx, y) = c*K(x, y) when sigma_0^2 = 0.
This makes norm and direction completely decoupled in the GP posterior.
Gradient ascent exploits the norm degree of freedom to increase variance (and thus entropy and utility) without sacrificing the conditioning benefit provided by directional alignment.

**This is NOT a bug** -- it is an intrinsic property of the DA utility formula with homogeneous kernels.

**Formal proofs**: `proof_divergence_theorems.tex` (Sections 2-6)
**Utility decomposition and gradient dynamics**: `proof_kernel_solutions.tex` (Sections 1-2)

---

## Laplace Validity Limit

The entropy computation uses a Laplace approximation truncated at r_max = 100.
It breaks when the GP posterior implies high firing rates.

Pre-check (no Laplace needed):

    z_safe = (log(r_max) - logf_mean) / sqrt(logf_var)
    where logf_mean = A*mu + lambda0,  logf_var = A^2 * sigma2

| z_safe | Status |
|--------|--------|
| > 2.0 | Safe |
| 1.5 - 2.0 | Borderline |
| < 1.5 | Unreliable (truncation corrupts entropy) |
| < 0.5 | Broken (utility can be negative or NaN) |

A confident GP on a low-firing cell gives a large z_safe -- but that is NOT an
operating-range guarantee. sigma2_g = A^2 * var(lambda) reaches ~20-50 in real
experiments (uncertain cells, especially the first few reps); there z_safe falls
to ~0.9-1.4 (Unreliable/Broken by the table above). Trouble arises both from
artificially amplified images (c >> 1) AND from genuinely uncertain predictions
(large sigma2_g). The narrow logf_var range quoted below is one cell's snapshot,
NOT the operating range -- do not use it as a safe ceiling.

See `entropy_landscape.md` for detailed empirical analysis and MC comparison.

---

## Cross-Kernel Investigation Results

### Arc-Cosine Kernel (unnormalized) -- DEFAULT
- K(x,x) ~ ||x||_C^2 -- norm grows without bound.
- DA utility grows monotonically with norm: higher ||x|| -> higher lambda_m -> higher firing rate -> higher H_marg -> higher utility.
- Gradient ascent exploits this by amplifying images rather than finding angularly similar ones.
- This is intrinsic to the DA utility formula, not a kernel bug.

### Normalized Arc-Cosine Kernel (DEPRECATED)
- K(x,x) = 1.0 exactly -- eliminates norm dependence.
- Utility depends only on angular structure (RF alignment).
- But test_r drops ~25% (0.79 -> 0.59 on PNAS cell 8, M=100). Image norm is genuinely informative for neural encoding.
- Not included as a kernel option in the unified scripts.
- **Analysis**: `proof_kernel_solutions.tex` (Section 3) and `proof_moments_and_conditioning.tex` (Section 8)

### Arc-Sine Kernel (Williams 1998)
- K(x,x) saturates toward 1 via erf activation. Training images: K_sat ranges 0.07-0.95.
- Saturation limits but does NOT eliminate norm-driven utility growth. The transition zone before saturation still allows lambda_m growth.
- Firing rates are modest (~6 spikes). The f_max guard doesn't fire.
- LBFGS converges fast (2-3 outer steps), then flat.
- **Analysis**: `proof_kernel_solutions.tex` (Section 5)

### LocalRBF Kernel
- K(x,x) = 1.0 (stationary). Utility depends on distance to conditioning image.
- With multi-image conditioning (50 images), gradients cancel out. Utility barely moves.
- Test_r lower than arc-cosine (0.25 vs 0.78 at M=50/N_TRAIN=50).

### Cross-Kernel Conclusions
- The core issue is shared across all kernels: DA utility rewards high marginal entropy H_marg, which correlates with predicted firing rate, not epistemic uncertainty.
- Only the normalized kernel truly eliminates norm dependence, but it sacrifices predictive accuracy.
- The f_max firing rate guard (default 100.0) prevents extreme rate exploitation. It's effective for arc-cosine but unnecessary for arc-sine/RBF where rates are naturally bounded.

---

## Trained Model Parameters (seed=42, cell=8, M=50, n_train=50)

- A = 0.0265 (firing rate scaling -- very small, compresses GP moments)
- lambda0 = -1.686 (firing rate offset)
- sigma_0 = 0.989 (kernel bias)
- K(x_target, x_target) = 669 (self-kernel of a typical natural image)
- Natural images, THIS cell, confident predictions: logf_mean in [-1.8, -1.3],
  logf_var in [0.04, 0.19]. NOTE: one cell's snapshot, NOT the operating range --
  across experiments logf_var (= sigma2_g) reaches ~20-50 (uncertain cells, early
  reps). Do not read [0.04, 0.19] as a safe sigma2_g ceiling.

---

## Proof Files (Full Derivations)

| File | Contents |
|------|----------|
| `proof_moments_and_conditioning.tex` | GP posterior formulas (mu, sigma2, rho2), Gaussian conditioning, log-firing-rate transform, DA utility derivation, scaling analysis with sigma_0^2 > 0, normalized kernel derivation |
| `proof_divergence_theorems.tex` | Theorem 1 (cross-kernel proportionality), Corollary 1 (rho=1), Corollary 2 (moment scaling), Remark 1 (sigma_0 effect), Proposition 1 (Gaussian log(c) growth), numerical evidence table, summary of divergence mechanism |
| `proof_kernel_solutions.tex` | Utility decomposition (variance vs correlation terms), optimization conflict analysis, Solution 1 (normalized kernel) with invariance proof, Solution 2 (arc-sine saturation kernel) |
