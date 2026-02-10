# Key Facts About Utility Behavior

Reference math: `norm_scaling_analysis.tex`
Workbench script: `explore_utility.py`

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

## The Arc-Cosine Kernel Factorization

K(x, y) = (1/pi) * ||x||_C * ||y||_C * J(angle)

where:
- ||x||_C = sqrt(x^T C x + sigma_0^2) is the C-weighted norm (amplitude)
- angle = arccos( (x^T C y + sigma_0^2) / (||x||_C * ||y||_C) ) is the direction similarity
- J(theta) = sin(theta) + (pi - theta)*cos(theta) is the angular factor
- C encodes the receptive field structure

Two images can differ in TWO independent ways:
1. **Angle** (direction in C-space): changes J(angle), affects cross-covariance
2. **Amplitude** (norm in C-space): scales K linearly, affects variance

---

## What Happens When You Scale an Image

For x* = c * x_target (same direction, different amplitude):

| Quantity | Scaling | Why |
|----------|---------|-----|
| K(x*, z) | ~ c * K(x_t, z) | Kernel is 1-homogeneous in first arg |
| mu(x*) | ~ c * mu(x_t) | Mean is linear in cross-kernel |
| sigma2(x*) | ~ c^2 * sigma2(x_t) | Variance is quadratic in cross-kernel |
| angle(x*, x_t) | ~ constant | Scaling preserves direction |
| rho^2(x*, x_t) | ~ constant (~0.99) | Correlation depends on angle, not norm |

These are exact for sigma_0^2 = 0 and approximate (0.6% error) for sigma_0^2 > 0.

---

## How Angle Affects Conditioning

The DA utility conditions on lambda(x_sample) to get updated moments:
- sigma2_cond = sigma2_marg * (1 - rho^2)
- rho^2 depends on the angle between x* and x_sample

| Angle (rad) | Angle (deg) | rho^2 | Var reduction | Conditioning effect |
|-------------|-------------|-------|---------------|---------------------|
| 0.0 | 0 | ~1.0 | ~100% | Perfect — sigma2_cond ~ 0 |
| 0.1 | 6 | ~0.99 | ~99% | Very strong |
| 0.6 | 34 | ~0.05 | ~5% | Weak |
| 1.0 | 57 | ~0.05 | ~5% | Weak |
| 1.5 | 86 | ~0.001 | ~0.1% | None |

When conditioning is strong: H_cond << H_marg, so U_DA is large.
When conditioning is weak: H_cond ~ H_marg, so U_DA ~ 0.

---

## Why Utility Grows With Amplitude

Scaling x* by c increases sigma2(x*) ~ c^2. This increases H_marg (more GP uncertainty).

For DA utility, the key question is: does H_cond grow as fast?
- If angle is small (rho ~ 1): sigma2_cond ~ 0 regardless of c.
  H_cond stays low while H_marg grows. U_DA grows ~ c^2.
- If angle is large (rho ~ 0): sigma2_cond ~ sigma2_marg.
  H_cond grows as fast as H_marg. U_DA stays near 0.

For standard utility: H_noise depends on the mean firing rate, not on
conditioning. It grows slower than H_marg, so U_std also grows with amplitude
(but for different reasons — no angle dependence).

---

## The Laplace Validity Limit

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

For natural images from the pool: z_safe >> 10 (always safe).
Trouble only arises with artificially amplified images (c >> 1).

---

## Trained Model Parameters (seed=42, cell=8, M=50, n_train=50)

- A = 0.0265 (firing rate scaling — very small, compresses GP moments)
- lambda0 = -1.686 (firing rate offset)
- sigma_0 = 0.989 (kernel bias)
- K(x_target, x_target) = 669 (self-kernel of a typical natural image)
- Natural images: logf_mean in [-1.8, -1.3], logf_var in [0.04, 0.19]
