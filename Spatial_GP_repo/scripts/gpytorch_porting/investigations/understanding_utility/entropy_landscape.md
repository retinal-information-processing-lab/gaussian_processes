# Entropy Landscape Investigation

**Script**: `entropy_landscape.py`
**Plot**: `entropy_landscape.png`

## What this checks

The entropy H(R | mu_g, sigma2_g) is computed via the Laplace approximation, truncated at r_max. For large mu_g or large sigma2_g, the Poisson probability mass sits above r_max and the truncated sum misses it, making H collapse to ~0. This is a numerical artifact — the true entropy increases monotonically with both mu_g and sigma2_g.

The script computes:
1. **Entropy heatmap** using adaptive r_max per grid row (up to 10000).
2. **Empirical truncation boundaries** for milestone r_max values (100, 500, 1000, 10000): the contour where sum(p_r) = 0.99.
3. **Analytical 3-sigma boundaries** for comparison: mu_g = log(r_max) - 3*sqrt(sigma2_g).

## What we found

### The old computation (r_max=100) was wrong above mu_g ~ 4.6

With fixed r_max=100, entropy collapses to ~0 for mu_g > log(100) ~ 4.6 at small sigma2_g, and at even lower mu_g for large sigma2_g. This is purely a truncation artifact. The correct entropy continues increasing monotonically (confirmed: zero monotonicity violations with adaptive r_max).

### The truncation boundary is a curve, not a horizontal line

Both mu_g and sigma2_g determine when truncation bites. Large sigma2_g means the upper tail of the log-firing rate distribution pushes the Poisson peak well above r_max, even at moderate mu_g. The computable region is roughly triangular.

### The analytical 3-sigma formula is conservative but wrong in shape

The formula mu_g = log(r_max) - 3*sqrt(sigma2_g) predicts a steeper decline than the empirical boundaries. At large sigma2_g, the analytical curve drops much lower than where truncation actually bites. The discrepancy is 1-3 mu_g units depending on the region.

### The sum(p_r) = 0.99 contour is noisy

The Laplace approximation produces unnormalized probabilities — sum(p_r) can exceed 1.0 (we observed up to 1.015). This makes sum(p_r) non-monotonic across the grid, and the 0.99 contour cuts through this noisy field, producing squiggly/S-shaped artifacts that don't reflect the true truncation boundary.

## Limitations

### The adaptive r_max is itself heuristic

The adaptive computation chooses r_max via:

    upper_logf = mu_g + 3 * sqrt(sigma2_g_max)
    upper_rate = exp(upper_logf)
    needed_rmax = upper_rate + 5 * sqrt(upper_rate) + 10

This is the 3-sigma upper tail of g, converted to a firing rate, plus 5 Poisson standard deviations. It is more conservative than the plain 3-sigma formula (double safety margin), but it is still a heuristic. We have not verified that the resulting H is converged.

If the adaptive H is wrong in some region, then any boundary drawn relative to it (whether sum(p_r) or relative H error) is drawn against a bad reference.

### sum(p_r) conflates two failure modes

sum(p_r) can be below 1 for two independent reasons:
1. **Truncation**: probability mass above r_max is missed.
2. **Laplace approximation error**: the saddle-point expansion under-estimates some probabilities.

And it can be above 1 due to Laplace over-estimation. Using sum(p_r) as the truncation boundary criterion conflates these effects.

## Suggested fix: convergence test

The correct way to validate the adaptive r_max (and to draw clean boundary curves) is a convergence test:

1. For each grid point, compute H at multiple r_max values: e.g., r_max = {1000, 2000, 5000, 10000}.
2. H is converged at a point if |H(r_max) - H(r_max_prev)| / H(r_max) < epsilon (e.g., 0.1%).
3. Use the converged H as ground truth.
4. For each milestone r_max, draw the contour where |H_fixed - H_converged| / H_converged = 1%.

This approach:
- Directly measures the quantity we care about (entropy error, not probability mass).
- Does not depend on a heuristic for "needed r_max" — convergence is self-certifying.
- Produces smooth contours (H is a smooth function, unlike sum(p_r)).
- Cannot draw the boundary beyond the region where convergence is achieved (honest about its limits).

**Cost**: ~4x the computation of the current approach (4 r_max values instead of 1). For a 400x300 grid this is seconds on GPU, not a practical concern.

**Limitation**: For the r_max=10000 boundary, we need convergence up to that point, which may require computing at r_max=20000 or 50000 as the reference. This is feasible for the grid range we use (mu_g up to 12, sigma2_g up to 15) but becomes expensive at very high mu_g.

---

## Monte Carlo Entropy Estimation (Panel 2)

**Updated**: February 2026 — the script now produces a 2-panel plot comparing Laplace sum (Panel 1) vs Monte Carlo estimation (Panel 2).

The right panel shows entropy computed with `compute_H_MC()`, a Monte Carlo estimator that avoids the truncation problem entirely.

### Algorithm

Instead of summing p(r) log p(r) for r=0..r_max, the MC approach samples from the distribution and evaluates log p(r) only at the sampled values:

1. Sample g ~ N(mu_g, sigma2_g)  [log-firing rate from GP posterior]
2. Sample r ~ Poisson(exp(g))      [spike count given firing rate]
3. Evaluate log p(r) via Laplace   [single r value, not a sum]
4. H ≈ -mean(clamp(-log p(r), max=50))  [MC estimator with clipping]

**Key difference**: No r_max parameter. The cost is O(S) per point, independent of (mu_g, sigma2_g). We use S=2000 samples per grid point.

### Variance Reduction: Clipping

For large sigma2_g, the Poisson-log-normal distribution becomes extremely heavy-tailed. Without clipping, ~2% of samples from the tail can have -log p(r) ~ 10^9, completely dominating the average (see `diagnose_MC.py` in git history for detailed analysis).

We clip -log p(r) at 50 nats. This introduces bias for large sigma2_g but makes the estimator practical. The bias manifests as:
- H values ~10-20% higher than the true entropy in the upper-right region (large mu_g, large sigma2_g)
- Smooth continuation where Laplace collapses (no truncation artifacts)

### Advantages

- **Works at any scale**: No r_max truncation boundary
- **Computational efficiency**: O(S) per point, same cost regardless of (mu_g, sigma2_g)
- **Smooth landscape**: No collapse artifacts in the upper-right region

### Limitations

- **Clipping introduces bias**: For sigma2_g > ~10, H is overestimated by 10-20%
- **Not differentiable**: Uses discrete Poisson sampling (no gradient flow)
- **Only for analysis**: Not suitable for production acquisition functions

### Comparison with Laplace (Panel 1 vs Panel 2)

- **Agreement region**: For z_safe > 2 (roughly mu_g < 4 and sigma2_g < 5), panels show similar H values (relative difference < 5%)
- **Laplace collapses**: Upper-right region where Panel 1 shows truncation artifacts (H → 0 incorrectly)
- **MC extends smoothly**: Panel 2 continues with plausible H values (growing with both mu_g and sigma2_g)
- **Natural images** (cyan marker): Both methods agree (safe zone)
- **DA-optimized** (orange marker): Both methods agree (near the boundary)

### Validation

See `test_compute_H_MC.py` for:
- Agreement test with Laplace in safe region (< 2% difference)
- Scaling behavior H(c) for c=1..50
- Gradient flow validation (for utility optimization)

### Deleted Scripts

The following scripts were consolidated into `entropy_landscape.py`:
- `compare_H_landscape.py` — functionality now in Panel 2
- `diagnose_MC.py` — findings documented here and in `test_compute_H_MC.py`

Both are archived in git history for reference.
