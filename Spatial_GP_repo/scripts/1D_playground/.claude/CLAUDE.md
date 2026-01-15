# 1D GP Playground - Claude Context

When working in this folder, read `GP_PLAYGROUND_CONTEXT.md` for full documentation.

## Quick Reference

**Utility functions** (in `Spatial_GP_repo/utility.py`):
- `nd_utility_new()` - Laplace approximation (use this one - numerically stable)
- `nd_utility_MC()` - Monte Carlo ground truth
- `nd_utility_MC_batched()` - MC with batched sampling for high sample counts (100k+)
- `nd_utility_NUMERICAL()` - Gauss-Hermite quadrature ground truth
- `distribution_aware_utility_gpytorch()` - Accounts for p(x), supports uniform/Gaussian

**Key scripts**:
- `gp_utility_playground.py` - Main playground (GPyTorch VariationalGP + Poisson likelihood)
- `test_distribution_aware_utility.py` - Compares both utilities with visualization
- `test_nd_utility_MC.py` - Compares Laplace vs MC vs Numerical
- `active_learning_loop.py` - Iterative acquisition demo

**Mathematical derivations**:
- `~/IDV_code/Papers/latex_summaries/distribution_aware_utility_pietro.tex` - Main derivation of distribution-aware acquisition function
- `~/IDV_code/Papers/latex_summaries/predictive_distribution_conditioned_on_observation.tex` - Derivation of conditional GP moments after observing λ(x)

---

## Distribution-Aware Utility: Complete Mathematical Reference

### Problem Setup

Neural responses `r` to stimulus `x` follow **Poisson(f(x))** where:
```
f(x) = exp(A·λ(x) + λ₀)
```
- `λ(x)` is the latent GP function we learn
- `g(x) = A·λ(x) + λ₀` is the log-firing rate (GP posterior: `g ~ N(μ, σ²)`)

### Goal
Find next stimulus `x*` that maximally improves prediction of responses to **natural stimuli** `x ~ p(x)`.

### The Utility Formula

```
U(x*) = H_marg(x*) - H_cond(x*)
```

Where:
- **H_marg(x*)** = `H(R | x*, D)` — marginal entropy of Poisson response at `x*`
- **H_cond(x*)** = `E_{x~p(x), λ(x)~q}[H(R | x*, λ(x), D)]` — expected conditional entropy

### Computing H_marg

The marginal response distribution:
```
p(r | x*, D) = ∫ Poiss(r | exp(g)) · N(g | μ, σ²) dg
```

This integral is approximated via **Laplace approximation**:
```
log p(r | x*, D) ≈ ḡ_r·r - exp(ḡ_r) - (ḡ_r-μ)²/(2σ²) - 0.5·log(1 + σ²·exp(ḡ_r)) - log(r!)
```

The mode `ḡ_r` is found using the **Lambert W function**:
```
ḡ_r = r·σ² + μ - W₀(σ²·exp(r·σ² + μ))
```

Then: `H_marg = -Σᵣ p(r|x*,D) log p(r|x*,D)`

### Computing H_cond (The Key Part)

H_cond requires computing the **conditional GP moments** after observing `λ(x)` at a sampled natural image.

#### Sparse Variational GP Setup

- Inducing points `λ̃` with variational posterior: `q(λ̃ | D) = N(m, V)`
- **K** = kernel matrix at inducing points
- **k(x)** = kernel vector from inducing points to x
- Projection vector: `u = K⁻¹ k(x)`
- Schur complement (prior conditional variance): `s = k(x,x) - k(x)ᵀ K⁻¹ k(x)`

#### Updated Posterior After Observing λ(x)

**CRITICAL: Distinguish between (m, V) and (m', V')**

When we observe a specific value `λ(x)` at sample point `x`:
- Original posterior: `q(λ̃) = N(m, V)`
- Updated posterior: `q(λ̃ | λ(x)) = N(m', V')`

**Updated mean (m → m'):**
```
m' = m + (V·u / (s + uᵀVu)) · (λ(x) - uᵀm)
```

**Updated covariance (V → V'):**
```
V' = V - (V·u·uᵀV) / (s + uᵀVu)
```

The denominator `s + uᵀVu` is the total marginal variance at x.

#### Conditional Moments at x*

For query point x*, define: `u* = K⁻¹ k(x*)` and `s* = k(x*,x*) - k(x*)ᵀ K⁻¹ k(x*)`

**Conditional mean (uses m'):**
```
μ_cond(x*) = u*ᵀ m'
```

**Conditional variance (uses V'):**
```
σ²_cond(x*) = s* + u*ᵀ V' u*
```

Or equivalently:
```
σ²_cond(x*) = k(x*,x*) - u*ᵀ (K - V') u*
```

### Monte Carlo Estimation of H_cond

```python
H_cond_sum = 0
for _ in range(n_mc_samples):
    # 1. Sample x from p(x) (e.g., Gaussian)
    x_i ~ p(x)

    # 2. Sample λ from GP posterior at x_i
    μ_i, σ²_i = GP_marginal(x_i)
    λ_i ~ N(μ_i, σ²_i)

    # 3. Compute updated posterior (m', V')
    u_i = K⁻¹ k(x_i)
    s_i = k(x_i, x_i) - k(x_i)ᵀ K⁻¹ k(x_i)
    denom = s_i + u_i.T @ V @ u_i
    m' = m + (V @ u_i / denom) * (λ_i - u_i.T @ m)
    V' = V - (V @ u_i @ u_i.T @ V) / denom

    # 4. Compute conditional moments at x*
    μ_cond = u*.T @ m'
    σ²_cond = s* + u*.T @ V' @ u*

    # 5. Compute conditional entropy H(R | x*, λ(x_i), D)
    H_cond_i = compute_H(μ_cond, σ²_cond, r_max)
    H_cond_sum += H_cond_i

H_cond = H_cond_sum / n_mc_samples
```

### Implementation in gp_utility_playground.py

The `evaluate_distribution_aware_utility()` function implements this, but uses GPyTorch's `covariance_matrix` for conditioning instead of explicit (m, V) → (m', V') updates:

```python
def get_conditional_moments(model, x_star, x_sample, lambda_sample):
    # Get full covariance for [x_sample, x_star]
    all_x = torch.cat([x_sample, x_star])
    full_covar = model(all_x).covariance_matrix

    # Standard Gaussian conditioning
    var_sample = full_covar[0, 0]
    cross_cov = full_covar[0, 1:]
    mu_cond = mu_star + cross_cov * (lambda_sample - mu_sample) / var_sample
    sigma2_cond = var_star - (cross_cov ** 2) / var_sample

    return mu_cond, sigma2_cond
```

### Cross-Covariance Structure

The cross-covariance `Σ_{x,x*}` between λ(x) and λ(x*) has TWO components:
```
Σ_{x,x*} = s_cross + u*ᵀ V u
```
where:
- `s_cross = k(x,x*) - uᵀ K u*` — **residual correlation** (direct kernel correlation)
- `u*ᵀ V u` — **epistemic correlation** (through inducing points)

The standard `nd_utility` only considers epistemic correlation. Distribution-aware utility accounts for both.

## Key Implementation Notes

- Uses GPyTorch `ApproximateGP` with `VariationalStrategy` (Poisson likelihood requires variational inference)
- Custom `PoissonLikelihood` class (GPyTorch doesn't have built-in Poisson)
- `utility.py` sets `torch.set_grad_enabled(False)` globally - use `torch.enable_grad()` for training

## Recent Fixes (Dec 2024)

**Laplace Approximation Overflow Bug (FIXED)**:
- `nd_utility()` had catastrophic normalization failure in high-uncertainty regions
- Root cause: overflow handling in `argmax_g_old()` assigned constant p=0.27 to all large r values
- Fix: `nd_utility_new()` uses log-space Lambert W (`lambertw0_log`) - no overflow, Σp≈1.0 always
- See `OVERFLOW_BUG_ANALYSIS.md` for full details

**Monte Carlo Implementation**:
- `nd_utility_MC_batched()` added for high sample counts (100k+) without GPU memory issues

## Choosing r_max for Utility Computation

**Critical**: All utility methods (Laplace, NUMERICAL, hybrid, MC) require truncating the infinite Poisson sum at `r_max`. If `r_max` is too small relative to the firing rate, **all methods fail** - including NUMERICAL which is supposed to be ground truth.

**Why this happens**: For Poisson(f), the probability mass is concentrated around r ≈ f. If r_max << f, we capture ~0% of the probability, making Σp(r) << 1 and entropy calculations meaningless.

**Rule for 99.99% probability coverage**:
```
r_max > f_max + 5 * sqrt(f_max)
```
where `f_max = max(exp(μ + σ²/2))` is the maximum expected firing rate.

**Important**: Since λ ~ N(μ, σ²), actual samples of f = exp(λ) can significantly exceed E[f] due to log-normal skewness. Use conservative estimates.

**Recommended r_max values**:

| f_max | Formula: f + 5√f | Recommended r_max |
|-------|------------------|-------------------|
| 20    | 20 + 22 = 42     | 75                |
| 50    | 50 + 35 = 85     | 150               |
| 100   | 100 + 50 = 150   | 250               |
| 200   | 200 + 71 = 271   | 450               |
| 400   | 400 + 100 = 500  | 800               |

**Rule of thumb**: When in doubt, use `r_max ≈ 2 * f_max`.

**Adaptive selection** (future improvement): Compute `f_max` from the GP posterior before utility evaluation and set `r_max` dynamically.

## Investigation (Dec 2024): Distribution-Aware Utility Peaks - RESOLVED

**Problem**: `distribution_aware_utility_gpytorch()` with Gaussian p(x) shows utility peaks OUTSIDE the Gaussian rather than inside.

**Resolution**: The behavior is mathematically CORRECT. The initial intuition was wrong.

**Key findings from diagnostic scripts**:

1. **Utility = H_marg - H_cond depends on ABSOLUTE values, not just ratios**:
   - Inside Gaussian: H_marg=1.48, H_cond=1.39 → U=0.09 (58% var reduction)
   - Outside Gaussian: H_marg=3.66, H_cond=3.41 → U=0.24 (25% var reduction)
   - Higher H_marg outside dominates, giving higher utility despite less variance reduction

2. **Cross-covariance can be NEGATIVE** due to GP posterior structure:
   - σ²_cond = Σ_** - Σ_x*²/Σ_xx uses SQUARED cross-covariance
   - Both positive AND negative correlations reduce conditional variance

3. **Correct interpretation**:
   - Utility measures information gain about f(X) for X~p(x) from observing R at x*
   - When H_marg is high (uncertain regions), even modest reduction gives good utility
   - The algorithm correctly finds that querying at high-uncertainty points (outside Gaussian) is valuable

**Diagnostic scripts**:
- `diagnose_single_sample.py` - Traces single (x_sample, x_star) pairs step-by-step
- `diagnose_utility_decomposition.py` - Full 6-panel visualization of all components

**Mathematical analysis**: `distribution_aware_utility_analysis.tex` - Self-contained LaTeX document explaining the resolution

**See**: `/home/idv-eqs8-pza/.claude/plans/encapsulated-shimmying-key.md` for full analysis

## Bug Investigation (Jan 2025): Train Border Utility Peaks

**Folder**: `train_border_bug/` - Investigating spurious utility peaks at the border of training data caused by manual cross-covariance formula differing from GPyTorch's `covariance_matrix`.
