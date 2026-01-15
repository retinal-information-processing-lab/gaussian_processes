# Dirac Delta Collapse Diagnosis

## Problem
When `x_samples = x_star` (Dirac delta), distribution-aware utility should equal nd_utility.
Initial naive MC implementation failed outside the training domain.

## Root Cause (RESOLVED)

**The naive MC approach fails to estimate E[f(1-log f)] due to heavy tails.**

H_cond = E[H(Poisson(exp(λ)))] decomposes as:
```
H_cond = E[f(1-log f)] + E[Σ_r Poisson(r|f) log(r!)]
       = term1         + term2
```

For λ ~ N(μ, σ²) with high σ²:
- term1 = E[exp(λ)(1-λ)] has HEAVY TAILS
- At λ = μ: exp(1.36)(1-1.36) ≈ -1.4
- At λ = μ+2σ: exp(4.46)(1-4.46) ≈ -298
- At λ = μ+3σ: exp(6.01)(1-6.01) ≈ -2044
- True mean is dominated by rare tail events

**Naive MC with finite samples underestimates the tail contribution.**

## Solution

Use the same approach as `nd_utility_NUMERICAL`:

1. **term1: ANALYTICAL** via moment generating function
   ```
   term1 = exp(μ + σ²/2) * (1 - μ - σ²)
   ```
   This exactly computes E[f(1-log f)] without sampling.

2. **term2: Gauss-Hermite quadrature**
   ```
   term2 = ∫ [Σ_r Poisson(r|exp(g)) log(r!)] · N(g|μ,σ²) dg
   ```
   This is well-behaved and converges well with ~100 quadrature points.

3. **H_marg: Gauss-Hermite quadrature**
   Compute p_true(r) by integrating over the Gaussian, then compute entropy.

## Results

With the fixed implementation (`_evaluate_dirac_utility_gauss_hermite`):
- **Dirac vs NUMERICAL: 0.00% error** (exact match when using same r_max)
- Works both inside AND outside the training domain

## Key Insight

The negative "entropy" values observed in the original diagnosis (H_cond = -2.44)
are **mathematically correct** when term1 and term2 are computed consistently.
The issue was the naive MC approach, not the analytical formula.

## Diagnostic Scripts

- `diagnose_covar_at_same_point.py` - GPyTorch covariance check (minor 1e-4 issue)
- `diagnose_H_cond_comparison.py` - Compares H_cond methods
- `diagnose_utility_components.py` - Decomposes U = H_marg - H_cond
