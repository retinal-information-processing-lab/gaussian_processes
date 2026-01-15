# Train Border Bug Investigation

This folder contains scripts investigating a numerical stability bug in the distribution-aware utility computation for variational Gaussian processes.

## Overview

We discovered that the manual formulas from the LaTeX derivations are **mathematically correct** but **numerically unstable** when the kernel matrix K is ill-conditioned. The instability comes from explicit K^{-1} computation.

## Formulas Under Investigation

### Marginal Moments (at query point x*)

From `predictive_distribution_derivation_temp.tex`:

```
μ* = u*^T m
σ²* = s* + u*^T V u*
```

Where:
- `u* = K^{-1} k(z, x*)` - projection vector
- `s* = k(x*,x*) - k(x*)^T K^{-1} k(x*)` - Schur complement (prior conditional variance)
- `m` - variational posterior mean over inducing points (M,)
- `V` - variational posterior covariance over inducing points (M×M)
- `K` - inducing point kernel matrix (M×M)

### Conditional Moments (at x* given observation λ(x))

From `predictive_distribution_derivation_temp.tex` Section 3:

**Step 1: Update the inducing point posterior** (condition on observing λ(x))
```
m' = m + (V u / denom) * (λ(x) - u^T m)
V' = V - (V u u^T V) / denom
```

Where:
- `denom = s + u^T V u`
- `s = k(x,x) - u^T K u` (Schur complement at sample point x)
- `u = K^{-1} k(z, x)` (projection at sample point x)

**Step 2: Compute predictive moments using updated posterior**
```
μ_cond = u*^T m'           [Eq. 113 in LaTeX]
σ²_cond = s* + u*^T V' u*  [Eq. 130 in LaTeX]
```

**Important**: Both formulas use the UPDATED posterior (m', V'), not the original (m, V).

## Key Finding

The numerical instability arises from:

1. **Ill-conditioned K matrix**: When K has a high condition number (e.g., 3.7e6 in our tests), the projection u* = K^{-1} k(z, x*) produces incorrect values outside the training region.

2. **Projection vectors outside training region**: For query points x* far from the inducing points, u* can have large magnitude despite k(z, x*) being nearly zero.

3. **`torch.linalg.solve()` does NOT fix the issue**: We tested using `solve()` instead of explicit `inv()` - both produce identical (incorrect) results. The issue is that solving the ill-conditioned system gives the same wrong answer regardless of method.

4. **GPyTorch uses a different formula**: GPyTorch doesn't just use a more stable way to compute K^{-1} - it uses a **different variational formulation** internally (likely Nyström approximation or similar) that avoids this issue entirely.

## Scripts

### `investigate_moment_formulas.py`

Compares the manual LaTeX formulas with GPyTorch's computations:
- 4 subplots comparing marginal mean, marginal variance, conditional mean, conditional variance
- Shows discrepancy between methods, especially outside the training region
- Includes both `stable=False` (explicit K⁻¹) and `stable=True` (torch.linalg.solve) - both produce same (incorrect) results

**Run:**
```bash
conda activate pytorch_gpytorch
python investigate_moment_formulas.py
```

### `simple_utility_gpytorch.py`

Original diagnostic script with additional utility computations.

## Source LaTeX Files

- `/home/idv-eqs8-pza/IDV_code/Papers/latex_summaries/predictive_distribution_derivation_temp.tex`
- `/home/idv-eqs8-pza/IDV_code/Papers/latex_summaries/distribution_aware_utility_pietro.tex`

## Recommendation

For production code, use GPyTorch's built-in `covariance_matrix` method rather than manual formulas. The mathematical derivations are correct, but the numerical implementation requires care to avoid instability from K^{-1}.
