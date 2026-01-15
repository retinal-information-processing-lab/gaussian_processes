# Utility Optimization: Analysis and Known Issues

This document describes the gradient optimization behavior for the distribution-aware utility function U(x*) and recommended approaches.

---

## Problem: Gradient Fails Far From Peak

When optimizing U(x*) to find the maximum (at x_sample), gradient descent **fails when starting far from the peak**. The gradient points in the wrong direction in flat regions.

### Observed Gradient Behavior

| x* | U(x*) | gradient | direction | correct? |
|----|-------|----------|-----------|----------|
| -1.50 | 0.000004 | -0.000117 | left | WRONG |
| -0.50 | 0.000017 | -0.002044 | left | WRONG |
| -0.25 | 0.010503 | +0.128343 | right | OK |
| 0.00 | 0.088658 | +0.505259 | right | OK |
| 0.25 | 0.189599 | ~0 | (peak) | OK |

**Key observation**: Gradient is correct within ~1 GP lengthscale of x_sample, wrong outside.

---

## Root Cause: Numerical Cancellation

### Mathematical Structure

```
U(x*) = H_marg(x*) - H_cond(x*)
```

- H_marg(x*) = entropy of response at x* under GP posterior
- H_cond(x*) = entropy at x* after conditioning on knowing λ(x_sample)

### Why U ≈ 0 Far From x_sample

The RBF kernel correlation decays exponentially:
```
k(x*, x_sample) = exp(-|x* - x_sample|² / (2 * lengthscale²))
```

When |x* - x_sample| >> lengthscale:
- k(x*, x_sample) → 0
- Knowing λ(x_sample) provides **no information** about λ(x*)
- Therefore: H_cond(x*) ≈ H_marg(x*)
- Result: U(x*) = H_marg - H_cond ≈ 0

### The Cancellation Problem

The gradient ∂U/∂x* = ∂H_marg/∂x* - ∂H_cond/∂x*

When H_marg ≈ H_cond (both ~1.35):
- We compute the **difference of two nearly-equal quantities**
- Classic **catastrophic cancellation**
- True gradient (~10⁻⁶) is smaller than numerical noise
- Computed gradient is dominated by floating-point errors

### Classification

| Category | Is it the cause? |
|----------|------------------|
| Algorithm (Adam, SGD, etc.) | No - any gradient method fails in flat regions |
| Numerical precision | Contributes, but not fixable with higher precision |
| **Problem structure** | **YES** - the landscape IS genuinely flat far from peak |

---

## Utility Landscape Shape

```
         U(x*)
           ^
    0.19   |           *
           |          * *
           |         *   *
    0.05   |        *     *
           |       *       *
    ~0     |******           ******
           +-------------------------> x*
              -2   -0.5  0.25  1    2
                         ^
                      x_sample
```

1. **Sharp peak** at x* = x_sample (width ~1 lengthscale)
2. **Flat plateau** at U ≈ 0 everywhere else
3. **Transition region** where gradient becomes meaningful (~0.5 from peak)

---

## Recommended Optimization Strategies

### For 1D: Grid Search + Local Refinement (RECOMMENDED)

```python
# Phase 1: Grid search to find approximate max
x_grid = torch.linspace(X_MIN, X_MAX, 50)
utilities = [compute_utility(x) for x in x_grid]
x_approx = x_grid[argmax(utilities)]

# Phase 2: Local gradient refinement (optional)
x_final = gradient_descent(start=x_approx, steps=50)
```

**Why this works:**
- Grid search always finds the peak region (O(N) evaluations)
- Gradient descent refines from a good starting point
- Simple, no hyperparameter tuning needed

### For Higher Dimensions

| Method | Best for | Pros | Cons |
|--------|----------|------|------|
| Multi-start gradient descent | 2-10D | Parallelizable, simple | May miss peak |
| Bayesian optimization | Expensive U(x) | Sample-efficient | Overhead of surrogate |
| Direct search (Nelder-Mead) | Low D, no gradients | Derivative-free | Slow |

---

## Practical Implications

1. **Don't start gradient descent far from expected peak** - the flat region has unreliable gradients

2. **Use grid search first** - it's fast for 1D and guarantees finding the peak region

3. **The "wrong gradient" is not a bug** - it's a fundamental property of the utility landscape

4. **Peak width scales with lengthscale** - shorter lengthscale = narrower peak = harder to find

---

## Open Questions for Future Investigation

1. **Approximation quality**: How much error does using E[λ] instead of MC sampling introduce?

2. **Higher dimensions**: How does this scale when x is multi-dimensional (e.g., image patches)?

3. **Alternative formulations**: Can U(x*) be reformulated to avoid the cancellation issue?
