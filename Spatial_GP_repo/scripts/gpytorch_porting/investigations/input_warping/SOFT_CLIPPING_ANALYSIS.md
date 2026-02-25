# Soft-Clipping Analysis for Bounded Pixel Domain

**Date**: 2026-02-25
**Context**: Replacing tanh warping in the kernel (which hurts test_r by 5-12%) with a function that is exactly identity inside [vmin, vmax].

---

## 1. Why Tanh Fails

The current implementation uses scaled tanh:

```
w(x) = mid + half_range * tanh(a * (x - mid) / half_range)
```

This is a global S-curve. The steepness parameter `a` controls the slope at the center: `dw/dx|_{x=mid} = a`. This creates a fundamental problem:

**Higher steepness does NOT mean "more identity-like inside bounds."** It means a steeper S-curve that pushes interior pixels toward the extremes.

### Numerical evidence

Let u = (x - mid) / half_range, so u in [-1, 1] corresponds to x in [vmin, vmax].
The warped normalized position is tanh(a * u).

| u (position) | a=1    | a=3    | a=6    | a=10   |
|---------------|--------|--------|--------|--------|
| 0.0 (center)  | 0.000  | 0.000  | 0.000  | 0.000  |
| 0.1           | 0.100  | 0.291  | 0.537  | 0.762  |
| 0.3           | 0.291  | 0.716  | 0.947  | 0.995  |
| 0.5           | 0.462  | 0.905  | 0.995  | 1.000  |
| 0.8           | 0.664  | 0.980  | 1.000  | 1.000  |
| 1.0 (boundary)| 0.762  | 0.995  | 1.000  | 1.000  |

Reading: at a=6, a pixel 30% of the way from center to edge (u=0.3) gets mapped to 94.7% of the way. The kernel sees most pixels clustered at the extremes -- like binarized images.

### Key insight: derivative w.r.t. x is irrelevant during training

During training and evaluation, gradients w.r.t. input images x are never computed:
- M-step: `torch.autograd.grad(loss, kernel_params, ...)` -- only kernel hyperparameters
- E-step: Newton update for variational parameters (m_b, V_b), no backprop through x
- F-step: optimizes A, lambda0 only
- Prediction: wrapped in `torch.no_grad()`

So `dw/dx` does not matter. Only the VALUES `w(x)` matter. And those values are catastrophically distorted for any steepness > 1.

### Experimental confirmation

| Steepness | vargp_direct M=50 test_r | Baseline (no warp) |
|-----------|--------------------------|-------------------|
| 3.0       | 0.7549                   | 0.6914            |
| 6.0       | 0.6141                   | 0.6914            |

Higher steepness is worse, confirming value distortion is the mechanism.

(Note: a=3.0 M=50 looks better than baseline, but across the full canonical matrix it averages -5% for vargp_direct and -12% for default_gpy.)

---

## 2. Requirements for a Replacement

1. **Exactly identity** inside [vmin, vmax]: w(x) = x for all in-bounds pixels
2. **Monotonically increasing** everywhere (preserves pixel ordering)
3. **Saturates** outside bounds (prevents unbounded growth)
4. **Differentiable** everywhere (gradient-based utility optimization needs this)
5. **Continuous first derivative** at boundaries (LBFGS approximates Hessian from gradient history; gradient jumps confuse it)

### What is mathematically impossible

An analytic function (convergent Taylor series) that equals x on an interval must equal x everywhere (identity theorem for real-analytic functions). Therefore:

- C^infinity AND analytic: impossible
- C^infinity non-analytic (bump functions): possible in theory, numerically terrible
- C^1 (continuous first derivative): achievable with simple formulas
- C^2 and monotone: difficult with simple formulas (requires non-trivial constructions)

C^1 is the practical target.

---

## 3. Softplus Soft-Clipping (from INPUT_WARPING_REFERENCE.md Section 3.4)

The reference document proposed:

```
w(x) = x                                            if vmin <= x <= vmax
     = vmax - (1/a) * log(1 + exp(-a*(x - vmax)))   if x > vmax
     = vmin + (1/a) * log(1 + exp(a*(x - vmin)))    if x < vmin
```

### Derivative analysis

For x > vmax, let t = x - vmax > 0:

```
w(t) = vmax - (1/a) * log(1 + exp(-a*t))
w'(t) = exp(-a*t) / (1 + exp(-a*t)) = sigmoid(-a*t)
w'(0) = sigmoid(0) = 0.5
```

From inside: w'(vmax-) = 1.

**The first derivative jumps from 1.0 to 0.5 at the boundary.** This is C^0 (continuous), NOT C^1.

The reference document's claim of "C^1 but not C^inf" was incorrect.

### Saturation

w(x) -> vmax as x -> +inf (exact saturation at vmax). This is an advantage.

---

## 4. Exponential Soft-Clipping (C^1)

### Formula

```
w(x) = x                                                if vmin <= x <= vmax
     = vmax + (1/a) * (1 - exp(-a * (x - vmax)))        if x > vmax
     = vmin - (1/a) * (1 - exp(-a * (vmin - x)))        if x < vmin
```

### Continuity check at x = vmax

**Value** (C^0):
- From inside: w(vmax) = vmax
- From outside: vmax + (1/a)(1 - exp(0)) = vmax + 0 = vmax. Continuous.

**First derivative** (C^1):
- From inside: w'(vmax-) = 1
- From outside: w'(x) = exp(-a*(x - vmax)), so w'(vmax+) = exp(0) = 1. Continuous.

**Second derivative** (NOT C^2):
- From inside: w''(vmax-) = 0
- From outside: w''(x) = -a * exp(-a*(x - vmax)), so w''(vmax+) = -a. Jump of magnitude a.

The same analysis holds at vmin by symmetry.

### Properties

| Property | Value |
|----------|-------|
| In-bounds behavior | Exactly identity: w(x) = x |
| Smoothness | C^1 (continuous first derivative), NOT C^2 |
| Monotonicity | Yes (w'(x) > 0 everywhere) |
| Upper saturation | vmax + 1/a (NOT exactly vmax) |
| Lower saturation | vmin - 1/a (NOT exactly vmin) |
| Parameter `a` | Controls outside decay rate. Larger a = faster saturation but larger 2nd-deriv jump |

### Saturation overshoot

The function saturates at vmax + 1/a, not vmax. This is the tradeoff vs softplus (which saturates exactly at vmax).

| a   | Overshoot (1/a) | % of half_range (2.44) |
|-----|-----------------|----------------------|
| 5   | 0.200           | 8.2%                 |
| 10  | 0.100           | 4.1%                 |
| 20  | 0.050           | 2.0%                 |
| 50  | 0.020           | 0.8%                 |
| 100 | 0.010           | 0.4%                 |

This overshoot means the kernel CAN see values slightly outside [vmin, vmax]. In practice, this is acceptable: the purpose is to prevent unbounded growth during utility optimization, not to enforce exact bounds. The projector's physical range has some tolerance too.

### Numerical values (a=10)

| x       | w(x)      | w(x) - x   | Region    |
|---------|-----------|-------------|-----------|
| -5.0000 | -2.5013   | +2.4987     | saturated |
| -3.0000 | -2.5010   | +0.4990     | outside   |
| -2.5000 | -2.4640   | +0.0360     | transition|
| -2.4013 | -2.4013   | 0.0000      | vmin      |
| -2.0000 | -2.0000   | 0.0000      | in-bounds |
|  0.0000 |  0.0000   | 0.0000      | in-bounds |
| +2.0000 | +2.0000   | 0.0000      | in-bounds |
| +2.4780 | +2.4780   | 0.0000      | vmax      |
| +2.5000 | +2.4977   | -0.0023     | transition|
| +3.0000 | +2.5775   | -0.4225     | outside   |
| +5.0000 | +2.5780   | -2.4220     | saturated |

---

## 5. Comparison of Three Approaches

| Property | Tanh (current) | Softplus clip | Exponential clip |
|----------|---------------|---------------|-----------------|
| In-bounds distortion | Nonzero, worse with higher a | Exactly zero | Exactly zero |
| Training cost (test_r) | -5% to -12% | Zero | Zero |
| Smoothness class | C^inf | C^0 (deriv jumps 1->0.5) | C^1 (2nd deriv jumps) |
| Saturation target | Exactly [vmin, vmax] | Exactly [vmin, vmax] | [vmin - 1/a, vmax + 1/a] |
| Monotone | Yes | Yes | Yes |
| Formula complexity | Single expression | Piecewise (3 branches) | Piecewise (3 branches) |
| Gradient at boundary | Smooth decay | Halves suddenly | Smooth decay from 1 |

### Recommendation

**Exponential soft-clipping** is the best option:
- Zero training cost (exact identity in bounds)
- C^1 smooth (LBFGS handles this well)
- Simple formula
- Small saturation overshoot (controllable via `a`, negligible for a >= 10)

The C^2 discontinuity at the boundary only matters during utility optimization (deferred), and even then LBFGS approximates the Hessian from gradient history rather than requiring exact second derivatives.

---

## 6. Implementation Notes

### Vectorized PyTorch formula (no branching)

The piecewise formula can be written without explicit if/else using masks or by exploiting `torch.clamp` and `torch.exp`:

```python
def soft_clip_exp(x, vmin, vmax, a):
    # Compute how far outside each bound
    above = torch.clamp(x - vmax, min=0)   # 0 inside, positive outside upper
    below = torch.clamp(vmin - x, min=0)    # 0 inside, positive outside lower

    # Exponential saturation outside bounds
    x_clipped = x - above + (1/a) * (1 - torch.exp(-a * above))  \
                  + below - (1/a) * (1 - torch.exp(-a * below))

    return x_clipped
```

Equivalently, more readable:

```python
def soft_clip_exp(x, vmin, vmax, a):
    above = torch.clamp(x - vmax, min=0)
    below = torch.clamp(vmin - x, min=0)
    return x - above - below \
           + (1/a) * (1 - torch.exp(-a * above)) \
           + (1/a) * (1 - torch.exp(-a * below))
```

Verification: inside bounds, above=0 and below=0, so returns x. Outside upper, below=0 and above = x - vmax, so returns x - (x-vmax) + (1/a)(1-exp(-a(x-vmax))) = vmax + (1/a)(1-exp(-a(x-vmax))). Correct.

Note: `torch.clamp(..., min=0)` returns exactly 0 for in-bounds values. No floating-point fuzz — `clamp` returns 0.0 exactly when x <= vmax.

### Gradient flow

Autograd handles piecewise functions correctly via `torch.clamp`. The gradient of `clamp(x, min=0)` is 0 when x < 0 and 1 when x > 0 (undefined at 0, PyTorch uses 0). Since we never optimize x values AT the boundary during training, this is not a concern.

### Parameter `a`

The parameter `a` controls only the outside behavior (how fast saturation occurs). It has zero effect on in-bounds computation. Suggested default: a=10 (fast saturation, 4% overshoot).

This parameter could be renamed from `steepness` to something like `decay_rate` to distinguish from the tanh steepness which has a very different meaning.
