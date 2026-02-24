# Input Warping and Bounded-Domain Gaussian Processes: Reference Document

**Created**: 2026-02-24
**Purpose**: Comprehensive literature survey and mathematical reference for input/output warping techniques in Gaussian processes, with focus on enforcing bounded input domains.
**Context**: In our active learning GP, images optimized to maximize utility can have pixel values outside the physical display range [vmin, vmax]. This document surveys approaches to making the GP aware of input bounds.

---

## Table of Contents

1. [The Problem: Unbounded Input Space](#1-the-problem)
2. [Taxonomy of Approaches](#2-taxonomy)
3. [Input Warping](#3-input-warping)
   - 3.1 General Framework
   - 3.2 Beta CDF Warping (Snoek et al. 2014)
   - 3.3 Kumaraswamy CDF Warping (BoTorch / HEBO)
   - 3.4 Fixed Nonlinear Warping (tanh, sigmoid)
4. [Output Warping](#4-output-warping)
   - 4.1 Sum-of-Tanh (Snelson & Rasmussen 2003)
   - 4.2 Compositional Warping (Rios & Tobar 2019)
5. [Constrained GP Approaches](#5-constrained-gps)
   - 5.1 Truncated Posterior Projection
   - 5.2 Truncated Gaussian Likelihood
   - 5.3 Multiplicative Penalty Kernel
6. [Jacobian Requirements: Input vs Output Warping](#6-jacobians)
7. [Interaction with Sparse / Inducing Point Methods](#7-sparse-gps)
8. [Practical Considerations](#8-practical)
9. [Comparison Table](#9-comparison)
10. [Application to Our Problem](#10-our-problem)
11. [References](#11-references)

---

## 1. The Problem: Unbounded Input Space <a name="1-the-problem"></a>

A standard GP kernel k(x, y) treats inputs x as living on R^D (the entire real line per dimension). When we optimize an image x* to maximize an acquisition function (e.g., distribution-aware utility), the optimizer is free to push pixel values to any magnitude. The kernel has no concept of physical bounds.

In our setup, the DMD projector has a finite physical range [0, 255]. In normalized data, this maps to [dataset_global_min, dataset_global_max]. Pixel values outside this range are **physically unrealizable** -- the projector clips them.

The problem is amplified by our kernel structure: the C matrix (a spatial smoother) sees `C @ x`, where C is low-rank. Individual pixel bound violations are invisible to C -- a pixel at the boundary and one slightly past it produce nearly identical `C @ x`. The utility landscape has no "walls" at pixel bounds.

**Current mitigation**: sigmoid reparameterization during optimization (`x = vmin + (vmax-vmin) * sigmoid(z)`). This constrains the optimizer but does not change the model. The GP still predicts high utility past the bounds; the sigmoid just prevents the optimizer from reaching there.

**Desired solution**: Make the GP model itself unable to distinguish "at the bound" from "past the bound," so the utility landscape naturally flattens at boundaries.

---

## 2. Taxonomy of Approaches <a name="2-taxonomy"></a>

```
                      Bounded-Domain GP Techniques
                                |
              +-----------------+-----------------+
              |                 |                 |
        Input Warping      Output Warping    Constrained GP
      (transform x)      (transform y)     (modify posterior)
              |                 |                 |
     +--------+--------+  +----+----+    +-------+-------+
     |        |        |  |         |    |       |       |
   Learned  Fixed   CDF  Sum-of  Compo-  Trunc. Trunc.  Penalty
   (Beta,   (tanh,  only  tanh   sitional Proj.  Likeli- Kernel
   Kuma.)   sigm.)       (2003)  (2019)         hood
   (2014)
```

**Critical distinction between the three families**:

- **Input warping**: Applies w(x) before the kernel. The GP trains on w(x), so its learned parameters (lengthscales, inducing points, etc.) live in warped space. Changes what the model sees.
- **Output warping**: Applies g(y) to observations after the GP. Changes what the model predicts.
- **Constrained GP**: Modifies the posterior or likelihood to enforce bounds. Changes how the model reasons.

For **enforcing input bounds** (our problem), input warping is the natural fit. Output warping and constrained GPs address bounded outputs, which is a different problem.

---

## 3. Input Warping <a name="3-input-warping"></a>

### 3.1 General Framework

Given a standard GP with kernel k, input warping defines a new kernel:

```
k_w(x, y) = k(w(x), w(y))
```

where w: R^D -> R^D is a (typically componentwise) warping function. If w is applied independently per dimension:

```
w(x) = (w_1(x_1), w_2(x_2), ..., w_D(x_D))
```

The GP is then:

```
f(x) ~ GP(mu(w(x)), k(w(x), w(x')))
```

**Key property**: No Jacobian correction is needed in the marginal likelihood. The change of variables is absorbed into the kernel -- we are simply evaluating the kernel at different points. The observations y are unchanged; only the locations where we evaluate are transformed. See Section 6 for the detailed argument.

**Effect on optimization**: When maximizing utility U(x*), the gradient is:

```
dU/dx* = dU/dw * dw/dx*
```

If w(x) saturates outside [vmin, vmax], then dw/dx* -> 0 outside bounds, and dU/dx* -> 0. The optimizer naturally decelerates and stops at the walls.


### 3.2 Beta CDF Warping (Snoek et al. 2014)

**Paper**: "Input Warping for Bayesian Optimization of Non-Stationary Functions"
Jasper Snoek, Kevin Swersky, Richard S. Zemel, Ryan P. Adams.
ICML 2014, PMLR 32:1674-1682.

**Goal**: Not bound enforcement, but handling non-stationarity. Standard stationary kernels (RBF) assume the function varies uniformly across the input space. Many real functions are smoother in some regions and rougher in others. The idea: if we warp the input space, a non-stationary function in the original space becomes (approximately) stationary in the warped space, where a standard kernel works well.

**Warping function**: The regularized incomplete Beta function (Beta CDF):

```
w_i(x_i) = I_{x_i}(alpha_i, beta_i) = B(x_i; alpha_i, beta_i) / B(1; alpha_i, beta_i)

where B(x; a, b) = integral_0^x t^{a-1} (1-t)^{b-1} dt   (incomplete beta function)
```

This maps [0, 1] -> [0, 1] with shape controlled by (alpha_i, beta_i):

- alpha = beta = 1: identity (no warping)
- alpha < 1, beta > 1: stretches left side (like log transform)
- alpha > 1, beta < 1: stretches right side (like exp transform)
- alpha > 1, beta > 1: compresses tails, expands center (sigmoidal)
- alpha < 1, beta < 1: expands tails, compresses center

Each input dimension gets independent (alpha_i, beta_i), for a total of 2D extra parameters.

**Prior specification**: Log-normal priors on (alpha, beta) to encode expected warping behavior:

| Behavior          | mu_alpha | sigma_alpha | mu_beta | sigma_beta |
|-------------------|----------|-------------|---------|------------|
| Slight/no warping | 0.0      | 0.5         | 0.0     | 0.5        |
| Logarithmic       | 0.0      | 0.25        | 1.0     | 1.0        |
| Exponential       | 1.0      | 1.0         | 0.0     | 0.25       |
| Sigmoidal         | 2.0      | 0.5         | 2.0     | 0.5        |

(mu, sigma are parameters of the log-normal distribution on alpha and beta.)

**Optimization**: Parameters optimized by maximizing the GP marginal likelihood. Snoek et al. used a fully Bayesian approach with MCMC (slice sampling) to marginalize over warping parameters alongside kernel hyperparameters.

**Computational limitation**: The Beta CDF (regularized incomplete beta function) has **no closed-form expression**. It requires numerical integration or special function evaluation. This makes automatic differentiation non-trivial and computation slower than alternatives.

**Inputs must be in [0, 1]**: The Beta CDF is defined on [0, 1]. Inputs must be normalized to this range before warping.


### 3.3 Kumaraswamy CDF Warping (BoTorch / HEBO)

**Origin**: The Kumaraswamy distribution was proposed as a computationally efficient alternative to the Beta distribution. Used in BoTorch (built on GPyTorch) and HEBO (Cowen-Rivers et al. 2022).

**CDF formula** (closed form):

```
K(x; a, b) = 1 - (1 - x^a)^b     for x in [0, 1],  a > 0,  b > 0
```

**PDF**:

```
k(x; a, b) = a * b * x^{a-1} * (1 - x^a)^{b-1}
```

**Inverse CDF** (closed form):

```
K^{-1}(u; a, b) = (1 - (1 - u)^{1/b})^{1/a}
```

**Why preferred over Beta CDF**:

1. **Closed-form CDF**: Elementary algebraic operations only (powers, subtraction). No special functions.
2. **Closed-form inverse**: Enables efficient sampling and quantile computation.
3. **Closed-form derivative**: dK/dx = a * b * x^{a-1} * (1 - x^a)^{b-1}. Clean gradient flow.
4. **2-9x faster** than Beta CDF in practice (Cowen-Rivers et al. 2022).
5. **Similar shape flexibility** to Beta: can approximate log, exp, sigmoidal, identity warps.

**Special cases**:

- a = 1, b = 1: K(x) = x (identity, no warping)
- a = 1: K(x) = 1 - (1-x)^b (power of complement)
- b = 1: K(x) = x^a (simple power)

**BoTorch implementation** (`botorch.models.transforms.input.Warp`):

```python
# Default initialization
Warp(
    d=input_dim,
    concentration1_prior=LogNormalPrior(0.0, sqrt(0.75)),  # prior on 'a'
    concentration0_prior=LogNormalPrior(0.0, sqrt(0.75)),  # prior on 'b'
)
```

Default priors: median = exp(0) = 1.0 (identity warping), scale ~ 0.87 on log scale.

**Per-dimension**: Like Beta CDF, each dimension gets independent (a_i, b_i). For D dimensions, 2D extra parameters.

**GPyTorch status**: GPyTorch itself does NOT provide input warping (as of 2025). BoTorch adds it as a model transform layer. GPyTorch maintainers declined to add it natively (GitHub Discussion #2124), recommending custom likelihood + `added_loss_terms` or Deep Kernel Learning as workarounds.


### 3.4 Fixed Nonlinear Warping (tanh, sigmoid)

For the specific goal of **bounding the input domain** (not learning non-stationarity), a fixed (non-learnable) warping function suffices. The function should:

1. Be approximately identity within [vmin, vmax]
2. Saturate smoothly outside [vmin, vmax]
3. Be differentiable everywhere (for gradient-based training and optimization)
4. Be monotonic (to preserve ordering)

**Scaled tanh**:

```
w(x) = mid + (range/2) * tanh(a * (x - mid) / (range/2))

where mid = (vmin + vmax) / 2
      range = vmax - vmin
      a = steepness parameter (a > 1 gives sharper walls)
```

Properties:
- w(mid) = mid (center preserved)
- w(x) -> vmin as x -> -inf, w(x) -> vmax as x -> +inf
- Approximately linear near mid when a ~ 1
- Derivative: dw/dx = a * (1 - tanh(...)^2)  (sech^2, goes to 0 at extremes)

**Scaled sigmoid**:

```
w(x) = vmin + (vmax - vmin) * sigmoid(a * (x - mid) / (range/2))

where sigmoid(z) = 1 / (1 + exp(-z))
```

Properties:
- w(mid) = mid (center preserved, since sigmoid(0) = 0.5)
- w(x) -> vmin as x -> -inf, w(x) -> vmax as x -> +inf
- Derivative: dw/dx = a/range * sigmoid(z) * (1 - sigmoid(z))

**Comparison**: tanh and sigmoid are related by tanh(z) = 2*sigmoid(2z) - 1. They produce equivalent warping up to rescaling of the steepness parameter.

**Soft clipping** (piecewise smooth):

```
w(x) = x                                          if vmin <= x <= vmax
      = vmax - (1/a) * log(1 + exp(-a*(x - vmax)))  if x > vmax   (softplus decay)
      = vmin + (1/a) * log(1 + exp(a*(x - vmin)))   if x < vmin   (softplus decay)
```

Properties:
- **Exactly** identity within [vmin, vmax] (zero distortion inside bounds)
- Smooth transition at boundaries (no kink in derivative)
- Derivative continuous but not smooth at boundaries (C^1 but not C^inf)
- More complex formula but conceptually cleaner: "identity inside, softplus outside"

**Choosing steepness 'a'**: The steepness controls the transition width. For a scaled tanh centered at mid with range R:
- At x = vmin (or vmax): w(x) differs from x by approximately R/2 * (tanh(a) - a) / a
- For a = 1: gentle saturation, ~24% compression at bounds
- For a = 2: moderate, ~4% compression at bounds
- For a = 3: sharp, ~0.5% compression at bounds
- For a >> 1: approaches hard clipping (non-differentiable in the limit)

In practice, a = 2-4 provides a good balance: nearly identity in bounds, strong saturation outside, smooth enough for gradient-based optimization.

---

## 4. Output Warping <a name="4-output-warping"></a>

Output warping transforms the **observations** y rather than the inputs x. It addresses non-Gaussian observation distributions, not bounded inputs. Included here for completeness and to prevent confusion with input warping.

### 4.1 Sum-of-Tanh (Snelson & Rasmussen 2003)

**Paper**: "Warped Gaussian Processes"
Edward Snelson, Carl Edward Rasmussen.
NIPS 2003.

**Model**:

```
f ~ GP(0, k(x, x'))         (latent GP in "warped" space)
y = g^{-1}(f)               (observations are inverse-warped latent values)
equivalently: g(y) = f      (warped observations are Gaussian)
```

**Warping function** (y -> f, monotonic):

```
g(y) = d * y + sum_{i=1}^{I} a_i * tanh(b_i * (y + c_i))
```

- Linear term d*y: captures overall trend
- Each tanh basis: adds local nonlinear warping
- Parameters: {d, a_i, b_i, c_i} for i = 1..I, where I is the number of tanh terms
- Monotonicity requires d > 0, a_i >= 0

**Log-marginal likelihood** (requires Jacobian):

```
log p(y | X) = -1/2 g(y)^T K^{-1} g(y) - 1/2 log|K| - N/2 log(2pi) + sum_n log|dg/dy_n|
                                                                          ^^^^^^^^^^^^^^^^^
                                                                          Jacobian correction
```

The Jacobian term is:

```
dg/dy_n = d + sum_i a_i * b_i * sech^2(b_i * (y_n + c_i))
```

**Prediction requires inverse warping**: To predict in observation space, we need g^{-1}(f*). Since g is a sum-of-tanh, it has **no analytical inverse**. Must use Newton-Raphson iteration to numerically invert g at each prediction point.

**Limitation**: Newton-Raphson can fail to converge, is computationally expensive, and introduces numerical error into predictions.


### 4.2 Compositional Warping (Rios & Tobar 2019)

**Paper**: "Compositionally-Warped Gaussian Processes"
Marcela Rios, Felipe Tobar.
Neural Networks 118:235-246, 2019.

**Key innovation**: Build the warping as a composition of simple bijections, each with a **known analytical inverse**:

```
g = f_1 o f_2 o ... o f_K
```

Inverse:

```
g^{-1} = f_K^{-1} o ... o f_2^{-1} o f_1^{-1}
```

**Elementary bijections**:

| Name           | f(y)                              | f^{-1}(u)                        | Jacobian |df/dy|            |
|----------------|-----------------------------------|-----------------------------------|--------------------------------------|
| Affine         | alpha + beta * y                  | (u - alpha) / beta                | |beta|                                |
| Box-Cox        | (y^lambda - 1) / lambda           | (1 + lambda * u)^{1/lambda}       | y^{lambda - 1}                       |
| Box-Cox (l=0)  | log(y)                            | exp(u)                            | 1/y                                  |
| Sinh-Arcsinh   | sinh(delta * arcsinh(y) - epsilon)| sinh((arcsinh(u) + epsilon)/delta)| delta*cosh(delta*arcsinh(y)-eps)/sqrt(1+y^2)|
| Arcsinh        | arcsinh(y)                        | sinh(u)                           | 1/sqrt(1 + y^2)                      |

**Jacobian of composition** (chain rule):

```
|dg/dy| = |df_1/df_2| * |df_2/df_3| * ... * |df_K/dy|
```

Each factor is elementary, so the total Jacobian is a product of simple expressions.

**Advantages over sum-of-tanh**:
1. **Analytical inverse**: No Newton-Raphson for prediction.
2. **Stable Jacobian**: Product of simple terms (no catastrophic cancellation).
3. **Interpretable layers**: Each elementary function has clear purpose (shift, scale, shape).
4. **Modular**: Add/remove layers to control expressiveness.

---

## 5. Constrained GP Approaches <a name="5-constrained-gps"></a>

These methods enforce constraints on the GP posterior or likelihood without transforming inputs or outputs. They are relevant when the constraint is on the GP's **output** (e.g., prediction must lie in [a, b]), not on the input domain.

### 5.1 Truncated Posterior Projection

**References**: Da Veiga & Marrel 2012, Maatouk & Bay 2017.

**Idea**: Start with an unconstrained GP posterior, then project it onto the constrained space. For bound constraints y in [a, b]:

```
p_constrained(f | D) = p_GP(f | D) * I(a <= f <= b) / Z
```

where I is the indicator function and Z is a normalizing constant.

The constrained posterior mean is the conditional expectation of a truncated multivariate normal. This requires numerical integration (no closed form for multivariate truncated normals).

**Practical issues**:
- Computationally expensive (multivariate truncated normal moments)
- Not differentiable (indicator function)
- Applicable to bounded outputs, not bounded inputs

### 5.2 Truncated Gaussian Likelihood

**Idea**: Replace the standard Gaussian likelihood with a truncated Gaussian:

```
p(y | f) = N(y; f, sigma^2) * I(a <= y <= b) / Z(f, sigma^2, a, b)

where Z = Phi((b-f)/sigma) - Phi((a-f)/sigma)     (normalizing constant)
      Phi = standard normal CDF
```

**Consequences**:
- The posterior is no longer Gaussian (non-conjugate likelihood)
- Requires approximate inference: Laplace approximation, expectation propagation, or MCMC
- No Jacobian term (different likelihood, not a transformation)

**When useful**: Bounded observations with known hard limits and small noise.

### 5.3 Multiplicative Penalty Kernel

**Idea**: Multiply the base kernel by a penalty that decays outside bounds:

```
k_bounded(x, y) = k_base(x, y) * prod_i phi(x_i) * prod_i phi(y_i)
```

where phi(x_i) is a per-pixel penalty:

```
phi(x_i) ~ 1      if vmin <= x_i <= vmax
phi(x_i) -> 0     if x_i outside [vmin, vmax]
```

For example, phi(x_i) = exp(-gamma * max(0, x_i - vmax)^2 - gamma * max(0, vmin - x_i)^2).

**Effect**: Points with OOB pixels have near-zero kernel correlation with everything. The GP's posterior variance explodes outside bounds (no information), and utility optimization sees no benefit in going there.

**Practical concerns for high-dimensional inputs**:
- Product over D = 11,664 dimensions: phi(x) = prod_i phi(x_i) can underflow to zero even for mild OOB
- Must work in log space: log phi(x) = sum_i log phi(x_i)
- Could apply only to RF-masked pixels (~2500 dimensions) instead of all 11,664
- Not standard in the literature; would require careful implementation and validation

---

## 6. Jacobian Requirements: Input vs Output Warping <a name="6-jacobians"></a>

This is a frequently confused point. Here is the precise argument.

### Output warping REQUIRES a Jacobian correction

When we observe y but model g(y) = f ~ GP:

```
p(y | X) = p(g(y) | X) * |dg/dy|                   (change of variables in probability)
log p(y | X) = log p(f | X) + sum_n log|dg/dy_n|    (log-likelihood)
```

The Jacobian |dg/dy| accounts for the stretching/compression of probability mass under the transformation. Without it, the model assigns incorrect probability to observations.

### Input warping does NOT require a Jacobian correction

When we evaluate the kernel at w(x) instead of x, we are changing where in function space we evaluate, not transforming a probability density:

```
f(x) ~ GP(mu(w(x)), k(w(x), w(x')))
p(y | f(x)) unchanged (still Gaussian likelihood around f)
```

The marginal likelihood is:

```
log p(y | X) = -1/2 y^T K_w^{-1} y - 1/2 log|K_w| - N/2 log(2pi)
```

where [K_w]_{ij} = k(w(x_i), w(x_j)). This is a standard GP marginal likelihood with a modified kernel matrix. No Jacobian.

**Intuition**: Input warping changes the **covariance structure** (how similar two points look to the GP), not the **observation model** (how observations relate to the latent function). The kernel absorbs the warping; the likelihood stays clean.

**Exception**: If warping parameters are treated as random variables in a fully Bayesian model, their prior enters the marginal likelihood. But this is a prior term, not a Jacobian.

---

## 7. Interaction with Sparse / Inducing Point Methods <a name="7-sparse-gps"></a>

Both our training modes (vargp_direct and default_gpy) use inducing points. How does warping interact?

### Variational sparse GP (ELBO)

The ELBO for a sparse GP is:

```
ELBO = E_q(f) [log p(y | f)] - KL[q(u) || p(u)]

where u = f(Z)    (function values at inducing locations Z)
      q(u) = N(m, V)
```

With input warping:

```
u = f(w(Z))       (evaluate at warped inducing locations)
K_uu = k(w(Z), w(Z))
K_uf = k(w(Z), w(X))
K_ff_diag = k(w(X), w(X))_diag
```

The ELBO formula is unchanged -- we just compute kernel matrices at warped locations. No Jacobian term.

### Where inducing points live

**Two choices**:

1. **Store inducing points in original space**, warp on the fly:
   - Z stored as raw pixel values
   - Every kernel call computes k(w(Z), w(X))
   - Inducing point optimization (if used) operates in original space
   - Pro: inducing points remain interpretable as images
   - Con: slight overhead from warping at every kernel call

2. **Store inducing points in warped space**:
   - Z_w = w(Z) stored directly
   - Kernel calls compute k(Z_w, w(X))
   - Pro: inducing point kernel k(Z_w, Z_w) computed once without re-warping
   - Con: inducing points no longer directly interpretable as images

For our project, option 1 is simpler and consistent with existing code.

### Eigenspace decomposition (vargp_direct)

In vargp_direct mode, we eigendecompose K_tilde = k(Z, Z) and work in the reduced eigenspace. With warping:

```
K_tilde = k(w(Z), w(Z))
```

The eigendecomposition proceeds identically on the warped kernel matrix. All eigenspace operations (m_b, V_b, etc.) are unchanged -- they depend on K_tilde's eigenvalues, not on whether K_tilde was computed from warped or unwarped inputs.

---

## 8. Practical Considerations <a name="8-practical"></a>

### 8.1 Where warping goes in the compute pipeline

For a kernel with structure `k(x, y) = f(C @ x, C @ y)` (our arc-cosine kernel), the warping should go **before C**:

```
x -> w(x) -> mask pixels -> C @ w(x) -> kernel math
```

Rationale: We want per-pixel saturation before spatial smoothing. If warping went after C, the smoothed representation `C @ x` could still contain OOB contributions.

An alternative placement is **after masking but before C**:

```
x -> mask pixels -> w(x_masked) -> C @ w(x_masked) -> kernel math
```

This is equivalent if w is applied elementwise and the mask is applied before warping. It is slightly more efficient (warp ~2500 pixels instead of 11,664).

### 8.2 Distortion within bounds

Any sigmoidal warping introduces some distortion even within [vmin, vmax]. For a scaled tanh with steepness a:

- The warping is approximately linear near the center: w(x) ~ x for |x - mid| << range/(2a)
- Near the boundaries, there is compression: w(vmin) and w(vmax) are not exactly at vmin and vmax

The distortion is controlled by the steepness parameter a:
- Large a (sharp walls): negligible distortion inside, but dw/dx has a very steep gradient at boundaries (may cause numerical issues)
- Small a (gentle saturation): more distortion inside, but smoother everywhere
- a ~ 2-4: practical sweet spot for pixel-bound enforcement

One can also construct a piecewise function that is **exactly** identity inside [vmin, vmax] and smoothly saturates outside (the "soft clipping" variant in Section 3.4). This avoids any in-bounds distortion at the cost of a more complex formula.

### 8.3 Effect on learned kernel hyperparameters

The model trains on w(x), so all learned parameters adapt to the warped input space. If w(x) ~ x within bounds (low distortion), the effect on hyperparameters should be small:

- **C matrix (RF structure)**: The C matrix learns spatial smoothing in the warped pixel space. Since warping is per-pixel and approximately identity in bounds, the spatial structure is preserved.
- **Kernel lengthscales / beta / rho**: These control how quickly the kernel decays with distance in the input space. In warped space, distances near the boundaries are compressed. The kernel may learn slightly different parameters to compensate.
- **Inducing points**: Selected from training data, which is within bounds. Warping barely affects them.
- **Variational parameters (m, V)**: Adapted during E-step to the warped kernel matrix. Should converge similarly.

### 8.4 Effect on gradient-based utility optimization

The gradient of utility w.r.t. the optimized image x* passes through the warping:

```
dU/dx* = dU/dw(x*) * dw/dx*
```

Since dw/dx* -> 0 as x* -> +/- inf, the gradient naturally vanishes outside bounds. This means:

- **No explicit constraint needed** in the optimizer (no sigmoid reparameterization, no projection)
- The optimizer can start from any point and will be guided toward the bounded region
- Convergence near boundaries may slow down (gradient compression), similar to the "vanishing gradient" phenomenon in deep learning

### 8.5 Numerical stability

**tanh saturation**: For large |z|, tanh(z) ~ +/- 1 with floating point precision issues. In float32, tanh(z) = +/- 1.0 exactly for |z| > ~9. This means w(x) = vmax (or vmin) exactly for sufficiently OOB pixels. Not a problem -- this is the desired behavior.

**Gradient underflow**: dw/dx = a * sech^2(z). For |z| > 9, sech^2(z) ~ 0 in float32. The gradient is effectively zero far from bounds. This is fine for utility optimization (we want it to stop), but could cause issues for kernel hyperparameter gradients if inducing points drift out of bounds. Since inducing points are selected from training data (which is in bounds), this should not arise in practice.

### 8.6 Interaction with analytical gradients

The analytical gradient modes (VJP, Jacobian) compute kernel gradients with explicit formulas that assume the kernel sees raw pixels. With input warping, these formulas would need to be extended to account for the chain rule through w(x). This requires:

```
dk/dx = dk/dw * dw/dx
```

For autograd mode, this is handled automatically. For analytical modes, the Jacobian dw/dx (which is diagonal, since w is elementwise) would need to be incorporated into the gradient expressions.

Since dw/dx is diagonal, the modification is a simple elementwise multiplication of the existing gradient by dw/dx. This is straightforward but does require changes to the analytical gradient code.

### 8.7 Warping as a learnable parameter (future direction)

The steepness parameter a can be made learnable by including it in the set of kernel hyperparameters optimized via marginal likelihood. This allows the model to determine how "hard" the walls should be:

- If the data provides no information about out-of-bounds behavior, the model can learn soft walls
- If in-bounds data strongly constrains boundary behavior, the model can learn hard walls

The implementation would add a single scalar parameter (shared across all pixels) to the kernel, optimized alongside beta, rho, sigma_0, etc.

A fully per-pixel warping (different (a_i) per pixel) is possible but adds 11,664 parameters. This is the Snoek/BoTorch approach and is likely overkill for bound enforcement (all pixels share the same physical bounds).

---

## 9. Comparison Table <a name="9-comparison"></a>

| Method | Type | Goal | Learnable Params | Jacobian Needed | Closed Form | Inverse | Complexity |
|--------|------|------|-------------------|-----------------|-------------|---------|------------|
| Beta CDF (Snoek 2014) | Input | Non-stationarity | 2D (alpha_i, beta_i) | No | No (incomplete beta) | Via quantile function | Medium |
| Kumaraswamy (BoTorch) | Input | Non-stationarity | 2D (a_i, b_i) | No | Yes | Yes (closed form) | Low |
| Fixed tanh/sigmoid | Input | Bound enforcement | 0-1 (steepness) | No | Yes | Yes (arctanh/logit) | Very low |
| Sum-of-tanh (Snelson 2003) | Output | Non-Gaussian obs. | 4I (a,b,c,d per tanh) | Yes | Yes (forward) | No (Newton-Raphson) | High |
| Compositional (Rios 2019) | Output | Non-Gaussian obs. | Per-layer params | Yes | Yes | Yes | Medium |
| Truncated posterior | Constraint | Bounded output | 0 | No | No | N/A | High |
| Truncated likelihood | Constraint | Bounded output | 0 | No | Partial | N/A | Medium |
| Penalty kernel | Constraint | Bounded input | 1 (decay rate) | No | Yes | N/A | Low* |

*Low complexity per se, but numerical issues in high dimensions (product over D terms).

---

## 10. Application to Our Problem <a name="10-our-problem"></a>

### What we need

1. Pixel values in optimized images should respect [vmin, vmax] without explicit optimizer constraints.
2. The GP model should be unable to distinguish "at the bound" from "past the bound."
3. Minimal distortion to the model's behavior within bounds.
4. Must work with both vargp_direct and default_gpy modes.
5. Must be differentiable for gradient-based training and utility optimization.

### Recommended approach: Fixed scaled tanh (Section 3.4)

**Why**:
- Zero extra learnable parameters (simplest possible implementation).
- Applies identically to all pixels (same physical bounds everywhere).
- Differentiable, clean gradient flow.
- No Jacobian correction needed (input warping).
- Compatible with both training modes and all kernel types.
- The steepness parameter can be made learnable later if needed (Section 8.7).

**Pipeline change**:

```
Current:    x -> mask -> C @ x -> kernel math
Proposed:   x -> mask -> w(x_masked) -> C @ w(x_masked) -> kernel math
```

Where:

```
w(x_i) = mid + (range/2) * tanh(a * (x_i - mid) / (range/2))
mid = (vmin + vmax) / 2
range = vmax - vmin
a = steepness (fixed constant, e.g., 3)
```

### What changes

1. **kernels.py**: Add warping before C matrix application in forward().
2. **Model training**: The model trains on warped pixels. Kernel hyperparameters adapt.
3. **Utility optimization**: No sigmoid reparameterization needed. Gradient naturally vanishes at bounds.
4. **Analytical gradients** (VJP/Jacobian): Need chain rule through w. Autograd handles this; analytical modes need manual update.
5. **Test metrics**: test_r should be compared with and without warping to verify no degradation.

### What does NOT change

1. **ELBO formula**: Unchanged (no Jacobian).
2. **E-step, M-step, F-step**: Same algorithms, just different kernel matrix values.
3. **Inducing point selection**: Still selected from training data.
4. **Eigenspace decomposition**: Same procedure on the warped K_tilde.
5. **Plotting**: Images are displayed in original (unwarped) space.

---

## 11. References <a name="11-references"></a>

### Primary references

1. **Snoek, J., Swersky, K., Zemel, R. S., & Adams, R. P.** (2014). "Input Warping for Bayesian Optimization of Non-stationary Functions." *ICML 2014*, PMLR 32:1674-1682. [arXiv:1402.0929](https://arxiv.org/abs/1402.0929)

2. **Snelson, E. & Rasmussen, C. E.** (2003). "Warped Gaussian Processes." *NIPS 2003*. [Paper PDF](https://proceedings.neurips.cc/paper_files/paper/2003/file/6b5754d737784b51ec5075c0dc437bf0-Paper.pdf)

3. **Rios, M. & Tobar, F.** (2019). "Compositionally-Warped Gaussian Processes." *Neural Networks* 118:235-246. [arXiv:1906.09665](https://arxiv.org/abs/1906.09665)

4. **Cowen-Rivers, A. I., et al.** (2022). "HEBO: Pushing the Limits of Sample-Efficient Hyper-parameter Optimisation." *JAIR* 74:1269-1349. [arXiv:2012.03826](https://arxiv.org/abs/2012.03826)

### Constrained GPs

5. **Da Veiga, S. & Marrel, A.** (2012). "Gaussian Process Modeling with Inequality Constraints." *Annales de la Faculte des sciences de Toulouse: Mathematiques* 21(3):529-555. [Link](https://www.numdam.org/item/AFST_2012_6_21_3_529_0/)

6. **Swiler, L. P., et al.** (2020). "A Survey of Constrained Gaussian Process Regression: Approaches and Implementation Challenges." [arXiv:2006.09319](https://arxiv.org/abs/2006.09319)

### Software

7. **BoTorch** input warping: `botorch.models.transforms.input.Warp`. [Tutorial](https://botorch.org/docs/tutorials/bo_with_warped_gp/)

8. **GPyTorch** Discussion #2124: Warped GP feature request (declined). [GitHub](https://github.com/cornellius-gp/gpytorch/discussions/2124)

### Background

9. **Rasmussen, C. E. & Williams, C. K. I.** (2006). *Gaussian Processes for Machine Learning*. MIT Press. Chapter 2 (GP regression), Chapter 9 (approximate inference).

10. **Kumaraswamy, P.** (1980). "A Generalized Probability Density Function for Double-bounded Random Processes." *Journal of Hydrology* 46(1-2):79-88.

---

*This document is a general reference. Implementation decisions and code changes are tracked separately.*
