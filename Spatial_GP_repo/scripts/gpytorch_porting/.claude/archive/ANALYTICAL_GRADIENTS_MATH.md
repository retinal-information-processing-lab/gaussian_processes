# Analytical Gradient Formulas for M-step

This document provides the precise mathematical formulas for computing ∂K/∂θ (kernel gradients w.r.t. hyperparameters) that will be implemented to replace autograd in the M-step.

---

## 1. Notation

### 1.1 Hyperparameters

| Symbol | Code name | Description |
|--------|-----------|-------------|
| σ₀ | `sigma_0` | Bias variance |
| Amp | `Amp` | Amplitude (handled by ScaleKernel, gradient is K/Amp) |
| β | `-2log2beta` | RF size, parameterized as `-2log(2β)` |
| ρ | `-log2rho2` | Smoothness, parameterized as `-log(2ρ²)` |
| ξ₀ = (ε₀ₓ, ε₀ᵧ) | `eps_0x`, `eps_0y` | RF center position |

### 1.2 Intermediate Quantities

For inputs x₁ ∈ ℝⁿˣ and x₂ ∈ ℝⁿˣ:

```
v₁ = x₁ᵀ C x₁ + σ₀²          (scalar, "variance" of x₁)
v₂ = x₂ᵀ C x₂ + σ₀²          (scalar, "variance" of x₂)
c₁₂ = x₁ᵀ C x₂ + σ₀²         (scalar, cross-term)

X₁ = √v₁                      (scalar)
X₂ = √v₂                      (scalar)
X₁X₂ = X₁ · X₂ = √(v₁ · v₂)  (scalar, "magnitude")

cos(δ) = c₁₂ / X₁X₂          (scalar, normalized inner product)
δ = arccos(cos(δ))           (scalar, angle)
sin(δ) = √(1 - cos²(δ))      (scalar)

J(δ) = sin(δ) + (π - δ)cos(δ)  (scalar, angular term)
```

### 1.3 Kernel Function

```
K(x₁, x₂) = (1/π) · X₁X₂ · J(δ)
```

Or equivalently:
```
K(x₁, x₂) = (1/π) · √(v₁ · v₂) · [sin(δ) + (π - δ)cos(δ)]
```

---

## 2. C Matrix Definition

The structured covariance matrix C encodes receptive field properties.

### 2.1 Pixel Coordinates

For an image with `n_px_side` pixels per side:
```
ξᵢ = (xᵢ, yᵢ) ∈ [-1, 1] × [-1, 1]
```

Grid construction:
```python
ycord, xcord = meshgrid(linspace(-1, 1, n_px_side), linspace(-1, 1, n_px_side))
```

### 2.2 C Matrix Formula

```
Cᵢⱼ = Amp · αᵢ · Cᵢⱼˢᵐᵒᵒᵗʰ · αⱼ
```

Where:
```
αᵢ = exp(-β · dᵢ²)                    (locality weight)
dᵢ² = (xᵢ - ε₀ₓ)² + (yᵢ - ε₀ᵧ)²      (squared distance to RF center)

Cᵢⱼˢᵐᵒᵒᵗʰ = exp(-ρ² · sᵢⱼ²)          (smoothness kernel)
sᵢⱼ² = (xᵢ - xⱼ)² + (yᵢ - yⱼ)²       (squared distance between pixels)
```

### 2.3 Log-space Parameterization

The code uses log-space parameters for numerical stability:

```
θ_β = -2log(2β)     →  β = exp(-θ_β/2) / 2  →  exp(θ_β) = 1/(4β²)
θ_ρ = -log(2ρ²)     →  ρ² = exp(-θ_ρ) / 2   →  exp(θ_ρ) = 1/(2ρ²)
```

In log-space:
```
log(αᵢ) = -exp(θ_β) · dᵢ²
log(Cᵢⱼˢᵐᵒᵒᵗʰ) = -exp(θ_ρ) · sᵢⱼ²
```

---

## 3. Gradients of C Matrix

These formulas are from `C_gradients_hyp()` in `kernels/kernels.py`.

### 3.1 ∂C/∂Amp

```
∂Cᵢⱼ/∂Amp = Cᵢⱼ / Amp
```

### 3.2 ∂C/∂ε₀ₓ (RF center x-coordinate)

```
∂Cᵢⱼ/∂ε₀ₓ = Cᵢⱼ · 2·exp(θ_β) · [(xᵢ - ε₀ₓ) + (xⱼ - ε₀ₓ)]
           = Cᵢⱼ · 2·exp(θ_β) · [xᵢ + xⱼ - 2ε₀ₓ]
```

### 3.3 ∂C/∂ε₀ᵧ (RF center y-coordinate)

```
∂Cᵢⱼ/∂ε₀ᵧ = Cᵢⱼ · 2·exp(θ_β) · [yᵢ + yⱼ - 2ε₀ᵧ]
```

### 3.4 ∂C/∂θ_β (locality parameter in log-space)

```
∂Cᵢⱼ/∂θ_β = Cᵢⱼ · [log(αᵢ) + log(αⱼ)]
           = Cᵢⱼ · [-exp(θ_β)·dᵢ² - exp(θ_β)·dⱼ²]
```

Note: In code, `logalpha[i] = -exp(θ_β) · dᵢ²`, so:
```
∂C/∂θ_β = C * (logalpha[:, None] + logalpha[None, :])
```

### 3.5 ∂C/∂θ_ρ (smoothness parameter in log-space)

```
∂Cᵢⱼ/∂θ_ρ = Cᵢⱼ · log(Cᵢⱼˢᵐᵒᵒᵗʰ)
           = Cᵢⱼ · [-exp(θ_ρ) · sᵢⱼ²]
```

Note: In code, `logCsmooth[i,j] = -exp(θ_ρ) · sᵢⱼ²`, so:
```
∂C/∂θ_ρ = C * logCsmooth
```

---

## 4. Gradients of K w.r.t. σ₀

This is the direct gradient (σ₀ appears in K independently of C).

### 4.1 Derivatives of Intermediate Quantities

```
∂v₁/∂σ₀ = 2σ₀
∂v₂/∂σ₀ = 2σ₀
∂c₁₂/∂σ₀ = 2σ₀

∂X₁/∂σ₀ = σ₀/X₁
∂X₂/∂σ₀ = σ₀/X₂

∂(X₁X₂)/∂σ₀ = σ₀·(X₂/X₁ + X₁/X₂) = σ₀²·(X₂/X₁ + X₁/X₂)/σ₀
```

Let:
```
dX₁X₂ = σ₀² · (X₂/X₁ + X₁/X₂)
```

### 4.2 Derivative of cos(δ)

```
cos(δ) = c₁₂ / (X₁X₂)

∂cos(δ)/∂σ₀ = [∂c₁₂/∂σ₀ · X₁X₂ - c₁₂ · ∂(X₁X₂)/∂σ₀] / (X₁X₂)²
             = [2σ₀ · X₁X₂ - c₁₂ · dX₁X₂] / (X₁X₂)²
             = [2σ₀² - cos(δ) · dX₁X₂] / (X₁X₂)
```

Let:
```
dcos(δ) = [2σ₀² - cos(δ) · dX₁X₂] / (X₁X₂)
```

### 4.3 Derivative of J(δ)

```
J(δ) = sin(δ) + (π - δ)cos(δ)

dJ/dδ = cos(δ) - cos(δ) + (π - δ)·(-sin(δ)) = -(π - δ)sin(δ)
```

Using chain rule through cos(δ):
```
∂δ/∂cos(δ) = -1/sin(δ)

∂J/∂σ₀ = (dJ/dδ) · (∂δ/∂cos(δ)) · (∂cos(δ)/∂σ₀)
       = -(π - δ)sin(δ) · (-1/sin(δ)) · dcos(δ)
       = (π - δ) · dcos(δ)
```

Let:
```
dJ = -(δ - π) · dcos(δ) / π    [note: code divides by π here]
```

### 4.4 Final Formula for ∂K/∂σ₀

```
K = X₁X₂ · J / π

∂K/∂σ₀ = [∂(X₁X₂)/∂σ₀ · J + X₁X₂ · ∂J/∂σ₀] / π
       = [dX₁X₂ · J + X₁X₂ · (π - δ) · dcos(δ)] / π
```

**Dividing by σ₀ to get the final form (as in code):**
```
∂K/∂σ₀ = (X₁X₂ · dJ + dX₁X₂ · J) / σ₀
```

Where:
```
dX₁X₂ = σ₀² · (X₂/X₁ + X₁/X₂)
dcos(δ) = (2σ₀² - cos(δ) · dX₁X₂) / (X₁X₂)
dJ = -(δ - π) · dcos(δ) / π
```

---

## 5. Gradients of K w.r.t. C-dependent Hyperparameters

For any hyperparameter θ that affects K only through C (i.e., θ ∈ {Amp, ε₀ₓ, ε₀ᵧ, θ_β, θ_ρ}):

```
∂K/∂θ = ∂K/∂C : ∂C/∂θ
```

Where `:` denotes the Frobenius inner product (element-wise multiply and sum).

### 5.1 Chain Rule Through C

We need ∂K/∂Cᵢⱼ. Since C appears in v₁, v₂, and c₁₂:

```
∂v₁/∂Cᵢⱼ = x₁[i] · x₁[j]     (element of outer product x₁x₁ᵀ)
∂v₂/∂Cᵢⱼ = x₂[i] · x₂[j]
∂c₁₂/∂Cᵢⱼ = x₁[i] · x₂[j]
```

### 5.2 Derivatives of X₁, X₂, X₁X₂ w.r.t. C

For the full dC matrix (not element-wise):
```
∂X₁/∂C = (1/2X₁) · ∂v₁/∂C = (1/2X₁) · x₁x₁ᵀ
```

In code, using dC[key] (a matrix):
```
dX₁[key] = (1/2X₁) · sum(x₁ ⊙ (dC[key] @ x₁))
         = 0.5 · sum(x₁ * matmul(dC[key], x₁), dim=0) / X₁
```

Similarly:
```
dX₂[key] = 0.5 · sum(x₂ * matmul(dC[key], x₂), dim=0) / X₂
```

And:
```
dX₁X₂[key] = dX₁[key] · X₂ + X₁ · dX₂[key]
```

### 5.3 Derivative of cos(δ) w.r.t. C

```
cos(δ) = c₁₂ / (X₁X₂)

∂cos(δ)/∂C = [∂c₁₂/∂C · X₁X₂ - c₁₂ · ∂(X₁X₂)/∂C] / (X₁X₂)²
```

In code:
```
dc₁₂[key] = x₁ᵀ @ dC[key] @ x₂

dcos(δ)[key] = (dc₁₂[key] - cos(δ) · dX₁X₂[key]) / (X₁X₂)
```

### 5.4 Derivative of J w.r.t. C

Same as before:
```
dJ[key] = -(δ - π) · dcos(δ)[key] / π
```

### 5.5 Final Formula for ∂K/∂θ (C-dependent)

```
∂K/∂θ = (dX₁X₂[θ] · J + X₁X₂ · dJ[θ]) / π
```

Or in the form used in code:
```
dK[θ] = dX₁X₂[θ] · J + X₁X₂ · dJ[θ]
```

(The 1/π factor is absorbed elsewhere or in final assembly.)

---

## 6. Matrix Form for Batch Computation

For matrices X₁ ∈ ℝⁿ¹ˣⁿˣ and X₂ ∈ ℝⁿ²ˣⁿˣ:

### 6.1 Forward Pass (Reference)

```python
# Transpose to (nx, n1) and (nx, n2) for original code compatibility
x1 = X1.T  # (nx, n1)
x2 = X2.T  # (nx, n2)

# Variances: V1[i] = x1[:,i]ᵀ C x1[:,i] + σ₀²
CX1 = C @ x1                           # (nx, n1)
V1 = (x1 * CX1).sum(dim=0) + σ₀²      # (n1,)
V2 = (x2 * (C @ x2)).sum(dim=0) + σ₀²  # (n2,)

X1_mag = sqrt(V1)  # (n1,)
X2_mag = sqrt(V2)  # (n2,)

# Magnitude matrix
X1X2 = outer(X1_mag, X2_mag)  # (n1, n2)

# Cross-term matrix
c12 = x1.T @ C @ x2 + σ₀²  # (n1, n2)

# Angle
cos_delta = clip(c12 / X1X2, -1, 1)  # (n1, n2)
delta = arccos(cos_delta)            # (n1, n2)
sin_delta = sqrt(1 - cos_delta²)     # (n1, n2)

# Angular term
J = (sin_delta + (π - delta) * cos_delta) / π  # (n1, n2)

# Kernel
K = X1X2 * J  # (n1, n2)
```

### 6.2 Gradient w.r.t. σ₀

```python
# dX1X2/dσ₀ (element-wise)
dX1X2_sigma = σ₀² * (X2_mag / X1_mag[:, None] + X1_mag[:, None] / X2_mag)  # (n1, n2)

# dcos/dσ₀
dcos_sigma = (2*σ₀² - cos_delta * dX1X2_sigma) / X1X2  # (n1, n2)

# dJ/dσ₀
dJ_sigma = -(delta - π) * dcos_sigma / π  # (n1, n2)

# Final
dK_sigma0 = (X1X2 * dJ_sigma + dX1X2_sigma * J) / σ₀  # (n1, n2)
```

### 6.3 Gradient w.r.t. C-dependent Parameter θ

```python
# Given: dC[θ] matrix of shape (nx, nx)

# dX1/dθ for each point
dX1_theta = 0.5 * (x1 * (dC[θ] @ x1)).sum(dim=0) / X1_mag  # (n1,)
dX2_theta = 0.5 * (x2 * (dC[θ] @ x2)).sum(dim=0) / X2_mag  # (n2,)

# dX1X2/dθ (broadcast to matrix)
dX1X2_theta = dX1_theta[:, None] * X2_mag + X1_mag[:, None] * dX2_theta  # (n1, n2)

# dc12/dθ
dc12_theta = x1.T @ dC[θ] @ x2  # (n1, n2)

# dcos/dθ
dcos_theta = (dc12_theta - cos_delta * dX1X2_theta) / X1X2  # (n1, n2)

# dJ/dθ
dJ_theta = -(delta - π) * dcos_theta / π  # (n1, n2)

# Final
dK[θ] = dX1X2_theta * J + X1X2 * dJ_theta  # (n1, n2)
```

---

## 7. Summary: Complete Gradient Computation

For each hyperparameter, the gradient ∂K/∂θ is a matrix of shape (n1, n2) matching K.

| Parameter | Formula |
|-----------|---------|
| σ₀ | `(X1X2 * dJ_sigma + dX1X2_sigma * J) / σ₀` |
| Amp | `K / Amp` (handled by ScaleKernel) |
| ε₀ₓ | `dX1X2_eps0x * J + X1X2 * dJ_eps0x` |
| ε₀ᵧ | `dX1X2_eps0y * J + X1X2 * dJ_eps0y` |
| θ_β | `dX1X2_beta * J + X1X2 * dJ_beta` |
| θ_ρ | `dX1X2_rho * J + X1X2 * dJ_rho` |

Where each `dX1X2_*` and `dJ_*` is computed using the chain rule through the corresponding `dC[*]` matrix from `C_gradients_hyp()`.

---

## 8. Implementation Notes

### 8.1 Numerical Stability

1. **Clamp cos(δ)** to [-1, 1] before arccos to avoid NaN
2. **Handle δ ≈ 0 or δ ≈ π** where sin(δ) → 0 (division issues in some formulas)
3. **Use float64** throughout (kernel values ~10,000 for PNAS data)

### 8.2 Code Structure

The implementation should:
1. Compute C and dC using existing `C_gradients_hyp()` or `LocalkerCleanFunction`
2. Compute K and intermediate quantities (X1, X2, cos_delta, delta, J)
3. Compute dK for each hyperparameter using formulas above
4. Return K and dict of dK matrices

### 8.3 Integration with GPyTorch

Option A: `torch.autograd.Function`
- Forward: return K
- Backward: given grad_output (∂L/∂K), return `(grad_output * dK[θ]).sum()` for each θ

Option B: Manual gradient in training loop
- Compute dK explicitly
- Chain with ∂L/∂K from likelihood
- Assign to `param.grad` directly (like original varGP)

---

*Document created: January 2025*
*Reference: utils.py:acosker() lines 3661-3741, kernels/kernels.py*
