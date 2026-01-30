# VJP-Based Analytical Gradients for Arc-Cosine Kernel

## Problem Statement

The current implementation materializes full Jacobian matrices:
- 5 dC matrices (each nx × nx ≈ 2500 × 2500)
- 5 dK matrices (each n1 × n2 ≈ 500 × 50)
- 15 large matrix multiplications per kernel call

**Goal**: Compute gradients using Vector-Jacobian Products (VJPs) without materializing intermediate Jacobians.

---

## Forward Pass (what we compute and save)

### Stage 1: C Matrix
```
Input: Amp, beta, rho, eps_0x, eps_0y, pixel coordinates (xcord, ycord)

dist_center[i] = (xcord[i] - eps_0x)² + (ycord[i] - eps_0y)²
logalpha[i] = -beta * dist_center[i]
alpha[i] = exp(logalpha[i])

dist_pairwise[i,j] = (xcord[i] - xcord[j])² + (ycord[i] - ycord[j])²
logS[i,j] = -rho² * dist_pairwise[i,j]
S[i,j] = exp(logS[i,j])                    # "C_smooth" in code

C[i,j] = Amp * alpha[i] * S[i,j] * alpha[j]
```

**Save**: alpha, S, logalpha, logS, dist_center, dist_pairwise

### Stage 2: K Matrix
```
Input: x1 (n1, nx), x2 (n2, nx), C (nx, nx), sigma_0

Note: code uses transposed convention x1_t = x1.T, so x1_t is (nx, n1)

V1[i] = x1[:,i]ᵀ C x1[:,i] + σ₀²
V2[j] = x2[:,j]ᵀ C x2[:,j] + σ₀²
X1[i] = √V1[i]
X2[j] = √V2[j]
X1X2[i,j] = X1[i] * X2[j]

x1Cx2[i,j] = x1[:,i]ᵀ C x2[:,j] + σ₀²

cosdelta[i,j] = x1Cx2[i,j] / X1X2[i,j]
delta[i,j] = arccos(cosdelta[i,j])
sindelta[i,j] = √(1 - cosdelta²[i,j])

J[i,j] = (sindelta[i,j] + (π - delta[i,j]) * cosdelta[i,j]) / π

K[i,j] = X1X2[i,j] * J[i,j]
```

**Save**: X1, X2, X1X2, cosdelta, delta, sindelta, J, V1, V2, x1Cx2
**Also need**: Cx1 = C @ x1_t, Cx2 = C @ x2_t (for efficient backward)

---

## Backward Pass (VJP Chain)

Given: G = dL/dK (shape n1 × n2)

### Step 1: dL/dK → dL/dJ, dL/dX1X2

```
K = X1X2 ⊙ J      (⊙ = element-wise multiply)

dL/dJ = G ⊙ X1X2
dL/dX1X2_from_K = G ⊙ J
```

### Step 2: dL/dJ → dL/dcosdelta

The J formula with cosdelta as the only free variable:
```
sindelta = √(1 - cosdelta²)
delta = arccos(cosdelta)
J = (sindelta + (π - delta) * cosdelta) / π

By chain rule:
dJ/dcosdelta = (1/π) * [d(sindelta)/dcosdelta + (π - delta) + cosdelta * d(π-delta)/dcosdelta]
             = (1/π) * [-cosdelta/sindelta + (π - delta) + cosdelta/sindelta]
             = (π - delta) / π

dL/dcosdelta = dL/dJ ⊙ (π - delta) / π
```

### Step 3: dL/dcosdelta → dL/dx1Cx2, additional dL/dX1X2

```
cosdelta = x1Cx2 / X1X2

dL/dx1Cx2 = dL/dcosdelta / X1X2
dL/dX1X2_from_cos = -dL/dcosdelta ⊙ cosdelta / X1X2

Total: dL/dX1X2 = dL/dX1X2_from_K + dL/dX1X2_from_cos
                = G ⊙ J - dL/dcosdelta ⊙ cosdelta / X1X2
```

### Step 4: dL/dX1X2 → dL/dX1, dL/dX2

```
X1X2[i,j] = X1[i] * X2[j]

dL/dX1[i] = Σⱼ dL/dX1X2[i,j] * X2[j]   = (dL/dX1X2 @ X2)
dL/dX2[j] = Σᵢ dL/dX1X2[i,j] * X1[i]   = (dL/dX1X2.T @ X1)
```

### Step 5: dL/dX1, dL/dX2 → dL/dV1, dL/dV2

```
X1 = √V1, so dX1/dV1 = 1/(2*X1)

dL/dV1 = dL/dX1 / (2 * X1)   (element-wise)
dL/dV2 = dL/dX2 / (2 * X2)
```

### Step 6: dL/dV1, dL/dV2, dL/dx1Cx2 → dL/dC

```
V1[i] = x1[:,i]ᵀ C x1[:,i] + σ₀²
V2[j] = x2[:,j]ᵀ C x2[:,j] + σ₀²
x1Cx2[i,j] = x1[:,i]ᵀ C x2[:,j] + σ₀²

For symmetric C, gradient of xᵀCx w.r.t. C is x⊗x (outer product).

dL/dC = Σᵢ dL/dV1[i] * x1[:,i] ⊗ x1[:,i]
      + Σⱼ dL/dV2[j] * x2[:,j] ⊗ x2[:,j]
      + Σᵢⱼ dL/dx1Cx2[i,j] * x1[:,i] ⊗ x2[:,j]

In matrix form:
dL/dC = x1_t @ diag(dL/dV1) @ x1_t.T
      + x2_t @ diag(dL/dV2) @ x2_t.T
      + x1_t @ dL/dx1Cx2 @ x2_t.T

Note: This is ONE (nx × nx) matrix, computed ONCE.
```

### Step 7: dL/dC → dL/d(hyperparameters)

Now we chain through the C structure. Recall:
```
C = Amp * alpha[:,None] * S * alpha[None,:]
```

Let `A = alpha[:,None] * alpha[None,:]` (outer product)

**dL/dAmp:**
```
dC/dAmp = A ⊙ S = C / Amp

dL/dAmp = Tr(dL/dC.T @ dC/dAmp)
        = Σᵢⱼ dL/dC[i,j] * C[i,j] / Amp
        = (dL/dC ⊙ C).sum() / Amp
```

**dL/dalpha** (intermediate):
```
C = Amp * A ⊙ S where A = alpha ⊗ alpha

dL/dA = dL/dC ⊙ Amp ⊙ S = dL/dC ⊙ C / A  (element-wise, where A > 0)
     or more simply: dL/dA = dL/dC ⊙ (Amp * S)

For A = alpha ⊗ alpha:
dL/dalpha = 2 * dL/dA @ alpha
          = 2 * (dL/dC ⊙ Amp ⊙ S) @ alpha
```

**dL/dbeta** (through alpha):
```
alpha[i] = exp(logalpha[i])
logalpha[i] = -beta * dist_center[i]

dalpha/dbeta = alpha * (-dist_center)

dL/dbeta = dL/dalpha · dalpha/dbeta
         = Σᵢ dL/dalpha[i] * alpha[i] * (-dist_center[i])
         = -(dL/dalpha ⊙ alpha ⊙ dist_center).sum()
```

**dL/deps_0x, dL/deps_0y** (through alpha):
```
dist_center[i] = (xcord[i] - eps_0x)² + (ycord[i] - eps_0y)²
d(dist_center)/d(eps_0x) = -2 * (xcord - eps_0x)

dlogalpha/deps_0x = -beta * d(dist_center)/d(eps_0x) = 2 * beta * (xcord - eps_0x)
dalpha/deps_0x = alpha * dlogalpha/deps_0x

dL/deps_0x = Σᵢ dL/dalpha[i] * alpha[i] * 2 * beta * (xcord[i] - eps_0x)
           = 2 * beta * (dL/dalpha ⊙ alpha ⊙ (xcord - eps_0x)).sum()
```

**dL/drho²** (through S):
```
S[i,j] = exp(logS[i,j])
logS[i,j] = -rho² * dist_pairwise[i,j]

dS/drho² = S ⊙ (-dist_pairwise)
dL/dS = dL/dC ⊙ Amp ⊙ A

dL/drho² = Tr(dL/dS.T @ dS/drho²)
         = -Σᵢⱼ dL/dS[i,j] * S[i,j] * dist_pairwise[i,j]
         = -(dL/dS ⊙ S ⊙ dist_pairwise).sum()
         = -(dL/dC ⊙ C ⊙ dist_pairwise).sum()   (since dL/dS ⊙ S = dL/dC ⊙ C / A ⊙ A = dL/dC ⊙ C)

Wait, let me redo this:
dL/dS = dL/dC ⊙ (Amp * A) = dL/dC ⊙ Amp ⊙ alpha[:,None] ⊙ alpha[None,:]

dL/drho² = -(dL/dS ⊙ S ⊙ dist_pairwise).sum()
```

**dL/dσ₀:**
```
σ₀ appears in V1, V2, x1Cx2:
V1[i] = ... + σ₀²
V2[j] = ... + σ₀²
x1Cx2[i,j] = ... + σ₀²

dV1/dσ₀ = 2σ₀
dV2/dσ₀ = 2σ₀
dx1Cx2/dσ₀ = 2σ₀

dL/dσ₀ = 2σ₀ * [dL/dV1.sum() + dL/dV2.sum() + dL/dx1Cx2.sum()]
```

---

## Complexity Analysis

### Current Implementation (Jacobian materialization)
- Compute 5 dC matrices: each involves element-wise ops on (nx, nx) → 5 * O(nx²)
- For each dK: 3 matrix multiplies involving (nx, nx) @ (nx, n) → 5 * 3 * O(nx² * n)
- **Total: O(15 * nx² * n)**

### VJP Implementation
- Backward through K: mostly element-wise ops on (n1, n2) → O(n1 * n2)
- Compute dL/dC: 3 terms, each O(nx² * n) → O(nx² * n)
- Chain to hyperparameters: element-wise ops on (nx, nx) or (nx,) → O(nx²)
- **Total: O(nx² * n) + O(nx²)**

### Speedup
VJP is approximately **15x faster** for the gradient computation (one O(nx² * n) instead of 15).

---

## Implementation Notes

1. **Forward pass**: Save all intermediates (X1, X2, cosdelta, J, alpha, S, dist_center, dist_pairwise, Cx1, Cx2)

2. **Backward pass**:
   - Compute scalar dL/dX1, dL/dX2, dL/dV1, dL/dV2, dL/dx1Cx2 first
   - Compute dL/dC matrix once
   - Extract all hyperparameter gradients via element-wise ops with dL/dC

3. **Memory tradeoff**: We save more intermediates in forward, but avoid storing 5 dC and 5 dK matrices.

4. **Numerical stability**: Same clipping as current implementation for arccos, etc.
