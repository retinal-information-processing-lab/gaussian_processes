# Mathematical Reference - GPyTorch Porting Project

This document contains the mathematical foundations for the variational GP implementation.
For project status and decisions, see `CLAUDE.md`.

---

## 1. Mathematical Foundation

### 1.1 The Model

**Observation model** (Poisson likelihood with exponential link):
```
r(x) ~ Poisson(f(x))
f(x) = exp(A·λ(x) + λ₀)
```
where:
- `r(x)` = observed spike count for stimulus x
- `f(x)` = firing rate (Poisson rate parameter)
- `λ(x)` = latent GP function
- `A` = gain parameter
- `λ₀` = bias parameter (baseline log-firing rate)

**Latent GP prior**:
```
λ ~ GP(0, K_θ)
```
where K_θ is the kernel with hyperparameters θ.

### 1.2 Variational Inference (Sparse GP)

Since the Poisson likelihood makes the posterior intractable, we use variational inference with inducing points.

**Inducing points**: A set of M << N pseudo-inputs {z̃₁, ..., z̃_M} with corresponding latent values λ̃ = {λ(z̃₁), ..., λ(z̃_M)}.

**Variational approximation**:
```
q(λ̃) = N(m, V)
```
where m ∈ ℝᴹ and V ∈ ℝᴹˣᴹ are variational parameters.

**Approximate posterior** for any point x:
```
q(λ(x)) = ∫ p(λ(x)|λ̃) q(λ̃) dλ̃

Mean:     μ(x) = k(x)ᵀ K̃⁻¹ m
Variance: σ²(x) = k(x,x) + k(x)ᵀ K̃⁻¹ (V - K̃) K̃⁻¹ k(x)
```
where:
- k(x) = [K(x, z̃₁), ..., K(x, z̃_M)]ᵀ  (cross-covariance vector)
- K̃ = K(Z̃, Z̃)  (inducing point kernel matrix)

### 1.3 Evidence Lower Bound (ELBO)

The objective to maximize:
```
L = E_q[log p(Y|λ)] - KL(q(λ̃) || p(λ̃))
```

**KL divergence term** (between two Gaussians):
```
-KL = ½ log|V| - ½ log|K̃| - ½ mᵀK̃⁻¹m - ½ Tr(K̃⁻¹V) + const
```

**Expected log-likelihood term** (for Poisson with exponential link):
```
E_q[log p(rᵢ|λᵢ)] = rᵢ(A·μᵢ + λ₀) - exp(A·μᵢ + ½A²σᵢ² + λ₀) + const
```

### 1.4 EM Algorithm

**E-step**: Update variational parameters (m, V) for fixed hyperparameters.

Newton update (closed-form when α=1):
```
g = A · (K·K̃⁻¹)ᵀ @ (r - f_mean)
G = A² · (K·K̃⁻¹)ᵀ @ diag(f_mean) @ (K·K̃⁻¹)

V_new = solve(I + K̃·G, K̃)
m_new = V_new @ (G·m + g)
```

**M-step**: Update kernel hyperparameters θ for fixed (m, V).
- Gradient-based optimization of ELBO w.r.t. θ

**F-step**: Update firing rate parameters (A, λ₀).
- Can be done with E-step (same loop) since no kernel recomputation needed

### 1.5 Arc-Cosine Kernel

Non-stationary kernel derived from infinite-width 2-layer ReLU network:
```
K(x, x') = (1/π) · M · J(θ)

where:
  v_x = xᵀCx + σ₀²
  v_x' = x'ᵀCx' + σ₀²
  M = √(v_x · v_x')
  cos(θ) = (xᵀCx' + σ₀²) / M
  J(θ) = sin(θ) + (π - θ)cos(θ)
```

**Structured covariance C** (encodes receptive field properties):
```
C_ij = α_i^local · α_j^local · C_ij^smooth

α_i^local = exp(-‖ξ_i - ξ₀‖² / 4β²)     [locality/RF size]
C_ij^smooth = exp(-‖ξ_i - ξ_j‖² / 2ρ²)  [smoothness]
```

Hyperparameters:
- ξ₀ = (eps_0x, eps_0y): RF center position
- β: RF size
- ρ: smoothness scale
- σ₀: bias variance
- Amp: amplitude

**Pixel coordinate grid** (from `localker_clean()`):
```python
# Normalized grid on [-1, 1] × [-1, 1]
ycord, xcord = torch.meshgrid(
    torch.linspace(-1, 1, n_px_side),  # n_px_side = 108 for PNAS
    torch.linspace(-1, 1, n_px_side),
    indexing='ij'
)
xcord = xcord.flatten()  # (n_px_side², ) = (11664,)
ycord = ycord.flatten()
```
- Center of image: (eps_0x, eps_0y) = (0, 0)
- Corners: (±1, ±1)

**Log-space parameterization** (for numerical stability):
```
Code parameter        →  Math symbol  →  Transform
-2log2beta            →  β            →  β = exp(-2log2beta) / 2
-log2rho2             →  ρ²           →  ρ² = exp(-log2rho2) / 2

Example: beta=0.1, rho=0.1
  -2log2beta = -2 * log(2 * 0.1) = -2 * log(0.2) ≈ 3.22
  -log2rho2  = -log(2 * 0.1²)    = -log(0.02)    ≈ 3.91
```

**Implementation details** (from `kernels/kernels.py:localker_clean()`):

1. **Mask computation with detached theta**:
   ```python
   # Mask computed with DETACHED hyperparameters
   # This keeps mask topology fixed during backprop (structural stability)
   dist_sq = (xcord - eps_0x.detach())**2 + (ycord - eps_0y.detach())**2
   alpha_for_mask = torch.exp(-beta.detach() * dist_sq)
   mask = alpha_for_mask >= 0.001  # Threshold: include if α ≥ 0.001
   ```
   Rationale: Prevents the set of active pixels from changing during optimization.

2. **C matrix assembly**:
   ```python
   # Locality weights (using NON-detached theta for gradients)
   dist_sq_center = (xcord - eps_0x)**2 + (ycord - eps_0y)**2
   logalpha = -beta * dist_sq_center
   alpha = torch.exp(logalpha)  # (n_px,)

   # Smoothness kernel (pairwise distances)
   dx = xcord[:, None] - xcord[None, :]
   dy = ycord[:, None] - ycord[None, :]
   dist_sq_pairwise = dx**2 + dy**2
   C_smooth = torch.exp(-rho2 * dist_sq_pairwise)  # (n_px, n_px)

   # Full C matrix
   C = alpha[:, None] * C_smooth * alpha[None, :]
   ```

3. **Symmetrization** (numerical stability):
   ```python
   C = (C + C.T) / 2  # Enforce exact symmetry
   ```

### 1.6 Eigenspace Projection (Numerical Stability)

The custom implementation projects all quantities into the eigenspace of K̃ for stability:
```
K̃ = B · Λ · Bᵀ  (eigendecomposition)
Keep only eigenvalues > threshold

Projected quantities:
  K̃_b = Λ_kept  (diagonal!)
  m_b = Bᵀ @ m
  V_b = Bᵀ @ V @ B
  K_b = K @ B
```

This avoids explicit K̃⁻¹ computation - uses element-wise division with diagonal K̃_b.

---

## 2. E-Step Formula Analysis

### 2.1 Correct Formulas (Rigorous Derivation)

**V Update** (closed form from gradient = 0):
```
V = K̃(K̃ + G)⁻¹K̃
```

**m Update** (Newton: m_new = m - H⁻¹∇L):
```
m_new = m + K̃(K̃ + G)⁻¹(g - m)
```

### 2.2 Code Implementation vs Correct Formulas

| Formula | Correct | Code | Status |
|---------|---------|------|--------|
| **V** | K̃(K̃+G)⁻¹K̃ | K̃(K̃+G)⁻¹K̃ | **CORRECT** |
| **m** | m + K̃(K̃+G)⁻¹(g-m) | K̃(K̃+G)⁻¹(GK̃⁻¹m+g) | Has discrepancy |

The code uses transformed g and G (pre-multiplied by K̃⁻¹), which changes the m formula. The discrepancy is:
```
m_code - m_correct = [K̃(K̃+G)⁻¹G K̃⁻¹ - G(K̃+G)⁻¹]m
```

### 2.3 Why Code Works Despite m Discrepancy

1. In eigenspace where K̃_b = Λ (diagonal), non-commutativity error is reduced
2. V is correct, which is crucial for variance estimates
3. Iterative convergence: multiple E-steps may still converge to good solution
4. Code symmetrizes V after each update

### 2.4 Implementation Guidelines

**Option 1 (direct):**
```python
V_new = K_tilde @ solve(K_tilde + G, K_tilde)
m_new = m + K_tilde @ solve(K_tilde + G, g - m)
```

**In eigenspace:**
```
V_b = Λ(Λ + G_b)⁻¹Λ
m_b = m_b + Λ(Λ + G_b)⁻¹(g_b - m_b)
```

---

## Related Documentation

- `ANALYTICAL_GRADIENTS_REFERENCE.md` - Kernel gradient formulas and VJP implementation
