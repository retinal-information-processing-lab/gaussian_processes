# High-Dimensional GP Scripts - Claude Context

This folder contains scripts for active learning with high-dimensional image data using the custom variational GP implementation (not GPyTorch).

---

## High-Dimensional Implementation: `one_cell_active_training_distribution_aware.py`

**Location**: `Spatial_GP_repo/scripts/one_cell_active_training_distribution_aware.py`

This script implements active learning for real neural data (images → spike counts) using the distribution-aware utility. Unlike the 1D playground (GPyTorch), this uses the custom variational GP implementation in `utils.py`.

### Script Overview

1. **Load Data**: Natural images (108×108 = 11,664 pixels) and neural responses from PNAS dataset
2. **Initialize GP**: Load pre-trained model with 500 inducing points
3. **Active Learning Loop** (50 iterations):
   - Select best image using `batch_utility_w_grad` (standard utility)
   - Evaluate/compare with `optimize_with_conditioned_utility` (distribution-aware)
   - Add ORIGINAL image to training (not optimized version)
   - Refit GP model

### Key Functions

#### `conditioned_utility_clean()` (utility.py:1991)

Computes distribution-aware utility for a single query point x*.

**Inputs:**
- `x_star`: Query image being evaluated `(1, n_pixels)`
- `model`: GPModel with fitted parameters
- `remaining_imgs`: Pool of natural images to sample from `(n_remaining, n_pixels)`
- `N`: Number of natural images to sample for MC estimate
- `lambda_samples_per_x`: Number of λ samples per image

**Algorithm:**

```
1. MARGINAL ENTROPY H_marg at x*:
   ├── Compute GP moments: λ_m*, λ_var* = GP_marginal(x*)
   ├── Convert to log-firing: μ = A·λ_m* + λ₀, σ² = A²·λ_var*
   ├── Laplace approximation: p(r|x*,D) for r = 0,1,...,r_cutoff
   └── H_marg = -Σᵣ p(r) log p(r)

2. CONDITIONAL ENTROPY H_cond (MC estimate):
   FOR each sampled image x_i from remaining_imgs:
   │
   ├── 2a. Compute marginal moments at x_i:
   │       λ_m_i, λ_var_i = GP_marginal(x_i)
   │
   ├── 2b. Sample λ(x_i) from posterior:
   │       λ_i ~ N(λ_m_i, λ_var_i)
   │
   ├── 2c. Update inducing point posterior (m,V) → (m',V'):
   │       δ = λ_i - uᵀm                    (innovation)
   │       denom = s + uᵀVu                 (marginal variance at x_i)
   │       m' = m + (Vu/denom)·δ            (updated mean)
   │       V' = V - (Vu·uᵀV)/denom          (updated covariance)
   │
   ├── 2d. Conditional moments at x* using AUGMENTED SYSTEM:
   │       U_star = k_aug(x*)ᵀ @ K_aug⁻¹
   │       λ_m2 = U_starᵀ @ [λ_i; m']       (conditional mean)
   │       λ_var2 = S_* + U_star[1:]ᵀV'U_star[1:]  (conditional variance)
   │
   ├── 2e. Compute conditional entropy:
   │       μ_cond = A·λ_m2 + λ₀
   │       σ²_cond = A²·λ_var2
   │       H_cond_i = entropy(Laplace_approx(μ_cond, σ²_cond))
   │
   └── Average: H_cond = mean(H_cond_i)

3. UTILITY:
   U(x*) = H_marg - H_cond
```

### Eigenspace Projection (Critical Implementation Detail)

All matrices are projected into the eigenspace of K_tilde for numerical stability:

```python
# Eigendecomposition of K_tilde
eigvals, eigvecs = torch.linalg.eigh(K_tilde)
B = eigvecs[:, eigvals > threshold]  # Keep large eigenvalues only

# Projected quantities:
K_tilde_b = diag(eigvals[kept])       # DIAGONAL in eigenspace
K_tilde_inv_b = diag(1/eigvals[kept]) # DIAGONAL inverse
m_b = Bᵀ @ m                          # Projected mean (n_b,)
V_b = Bᵀ @ V @ B                      # Projected covariance (n_b, n_b) - NOT diagonal!
k_star_b = k(x*, x_tilde) @ B         # Projected kernel vector
u_b = k_b * K_tilde_inv_b.diag()      # Element-wise (K_tilde_b is diagonal!)
```

**Key insight**: Because `K_tilde_b` is diagonal, `u_b = K_b⁻¹ k_b` is computed via element-wise multiplication, NOT matrix solve.

### Augmented Matrix System

To capture exact correlations between x_i and x*, the implementation uses an augmented kernel matrix:

```
K_aug = [k(x_i,x_i)   k(x_i,x_tilde)]  = [s + uᵀKu    kᵀ   ]
        [k(x_tilde,x_i)  K_tilde    ]    [k           K_tilde]
```

The augmented projection vector for x* is:
```
k_aug(x*) = [k(x_i, x*); k(x*, x_tilde)@B]
U_star = k_aug(x*)ᵀ @ K_aug⁻¹
```

`compute_U_star_direct()` computes this efficiently using the block inverse formula without forming K_aug explicitly.

### `optimize_with_conditioned_utility()` (utility.py:2197)

Wraps `conditioned_utility_clean()` with Adam optimizer to maximize utility by modifying x*.

**Key behavior:**
- Computes gradients through the entire utility computation
- `compute_U_star_direct()` carefully detaches quantities independent of x* for correct gradient flow
- Supports sigmoid pixel constraints to keep optimized images in valid range

### λ_moments() Function (utils.py:3904)

Computes GP posterior moments at arbitrary points:

```python
# Inputs (in eigenspace):
#   K_tilde_b: (n_b, n_b) diagonal - inducing point kernel
#   k_b: (N, n_b) - cross-kernel to inducing points
#   k0: (N,) - self-kernel k(x,x)
#   m_b: (n_b,) - variational mean
#   V_b: (n_b, n_b) - variational covariance

# u = K_tilde_inv @ k (projection vector)
u_b = k_b * K_tilde_inv_b.diag()  # Element-wise because diagonal

# Mean: E[λ(x)] = uᵀm
lambda_m = u_b @ m_b

# Variance: Var[λ(x)] = k(x,x) - kᵀK⁻¹k + uᵀVu
#         = k0 - uᵀKu + uᵀVu = k0 + uᵀ(V-K)u
lambda_var = k0 + torch.sum(-k_b.T * u_b.T + u_b.T * (V_b @ u_b.T), dim=0)
```

### Code-to-Math Mapping

| Code Variable | Math Symbol | Shape | Description |
|--------------|-------------|-------|-------------|
| `m_b` | m | (n_b,) | Variational mean (projected) |
| `V_b` | V | (n_b, n_b) | Variational covariance (projected, NOT diagonal) |
| `K_tilde_b` | K | (n_b, n_b) | Inducing kernel (diagonal in eigenspace) |
| `K_tilde_inv_b` | K⁻¹ | (n_b, n_b) | Inverse (diagonal) |
| `k_star_b` | k(x*) | (1, n_b) | Cross-kernel x* to inducing points |
| `u_b` | u = K⁻¹k(x_i) | (N, n_b) | Projection vector for sampled images |
| `s` | s = k(x,x) - kᵀK⁻¹k | (N,) | Prior Schur complement |
| `denom` | s + uᵀVu | (N,) | Total marginal variance |
| `Vu` | V·u | (N, n_b) | Needed for m' update |
| `delta` | λ(x) - uᵀm | (N,) | Innovation |
| `m_prime_b` | m' | (N, n_b) | Updated mean per sample |
| `U_star` | k_augᵀK_aug⁻¹ | (N, n_b+1) | Augmented projection |
| `S_star` | s* | (N,) | Conditional prior variance at x* |
| `V_prime_term` | u*ᵀV'u* | (N,) | Expanded V' contribution |
| `lambda_m2` | μ_cond | (N,) | Conditional mean at x* |
| `lambda_var2` | σ²_cond | (N,) | Conditional variance at x* |

### Debug Modes

- `DEBUG_SAME_X`: Set x_i = x* (tests limit case where observation = query)
- `DEBUG_FIXED_X`: Use first N images instead of random sampling
- `DEBUG_FIX_LAMBDA_i`: Use mean instead of sampling λ_i
- `DEBUG_FULL_FIELD`: Use uniform gray/white images for debugging

### Important Notes

1. **p(x) is empirical**: Samples from `remaining_imgs` (dataset), not continuous distribution
2. **Training uses ORIGINAL images**: The optimization is just for evaluation/comparison
3. **Gradients disabled globally**: Script starts with `torch.set_grad_enabled(False)`, enable locally for optimization
4. **r_cutoff=100**: Truncates Poisson sum; increase if firing rates are high

---

## Mathematical Reference

See `scripts/1D_playground/.claude/CLAUDE.md` for the full mathematical derivation of:
- Distribution-aware utility formula
- Updated posterior formulas (m', V')
- Laplace approximation details
- Cross-covariance structure

Key LaTeX documents:
- `~/IDV_code/Papers/latex_summaries/distribution_aware_utility_pietro.tex`
- `~/IDV_code/Papers/latex_summaries/predictive_distribution_conditioned_on_observation.tex`
