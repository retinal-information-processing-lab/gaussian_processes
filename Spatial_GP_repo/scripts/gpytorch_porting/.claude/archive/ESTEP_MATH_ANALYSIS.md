# E-Step Mathematical Analysis

**Purpose**: Document the correct E-step formulas vs what's implemented in the code, for reference when implementing custom E-step in GPyTorch.

**Created**: January 2025
**Context**: Analysis of `utils.py:Estep()` against rigorous derivations in LaTeX documents.

---

## Reference Documents

| Document | Path | Status |
|----------|------|--------|
| Correct derivation | `~/IDV_code/Papers/latex_summaries/Estep_corrected_mderivation.tex` | Ground truth |
| Old (partially wrong) | `~/IDV_code/Papers/latex_summaries/Gaussian_process_theory.tex` | Has errors |
| Code | `Spatial_GP_repo/utils.py:Estep()` lines 4215-4276 | V correct, m has discrepancy |

---

## Notation

| Symbol | Meaning | Shape |
|--------|---------|-------|
| K̃ | Inducing point kernel matrix K(Z̃, Z̃) | (M, M) |
| K | Cross-kernel from training to inducing points | (N, M) |
| kᵢ | Cross-covariance vector for point i: K(Z̃, xᵢ) | (M,) |
| m | Variational mean | (M,) |
| V | Variational covariance | (M, M) |
| g | A ∑ᵢ kᵢ(yᵢ - f̄ᵢ) | (M,) |
| G | A² ∑ᵢ f̄ᵢ kᵢkᵢᵀ | (M, M) |
| f̄ᵢ | exp(Aμᵢ + ½A²σᵢ² + λ₀), expected firing rate | scalar |
| μᵢ | kᵢᵀ K̃⁻¹ m, posterior mean at point i | scalar |
| σᵢ² | kᵢᵢ + kᵢᵀ K̃⁻¹(V - K̃)K̃⁻¹kᵢ, posterior variance | scalar |
| Λ | Diagonal matrix of K̃ eigenvalues (in eigenspace) | (M, M) |
| B | Matrix of K̃ eigenvectors | (M, M) |

---

## CORRECT Formulas (Rigorous Derivation)

### Gradient and Hessian for m

```
∇_m L = K̃⁻¹(g - m)

H_m = -K̃⁻¹(K̃ + G)K̃⁻¹
```

### V Update (closed form from ∇_V L = 0)

Starting point:
```
∇_V L = ½V⁻¹ - ½K̃⁻¹ - ½K̃⁻¹GK̃⁻¹ = 0
```

Solving:
```
V⁻¹ = K̃⁻¹ + K̃⁻¹GK̃⁻¹ = K̃⁻¹(K̃ + G)K̃⁻¹
```

**CORRECT V formula:**
```
V = [K̃⁻¹(K̃ + G)K̃⁻¹]⁻¹ = K̃(K̃ + G)⁻¹K̃
```

This is a symmetric "sandwich" product: K̃ · (symmetric matrix) · K̃ = symmetric.

### m Update (Newton: m_new = m - H⁻¹∇L)

```
H⁻¹ = -K̃(K̃ + G)⁻¹K̃

m_new = m - [-K̃(K̃ + G)⁻¹K̃] · [K̃⁻¹(g - m)]
      = m + K̃(K̃ + G)⁻¹K̃ · K̃⁻¹(g - m)
      = m + K̃(K̃ + G)⁻¹(g - m)
```

**CORRECT m formula:**
```
m_new = m + K̃(K̃ + G)⁻¹(g - m)
```

**Alternate expanded form:**
```
m_new = [I - K̃(K̃+G)⁻¹]m + K̃(K̃+G)⁻¹g
      = G(K̃ + G)⁻¹m + K̃(K̃ + G)⁻¹g
```

(Using: I - K̃(K̃+G)⁻¹ = (K̃+G-K̃)(K̃+G)⁻¹ = G(K̃+G)⁻¹)

---

## OLD LaTeX Formulas (Gaussian_process_theory.tex) - CONTAINS ERRORS

### V formula (lines 695, 920) - WRONG
```
V = (K̃ + G)⁻¹K̃     ← INCORRECT
```

**Why wrong**: Product of two symmetric matrices is symmetric only if they commute. K̃ and (K̃+G)⁻¹ don't commute in general, so this produces a non-symmetric V.

### m formula (line 677) - WRONG
```
m = K̃(K̃ + G)⁻¹(GK̃⁻¹m + g)     ← INCORRECT
```

**Why wrong**: Line 673 correctly states `m + K̃(K̃+G)⁻¹(g-m)`, but the claim that this simplifies to the above formula (line 677) is algebraically incorrect.

---

## CODE Implementation (utils.py:Estep)

### Transformed g and G

The code uses pre-multiplied versions (lines 4234-4235):
```python
g_code = A * KKtilde_inv.T @ (r - f_mean)
G_code = A*A * KKtilde_inv.T @ (KKtilde_inv * f_mean[:,None])
```

where `KKtilde_inv = K @ K̃⁻¹`.

**Relationship to standard g, G:**
```
g_code = K̃⁻¹ g_standard
G_code = K̃⁻¹ G_standard K̃⁻¹
```

### V Update (code, line 4244)
```python
V_new = torch.linalg.solve(eye + K_tilde @ G, K_tilde)
```

**Derivation of what this computes:**
```
V_code = (I + K̃·G_code)⁻¹ K̃
       = (I + K̃·K̃⁻¹G K̃⁻¹)⁻¹ K̃
       = (I + G K̃⁻¹)⁻¹ K̃
       = K̃(K̃ + G)⁻¹ K̃          [using identity below]
```

**VERDICT: V_code = K̃(K̃ + G)⁻¹K̃ = V_correct ✓**

### m Update (code, line 4245)
```python
m_new = V_new @ (G @ m + g)
```

**Derivation of what this computes:**
```
m_code = V_code · (G_code·m + g_code)
       = K̃(K̃+G)⁻¹K̃ · (K̃⁻¹G K̃⁻¹m + K̃⁻¹g)
       = K̃(K̃+G)⁻¹K̃ · K̃⁻¹(G K̃⁻¹m + g)
       = K̃(K̃+G)⁻¹(G K̃⁻¹m + g)
```

**Expanded:**
```
m_code = K̃(K̃+G)⁻¹G K̃⁻¹m + K̃(K̃+G)⁻¹g
```

**Correct expanded:**
```
m_correct = G(K̃+G)⁻¹m + K̃(K̃+G)⁻¹g
```

**VERDICT: m_code ≠ m_correct (unless K̃ and G commute)**

---

## Summary Comparison Table

| Formula | Correct | Old LaTeX | Code | Code Status |
|---------|---------|-----------|------|-------------|
| **V** | K̃(K̃+G)⁻¹K̃ | (K̃+G)⁻¹K̃ | K̃(K̃+G)⁻¹K̃ | **CORRECT ✓** |
| **m** | m + K̃(K̃+G)⁻¹(g-m) | K̃(K̃+G)⁻¹(GK̃⁻¹m+g) | K̃(K̃+G)⁻¹(GK̃⁻¹m+g) | **MATCHES OLD (has discrepancy)** |

---

## Why Code Works Despite m Discrepancy

1. **Eigenspace representation**: In eigenspace where K̃_b = Λ (diagonal), the non-commutativity error is reduced

2. **The error term**:
   ```
   m_code - m_correct = [K̃(K̃+G)⁻¹G K̃⁻¹ - G(K̃+G)⁻¹]m
   ```
   When eigenvalues of K̃ are similar, this error is small.

3. **V is correct**: The covariance update is mathematically correct, which is crucial for variance estimates.

4. **Iterative convergence**: Multiple E-steps may still converge to a good solution.

5. **Symmetrization**: Code explicitly symmetrizes V (line 4253):
   ```python
   V_new = (V_new + V_new.T) / 2
   ```

---

## Key Mathematical Identity

```
(I + AB⁻¹)⁻¹ = B(B + A)⁻¹
```

**Proof:**
```
(I + AB⁻¹) · B(B+A)⁻¹ = B(B+A)⁻¹ + AB⁻¹B(B+A)⁻¹
                      = B(B+A)⁻¹ + A(B+A)⁻¹
                      = (B+A)(B+A)⁻¹
                      = I  ✓
```

---

## Recommendations for GPyTorch E-Step Implementation

### Use CORRECT formulas:

**Option 1 (direct):**
```python
# V update
V_new = K_tilde @ solve(K_tilde + G, K_tilde)

# m update
m_new = m + K_tilde @ solve(K_tilde + G, g - m)
```

**Option 2 (via V⁻¹):**
```python
# Compute V⁻¹ first
V_inv = solve(K_tilde, I) + solve(K_tilde, G @ solve(K_tilde, I))
V_new = solve(V_inv, I)

# m update
m_new = m + K_tilde @ solve(K_tilde + G, g - m)
```

### Always symmetrize:
```python
V_new = (V_new + V_new.T) / 2
```

### In eigenspace (if using projection):
```
V_b = Λ(Λ + G_b)⁻¹Λ
m_b = m_b + Λ(Λ + G_b)⁻¹(g_b - m_b)
```

where Λ is diagonal (eigenvalues), making solves efficient.

---

## Open Questions

1. **Should we fix the m update in existing code?** The current code works well empirically, so changing it could introduce regressions.

2. **Does the m discrepancy affect active learning?** The utility calculations depend on posterior variance, which uses V (correct). The mean m affects f_mean used in g and G for the next iteration.

3. **Quantify the error**: Could run experiments comparing m_code vs m_correct to measure practical impact.

---

*Last updated: January 2025*
