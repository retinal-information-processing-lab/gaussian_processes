# Amp Parameter Analysis

**Created by Claude - Investigation into potential Amp parameter bug**
**Date**: January 2025

---

## 1. Mathematical Definition (from LaTeX)

From `acosker_kernel_def_and_gradients.tex`:

**Kernel:**
```
K(x, x') ∝ sqrt(x^T C x + σ₀²) · sqrt(x'^T C x' + σ₀²) · J(θ)

where θ = arccos((x^T C x' + σ₀²) / M)
      M = sqrt((x^T C x + σ₀²)(x'^T C x' + σ₀²))
      J(θ) = sin(θ) + (π - θ)cos(θ)
```

**Structured Covariance Matrix C (from LaTeX):**
```
C_ij = α_i^local · α_j^local · C_ij^smooth

where:
  α_i^local = exp(-||ξ_i - ξ_0||² / (4β²))
  C_ij^smooth = exp(-||ξ_i - ξ_j||² / (2ρ²))
```

**CRITICAL OBSERVATION**: The LaTeX definition does NOT include `Amp` anywhere. The Amp parameter is an **addition** made in the implementation.

---

## 2. Implementation Comparison

### 2.1 Production Code (utils.py)

**localker() at line 3612:**
```python
C = theta['Amp'] * alpha_local[:, None] * C_smooth * alpha_local[None, :]
```

**acosker() at lines 3702-3706:**
```python
X1 = torch.sqrt(torch.sum(x1*(C @ x1), dim=0) + sigma_0 ** 2)  # v_x = x^T C x + σ₀²
X2 = torch.sqrt(torch.sum(x2*(C @ x2), dim=0) + sigma_0 ** 2)  # v_x' = x'^T C x' + σ₀²
x1x2 = x1.T @ C @ x2 + sigma_0 ** 2  # c_xx' = x^T C x' + σ₀²
```

### 2.2 GPyTorch Implementation (kernels.py)

**_compute_C_matrix() at lines 403-405:**
```python
C = self.Amp * alpha[:, None] * C_smooth * alpha[None, :]
```

**forward() at lines 476-478:**
```python
CX1 = x1 @ C
V1 = (CX1 * x1).sum(dim=-1) + sigma_0_sq  # v_x = x^T C x + σ₀²
```

### 2.3 VJP Implementation (analytical_gradients_vjp.py)

**Lines 113-116:**
```python
C = Amp_val * alpha[:, None] * S * alpha[None, :]
```

**Lines 141-142:**
```python
V1 = torch.sum(x1_t * Cx1, dim=0) + sigma_0_sq  # v_x = x^T C x + σ₀²
```

### 2.4 Jacobian Implementation (analytical_gradients.py)

**compute_C_and_gradients() at line 324:**
```python
C = theta_Amp * alpha_local[:, None] * C_smooth * alpha_local[None, :]
```

**acosker_with_hyp_grad() at lines 164-165:**
```python
X1 = torch.sqrt(torch.sum(x1_t * (C @ x1_t), dim=0) + sigma_0 ** 2)
```

---

## 3. Key Mathematical Behavior

### 3.1 How Amp Affects the Kernel

When `C = Amp · C_base` (where C_base = α · C_smooth · α^T):

```
v_x = x^T (Amp · C_base) x + σ₀² = Amp · (x^T C_base x) + σ₀²

c_xx' = x^T (Amp · C_base) x' + σ₀² = Amp · (x^T C_base x') + σ₀²
```

**CRITICAL**: The `σ₀²` term is **NOT scaled by Amp**. This creates a **non-linear** interaction:

```
cos(θ) = (Amp · c + σ₀²) / sqrt((Amp · v_x + σ₀²)(Amp · v_x' + σ₀²))
```

This is fundamentally different from a linear output scaling like `ScaleKernel` would do.

### 3.2 Limiting Behavior

**When Amp → 0:**
- C → 0
- v_x → σ₀² (constant for all x)
- c_xx' → σ₀²
- cos(θ) → 1, θ → 0
- J(0) = π
- K → σ₀² · π/π = σ₀² (constant kernel)

**When Amp → ∞:**
- v_x ≈ Amp · (x^T C_base x), σ₀² negligible
- cos(θ) → (x^T C_base x') / sqrt((x^T C_base x)(x'^T C_base x'))
- The kernel converges to the σ₀²-free version

### 3.3 Semantic Interpretation

The current parameterization means:
- **Amp controls the strength of RF structure** relative to the constant σ₀² baseline
- When Amp=0, RF structure is completely ignored, kernel is constant σ₀²
- When Amp is large, σ₀² becomes negligible, pure RF-structured kernel

---

## 4. Gradient Verification

### 4.1 Reference Implementation (kernels/kernels.py:C_gradients_hyp)

```python
dC_Amp = C / theta_Amp
```

**Verification**: If C = Amp · C_base, then dC/dAmp = C_base = C/Amp ✓

### 4.2 VJP Implementation (analytical_gradients_vjp.py:323)

```python
grad_Amp = (dL_dC * C).sum() / Amp_val
```

**Verification**: dL/dAmp = Σᵢⱼ (dL/dC_ij) · (dC_ij/dAmp) = Σᵢⱼ (dL/dC_ij) · (C_ij/Amp) ✓

### 4.3 Jacobian Implementation (analytical_gradients.py)

Uses `C_gradients_hyp` which computes `dC_Amp = C / Amp`, then:
```python
dK[key] = X1X2 * dJ + dX1X2 * J
```

This matches the production acosker() gradient computation (utils.py:3747).

---

## 5. Consistency Check Summary

| Aspect | Production | GPyTorch | VJP | Jacobian | Status |
|--------|------------|----------|-----|----------|--------|
| C = Amp · α · S · α^T | ✓ | ✓ | ✓ | ✓ | Consistent |
| v_x = x^T C x + σ₀² | ✓ | ✓ | ✓ | ✓ | Consistent |
| σ₀² NOT scaled by Amp | ✓ | ✓ | ✓ | ✓ | Consistent |
| dC/dAmp = C/Amp | ✓ | (autograd) | ✓ | ✓ | Consistent |
| Diagonal K = v_x | ✓ | ✓ | ✓ | ✓ | Consistent |

---

## 6. Findings

### 6.1 No Implementation Bug Found

All four implementations (production, GPyTorch, VJP, Jacobian) are:
1. **Internally consistent** with each other
2. **Mathematically correct** for their definition of Amp
3. **Gradient computations are correct** for this parameterization

### 6.2 Potential Documentation Gap

The LaTeX document (`acosker_kernel_def_and_gradients.tex`) does NOT document the Amp parameter. This could cause confusion when comparing mathematical formulas to code.

### 6.3 Design Choice (Not a Bug)

The fact that `σ₀²` is independent of `Amp` is a **design choice**, not a bug. However, this creates semantics that may be non-intuitive:

- `Amp` does NOT simply scale the kernel output
- `Amp` scales the RF structure strength relative to the σ₀² baseline
- The effect is **non-linear** through sqrt and arccos operations

---

## 7. Questions for Further Investigation

1. **Was the Amp parameterization intentional?** Should σ₀² also be scaled by Amp?

2. **Is the current behavior desired?** When Amp→0, the kernel becomes constant σ₀². Is this expected?

3. **Should Amp be documented in the LaTeX?** The mathematical reference doesn't mention it at all.

4. **What is the typical learned Amp value?** If it's always ~1.0 or near bounds, the current parameterization might be suboptimal.

---

## 8. Alternative Parameterizations to Consider

### Option A: Current (Amp scales C only)
```
K ∝ sqrt(Amp·v + σ₀²) · sqrt(Amp·v' + σ₀²) · J(θ)
```
- Amp controls RF strength vs constant baseline
- Non-linear Amp dependence

### Option B: Amp as output scaling
```
K = Amp · (original kernel)
```
- Simple linear scaling
- Equivalent to ScaleKernel wrapper

### Option C: Amp scales everything including σ₀²
```
K ∝ sqrt(Amp·(v + σ₀²)) · sqrt(Amp·(v' + σ₀²)) · J(θ)
  = Amp · sqrt(v + σ₀²) · sqrt(v' + σ₀²) · J(θ)
```
- Equivalent to Option B (just linear output scaling)
- The σ₀² would need to be divided by Amp first, or Amp extracted from sqrt

The current Option A is mathematically valid but may not be what was intended.

---

## 9. Conclusion

**No bug found in the implementation.** All code is consistent and gradients are correct.

**However**, there may be a **conceptual question** about whether the current Amp parameterization is the intended behavior. The user should clarify:
- What is Amp supposed to represent semantically?
- Should σ₀² be affected by Amp or not?
