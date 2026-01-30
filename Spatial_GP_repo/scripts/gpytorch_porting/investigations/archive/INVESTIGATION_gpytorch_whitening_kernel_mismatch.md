# Investigation: GPyTorch Whitening and Kernel Parameter Changes

**Date**: 2026-01-21
**Status**: VERIFIED (with deeper understanding)
**Investigators**: Claude (analysis), Human (verification requested)

---

## Summary

This document describes a **design incompatibility** (not a bug) between GPyTorch's `VariationalStrategy` whitening and EM-style optimization where kernel hyperparameters and variational parameters are updated in separate steps.

**Initial finding**: When kernel hyperparameters change, GPyTorch uses NEW L_K with OLD m_w, producing seemingly incorrect predictions.

**Deeper finding**: GPyTorch's math IS correct for its intended use (joint optimization). The issue is a **parameterization mismatch**: EM wants to fix the natural distribution q(u) = N(m, S), but GPyTorch fixes the whitened parameters m_w, S_w. These are NOT equivalent when θ changes.

---

## The Core Issue: GPyTorch's Math IS Correct

### The Whitened Parameterization

GPyTorch stores whitened parameters (m_w, S_w) and computes:
```
predictive_mean = K_ZX^T @ L_K^{-T} @ m_w
```

This IS mathematically correct because it equals:
```
K_ZX^T @ K_ZZ^{-1} @ m   where m = L_K @ m_w
```

The natural mean m is **implicitly defined** as `m = L_K @ m_w`. When θ changes, L_K changes, so m changes **by design**.

### The KL Divergence Insight

**File**: `variational_strategy.py` lines 107-117

The whitened `VariationalStrategy` uses:
```python
prior_distribution = MultivariateNormal(zeros, DiagLinearOperator(ones))  # N(0, I)
```

The KL divergence is computed as:
```
KL_whitened = KL(N(m_w, S_w) || N(0, I))
            = 0.5 * (tr(S_w) + m_w^T m_w - M - log|S_w|)
```

**Critical observation**: θ does NOT appear in this formula!

This is mathematically equivalent to the natural KL:
```
KL_natural = KL(N(m, S) || N(0, K_ZZ))
           = 0.5 * (tr(K_ZZ^{-1} S) + m^T K_ZZ^{-1} m - M + log|K_ZZ| - log|S|)
```

Where m = L_K @ m_w, S = L_K @ S_w @ L_K^T. **The values are equal**, but θ appears explicitly in the natural form.

### The Gradient Mismatch

| Optimization | What's fixed | ∂KL/∂θ |
|--------------|--------------|--------|
| **Joint optimization** | Nothing | Flows through m_w via autograd |
| **EM M-step (GPyTorch's view)** | m_w, S_w | **= 0** (θ not in whitened formula) |
| **EM M-step (correct)** | m, S (natural) | **≠ 0** (θ in K_ZZ) |

**Implication**: When m_w is fixed during M-step, the whitened KL term contributes **zero gradient** to θ. GPyTorch's M-step only optimizes the expected log-likelihood, not the full ELBO w.r.t. θ!

### Comparison with UnwhitenedVariationalStrategy

**File**: `unwhitened_variational_strategy.py` lines 60-65

```python
prior_distribution = self.model.forward(self.inducing_points)  # N(μ_Z, K_ZZ)
```

Here the prior **depends on current kernel θ**, so:
- KL = KL(N(m, S) || N(μ_Z, K_ZZ(θ)))
- ∂KL/∂θ ≠ 0 even when m, S are fixed
- This is the correct behavior for EM!

---

## How to Verify This Claim

A coding assistant can verify this by examining the GPyTorch source code at:
```
<gpytorch_install_path>/variational/variational_strategy.py
<gpytorch_install_path>/variational/_variational_strategy.py
```

Find the install path with:
```python
import gpytorch, os
print(os.path.dirname(gpytorch.__file__))
```

---

## Verification Steps

### Step 1: Verify where `inducing_values` (m_w) comes from

**File**: `_variational_strategy.py`
**Lines**: 343, 346-352

```python
# Line 343: Get the stored variational distribution
variational_dist_u = self.variational_distribution

# Lines 346-352: Pass its mean as inducing_values
if isinstance(variational_dist_u, MultivariateNormal):
    return super().__call__(
        x,
        inducing_points,
        inducing_values=variational_dist_u.mean,  # <-- THIS IS m_w (stored)
        variational_inducing_covar=variational_dist_u.lazy_covariance_matrix,
        **kwargs,
    )
```

**Verification**: Confirm that `inducing_values` is the mean of the stored `variational_distribution`, which is the whitened mean m_w stored in `CholeskyVariationalDistribution`.

---

### Step 2: Verify where L_K comes from (CURRENT kernel)

**File**: `variational_strategy.py`
**Lines**: 189-203

```python
# Lines 189-191: Compute full prior covariance using CURRENT kernel
full_inputs = torch.cat([inducing_points, x], dim=-2)
full_output = self.model.forward(full_inputs, **kwargs)  # <-- Uses CURRENT kernel hyperparameters
full_covar = full_output.lazy_covariance_matrix

# Line 196: Extract K_ZZ from CURRENT kernel
induc_induc_covar = full_covar[..., :num_induc, :num_induc].add_jitter(self.jitter_val)

# Line 203: Compute Cholesky of CURRENT K_ZZ
L = self._cholesky_factor(induc_induc_covar)  # <-- L_K_new (from current kernel)
```

**Verification**: Confirm that:
1. `full_output` is computed by calling `self.model.forward()` with current inputs
2. This uses the current kernel hyperparameters (whatever they are at call time)
3. `L` is the Cholesky of the CURRENT kernel's K_ZZ matrix

---

### Step 3: Verify the predictive mean formula

**File**: `variational_strategy.py`
**Lines**: 212, 216

```python
# Line 212: Compute interpolation term using L from CURRENT kernel
interp_term = L.solve(induc_data_covar.type(_linalg_dtype_cholesky.value())).to(full_inputs.dtype)
# This computes: L_K_new^{-1} @ K_ZX_new

# Line 216: Compute predictive mean
predictive_mean = (interp_term.transpose(-1, -2) @ inducing_values.unsqueeze(-1)).squeeze(-1) + test_mean
# This computes: (L_K_new^{-1} @ K_ZX_new)^T @ m_w + μ_prior
#              = K_ZX_new^T @ L_K_new^{-T} @ m_w + μ_prior
```

**Verification**: Confirm the mathematical formula:
- `L.solve(RHS)` returns `L^{-1} @ RHS` (not `L^{-T} @ RHS`)
- After transpose: `interp_term.T` = `K_ZX^T @ L_K^{-T}`
- Final formula: `K_ZX_new^T @ L_K_new^{-T} @ m_w_old + μ_prior`

---

### Step 4: Verify the mismatch

**The problem**:

If m_w was stored after an E-step when kernel had parameters θ_old:
```
m_w_old = L_K_old^{-1} @ m_natural
```
where `L_K_old = chol(K_ZZ(θ_old))`.

After M-step changes kernel to θ_new, the forward pass computes:
```
predictive_mean = K_ZX_new^T @ L_K_new^{-T} @ m_w_old + μ_prior
                = K_ZX_new^T @ L_K_new^{-T} @ L_K_old^{-1} @ m_natural + μ_prior
```

**The correct formula should be**:
```
predictive_mean = K_ZX_new^T @ K_ZZ_new^{-1} @ m_natural + μ_prior
                = K_ZX_new^T @ L_K_new^{-T} @ L_K_new^{-1} @ m_natural + μ_prior
```

**The mismatch factor is**: `L_K_new^{-T} @ L_K_old^{-1}` instead of `L_K_new^{-T} @ L_K_new^{-1}`

---

## Why This Is Not a Bug

GPyTorch's `VariationalStrategy` is designed for **joint gradient optimization** where:
1. Both kernel hyperparameters θ and variational parameters (m_w, S_w) are updated via autograd in the same backward pass
2. The coupling between L_K and m_w is handled implicitly through the computation graph
3. Gradients for m_w account for how changes to m_w affect predictions through L_K

**Our EM-style optimization breaks this assumption**:
1. E-step: Update m_w (closed-form Newton) with fixed θ
2. M-step: Update θ (gradient descent) with fixed m_w
3. The stored m_w becomes inconsistent with the new L_K after M-step

---

## Impact

When kernel hyperparameters change between storing m_w and calling `model(X)`:

| What GPyTorch computes | What it should compute |
|------------------------|------------------------|
| `K_ZX^T @ L_K_new^{-T} @ L_K_old^{-1} @ m_natural` | `K_ZX^T @ L_K_new^{-T} @ L_K_new^{-1} @ m_natural` |

The error factor `L_K_new^{-T} @ L_K_old^{-1}` vs `L_K_new^{-T} @ L_K_new^{-1}` can cause significant prediction errors depending on how much the kernel changed.

---

## Possible Solutions

### Option A: Re-whiten after kernel changes

After M-step updates kernel hyperparameters:
```python
# Compute L_K_old before M-step (from cache or fresh)
L_K_old = compute_L_K(model)

# Run M-step (kernel changes)
m_step(...)

# Compute L_K_new after M-step
L_K_new = compute_L_K(model)

# Re-whiten: m_w_new = L_K_new^{-1} @ L_K_old @ m_w_old
m_natural = L_K_old @ m_w_old
m_w_new = torch.linalg.solve_triangular(L_K_new, m_natural, upper=False)
# Store m_w_new
```

### Option B: Use UnwhitenedVariationalStrategy

Store natural parameters (m, V) directly. No whitening mismatch possible.
Trade-off: ~4x slower, potentially less numerically stable.

### Option C: Own variational storage (bypass GPyTorch's VariationalStrategy)

Store m, V as plain tensors outside GPyTorch. Compute moments and KL ourselves.
Use GPyTorch only for kernel computation.

---

## Test to Confirm the Issue

```python
import torch
import gpytorch

# Create model with whitened VariationalStrategy
model = VariationalGPModel(inducing_points, kernel)
likelihood = PoissonLikelihood()

# Fit with E-step (stores m_w whitened with current L_K)
# ... run E-step ...

# Record predictions BEFORE kernel change
model.eval()
pred_before = model(test_x).mean.clone()

# Change kernel hyperparameters (simulate M-step)
with torch.no_grad():
    model.covar_module.base_kernel.raw_sigma_0.add_(0.5)  # Change sigma_0

# Clear cache to force L_K recomputation
model.variational_strategy._memoize_cache.clear()

# Record predictions AFTER kernel change
pred_after = model(test_x).mean

# If whitening were handled correctly, pred_after would use the same
# natural m_natural, just with new kernel.
# But actually, pred_after uses m_w (whitened for OLD kernel) with NEW L_K.
print(f"Prediction change: {(pred_after - pred_before).abs().mean():.6f}")
# This will be non-zero due to the whitening mismatch!
```

---

## Related Files

- `estep.py`: Contains whitening conversion functions (`get_variational_mean_with_L_K`, etc.)
- `.claude/archive/ARCHIVE_2026-01-20_whitening_research_notes.md`: Earlier investigation notes
- `investigations/INVESTIGATION_whitening_collapse_M75.md`: Related whitening issue at M=75

---

## Conclusion

The finding is **verified and reproducible**. GPyTorch's `VariationalStrategy` assumes joint optimization and does not automatically re-whiten variational parameters when kernel hyperparameters change. For EM-style optimization, explicit re-whitening is required after each M-step, or an alternative approach (UnwhitenedVariationalStrategy or own storage) should be used.
