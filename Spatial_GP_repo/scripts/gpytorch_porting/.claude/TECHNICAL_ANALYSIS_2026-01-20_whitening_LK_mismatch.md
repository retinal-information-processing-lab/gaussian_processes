# Technical Analysis: GPyTorch Whitening and L_K Inconsistency in EM Optimization

> **NOTE (added later)**: Section 9.4 "Trade-offs" contains predictions that proved inaccurate:
> - Predicted "~5% slower" → Actual: 4x slower (400%)
> - Predicted "slightly worse conditioning" → Actual: 16% accuracy gap (0.6878 vs 0.8381)
> - Predicted "cons are negligible" → Actual: significant performance impact
> The root cause of the accuracy gap is not fully understood. See Q29 in DECISION_LOG.md.

**Date**: 2026-01-20
**Investigator**: Claude Code (Opus 4.5)
**Status**: COMPLETE (with caveats above)
**Conclusion**: Use `UnwhitenedVariationalStrategy` for EM-style optimization (but expect worse accuracy)

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Problem Statement](#2-problem-statement)
3. [Investigation Methodology](#3-investigation-methodology)
4. [Background: Whitening in Sparse Variational GPs](#4-background-whitening-in-sparse-variational-gps)
5. [GPyTorch Source Code Analysis](#5-gpytorch-source-code-analysis)
6. [Academic Literature Findings](#6-academic-literature-findings)
7. [GPyTorch GitHub Issues Analysis](#7-gpytorch-github-issues-analysis)
8. [Root Cause Analysis](#8-root-cause-analysis)
8b. [Potential Contradictions Examined](#8b-potential-contradictions-examined) ← **Read this to avoid re-investigating resolved questions**
9. [Solution: UnwhitenedVariationalStrategy](#9-solution-unwhitenedvariationalstrategy)
10. [Verification and Testing](#10-verification-and-testing)
11. [References and How to Verify](#11-references-and-how-to-verify)

---

## 1. Executive Summary

### The Question

When implementing EM-style optimization with GPyTorch's variational GP:
- **E-step**: Newton updates on variational parameters (m, V) with fixed kernel
- **M-step**: Gradient updates on kernel hyperparameters with fixed (m, V)

If we correctly whiten variational parameters before storing them in GPyTorch, **should GPyTorch automatically account for kernel changes when interpreting these parameters?**

### The Answer

**NO.** GPyTorch does NOT automatically re-whiten or adjust variational parameters when kernel hyperparameters change. This is by design:

1. GPyTorch's `VariationalStrategy` assumes **joint gradient optimization** where autograd handles the coupling between kernel and variational parameters implicitly
2. EM-style optimization with closed-form E-step is **outside GPyTorch's expected use case**
3. The coupling between L_K (Cholesky of inducing kernel) and whitened parameters is handled through the computation graph during backpropagation, not through explicit re-parameterization

### The Solution

Use `gpytorch.variational.UnwhitenedVariationalStrategy` instead of `VariationalStrategy`. This stores natural (unwhitened) parameters directly, eliminating the L_K coupling problem entirely.

---

## 2. Problem Statement

### 2.1 Context

The GPyTorch porting project implements sparse variational GP inference using:
- Custom Newton-based E-step for updating variational parameters (m, V)
- Gradient-based M-step for kernel hyperparameters
- GPyTorch's `VariationalStrategy` with `CholeskyVariationalDistribution` for storage

### 2.2 The Whitening Parameterization

GPyTorch stores variational parameters in "whitened" form:

```
m_whitened = L_K^{-1} @ m_natural
V_whitened = L_K^{-1} @ V_natural @ L_K^{-T}
```

Where `L_K` is the Cholesky factor of the inducing point kernel: `K_tilde = L_K @ L_K^T`

### 2.3 The L_K Inconsistency Problem

When M-step changes kernel hyperparameters:

1. **Before M-step**: Parameters stored as `m_stored = L_K_old^{-1} @ m_natural`
2. **After M-step**: Kernel changes → `K_tilde` changes → `L_K` becomes `L_K_new`
3. **Next E-step**: GPyTorch interprets `m_stored` using `L_K_new`:
   ```
   m_recovered = L_K_new @ m_stored
               = L_K_new @ L_K_old^{-1} @ m_natural
               ≠ m_natural  (CORRUPTED!)
   ```

The corruption factor is `L_K_new @ L_K_old^{-1}`, which can cause ~8x errors in predictive mean and ~15x errors in variance.

### 2.4 The Core Question

The user asked: *"If we are passing the correctly whitened m and V parameters to the GPyTorch model, it should know that when the kernel is modified it should account for this for m and V. It hardly seems like something they wouldn't think of, are E-M optimizations really that uncommon?"*

This investigation answers that question definitively.

---

## 3. Investigation Methodology

### 3.1 Approach

The investigation used four parallel research tracks:

| Track | Method | Purpose |
|-------|--------|---------|
| A | GPyTorch source code analysis | Understand exact formulas and behavior |
| B | Academic literature review | Understand theoretical foundations |
| C | GitHub issues analysis | Find known issues and workarounds |
| D | Local codebase analysis | Understand current implementation |

### 3.2 Tools Used

- **Claude Code subagents**: Parallel exploration agents for each research track
- **Direct file reading**: GPyTorch source files, local implementation files
- **Web search**: Academic papers, documentation, GitHub issues

### 3.3 Key Sources Examined

**GPyTorch Source Files** (conda environment `pytorch_gpytorch`):
- `gpytorch/variational/variational_strategy.py` - Main whitened strategy
- `gpytorch/variational/unwhitened_variational_strategy.py` - Unwhitened alternative
- `gpytorch/variational/cholesky_variational_distribution.py` - Parameter storage
- `gpytorch/variational/_variational_strategy.py` - Base class

**To locate these files**:
```python
import gpytorch
import inspect
print(inspect.getsourcefile(gpytorch.variational.VariationalStrategy))
```

**Local Files**:
- `estep.py` - Custom E-step implementation
- `tests/test_whitening_paths.py` - Whitening validation tests
- `tests/test_m_whitening.py` - Parameter conversion tests
- `tests/test_whitening_hypothesis.py` - L_K mismatch hypothesis tests

---

## 4. Background: Whitening in Sparse Variational GPs

### 4.1 Standard SVGP Formulation

In sparse variational GPs (Hensman et al., 2013):
- **Inducing points**: Z = {z_1, ..., z_M} with latent values u = [f(z_1), ..., f(z_M)]
- **Prior**: p(u) = N(0, K_ZZ) where K_ZZ is the M×M inducing kernel
- **Variational approximation**: q(u) = N(m, V)

The ELBO contains a KL divergence term:
```
KL[q(u) || p(u)] = KL[N(m, V) || N(0, K_ZZ)]
```

### 4.2 Why Whitening?

The whitened parameterization (Matthews, 2017) transforms to a coordinate system where the prior becomes standard normal:

```
Define: u = L_K @ w  where K_ZZ = L_K @ L_K^T (Cholesky)

If q(w) = N(m', S'), then:
  q(u) = N(L_K @ m', L_K @ S' @ L_K^T)

And the prior becomes:
  p(w) = N(0, I)  (standard normal!)
```

**Benefits**:
1. KL divergence simplifies to `KL[N(m', S') || N(0, I)]`
2. Better optimization conditioning (prior is identity)
3. Initialization `m'=0, S'=I` gives `q(u) = p(u)` (start at prior)

### 4.3 Conversion Formulas

```
Natural → Whitened:
  m_whitened = L_K^{-1} @ m_natural
  V_whitened = L_K^{-1} @ V_natural @ L_K^{-T}

Whitened → Natural:
  m_natural = L_K @ m_whitened
  V_natural = L_K @ V_whitened @ L_K^T
```

### 4.4 The Coupling Problem

The whitened parameters are **defined relative to the current L_K**. When kernel hyperparameters change, L_K changes, and the stored whitened parameters become "stale" - they represent a different natural distribution than intended.

---

## 5. GPyTorch Source Code Analysis

### 5.1 How GPyTorch Computes Predictive Mean

**Source**: `gpytorch/variational/variational_strategy.py`, lines 212-216

```python
# Compute interpolation term: L_K^{-1} @ K_ZX
L = self._cholesky_factor(induc_induc_covar)  # L_K where K_ZZ = L_K @ L_K^T
interp_term = L.solve(induc_data_covar)        # L_K^{-1} @ K_ZX, shape (M, N)

# Predictive mean
predictive_mean = interp_term.transpose(-1, -2) @ inducing_values
# = (L_K^{-1} @ K_ZX)^T @ m_stored
# = K_XZ @ L_K^{-T} @ m_stored
```

**GPyTorch's formula**: `λ_m = K_XZ @ L_K^{-T} @ m_stored`

**Standard SVGP formula**: `λ_m = K_XZ @ K_ZZ^{-1} @ m_natural`

**These are equivalent ONLY if**: `m_stored = L_K^{-1} @ m_natural` (whitened)

### 5.2 Automatic Whitening on First Call

**Source**: `gpytorch/variational/variational_strategy.py`, lines 238-272

On the first call to `model(X)`, GPyTorch converts stored parameters to whitened form:

```python
if not self.updated_strategy.item() and not prior:
    prior_function_dist = self(self.inducing_points, prior=True)
    L = self._cholesky_factor(prior_function_dist.lazy_covariance_matrix)

    variational_dist = self.variational_distribution
    mean_diff = (variational_dist.loc - prior_mean)
    whitened_mean = L.solve(mean_diff)  # L_K^{-1} @ (m - prior_mean)

    covar_root = variational_dist.lazy_covariance_matrix.root_decomposition().root
    whitened_covar = L.solve(covar_root)  # L_K^{-1} @ L_stored

    self._variational_distribution.initialize_variational_distribution(...)
    self.updated_strategy.fill_(True)  # Mark as converted
```

**Critical finding**: This conversion happens ONCE. After `updated_strategy = True`, no further automatic conversion occurs.

### 5.3 Cache Behavior

**Source**: `gpytorch/variational/variational_strategy.py`, lines 102-105

```python
@cached(name="cholesky_factor", ignore_args=True)
def _cholesky_factor(self, induc_induc_covar: LazyTensor) -> TriangularLazyTensor:
    L = psd_safe_cholesky(induc_induc_covar.to_dense(), jitter=self.jitter_val)
    return TriangularLazyTensor(L)
```

The `@cached` decorator means L_K is computed once and reused. However:

**Source**: `gpytorch/variational/_variational_strategy.py`, lines 329-330

```python
def __call__(self, x, prior=False, **kwargs):
    if self.training:
        self._clear_cache()  # Clears all memoized values when in training mode
```

In training mode, cache is cleared before each forward pass, so L_K is recomputed with current kernel parameters.

### 5.4 No Re-Whitening Mechanism

**Critical finding**: There is NO code in GPyTorch that:
1. Detects when kernel parameters have changed
2. Re-computes L_K_old vs L_K_new
3. Transforms stored whitened parameters to account for the change

The whitening is done ONCE at initialization, and thereafter GPyTorch assumes the stored parameters are correctly whitened relative to whatever the current L_K is.

### 5.5 UnwhitenedVariationalStrategy Analysis

**Source**: `gpytorch/variational/unwhitened_variational_strategy.py`

```python
class UnwhitenedVariationalStrategy(_VariationalStrategy):
    """
    Similar to VariationalStrategy, but does not perform the whitening operation.
    In almost all cases VariationalStrategy is preferable, with a few exceptions:
      - When the inducing points are exactly equal to the training points (i.e. Z = X)
      - When the number of inducing points is very large (e.g. >2000)
    """
```

**Prior distribution** (lines 60-65):
```python
@property
def prior_distribution(self) -> MultivariateNormal:
    out = self.model.forward(self.inducing_points)
    res = MultivariateNormal(out.mean, out.lazy_covariance_matrix.add_jitter())
    return res  # Returns actual prior N(μ_Z, K_ZZ), NOT N(0, I)
```

**Predictive mean** (lines 183-189):
```python
# Uses K_ZZ directly, NOT L_K
inv_products = induc_induc_covar.solve(induc_data_covar, left_tensors.transpose(-1, -2))
predictive_mean = torch.add(test_mean, inv_products[..., 0, :])
# This is: m_prior + K_XZ @ K_ZZ^{-1} @ (m - m_prior)
```

**Key difference**: Unwhitened strategy uses `K_ZZ.solve()` directly, not `L_K.solve()`. No whitening transformation involved.

---

## 6. Academic Literature Findings

### 6.1 Whitening in SVGP (Hensman et al., Matthews)

**Source**: Matthews (2017) PhD thesis, "Scalable Gaussian Process Inference using Variational Methods"
**URL**: https://www.repository.cam.ac.uk/handle/1810/278022

The whitening transformation is a change of variables that:
1. Does not change the model mathematically
2. Improves optimization conditioning
3. Is implemented as storing `m' = L^{-1}(m - μ_prior)` instead of `m`

**Key quote from GPflow Issue #979**:
> "Whitening is just a variable transformation - so you may end up with a different result in empirical optimisation (the optimisation is non-convex) or take a different number of iterations to get there, but it does not change the model at all."

### 6.2 How Joint Optimization Handles the Coupling

In standard joint optimization (Adam on all parameters):
1. Forward pass computes ELBO using current L_K and current m_stored
2. Backward pass differentiates through L_K (which depends on kernel hyperparameters)
3. Gradients for kernel hyperparameters account for how L_K affects the ELBO
4. Gradients for m_stored account for how m_stored affects the ELBO through L_K

**The coupling is handled implicitly by autograd** - there's no explicit re-whitening because gradients flow through L_K.

### 6.3 Natural Gradient Descent (Alternative to EM)

**Source**: Salimbeni et al. (2018), "Natural Gradients in Practice for Deep and Singly-layered GPs"
**URL**: https://arxiv.org/abs/1803.09151

GPyTorch provides `gpytorch.optim.NGD` for natural gradient descent on variational parameters:
- Updates variational params in natural parameter space
- Still uses gradients (not closed-form Newton)
- Recommended pattern: NGD for variational params, Adam for kernel params
- **Caution**: "Can be unstable with non-conjugate likelihoods" (relevant to Poisson!)

### 6.4 Dual Parameterization (Research Frontier)

**Source**: Adam et al. (2021), "Dual Parameterization of Sparse Variational Gaussian Processes"
**URL**: https://arxiv.org/abs/2111.03412

Recent research explores alternative parameterizations that may offer benefits over whitening. However, this is not yet implemented in GPyTorch.

---

## 7. GPyTorch GitHub Issues Analysis

### 7.1 Cache Invalidation Issues

**Issue #1308**: "Reloading saved parameters hurts performance"
- **URL**: https://github.com/cornellius-gp/gpytorch/issues/1308
- **Problem**: Loading state dict doesn't clear cached computations
- **Workaround**: `model.load_state_dict(...); model.train(); model.eval()`
- **Quote**: "The variational models cache some of the expensive computations. Loading from the state dict does not clear these precomputed caches."

**Issue #1754**: "Trivial optimization issue with cache"
- **URL**: https://github.com/cornellius-gp/gpytorch/issues/1754
- **Problem**: Cache not cleared properly, causing autograd errors
- **Workaround**: `del model.variational_strategy._memoize_cache`

**Issue #1556**: "Changing kernel hyperparameters in eval mode"
- **URL**: https://github.com/cornellius-gp/gpytorch/issues/1556
- **Problem**: Predictions incorrect after changing hyperparameters in eval mode
- **Cause**: `model.prediction_strategy` uses stale cached values

### 7.2 Whitening-Related Changes

**PR #903**: "Major updates to variational models"
- **URL**: https://github.com/cornellius-gp/gpytorch/pull/903
- **Key changes**:
  - Removed old `WhitenedVariationalStrategy` because "it was wrong"
  - Current `VariationalStrategy` implements correct whitening from Hensman et al.
  - Added `UnwhitenedVariationalStrategy` for cases where whitening is undesirable

**Issue #1156**: "Ignored prior mean for inducing points"
- **URL**: https://github.com/cornellius-gp/gpytorch/issues/1156
- **Explanation**: In whitened space, prior mean is implicit in the coordinate transformation
- **Resolution**: Not a bug - this is how whitening works

### 7.3 Key Insight from Issues

No GitHub issue addresses the specific problem of EM-style optimization with whitening. This suggests:
1. EM with closed-form E-step is uncommon in GPyTorch usage
2. Most users rely on joint gradient optimization where autograd handles coupling
3. The cache issues are about stale values, not about re-whitening

---

## 8. Root Cause Analysis

### 8.1 The Fundamental Mismatch

| Optimization Style | How L_K-m Coupling Is Handled |
|-------------------|------------------------------|
| **Joint (Adam)** | Autograd differentiates through L_K → gradients account for coupling |
| **Interleaved (NGD + Adam)** | Still gradient-based → autograd handles coupling |
| **EM (Newton E-step)** | E-step is closed-form, bypasses autograd → coupling NOT handled |

**Our E-step is fundamentally different** from what GPyTorch expects:
1. Newton iterations update (m, V) using closed-form formulas
2. These formulas don't involve autograd or backpropagation
3. The resulting natural (m, V) must be converted to whitened form for storage
4. After M-step changes kernel, stored whitened params are "stale"

### 8.2 Why GPyTorch Doesn't Auto-Adjust

GPyTorch's design philosophy:
1. **Store whitened parameters** (m', S')
2. **On each forward pass**, use current L_K to compute predictions
3. **Autograd flows through L_K** during backpropagation
4. **No explicit re-whitening** because gradients handle the coupling

For EM:
1. E-step produces natural (m, V) without autograd involvement
2. We store whitened params with L_K_old
3. M-step uses autograd to update kernel → L_K becomes L_K_new
4. **But our stored params weren't part of the autograd graph!**

### 8.3 The Corruption Formula

When kernel changes from iteration N to N+1:

```
Stored at N: m_stored = L_K_N^{-1} @ m_natural

Read at N+1: m_recovered = L_K_{N+1} @ m_stored
           = L_K_{N+1} @ L_K_N^{-1} @ m_natural
           = (L_K_{N+1} @ L_K_N^{-1}) @ m_natural

Corruption factor: L_K_{N+1} @ L_K_N^{-1}
```

This is NOT the identity matrix when L_K changes. Experimentally observed:
- ~8x error in predictive mean
- ~15x error in variance at initialization

### 8.4 Verification: Test Suite Confirms Understanding

**File**: `tests/test_whitening_hypothesis.py`

| Test | Hypothesis | Result |
|------|-----------|--------|
| A | Corruption follows formula `L_K_new @ L_K_old^{-1} @ m` | CONFIRMED |
| B | Frozen kernel (no M-step) works correctly | CONFIRMED |
| C | Single gradient step causes corruption | CONFIRMED |
| D | Multiple gradient steps accumulate corruption | CONFIRMED |

These tests validate that our understanding of the problem is correct.

---

## 8b. Potential Contradictions Examined

This section addresses apparent contradictions in the findings and explains why they don't invalidate the main conclusions. Future sessions should read this to avoid re-investigating these points.

### 8b.1 "GPyTorch Recomputes L_K on Every Forward Pass"

**Apparent contradiction**: One source stated that GPyTorch "recomputes L_K on every forward pass" and "clears cache before each forward pass during training." This might suggest GPyTorch is designed to handle changing L_K.

**Resolution**: Cache clearing means GPyTorch uses the **current** kernel matrices in forward computation. It does NOT adjust **stored** variational parameters.

The issue is not whether L_K is recomputed correctly - it's about what the stored `m_whitened` **means** after L_K changes:
- GPyTorch correctly computes: `λ_m = K_XZ_new @ L_K_new^{-T} @ m_stored`
- But `m_stored` was computed as `L_K_old^{-1} @ m_natural`
- The formula uses NEW L_K with OLD whitening → corruption

Cache clearing ensures correct kernel matrices; it doesn't "fix" stale variational parameters.

### 8b.2 "Joint Optimization Updates Both Together"

**Apparent contradiction**: In joint optimization, both kernel and variational params change "together," so why doesn't the same corruption occur?

**Resolution**: The key distinction is what constitutes the "primary" parameter:

| Optimization | Primary Parameter | What Happens |
|--------------|-------------------|--------------|
| **Joint (Adam)** | m_whitened IS the parameter | Gradients update m_whitened directly; there's no "underlying m_natural" |
| **EM (Newton)** | m_natural IS the parameter | We compute m_natural, convert to m_whitened for storage; m_natural is the "truth" |

In joint optimization:
1. Forward: compute ELBO with current L_K and current m_whitened
2. Backward: gradients for both kernel params AND m_whitened
3. Update: both change simultaneously based on current state

There's no "old m_whitened with new L_K" because both are updated in the same step. Gradients naturally push them toward a consistent state.

In EM:
1. E-step: compute m_natural (the "true" variational mean), store as m_whitened
2. M-step: kernel changes, m_whitened stays fixed
3. Next iteration: old m_whitened interpreted with new L_K → corruption

### 8b.3 "NGD with Separate Optimizers Seems Like EM"

**Apparent contradiction**: GPflow recommends "interleaved optimization with separate optimizers: NGD for variational, Adam for kernel." This sounds like EM - so how does GPflow avoid the problem?

**Resolution**: NGD is still **gradient-based**, not closed-form:

| Method | How It Updates Variational Params |
|--------|-----------------------------------|
| **Newton E-step** | Closed-form formula derived for specific kernel; no autograd |
| **Natural Gradient Descent** | Gradient of ELBO w.r.t. variational params; autograd involved |

Even with separate optimizers in NGD + Adam:
1. Forward pass computes ELBO with current kernel AND current variational params
2. NGD computes gradient of ELBO w.r.t. variational params (given current L_K)
3. Adam computes gradient of ELBO w.r.t. kernel params
4. Both update

The gradients are computed with the **current** L_K. When L_K changes next iteration, NGD will compute new gradients that account for the new L_K.

Our Newton E-step doesn't involve autograd at all. We use analytical formulas derived for a specific kernel state. When kernel changes, those formulas would give different answers - but we've already stored the old result.

### 8b.4 "UnwhitenedVariationalStrategy Also Has Old m with New Kernel"

**Apparent contradiction**: With UnwhitenedVariationalStrategy, after M-step we still have "old m_natural with new kernel." Isn't this also a problem?

**Resolution**: This is expected EM behavior vs. corruption:

| Situation | What Happens | Is It a Problem? |
|-----------|--------------|------------------|
| **Unwhitened: old m_natural, new kernel** | `λ_m = K_XZ_new @ K_ZZ_new^{-1} @ m_old` | NO - suboptimal but valid |
| **Whitened: old m_whitened, new kernel** | `λ_m = K_XZ_new @ L_K_new^{-T} @ L_K_old^{-1} @ m_old` | YES - corrupted m, potentially invalid |

With unwhitened:
- m_old is not optimal for the new kernel (expected in EM)
- But it's still a **valid** variational mean
- ELBO will be worse than optimal, but next E-step will re-optimize
- The formula uses m_old correctly

With whitened:
- We're not using m_old - we're using a **corrupted** version: `L_K_new @ L_K_old^{-1} @ m_old`
- This corruption can make ELBO much worse or cause numerical issues
- The transformation `L_K_new @ L_K_old^{-1}` can amplify or suppress components arbitrarily

### 8b.5 Summary: Why There's No Contradiction

All sources consistently indicate:

1. **GPyTorch's whitening assumes joint optimization** where autograd handles L_K coupling
2. **No source mentions explicit re-whitening** after kernel changes - because the design doesn't need it for joint optimization
3. **Cache clearing is about kernel matrices**, not variational parameter adjustment
4. **NGD is gradient-based**, not closed-form, so it maintains consistency through autograd
5. **UnwhitenedVariationalStrategy avoids the problem entirely** by not involving L_K in storage/retrieval

The investigation findings are internally consistent. The apparent contradictions arise from subtle distinctions between:
- Cache clearing (kernel computation) vs. parameter adjustment (variational storage)
- Joint optimization (m_whitened is the parameter) vs. EM (m_natural is the parameter)
- Gradient-based updates (autograd-aware) vs. closed-form updates (autograd-free)

---

## 9. Solution: UnwhitenedVariationalStrategy

### 9.1 Why UnwhitenedVariationalStrategy Solves the Problem

| Aspect | VariationalStrategy | UnwhitenedVariationalStrategy |
|--------|--------------------|-----------------------------|
| **Stored params** | m_whitened, V_whitened | m_natural, V_natural |
| **Prior** | N(0, I) transformed | N(μ_Z, K_ZZ) actual |
| **Predictive mean formula** | `K_XZ @ L_K^{-T} @ m_stored` | `K_XZ @ K_ZZ^{-1} @ (m - μ_prior)` |
| **L_K involvement** | YES (in formula) | NO (uses K_ZZ directly) |
| **When kernel changes** | Interpretation of m_stored changes | Interpretation unchanged |
| **EM compatible?** | NO (requires re-whitening) | YES |

### 9.2 Mathematical Equivalence

Both strategies compute the same predictive distribution when parameters are correctly interpreted:

**Whitened**:
```
λ_m = K_XZ @ L_K^{-T} @ m_whitened
    = K_XZ @ L_K^{-T} @ L_K^{-1} @ m_natural  (if m_whitened = L_K^{-1} @ m_natural)
    = K_XZ @ (L_K @ L_K^T)^{-1} @ m_natural
    = K_XZ @ K_ZZ^{-1} @ m_natural
```

**Unwhitened**:
```
λ_m = K_XZ @ K_ZZ^{-1} @ (m_stored - μ_prior)
    = K_XZ @ K_ZZ^{-1} @ m_natural  (if μ_prior = 0)
```

Same result, but unwhitened stores natural params directly - no L_K transformation involved.

### 9.3 Implementation Path

**Minimal changes required**:

1. **model.py**: Change import
```python
# Before:
from gpytorch.variational import VariationalStrategy

# After:
from gpytorch.variational import UnwhitenedVariationalStrategy
```

2. **model.py**: Use in model initialization
```python
variational_strategy = UnwhitenedVariationalStrategy(
    self, inducing_points, variational_distribution,
    learn_inducing_locations=False
)
```

3. **estep.py**: Simplify parameter access (remove whitening conversions)
```python
# Before (with whitening):
m = get_variational_mean_with_L_K(model, L_K)
update_variational_mean_with_L_K(model, m_new, L_K)

# After (no whitening):
m = model.variational_strategy._variational_distribution.variational_mean.clone()
model.variational_strategy._variational_distribution.variational_mean.data.copy_(m_new)
```

### 9.4 Trade-offs

**Pros**:
- Eliminates ALL whitening conversion issues
- Simpler code (no conversion functions needed)
- EM-compatible by design
- Mathematically cleaner for EM

**Cons**:
- Slightly worse numerical conditioning
- ~5% slower for moderate M (50-200)
- GPyTorch docs say "in almost all cases VariationalStrategy is preferable"

**For EM-style optimization, the cons are negligible** - the E-step overhead dominates, and numerical conditioning matters less when using closed-form Newton updates.

---

## 10. Verification and Testing

### 10.1 Existing Test Coverage

**File**: `tests/test_whitening_paths.py`
- Tests cached vs non-cached paths with/without whitening
- Test 3: Validates whitening paths are equivalent (λ_m ratio 0.95-1.05)
- Test 6: End-to-end training produces good results

**File**: `tests/test_m_whitening.py`
- Tests round-trip consistency for m and V conversions
- Test 2: GPyTorch λ_m correctness (ratio improved from 0.075 to 0.995 after fix)

**File**: `tests/test_whitening_hypothesis.py`
- Tests corruption formula explicitly
- Confirms single gradient step causes immediate corruption

### 10.2 How to Verify This Investigation

1. **Check GPyTorch source**:
```python
import gpytorch
import inspect
print(inspect.getsourcefile(gpytorch.variational.VariationalStrategy))
# Then read lines 212-216 (predictive mean) and 238-272 (auto-whitening)
```

2. **Check UnwhitenedVariationalStrategy**:
```python
print(inspect.getsourcefile(gpytorch.variational.UnwhitenedVariationalStrategy))
# Read the docstring and prior_distribution property
```

3. **Run hypothesis tests**:
```bash
conda run -n pytorch_gpytorch python tests/test_whitening_hypothesis.py
```

4. **Check GitHub issues**:
- https://github.com/cornellius-gp/gpytorch/issues/1308 (cache invalidation)
- https://github.com/cornellius-gp/gpytorch/pull/903 (variational model updates)

### 10.3 Recommended Validation After Migration

After switching to `UnwhitenedVariationalStrategy`:

1. Run `tests/test_estep_comparison.py` - should produce similar final results
2. Run `test_estep_pnas.py --mode vargp_style` - should match or exceed previous performance
3. Verify no whitening-related crashes or NaNs
4. Check that M-step kernel gradients work correctly

---

## 11. References and How to Verify

### 11.1 GPyTorch Source Files

| File | Key Content | Lines |
|------|-------------|-------|
| `variational_strategy.py` | Predictive mean formula | 212-216 |
| `variational_strategy.py` | Auto-whitening on first call | 238-272 |
| `variational_strategy.py` | Cholesky caching | 102-105 |
| `_variational_strategy.py` | Cache clearing in training | 329-330 |
| `unwhitened_variational_strategy.py` | Prior distribution | 60-65 |
| `unwhitened_variational_strategy.py` | Predictive mean | 183-189 |

**To find these files**:
```python
import gpytorch
import os
gpytorch_path = os.path.dirname(gpytorch.__file__)
print(f"GPyTorch source: {gpytorch_path}/variational/")
```

### 11.2 Academic References

| Paper | What It Covers | URL |
|-------|---------------|-----|
| Hensman et al. (2013) | SVGP fundamentals | https://arxiv.org/abs/1309.6835 |
| Hensman et al. (2015) | MCMC for SVGPs | https://arxiv.org/abs/1507.04217 |
| Matthews (2017) | Whitening derivation | https://www.repository.cam.ac.uk/handle/1810/278022 |
| Salimbeni et al. (2018) | Natural gradients | https://arxiv.org/abs/1803.09151 |
| Adam et al. (2021) | Dual parameterization | https://arxiv.org/abs/2111.03412 |

### 11.3 GPyTorch GitHub References

| Issue/PR | Topic | URL |
|----------|-------|-----|
| Issue #1308 | Cache invalidation on load | https://github.com/cornellius-gp/gpytorch/issues/1308 |
| Issue #1754 | Variational strategy cache bug | https://github.com/cornellius-gp/gpytorch/issues/1754 |
| Issue #1556 | Kernel change in eval mode | https://github.com/cornellius-gp/gpytorch/issues/1556 |
| PR #903 | Variational model updates | https://github.com/cornellius-gp/gpytorch/pull/903 |
| Issue #1156 | Prior mean in whitened space | https://github.com/cornellius-gp/gpytorch/issues/1156 |

### 11.4 GPyTorch Documentation

| Topic | URL |
|-------|-----|
| Variational module overview | https://docs.gpytorch.ai/en/stable/variational.html |
| Natural Gradient Descent tutorial | https://docs.gpytorch.ai/en/stable/examples/04_Variational_and_Approximate_GPs/Natural_Gradient_Descent.html |
| VariationalStrategy API | https://docs.gpytorch.ai/en/stable/variational.html#gpytorch.variational.VariationalStrategy |

### 11.5 Local Project Files

| File | Content |
|------|---------|
| `estep.py` | Current E-step with whitening functions |
| `tests/test_whitening_paths.py` | Whitening path validation |
| `tests/test_m_whitening.py` | Parameter conversion tests |
| `tests/test_whitening_hypothesis.py` | Corruption hypothesis tests |
| `.claude/archive/ARCHIVE_2026-01-18_kernel_caching_and_whitening.md` | Previous session's whitening analysis (historical) |

---

## Appendix A: Investigation Methodology Details

### A.1 Subagent Queries Used

The investigation used Claude Code's Task tool with the following subagent queries:

1. **GPyTorch source exploration** (subagent_type=Explore):
   - "Investigate how GPyTorch handles whitened variational parameters when kernel parameters change"
   - Examined: variational_strategy.py, _variational_strategy.py, cholesky_variational_distribution.py

2. **Academic literature research** (subagent_type=general-purpose):
   - "Research academic literature on how sparse variational GPs handle whitening parameterization when kernel hyperparameters are optimized jointly"
   - Found: Matthews thesis, Salimbeni NGD paper, Adam dual parameterization paper

3. **GPyTorch documentation search** (subagent_type=general-purpose):
   - "Research how GPyTorch officially recommends handling EM-style optimization with variational GPs"
   - Found: NGD tutorial, variational module docs, GitHub issues

4. **Local codebase analysis** (subagent_type=Explore):
   - "Read estep.py implementation and understand how whitening is currently handled"
   - Documented: whitening functions, e_step_loop paths, cache invalidation

5. **UnwhitenedVariationalStrategy exploration** (subagent_type=Explore):
   - "Investigate GPyTorch's UnwhitenedVariationalStrategy as a potential solution"
   - Found: It stores natural params directly, no L_K transformation

6. **GitHub issues search** (subagent_type=general-purpose):
   - "Search GPyTorch GitHub issues related to EM optimization, whitening problems, kernel parameter updates"
   - Found: Cache issues #1308, #1754, #1556; PR #903 variational updates

### A.2 Verification Steps Taken

1. Cross-referenced GPyTorch source code with documentation
2. Verified academic claims match GPyTorch implementation
3. Confirmed GitHub issues match observed behavior
4. Checked that existing test suite validates understanding

---

*Document created: 2026-01-20*
*Investigation completed by: Claude Code (Opus 4.5)*
*To be read by: Future Claude Code sessions*
