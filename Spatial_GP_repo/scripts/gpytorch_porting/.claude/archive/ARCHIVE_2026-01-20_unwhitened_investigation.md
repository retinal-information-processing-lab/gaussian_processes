# Handoff: UnwhitenedVariationalStrategy Performance Investigation

> **DISCLAIMER (added later)**: This document contains a conclusion stating the accuracy
> gap is "INHERENT to UnwhitenedVariationalStrategy, not a bug". This conclusion is
> **tentative and not fully validated**. The observed KL divergence explosion and gradient
> instability are real, but the root cause explanation needs further investigation.
> See Q29 in DECISION_LOG.md for the current (more cautious) assessment.

**Date**: 2026-01-20
**Status**: Implementation COMPLETE, Performance Investigation IN PROGRESS

---

## PROMPT FOR NEXT SESSION

**CONTEXT**: We implemented `UnwhitenedVariationalStrategy` as an alternative to the default whitened strategy. The implementation is correct, but unwhitened performs significantly worse. We investigated why and need to continue.

**READ THESE FILES FIRST**:
1. `.claude/DECISION_LOG.md` - See Q26-Q28 for whitening decisions
2. `.claude/WHITENING_INVESTIGATION_2026-01-20.md` - Original whitening investigation
3. `model.py` - Has `whitening` parameter (lines 30-33, 56-71)
4. `estep.py` - Has auto-detection of whitening (lines 803-805, 1536-1538)

---

## WHAT WAS IMPLEMENTED

Added `UnwhitenedVariationalStrategy` as alternative via `whitening` parameter:

```python
# model.py
model = VariationalGPModel(inducing_points, kernel, whitening=False)

# CLI
python test_estep_pnas.py --mode vargp_style --unwhitened
```

**Files modified**:
- `model.py`: Added `whitening` parameter, conditional strategy selection
- `estep.py`: Auto-detect whitening from model, conditional L_K computation
- `test_estep_pnas.py`: Added `--unwhitened` flag
- `tests/test_estep_comparison.py`: Added `--unwhitened` flag

---

## PERFORMANCE COMPARISON (M=50, N=500)

| Strategy | Explained Var | Time | Final A | Final λ₀ |
|----------|---------------|------|---------|----------|
| varGP (reference) | 0.8748 | 5.4s | - | - |
| **Whitened (default)** | 0.8447 | 7.2s | 1.72 | -2.47 |
| **Unwhitened** | 0.6878 | 29.7s | 0.095 | -0.15 |

**Key observations**:
- Unwhitened is 4x slower (29.7s vs 7.2s)
- Unwhitened is 16% worse accuracy (0.6878 vs 0.8447)
- Unwhitened learns VERY different params (A is 18x smaller!)

---

## INVESTIGATION FINDINGS

### CONFIRMED: No Parameter Interpretation Bug

We verified that `UnwhitenedVariationalStrategy` correctly interprets parameters as natural (NOT whitened):

| Strategy | Stores | Formula |
|----------|--------|---------|
| VariationalStrategy | m_whitened = L_K⁻¹ @ m_natural | λ = K @ L_K⁻ᵀ @ m_whitened |
| UnwhitenedVariationalStrategy | m_natural directly | λ = K @ K̃⁻¹ @ m_natural |

Both are mathematically equivalent: `K @ L_K⁻ᵀ @ L_K⁻¹ @ m = K @ K̃⁻¹ @ m`

**Our code path is correct:**
1. With `whitening=False`, GPyTorch stores m_natural directly
2. Our `get_variational_mean()` returns m_natural
3. Our cached formula `λ = K @ K̃⁻¹ @ m` is correct for natural params
4. E-step produces m_new in natural space
5. We store m_new directly — correct for UnwhitenedVariationalStrategy

### CONFIRMED: Slowdown Cause

**Computational complexity**:
- Whitened: Uses Cholesky L_K, triangular solves O(M²)
- Unwhitened: Must compute K̃⁻¹ directly, O(M³) per operation

**KL divergence cost**:
- Whitened: `KL[N(m,V) || N(0,I)]` — simple, fast
- Unwhitened: `KL[N(m,V) || N(0,K̃)]` — requires K̃⁻¹, expensive

### UNKNOWN: Accuracy Gap Cause

Possible causes (not yet verified):
1. **Optimization landscape**: Whitened has better-conditioned gradients
2. **Prior effect**: N(0,I) vs N(0,K̃) regularizes differently
3. **Numerical precision**: Ill-conditioned K̃⁻¹ accumulates errors
4. **E-step formula**: May need modification for unwhitened prior

---

## INVESTIGATION COMPLETE (2026-01-20)

### Root Cause: KL Divergence Gradient Instability

A diagnostic script (`tests/diagnose_unwhitened_performance.py`) revealed the root cause:

**K_tilde Conditioning:**
- Condition number: 1.02e+04
- Eigenvalue range: [1.58e-03, 1.61e+01]

**Gradient Evolution Over Training Iterations:**

| Iter | Whitened KL | Unwhitened KL | Gradient Ratio |
|------|-------------|---------------|----------------|
| 0    | 0.00        | 0.16          | 0.64x          |
| 1    | 0.00        | 0.24          | 1.67x          |
| 2    | 0.00        | 1.07          | 7.42x          |
| 3    | 0.01        | 20.61         | 34.30x         |
| 4    | 0.02        | **423.26**    | **159.33x**    |

**Key Observation**: The unwhitened KL divergence **explodes** (0 → 423 in 5 iterations), causing:
1. Gradient magnitudes to escalate (2.9 → 689)
2. Loss to diverge (806 → 1230)

**Why This Happens**: The unwhitened KL term includes K̃⁻¹:
```
KL_unwhitened = 0.5 * (tr(K̃⁻¹ @ V) + mᵀ @ K̃⁻¹ @ m - M - log|V| + log|K̃|)
```
The K̃⁻¹ operator amplifies any deviation in m by O(cond(K̃)) ≈ 10⁴.

**In whitened space**: Prior is N(0, I), so KL gradient is simply ∂KL/∂m_w ≈ m_w (no K̃⁻¹).

### This Is NOT Fixable by Tuning

The instability is inherent to the unwhitened parameterization:
- Adaptive learning rates (Adam) help but don't eliminate the fundamental conditioning issue
- The E-step Newton update is correct, but M-step optimization suffers
- Any gradient-based kernel optimization will see imbalanced signals

### Conclusion

**The 16% accuracy gap (0.6878 vs 0.8381) is INHERENT to UnwhitenedVariationalStrategy**, not a bug.

Use UnwhitenedVariationalStrategy only when:
- L_K coupling after M-step is unacceptable (pure EM with kernel changes)
- Worse accuracy is acceptable for the application
- Consider more E-step iterations to partially compensate

### Reproducing These Findings

**1. Run the gradient/KL diagnostic script:**
```bash
cd /home/idv-eqs8-pza/IDV_code/ClosedLoopProject/gaussian_processes/Spatial_GP_repo/scripts/gpytorch_porting
conda run -n pytorch_gpytorch python tests/diagnose_unwhitened_performance.py --n-iterations 5
```

Expected output shows:
- K_tilde condition number: ~1.02e+04
- Unwhitened KL divergence explosion (0 → 400+ in 5 iterations)
- Gradient ratio escalation (0.6x → 159x)

**2. Run the full comparison test:**
```bash
conda run -n pytorch_gpytorch python tests/test_estep_comparison.py --unwhitened
```

Expected results (M=50, N=500):
| Implementation | Expl. Var | Time |
|----------------|-----------|------|
| varGP (reference) | 0.8748 | 5.3s |
| UnwhitenedVariationalStrategy | 0.6878 | 29.3s |
| Whitened (no-whiten conv) | 0.8381 | 6.5s |

**3. Vary M to see conditioning effect:**
```bash
conda run -n pytorch_gpytorch python tests/diagnose_unwhitened_performance.py --ntilde 100 --n-iterations 5
```

Higher M → higher condition number → worse gradient instability.

---

## HYPOTHESIS TO TEST

The E-step Newton formula is:
```
V_new = K̃(K̃ + G)⁻¹K̃
m_new = m + K̃(K̃ + G)⁻¹(g - m)
```

This formula is derived assuming **prior covariance = K̃** in natural space.

- For VariationalStrategy: Prior in whitened space is N(0, I), but when transformed back to natural space, it becomes N(0, K̃). So the formula is correct.
- For UnwhitenedVariationalStrategy: Prior in natural space is N(0, K̃). The formula should still be correct!

**If the E-step formula is correct for both**, then the performance gap must come from:
- M-step optimization dynamics (gradient landscape)
- KL regularization effect
- Numerical conditioning during optimization

---

## QUICK VERIFICATION COMMANDS

```bash
# Test whitened (default) - should get ~0.84
conda run -n pytorch_gpytorch python test_estep_pnas.py --mode vargp_style --ntilde 50 --save-plot none

# Test unwhitened - should get ~0.69
conda run -n pytorch_gpytorch python test_estep_pnas.py --mode vargp_style --ntilde 50 --unwhitened --save-plot none

# Run comparison
conda run -n pytorch_gpytorch python tests/test_estep_comparison.py --unwhitened
```

---

## KEY CODE LOCATIONS

| What | File | Lines |
|------|------|-------|
| Strategy selection | model.py | 56-71 |
| Whitening auto-detect | estep.py | 803-805, 1536-1538 |
| Conditional L_K computation | estep.py | 125-128 |
| Unwhitened E-step path | estep.py | 821-824 (read), 866-868 (write) |
| Moment computation (cached) | estep.py | 169-215 |
| Newton update | estep.py | 218-262 |

---

## NEXT STEPS

1. Create diagnostic script to compare whitened vs unwhitened at each iteration
2. Verify KL divergence values are equivalent at equivalent params
3. Check if optimization landscape (gradients) differs significantly
4. Determine if performance gap is inherent to parameterization or fixable
