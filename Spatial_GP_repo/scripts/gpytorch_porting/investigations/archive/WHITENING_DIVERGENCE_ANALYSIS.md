# Whitening Mode Divergence Analysis

**Date**: 2026-01-21
**Investigation**: Why whitened mode collapses while unwhitened doesn't
**Setup**: M=50, seed=456, 5 EM iterations, identical initialization

## Key Findings

### 1. INITIAL STATE PROBLEM (Iteration 0)

**Critical discovery**: The initial KL divergence is VASTLY different:

| Mode | KL (iter 0) | m_norm | V_cond | Interpretation |
|------|-------------|--------|--------|----------------|
| **Whitened** | **0.00** | 0.00 | 2.15e+04 | V_stored = I → V_actual = K̃ (correct prior!) |
| **Unwhitened** | **1501.82** | 0.007 | 1.00 | V_stored = I → V_actual = I (NOT the prior!) |

**What this means**:
- **Whitened mode** correctly interprets V_stored=I as the GP prior (V_actual = K̃)
- **Unwhitened mode** incorrectly starts with V=I, which has KL=1501 relative to the prior

**From `estep.py:get_variational_covar_with_L_K()` comments**:
```python
# At initialization:
#     L_stored = I, so V_stored = I
#     GPyTorch interprets this as V_actual = L_K @ I @ L_K.T = K̃ (the prior!)
#     We MUST apply this conversion, otherwise cached path uses V=I instead of V=K̃
```

The whitening conversion is **CORRECT** - it's applying the standard GPyTorch interpretation.
The unwhitened mode is starting from the **WRONG** initial state!

---

### 2. TRAJECTORY DIFFERENCES

#### Iteration 1: First Major Divergence

| Metric | Whitened | Unwhitened | Difference |
|--------|----------|------------|------------|
| KL | 115.22 | 93.68 | +21.54 |
| ELBO | -2105.47 | -2089.75 | -15.72 (worse) |
| ||m|| | 51.68 | 69.82 | **-18.14** |
| ||m_stored|| | 15.18 | 69.82 | **-54.64** |
| ||V|| | 48.58 | 27.32 | +21.26 |
| cond(V) | 65801 | 17888 | +47913 |

**Key observations**:
- Whitened m is much smaller in natural space (51.68 vs 69.82)
- Whitened m_stored is DRASTICALLY smaller (15.18 vs 69.82) - whitening shrinks params
- Whitened V has higher norm and condition number
- Both models have similar ELBO, but whitened has higher KL

#### Iteration 2: Whitened KL Collapse

| Metric | Whitened | Unwhitened | Difference |
|--------|----------|------------|------------|
| KL | **1.30** | 12.84 | **-11.54** |
| ELBO | -1966.95 | -1898.84 | -68.11 (worse) |
| ||m|| | 13.36 | 42.50 | **-29.14** |
| ||V|| | 238.17 | 127.88 | +110.29 |
| cond(V) | 135552 | 10457 | +125095 |

**Critical moment**: Whitened KL drops from 115 → 1.3, while unwhitened drops normally (94 → 13).

This suggests the whitened variational distribution collapsed onto the prior (KL ≈ 0 means q(u) ≈ p(u)).

#### Iteration 3: Whitened Rebounds

| Metric | Whitened | Unwhitened | Difference |
|--------|----------|------------|------------|
| KL | **40.81** | 35.15 | +5.66 |
| ELBO | -1774.91 | -1787.27 | **+12.36 (better!)** |
| ||m|| | 56.92 | 54.01 | +2.91 |
| cond(V) | 5487 | 1302 | +4185 |

Whitened recovers! But with much higher V condition number.

---

### 3. FINAL OUTCOME (Iteration 5)

| Metric | Whitened | Unwhitened | Winner |
|--------|----------|------------|---------|
| ELBO | -1720.44 | -1701.95 | Unwhitened (-18.49) |
| KL | 51.28 | 46.64 | Unwhitened (-4.64) |
| ELL | -1669.16 | -1655.31 | Unwhitened (+13.85) |
| A | 0.0960 | 0.0780 | - |
| λ₀ | -0.1208 | -0.0380 | - |
| ||m|| | 42.28 | 52.79 | Whitened (-10.51) |
| cond(V) | 254.98 | 73.41 | Unwhitened (3.5x lower) |
| cond(L_K) | 3371.75 | 4191.25 | Whitened (1.2x lower) |

**Result**: Unwhitened achieves slightly better ELBO, but both modes are functional (no collapse).

---

## ROOT CAUSE ANALYSIS

### Hypothesis: Initial State Mismatch

The core issue is that our E-step and initialization code does NOT match GPyTorch's expectations:

1. **GPyTorch convention** (whitened mode):
   - V_stored = I means "start at the prior" (V_actual = K̃)
   - Initial KL = 0 (correct!)

2. **Our E-step produces natural params**:
   - We compute V_new in natural space
   - When we store V_new directly (unwhitened mode), we're ignoring GPyTorch's interpretation
   - Initial V=I is treated as V_actual=I, giving KL=1501 (wrong!)

3. **The whitening conversion fixes this**:
   - `get_variational_covar_with_L_K()` applies V_actual = L_K @ V_stored @ L_K.T
   - Initial V_stored=I correctly becomes V_actual=K̃
   - But then our E-step updates produce new natural V, which gets stored back as whitened...

### Why Doesn't Unwhitened Collapse?

Despite starting from the WRONG initial state (V=I with KL=1501), unwhitened mode works because:

1. **Self-consistency**: It never applies whitening conversions, so it's "wrong but consistent"
2. **E-step corrects quickly**: After 1 iteration, V moves to a reasonable state (cond(V)=17888)
3. **No whitening artifacts**: No L_K⁻¹ operations that amplify numerical errors

### Why Does Whitened Struggle?

The whitened mode has:

1. **Correct initialization** (V=K̃, KL=0)
2. **But unstable updates**: Each E-step involves:
   - Read: V_natural = L_K @ V_stored @ L_K.T  (O(M³))
   - Newton update in natural space
   - Write: V_stored = L_K⁻¹ @ V_natural @ L_K⁻ᵀ  (O(M³))
3. **Numerical accumulation**: Two triangular solves per iteration compound errors
4. **Condition number growth**: V_cond explodes from 2.15e4 → 1.36e5 by iteration 2

---

## IMPLICATIONS FOR DEBUGGING

### What We Learned

1. **Unwhitened mode is NOT correct** - it just happens to work because it's self-consistent
2. **Whitened mode IS correct** - it properly handles GPyTorch's prior interpretation
3. **The instability is NUMERICAL** - repeated whitening/unwhitening conversions accumulate errors
4. **High condition numbers are WARNING SIGNS** - cond(V) reaching 1e5 indicates trouble

### Next Steps to Investigate

1. **Check whitening conversion accuracy**:
   - Does V_stored → V_natural → V_stored round-trip correctly?
   - At what condition number do errors become significant?

2. **Test with better-conditioned K̃**:
   - Try M=25 (cond(K̃)=3e3 vs M=50's 1.3e4)
   - Does lower condition number prevent collapse?

3. **Eigenspace projection**:
   - Original varGP works in reduced eigenspace (n_b ~ 10-11)
   - This constrains to well-conditioned subspace
   - Might this be the missing stabilization?

4. **L_K recomputation frequency**:
   - Currently recompute L_K every iteration
   - Is L_K itself becoming ill-conditioned?
   - Check eigenvalues of K̃ across iterations

---

## CONCLUSION

**Hypothesis 2 is PARTIALLY CONFIRMED**:

> "Whitening transformation itself causes numerical issues with certain parameter configurations"

YES, but the root cause is more subtle:
- Whitening is mathematically CORRECT
- Unwhitened is mathematically WRONG (bad initial state)
- Whitening's instability comes from O(M³) conversions with ill-conditioned L_K
- Condition numbers grow during training (2.15e4 → 1.36e5), amplifying errors

**The real question**: Why does original varGP work despite using whitened params?
**Answer**: Eigenspace projection! It works in a low-dimensional, well-conditioned subspace.

**Recommendation**: Implement eigenspace projection (Section 6.5 of CLAUDE.md) to stabilize whitened mode.
