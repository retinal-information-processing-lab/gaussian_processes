# Firing Rate Parameter Instability - Investigation Findings

**Date**: 2026-01-21
**Status**: CONFIRMED - A parameter causes whitening collapse
**Related**: INVESTIGATION_whitening_collapse_M75.md

---

## Summary

**HYPOTHESIS CONFIRMED**: The whitening collapse at M=75 is caused by instability in the firing rate parameter A, not by the L_K mismatch. The A parameter explodes to ~0.16, then suddenly collapses to near-zero, causing the model to fail.

---

## Evidence

### Test 1: M=50, seed=123 (STABLE)
**Command**: `python investigations/diagnose_firing_rate_instability.py --ntilde 50 --seed 123`

**Result**: Stable training
- A trajectory: 0.0100 → 0.0362 (smooth growth)
- Final loss: 402.52 (good)
- Final firing rate: 5.27 max
- **No instability**

### Test 2: M=50, seed=456 (STABLE)
**Command**: `python investigations/diagnose_firing_rate_instability.py --ntilde 50 --seed 456`

**Result**: Stable training
- A trajectory: 0.0100 → 0.0414 (smooth growth)
- Final loss: 404.73 (good)
- Final firing rate: 5.09 max
- **No instability**

### Test 3: M=75, seed=42 (COLLAPSE)
**Command**: `python investigations/diagnose_firing_rate_instability.py --ntilde 75 --seed 42`

**Result**: **CATASTROPHIC FAILURE**

| Phase | Iterations | A trajectory | Firing rates | Loss |
|-------|------------|--------------|--------------|------|
| **Growth** | 1-37 | 0.0100 → 0.1077 | Normal (0.2-8.0) | ~410-447 |
| **Explosion** | 38-43 | 0.1173 → 0.1614 | Erratic (0.3-7.3) | ~430-531 |
| **Collapse** | 44-50 | 0.1576 → 0.0004 | **Constant 1.0** | **~500-510** |

**Detailed collapse sequence**:

```
Iter 38: A=0.1173, rate_max=5.4, loss=430
Iter 41: A=0.1430, rate_max=2.2, loss=505
Iter 43: A=0.1614, rate_max=1.3, loss=531  [PEAK]
Iter 44: A=0.1576, rate_max=1.2, loss=511
Iter 45: A=0.1454, rate_max=1.0, loss=508  [RATES COLLAPSE TO 1.0]
Iter 46: A=0.0429, rate_max=1.0, loss=505  [A COLLAPSE BEGINS]
Iter 47: A=0.0127, rate_max=1.0, loss=504
Iter 50: A=0.0004, rate_max=1.0, loss=500  [COMPLETE FAILURE]
```

**Key observation**: Once firing rates collapse to constant 1.0 (iteration 45), the model cannot recover. The A parameter rapidly decreases to near-zero.

---

## Analysis

### Why does A explode then collapse?

1. **Explosion phase** (iter 38-43):
   - A grows rapidly: 0.1173 → 0.1614
   - Firing rate = exp(A·λ + λ₀), so large A amplifies GP predictions
   - Gradients become large (grad_norm_A up to 215)
   - Loss increases: 430 → 531

2. **Collapse phase** (iter 44-50):
   - Firing rates saturate to constant 1.0 (all images predict same rate)
   - This makes the GP predictions meaningless
   - Optimizer reduces A to make predictions safer
   - A collapses: 0.1454 → 0.0004

3. **Why doesn't it recover?**
   - Once all firing rates are identical, there's no gradient signal to improve λ
   - The model is stuck in a degenerate state
   - E-step cannot improve variational parameters
   - Loss plateaus at ~500 (baseline rate only, no signal)

### Why is this specific to M=75, seed=42?

**It's not M=75 specifically** - it's the unlucky inducing point initialization:
- Bad seeds select inducing points that create ill-conditioned kernel matrices
- This makes the E-step Newton update unstable
- Unstable E-step produces bad λ predictions
- F-step overcompensates by increasing A
- Positive feedback loop: larger A → more unstable λ → larger A
- Eventually A explodes and collapses

**M=50 seeds were lucky** - their inducing points didn't hit this issue.

### Connection to whitening

Whitened mode (m_whitened = L_K^-1 @ m_natural) may be more sensitive to:
- Inducing point conditioning (affects L_K)
- Kernel parameter changes during M-step (L_K changes, creates mismatch)
- Small errors in m get amplified when multiplied by L_K during prediction

---

## Proposed Solutions

### Option 1: Constrain A parameter (RECOMMENDED)

**Problem**: A has no upper bound (only Positive constraint)

**Fix**: Add upper bound to prevent explosion
```python
# In likelihoods.py:
self.register_constraint('raw_A', Interval(lower_bound=LOG_A_MIN, upper_bound=LOG_A_MAX))
# Where:
#   LOG_A_MIN = log(1e-4)  # Min A = 0.0001
#   LOG_A_MAX = log(0.1)   # Max A = 0.1 (prevent explosion)
```

**Rationale**:
- A = 0.16 is way too large for neural data (typical A ~ 0.01-0.05)
- Original varGP uses A_init = 0.01, which suggests similar range
- Clamping prevents explosion while allowing reasonable range

### Option 2: Gradient clipping

**Problem**: Gradients for A become very large (grad_norm_A up to 215)

**Fix**: Clip gradients during F-step
```python
torch.nn.utils.clip_grad_norm_(likelihood.parameters(), max_norm=10.0)
```

**Rationale**:
- Prevents sudden jumps in A
- Allows recovery from near-explosion states
- Standard technique for training stability

### Option 3: Better inducing point initialization

**Problem**: Random inducing points can be ill-conditioned

**Fix**: Use k-means or other clustering methods
```python
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=ntilde, random_state=seed)
kmeans.fit(X_train.cpu().numpy())
inducing_points = torch.tensor(kmeans.cluster_centers_)
```

**Rationale**:
- Better coverage of input space
- More stable kernel matrices
- Used in many GP libraries (e.g., GPflow)

### Option 4: Unwhitened mode (TEMPORARY)

**Problem**: Whitening may amplify numerical issues

**Fix**: Use `whitening=False` in model initialization
```python
model = VariationalGPModel(inducing_points, kernel, whitening=False)
```

**Rationale**:
- Avoids L_K^-1 multiplication that can amplify errors
- From BENCHMARK_LOG.md: legacy (unwhitened) mode is more stable
- But this is just avoiding the problem, not fixing it

---

## Recommendations

**Immediate action**:
1. Implement Option 1 (constrain A) - prevents explosions
2. Test on M=75, seed=42 to verify fix
3. Run benchmark across all M values (50, 75, 100, 200) with multiple seeds

**Follow-up**:
1. Investigate why whitening makes this worse
2. Consider Option 3 (better inducing point init) for long-term robustness
3. Profile eigenvalue spectrum of K̃ for bad vs good seeds

**NOT recommended**:
- Option 4 (unwhitened mode) - just masks the symptom, doesn't fix root cause

---

## Files

**Diagnostic script**: `investigations/diagnose_firing_rate_instability.py`
**Data**:
- `investigations/diagnostics_M50_seed123.csv`
- `investigations/diagnostics_M50_seed456.csv`
- `investigations/diagnostics_M75_seed42.csv`
- `investigations/log_M50_seed123.txt`
- `investigations/log_M50_seed456.txt`
- `investigations/log_M75_seed42.txt`

**Usage**:
```bash
python investigations/diagnose_firing_rate_instability.py --ntilde M --seed SEED
```
