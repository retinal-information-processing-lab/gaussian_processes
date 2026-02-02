# Performance Loss Investigation - Key Findings

**Date**: 2026-02-01
**Cell**: 8 (well-fitted cell)
**Config**: Float32, Seed 123

---

## Summary: Performance DECREASES with More Data

Contrary to expectations, **adding more training data (500→2000) DEGRADES performance** for M≥100:

| Mode | M | ntrain=500 | ntrain=2000 | Loss |
|------|---|------------|-------------|------|
| vargp_old | 100 | **0.869** | 0.767 | **-11.8%** ⬇️ |
| vargp_old | 200 | **0.814** | 0.735 | **-9.7%** ⬇️ |
| vargp_direct | 100 | **0.868** | 0.765 | **-11.9%** ⬇️ |
| vargp_direct | 200 | **0.815** | 0.731 | **-10.2%** ⬇️ |
| default_gpy | 50 | **0.845** | 0.605 | **-28.4%** ⬇️ |

Only M=50 shows modest gains (+1-2%) with more data.

---

## Pattern 1: Increasing M Degrades Performance (ntrain=2000)

With 2000 training points, **all methods show MONOTONIC DEGRADATION** as M increases:

### vargp_old (ntrain=2000):
- M=50: 0.853 ⬆️
- M=100: 0.767 ⬇️
- M=200: 0.735 ⬇️

### vargp_direct (ntrain=2000):
- M=50: 0.855 ⬆️
- M=100: 0.765 ⬇️
- M=200: 0.731 ⬇️

This is **counterintuitive**: more inducing points should improve capacity, not hurt it.

---

## Pattern 2: Interaction Effect (ntrain × M)

The performance loss is NOT independent - it's an **interaction**:

- **M=50**: More data helps (+1.4%)
- **M=100**: More data HURTS (-11.8%)
- **M=200**: More data HURTS (-9.7%)

The effect of adding data depends on M, suggesting the problem emerges at specific M/ntrain ratios.

---

## Pattern 3: default_gpy Collapses Catastrophically

Default GPyTorch (adam optimizer, 500 iterations) shows **massive failure** at ntrain=2000, M=50:
- ntrain=500: 0.845 ✓
- ntrain=2000: 0.605 ⚠️ **-28% loss**

This mode is fundamentally broken for large ntrain.

---

## Hypotheses

### H1: Optimization Failure (Most Likely)
- **Evidence**: Monotonic degradation with M suggests optimizer struggles in high-dimensional spaces
- **Mechanism**: LBFGS may converge to poor local minima with more parameters (larger M)
- **Test**: Try more iterations, different optimizers, or better initialization

### H2: Overfitting to Inducing Point Locations
- **Evidence**: Performance drops when M is large relative to data variance
- **Mechanism**: With random inducing point selection, bad configurations become more likely as M grows
- **Test**: Try different inducing point initialization strategies (k-means, greedy selection)

### H3: Numerical Instability
- **Evidence**: Float32 + large M + large ntrain = potential for ill-conditioning
- **Mechanism**: Kernel matrices become poorly conditioned, eigenvalue cutoff excludes important information
- **Test**: Check eigenspace dimensions (n_b vs M), try float64, check condition numbers

### H4: Underfitting with Fixed Iterations
- **Evidence**: More data needs more iterations to converge, but we use fixed 50 iterations
- **Mechanism**: Optimization doesn't have time to adapt all M inducing points + kernel params
- **Test**: Scale iterations with M or ntrain (e.g., 50 * (M/50) iterations)

### H5: Hyperparameter Mismatch
- **Evidence**: Same learning rates (0.1) for all configurations
- **Mechanism**: Optimal LR depends on problem scale (ntrain, M)
- **Test**: Tune learning rates for large M/ntrain configs

---

## Recommended Next Steps

1. **Check eigenspace dimensions**: Are we losing rank when M increases?
   ```python
   # Add logging to track n_b vs M
   print(f"M={M}, n_b={len(eigvals_b)}, rank_loss={M - len(eigvals_b)}")
   ```

2. **Run float64 comparison**: Does numerical precision matter?
   ```bash
   python run_single_mode.py --mode vargp_direct --ntilde 100 --n-train 2000 --cell 8
   ```

3. **Try more iterations**: Does optimization just need more time?
   ```bash
   python run_single_mode.py --mode vargp_old --ntilde 100 --n-train 2000 --n-iterations 200 --cell 8
   ```

4. **Inspect final parameters**: Are they reaching boundary constraints or diverging?

5. **Test on more cells**: Is this cell-specific or systematic?
   ```bash
   python run_systematic_test.py  # Run cells 8, 10, 15
   ```

---

## Timing Observations

- **vargp_direct is 40-45% faster** than vargp_old (8.3s vs 14.8s at M=200, ntrain=2000)
- **default_gpy is 2-3x faster** but accuracy is too poor to be useful
- Speed advantage of vargp_direct maintained across all configs

---

## Best Configurations (Cell 8)

| Rank | Mode | ntrain | M | test_r | Notes |
|------|------|--------|---|--------|-------|
| 1st | vargp_old | 500 | 100 | **0.869** | Best overall |
| 2nd | vargp_direct | 500 | 100 | **0.868** | Nearly identical |
| 3rd | vargp_direct | 2000 | 50 | **0.855** | Best with large data |

**Avoid**: M≥100 with ntrain=2000 (all methods degrade)

---

*Generated from run_systematic_test.py (Cell 8, float32, seed 123)*
