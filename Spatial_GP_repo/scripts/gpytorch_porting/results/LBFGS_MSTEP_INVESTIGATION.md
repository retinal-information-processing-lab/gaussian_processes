# LBFGS M-step Investigation (January 2025)

## Summary

Attempted to replace Adam with LBFGS for the M-step (kernel hyperparameter optimization) in `train_varGP_style()`. **LBFGS underperforms Adam significantly.**

## Results Comparison

| Mode | M=50 | M=75 | M=100 |
|------|------|------|-------|
| vargp_old (reference) | **0.87** | 0.70 | 0.66 |
| vargp_style (Adam M-step) | 0.83 | 0.82 | **0.87** |
| vargp_style (LBFGS M-step) | 0.61 | 0.39 | 0.43 |

## Root Cause: Gradient Scale Imbalance

After one E-step, the gradients for kernel parameters have vastly different magnitudes:

| Parameter | Gradient | Ratio (grad/param) |
|-----------|----------|-------------------|
| raw_outputscale | -16.7 | 1.82 |
| raw_m2log2beta (beta) | 26.2 | 8.14 |
| raw_mlog2rho2 (rho) | 7.8 | 2.01 |
| eps_0x | 0.8 | ∞ |
| **raw_sigma_0** | **-0.004** | **0.008** |

**Key finding:** `raw_sigma_0` gradient is ~1000-6000x smaller than other parameters!

### Why This Matters

- LBFGS uses a single step size for all parameters
- It follows the large gradients (beta, rho, outputscale) and barely moves sigma_0
- After 50 iterations: LBFGS sigma_0 = 0.97, Adam sigma_0 = 5.22
- The larger sigma_0 is important for good generalization

### Why Adam Works Better

Adam has **adaptive per-parameter learning rates** that automatically handle different gradient scales. Each parameter gets its own effective learning rate based on historical gradient statistics.

## Additional Findings

### 1. Bounds Checking for eps_0

Original varGP returns infinite loss when parameters exceed bounds. Implemented this in `m_step_lbfgs()`:
- Check eps_0x, eps_0y in [-0.99, 0.99]
- Set gradient to inf when bounds violated
- Return inf loss to force LBFGS to try smaller step

This prevents NaN propagation but doesn't fix the sigma_0 issue.

### 2. sigma_0 Initialization Experiment

With `sigma_0_init=5.0` (matching what Adam learns):
- LBFGS achieves r=0.79 (vs r=0.58 with sigma_0_init=1.0)
- Still worse than Adam's r=0.83, but much improved

This confirms sigma_0 is the key differentiator.

### 3. Original varGP Uses Analytical Gradients

The original `utils.py:varGP()` M-step:
- Uses `@torch.no_grad()` decorator
- Computes gradients analytically (not autograd)
- Sets gradients manually on parameters
- Uses LBFGS with strong_wolfe line search

The analytical gradients may have different scaling properties than autograd gradients.

## Possible Solutions

1. **Revert to Adam M-step** (recommended for now)
   - Works well (0.83-0.87 explained variance)
   - Simple, no changes needed

2. **Initialize sigma_0 larger when using LBFGS**
   - Partial fix: r=0.79 with sigma_0_init=5.0
   - Requires knowing the right initialization

3. **Use separate parameter groups with different LRs**
   - Give sigma_0 a much larger learning rate
   - More complex to tune

4. **Implement analytical gradients**
   - Match original varGP exactly
   - Most work, but might solve the issue

## Code Changes Made

### `estep.py`

Added `m_step_lbfgs()` function with:
- LBFGS optimizer (lr=0.1, strong_wolfe, history_size=100)
- Bounds checking for eps_0 parameters
- Gradient set to inf on bounds violation
- Debug mode for parameter tracking

Modified `train_varGP_style()` to call `m_step_lbfgs()` instead of `m_step()`.

**Current state:** LBFGS M-step is implemented but underperforms. Consider reverting to Adam.

## Test Commands

```bash
# LBFGS M-step (current)
python test_estep_pnas.py --mode vargp_style --ntilde 50 --save-plot none

# To test Adam M-step, temporarily patch in estep.py:
# Change line 633: m_step_lbfgs -> m_step
```

## Files Modified

- `estep.py`: Added `m_step_lbfgs()`, updated `train_varGP_style()`

## Grouped LBFGS Experiment (Failed)

Attempted block coordinate descent with 3 parameter groups:
1. sigma_0 (Adam, larger lr)
2. RF center eps_0x/eps_0y (LBFGS with bounds)
3. Other params (LBFGS)

**Result: FAILED**

The kernel matrix becomes non-positive-definite during LBFGS line search, regardless of:
- Learning rate settings
- Whether line search is enabled
- Which groups use LBFGS vs Adam

**Root cause:** LBFGS strong_wolfe line search explores parameter values that break K_tilde's positive-definiteness. The original varGP avoids this via:
1. Analytical gradients (not autograd)
2. Eigenspace projection (constrains variational params)
3. Returning inf loss on bounds violation (with analytical gradients set)

**Conclusion:** Grouped LBFGS doesn't work with autograd + GPyTorch's variational framework. Recommend using **Adam for M-step** which works well (0.83-0.87 explained variance).

## Recommendation

Revert `train_varGP_style()` to use `m_step()` (Adam) instead of `m_step_lbfgs()`.

The LBFGS M-step code is preserved in `m_step_lbfgs()` and `m_step_lbfgs_grouped()` for future investigation if analytical gradients are implemented.

## Date

2025-01-17
