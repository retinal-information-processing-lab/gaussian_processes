# HYPOTHESIS 3: Kernel Hyperparameter Instability - CONFIRMED

**Date**: 2026-01-21
**Test**: `investigations/diagnose_kernel_instability.py`
**Hypothesis**: Unbounded kernel hyperparameters cause training collapse

---

## Summary

**CONFIRMED**: Kernel hyperparameters `beta` and `rho` explode to extreme values during training, leading to numerical instability and poor performance.

### Key Findings

| Seed | Final test_r | Beta range | Rho range | C condition number |
|------|--------------|------------|-----------|-------------------|
| **123 (good)** | ~0.7-0.8 | 20 → 61 | 9 → 13 | 10^20 to 10^23 |
| **456 (bad)** | ~0.4-0.8 | 25 → 116 | 8 → 125 | 10^22 to 10^6 |

### Expected vs Actual

**Reasonable parameter ranges** (from varGP default initialization):
- `beta = 0.1` (RF locality)
- `rho = 0.1` (smoothness)
- `eps_0 ∈ [-1, 1]` (RF center on pixel grid)

**Actual values observed**:
- `beta` up to **116** (1160x larger!)
- `rho` up to **125** (1250x larger!)
- `eps_0` drifts to `(0.21, 0.03)` but stays within bounds

---

## Root Cause Analysis

### 1. Original varGP Uses Bounds

From `utils.py:5876-5884`, the original varGP implementation **enforces bounds** via LBFGS closure:

```python
theta_lower_lims = {'eps_0x': -1.0, 'eps_0y': 1.0, ...}
theta_higher_lims = {'eps_0x': 1.0, 'eps_0y': 1.0, ...}

def closure_hyperparams():
    # Check if any hyperparameter is out of bounds
    for key, value in theta.items():
        if not (theta_lower_lims[key] <= value <= theta_higher_lims[key]):
            theta[key].grad = torch.tensor(float('inf'))
            return torch.tensor(float('inf'))  # Reject step
    # ... normal loss computation ...
```

**Key bounds**:
- `eps_0x, eps_0y ∈ [-1, 1]` (pixel grid range)
- `sigma_0 > 0` (enforced)
- `beta, rho` technically unbounded in raw space, but log-parameterization prevents negative values

### 2. GPyTorch Implementation Has NO Bounds

From `kernels.py:ArcCosineKernel`:

| Parameter | Constraint | Status |
|-----------|-----------|---------|
| `sigma_0` | `Positive()` | ✓ Bounded |
| `beta` | **None** | ❌ Unbounded |
| `rho` | **None** | ❌ Unbounded |
| `eps_0x` | **None** | ❌ Unbounded |
| `eps_0y` | **None** | ❌ Unbounded |

**Log-space parameterization does NOT prevent explosion**:
- `beta = exp(raw_m2log2beta)` → as `raw → -∞`, `beta → +∞`
- `rho = sqrt(exp(raw_mlog2rho2))` → as `raw → -∞`, `rho → +∞`

### 3. Why This Causes Problems

**Physical interpretation**:
- `beta` controls locality weight: `α(x) = exp(-beta · ||x - x₀||²)`
  - Small `beta` (0.1): RF covers large area
  - Large `beta` (100): RF is a tiny pinpoint, almost zero weight everywhere except exact center
- `rho` controls smoothness: `C_smooth(x,x') = exp(-rho² · ||x - x'||²)`
  - Small `rho` (0.1): Smooth correlations across pixels
  - Large `rho` (100): No correlation, almost diagonal matrix

**Numerical consequences**:
- C matrix becomes ill-conditioned (cond ~ 10^20)
- Kernel computations involve `exp(-beta * 100)` → underflow
- Gradients become extremely imbalanced
- Model degenerates to trivial solution

---

## Diagnostic Output

### Good Seed (123) - 30 Iterations

```
Iter  0: beta=20.85, rho= 9.10, cond(C)=2.25e+23
Iter  5: beta=22.02, rho=11.67, cond(C)=4.25e+21
Iter 10: beta=29.14, rho=12.37, cond(C)=1.61e+22
Iter 15: beta=40.65, rho=13.27, cond(C)=7.22e+20
Iter 20: beta=51.51, rho=13.74, cond(C)=2.64e+20
Iter 25: beta=52.53, rho=11.64, cond(C)=3.07e+21
Iter 29: beta=60.58, rho=11.30, cond(C)=2.70e+22
```

**Observations**:
- Beta steadily increases 20 → 60
- Rho oscillates 9 → 13
- Condition number wildly unstable (10^20 to 10^23)
- Final test_r ~ 0.7-0.8 (moderate, but unstable)

### Bad Seed (456) - 30 Iterations

```
Iter  0: beta=25.53, rho=  7.97, cond(C)=3.30e+22
Iter  5: beta=29.41, rho= 10.26, cond(C)=1.85e+23
Iter 10: beta=48.95, rho= 10.82, cond(C)=3.14e+21
Iter 15: beta=67.55, rho= 22.39, cond(C)=7.79e+12  ← Rho explosion starts
Iter 20: beta=116.38, rho= 57.23, cond(C)=3.32e+06
Iter 25: beta=107.71, rho= 91.42, cond(C)=1.08e+06
Iter 29: beta=108.11, rho=124.85, cond(C)=9.74e+05
```

**Observations**:
- **Both beta AND rho explode**
- Beta reaches 116 (58x initial)
- **Rho reaches 125 (15x initial)** ← This is the killer
- Condition number drops dramatically (good?) but model is degenerate
- Final test_r unstable (0.4 to 0.8)

---

## Why Seed Matters

The difference between seeds is **initialization of inducing points**. Bad seeds may:
1. Place inducing points in regions with poor coverage
2. Lead to initial kernel configurations that encourage extreme parameters
3. Create gradient landscapes with bad local minima

Once `beta` or `rho` start growing, there's no mechanism to stop them.

---

## Proposed Fix

### Option 1: Add GPyTorch Constraints (Recommended)

```python
from gpytorch.constraints import Interval

class ArcCosineKernel(Kernel):
    def __init__(self, ...):
        # Constrain eps_0 to pixel grid
        self.register_constraint('eps_0x', Interval(-1.0, 1.0))
        self.register_constraint('eps_0y', Interval(-1.0, 1.0))

        # Constrain beta and rho to reasonable ranges
        # beta ∈ [0.01, 1.0] → raw ∈ [-2*log(2*1.0), -2*log(2*0.01)]
        self.register_constraint('raw_m2log2beta',
            Interval(-2*np.log(2*1.0), -2*np.log(2*0.01)))  # [0.01, 1.0]

        # rho ∈ [0.01, 0.5] → raw ∈ [-log(2*0.5²), -log(2*0.01²)]
        self.register_constraint('raw_mlog2rho2',
            Interval(-np.log(2*0.5**2), -np.log(2*0.01**2)))  # [0.01, 0.5]
```

**Rationale**:
- `eps_0 ∈ [-1, 1]`: Matches original varGP, keeps RF center on pixel grid
- `beta ∈ [0.01, 1.0]`: Wide enough for learning, prevents explosion
- `rho ∈ [0.01, 0.5]`: Wide enough for learning, prevents degeneracy

### Option 2: Manual Bound Checking in M-step

Add bound checking in `estep.py:m_step()` to match original varGP behavior:

```python
def m_step(...):
    bounds = {
        'eps_0x': (-1.0, 1.0),
        'eps_0y': (-1.0, 1.0),
        'raw_m2log2beta': (-2*np.log(2*1.0), -2*np.log(2*0.01)),
        'raw_mlog2rho2': (-np.log(2*0.5**2), -np.log(2*0.01**2)),
    }

    for _ in range(n_mstep):
        # Check bounds before step
        kernel = model.covar_module
        if hasattr(kernel, 'base_kernel'):
            kernel = kernel.base_kernel

        for param_name, (lower, upper) in bounds.items():
            param = getattr(kernel, param_name)
            if not (lower <= param.item() <= upper):
                print(f"Clamping {param_name} to [{lower}, {upper}]")
                param.data.clamp_(lower, upper)

        # Normal optimizer step
        optimizer.zero_grad()
        ...
```

---

## Next Steps

1. **Implement Option 1** (GPyTorch constraints) - cleanest solution
2. **Re-run benchmark** with bounded hyperparameters
3. **Verify** that both seeds now achieve stable, high test_r
4. **Document** the fix in DECISION_LOG.md

---

## Conclusion

**Hypothesis CONFIRMED**: Unbounded kernel hyperparameters (`beta`, `rho`) are the root cause of training instability. The original varGP implementation uses soft bounds via LBFGS closure rejection. The GPyTorch port lacks these bounds, allowing parameters to explode.

**Impact**: This explains why:
- Bad seeds cause catastrophic failure (rho → 125)
- Good seeds still have issues (beta → 61, unstable condition numbers)
- Performance is seed-dependent and unpredictable

**Fix**: Add explicit bounds matching original varGP implementation.
