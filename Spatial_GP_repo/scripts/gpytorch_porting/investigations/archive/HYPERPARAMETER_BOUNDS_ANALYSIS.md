# Kernel Hyperparameter Bounds Analysis

**Created**: 2026-01-21
**Hypothesis**: Unbounded kernel hyperparameters may cause training collapse on bad seeds

---

## Current Implementation in GPyTorch Port

### Hyperparameters in `kernels.py:ArcCosineKernel`

| Parameter | Raw Space | Natural Space | Transform | Constraint |
|-----------|-----------|---------------|-----------|------------|
| `raw_sigma_0` | Unconstrained | `sigma_0` (positive) | `softplus` | `Positive()` ✓ |
| `raw_m2log2beta` | Unconstrained | `beta = exp(raw/2)/2` | Exponential | **None** ❌ |
| `raw_mlog2rho2` | Unconstrained | `rho = sqrt(exp(raw)/2)` | Exponential | **None** ❌ |
| `eps_0x` | **Direct** | `eps_0x` | Identity | **None** ❌ |
| `eps_0y` | **Direct** | `eps_0y` | Identity | **None** ❌ |

**Key findings**:
1. Only `sigma_0` has a GPyTorch constraint (`Positive()`)
2. `beta` and `rho` use log-space parameterization but **no explicit bounds**
3. `eps_0x` and `eps_0y` are **directly optimized** (not reparameterized) with **no bounds**

### What the Transforms Do

**Intended behavior** (from code comments and math):
- `beta = 0.1` → `raw_m2log2beta = -2*log(2*0.1) ≈ 3.22`
  - Transform: `beta = exp(-raw/2) / 2`
  - This is supposed to keep `beta > 0`, but there's NO upper bound
- `rho = 0.1` → `raw_mlog2rho2 = -log(2*0.1²) ≈ 3.91`
  - Transform: `rho = sqrt(exp(-raw) / 2)`
  - Again, only ensures `rho > 0`, no upper bound

**Problem**: Without upper bounds:
- `raw_m2log2beta → -∞` makes `beta → +∞` (locality vanishes)
- `raw_mlog2rho2 → -∞` makes `rho → +∞` (smoothness vanishes)
- `eps_0x, eps_0y` can drift arbitrarily far from `[-1, 1]` grid

---

## Original varGP Implementation (utils.py:5876-5884)

### How Bounds are Enforced

The original code uses **LBFGS with manual bound checking** in the closure:

```python
# From utils.py line 3540-3541
theta_lower_lims  = {'sigma_0': 0, 'eps_0x': low_lim, 'eps_0y': low_lim,
                     '-2log2beta': -inf, '-log2rho2': -inf, 'Amp': 0}
theta_higher_lims = {'sigma_0': inf, 'eps_0x': upp_lim, 'eps_0y': upp_lim,
                     '-2log2beta': inf, '-log2rho2': inf, 'Amp': inf}
```

Where `low_lim = -1.0`, `upp_lim = 1.0` (from line 3559-3560).

**Enforcement in M-step closure** (lines 5876-5884):
```python
def closure_hyperparams():
    # Check if any hyperparameter is out of bounds
    return_infinite_loss = False
    for key, value in theta.items():
        if not (theta_lower_lims[key] <= value <= theta_higher_lims[key]):
            return_infinite_loss = True
            print(f"{key} = {value:.4f} is not within limits")
            if theta[key].requires_grad:
                theta[key].grad = torch.tensor(float('inf'))
    if return_infinite_loss:
        return torch.tensor(float('inf'))

    # ... normal loss computation ...
```

**Key insight**: varGP returns `inf` loss if bounds are violated. LBFGS's line search will reject the step and backtrack.

### Actual Bounds Used

| Parameter | Lower | Upper | Notes |
|-----------|-------|-------|-------|
| `sigma_0` | 0 | ∞ | Same as GPyTorch |
| `eps_0x` | **-1.0** | **+1.0** | **BOUNDED** (pixel grid range) |
| `eps_0y` | **-1.0** | **+1.0** | **BOUNDED** (pixel grid range) |
| `-2log2beta` | -∞ | ∞ | Unbounded in raw space |
| `-log2rho2` | -∞ | ∞ | Unbounded in raw space |

**Critical difference**: `eps_0` is bounded to `[-1, 1]` in varGP but **unbounded** in GPyTorch!

---

## Why This Matters

### 1. RF Center Can Drift Off-Grid

In varGP, `eps_0` is constrained to the `[-1, 1]` pixel grid. In GPyTorch, it can drift to arbitrary values like `(10.5, -7.2)`, which is:
- Far outside the image domain
- Makes the locality mask meaningless
- Can cause numerical issues in `exp(-beta * dist²)`

### 2. Beta and Rho Can Explode

While the log-space parameterization prevents `beta, rho < 0`, there's nothing stopping:
- `beta → 1000` (extreme localization, near-zero locality weights everywhere except RF center)
- `rho → 100` (no smoothness, C becomes nearly diagonal)

### 3. Gradients Can Be Unbalanced

From the LBFGS M-step investigation (`.claude/LBFGS_MSTEP_INVESTIGATION.md`):
> "sigma_0 gradient ~1000x smaller than others"

Without bounds, LBFGS/Adam might:
- Follow large gradients blindly
- Ignore small gradients even if they matter
- Lead to divergence on bad seeds

---

## Proposed Fix

### Option 1: Add GPyTorch Constraints (Cleanest)

```python
from gpytorch.constraints import Interval

class ArcCosineKernel(Kernel):
    def __init__(self, ...):
        # Constrain eps_0 to pixel grid
        self.register_constraint('eps_0x', Interval(-1.0, 1.0))
        self.register_constraint('eps_0y', Interval(-1.0, 1.0))

        # Constrain beta and rho to reasonable ranges
        # beta ∈ [0.01, 1.0], rho ∈ [0.01, 0.5]
        self.register_constraint('raw_m2log2beta',
            Interval(-2*np.log(2*1.0), -2*np.log(2*0.01)))
        self.register_constraint('raw_mlog2rho2',
            Interval(-np.log(2*0.5**2), -np.log(2*0.01**2)))
```

**Pros**:
- Clean, matches GPyTorch patterns
- Automatic constraint transforms
- No manual bound checking

**Cons**:
- Requires understanding GPyTorch constraint system
- May interact weirdly with transforms

### Option 2: Manual Bound Checking in M-step (Matches varGP)

Add bounds to `estep.py:m_step()`:

```python
def m_step(...):
    # Define bounds (matching varGP)
    bounds = {
        'eps_0x': (-1.0, 1.0),
        'eps_0y': (-1.0, 1.0),
        # Could add beta, rho bounds if needed
    }

    def closure():
        # Check bounds
        for param_name, (lower, upper) in bounds.items():
            param = getattr(kernel, param_name)
            if not (lower <= param.item() <= upper):
                print(f"Bound violation: {param_name}={param.item()}")
                return torch.tensor(float('inf'))

        # Normal loss computation
        ...
```

**Pros**:
- Exact match with varGP behavior
- Easy to understand and debug

**Cons**:
- Manual, not leveraging GPyTorch infrastructure
- Requires modifying training loop

### Option 3: Soft Constraints via Penalty

Add penalty term to loss when approaching bounds:

```python
loss = -elbo
# Add soft penalties
if abs(kernel.eps_0x) > 0.9:
    loss += 1000 * (abs(kernel.eps_0x) - 0.9)**2
if abs(kernel.eps_0y) > 0.9:
    loss += 1000 * (abs(kernel.eps_0y) - 0.9)**2
```

**Pros**:
- Gradual, smooth enforcement
- Works with any optimizer

**Cons**:
- Hyperparameter tuning (penalty strength)
- Not a hard constraint

---

## Next Steps

1. **Run diagnostic**: `python investigations/diagnose_kernel_instability.py --seed 123 --ntilde 50`
2. **Run with bad seed**: `python investigations/diagnose_kernel_instability.py --seed 456 --ntilde 50`
3. **Compare trajectories**:
   - Do `eps_0x, eps_0y` drift beyond `[-1, 1]` on bad seeds?
   - Do `beta, rho` hit extreme values?
   - Are gradients unusually large before collapse?
4. **Test fix**: Implement Option 1 or 2 and rerun

---

## Expected Diagnostic Results

**Good seed (123)**:
- `eps_0` stays near `[-0.2, 0.2]` (reasonable RF center)
- `beta` stays near `0.1 ± 0.05` (reasonable locality)
- `rho` stays near `0.1 ± 0.05` (reasonable smoothness)
- Gradients moderate (`<1e-2` in natural space)
- Condition numbers stable (`cond(C) < 1e4`)

**Bad seed (456)** - if hypothesis is correct:
- `eps_0` drifts beyond `[-1, 1]` (off-grid RF)
- OR `beta → 0` or `beta → ∞` (degenerate locality)
- OR `rho → ∞` (no smoothness)
- Large gradients spike before collapse
- Condition numbers explode (`cond(C) > 1e6`)

If we see these patterns, adding bounds (especially for `eps_0`) should fix the issue.
