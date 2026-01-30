# LogInterval Parameterization Attempt for A Parameter

**Date**: 2026-01-21
**Branch**: `pietro/lambda0-log-parameterization`
**Status**: ATTEMPTED - Prevented collapse but caused A to stick at lower bound
**Outcome**: Need alternative approach

---

## 1. Problem Statement

### Original Issue
The `vargp_style` training mode exhibits numerical instability for certain seeds. Specifically:
- **M=75, seed=42**: A parameter grows from 0.01 → 0.16, then collapses to 0.0004
- Firing rates become constant (std=0) at iteration 45
- Model enters degenerate state with no gradient signal

### Root Cause Analysis (from prior investigation)
The A parameter in the Poisson likelihood has no upper bound:
```python
# Original parameterization in likelihoods.py
A = exp(raw_A)  # raw_A unconstrained → A ∈ (0, ∞)
```

When A grows too large (~0.16), the firing rate `f = exp(A·λ + λ₀)` becomes unstable.

---

## 2. Proposed Solution: LogInterval Constraint

### Design Goals
1. **Bound A** to prevent explosion: A ∈ [A_min, A_max]
2. **Work in log space** for scale-invariant optimization
3. **Smooth gradients** everywhere (no hard boundaries)

### Mathematical Formulation

**Transform (raw → A):**
```
log_A = log(A_min) + (log(A_max) - log(A_min)) · σ(raw)
A = exp(log_A)
```

Where σ is the sigmoid function.

**Inverse transform (A → raw):**
```
log_A = log(A)
normalized = (log_A - log(A_min)) / (log(A_max) - log(A_min))
raw = logit(normalized) = log(normalized / (1 - normalized))
```

**Properties:**
- raw = -∞ → A = A_min
- raw = 0 → A = √(A_min · A_max) = geometric mean
- raw = +∞ → A = A_max

**Comparison with GPyTorch Interval (linear interpolation):**
```
# GPyTorch Interval:
A = A_min + (A_max - A_min) · σ(raw)
# At raw=0: A = (A_min + A_max) / 2 = arithmetic mean

# LogInterval:
A = exp(log(A_min) + (log(A_max) - log(A_min)) · σ(raw))
# At raw=0: A = √(A_min · A_max) = geometric mean
```

For A_min=0.001, A_max=0.15:
- Arithmetic mean = 0.0755 (too large for initialization)
- Geometric mean = 0.0122 (close to typical A_init=0.01)

---

## 3. Implementation

### 3.1 LogInterval Constraint Class

**File**: `likelihoods.py`

```python
class LogInterval(nn.Module):
    """Constraint that bounds A to [lower, upper] while working in log space."""

    def __init__(self, lower_bound=0.001, upper_bound=0.15):
        super().__init__()
        # Register bounds as buffers for device/dtype consistency
        dtype = torch.get_default_dtype()
        self.register_buffer("lower_bound", torch.as_tensor(lower_bound, dtype=dtype))
        self.register_buffer("upper_bound", torch.as_tensor(upper_bound, dtype=dtype))
        self.register_buffer("_log_lower", torch.as_tensor(math.log(lower_bound), dtype=dtype))
        self.register_buffer("_log_upper", torch.as_tensor(math.log(upper_bound), dtype=dtype))

    def transform(self, raw):
        """raw (unconstrained) → A (bounded)"""
        log_A = self._log_lower + (self._log_upper - self._log_lower) * torch.sigmoid(raw)
        return torch.exp(log_A)

    def inverse_transform(self, A):
        """A (bounded) → raw (unconstrained)"""
        A_clamped = A.clamp(min=lower * 1.001, max=upper * 0.999)
        log_A = torch.log(A_clamped)
        normalized = (log_A - self._log_lower) / (self._log_upper - self._log_lower)
        normalized = normalized.clamp(min=1e-6, max=1 - 1e-6)
        return torch.log(normalized / (1 - normalized))  # logit

    # Required by GPyTorch's register_constraint:
    @property
    def initial_value(self):
        return None

    @property
    def enforced(self):
        return True

    def check_raw(self, tensor):
        return self.check(self.transform(tensor))
```

### 3.2 Updated PoissonLikelihood

**Changes:**
1. Use `LogInterval` instead of `Positive` constraint for A
2. Change `lambda0` from Parameter to Buffer (it's computed analytically, not optimized)
3. Update default initializations to match varGP

```python
class PoissonLikelihood(Likelihood):
    def __init__(self, A_init=0.01, lambda0_init=1.0, A_min=0.001, A_max=0.15):
        super().__init__()

        # A parameter with LogInterval constraint
        self.register_parameter('raw_A', torch.nn.Parameter(torch.zeros(1)))
        self.register_constraint('raw_A', LogInterval(lower_bound=A_min, upper_bound=A_max))
        self.A = A_init  # Apply via property setter

        # lambda0 as BUFFER (not parameter) - computed analytically
        self.register_buffer('lambda0', torch.tensor([lambda0_init]))
```

---

## 4. Test Results

### 4.1 Unit Tests (PASSED)

**Command:**
```bash
conda run -n pytorch_gpytorch python likelihoods.py
```

**Results:**
```
=== Testing LogInterval constraint ===
raw -> A mapping:
  raw= -10.0 -> A=0.001000
  raw=  -2.0 -> A=0.001817
  raw=   0.0 -> A=0.012247
  raw=   2.0 -> A=0.082546
  raw=  10.0 -> A=0.149966

Geometric mean check: sqrt(0.001 * 0.15) = 0.012247
A at raw=0: 0.012247 ✓

Round-trip test (A -> raw -> A):
  A=0.0050 -> raw=-0.7482 -> A=0.005000 ✓
  A=0.0100 -> raw=-0.1622 -> A=0.010000 ✓
  A=0.0500 -> raw=1.2700 -> A=0.050000 ✓
  A=0.1000 -> raw=2.4299 -> A=0.100000 ✓

=== Testing PoissonLikelihood ===
Parameters: ['raw_A']
Buffers: ['lambda0', ...]  ✓ (lambda0 is buffer, not parameter)

A bounds enforcement:
  Set A=0.001 -> got A=0.001001 ✓
  Set A=0.15  -> got A=0.149850 ✓
  Set A=0.0001 (below) -> clamped to 0.001001 ✓
  Set A=0.5 (above) -> clamped to 0.149850 ✓
```

### 4.2 Baseline Test (BEFORE changes)

**Command:**
```bash
conda run -n pytorch_gpytorch python investigations/test_seed_stability.py \
    --seeds 123 456 42 --ntilde 75 --n-iterations 50
```

**Results (original parameterization):**
| Seed | Status | Final A | Final λ₀ | Test EV |
|------|--------|---------|----------|---------|
| 123 | OK | 0.092 | -1.93 | **0.825** |
| 456 | OK | 0.039 | -0.92 | **0.798** |
| 42 | **COLLAPSED** | 0.0004 | -0.02 | nan |

**Collapse trajectory for seed 42:**
```
Iter 37: A=0.1077, rates normal
Iter 43: A=0.1614 (PEAK), loss=531
Iter 45: A=0.1454, rates constant (std=0) ← COLLAPSE
Iter 50: A=0.0004, model degenerate
```

### 4.3 Test WITH LogInterval (AFTER changes)

**Command:**
```bash
conda run -n pytorch_gpytorch python investigations/test_seed_stability.py \
    --seeds 123 456 42 --ntilde 75 --n-iterations 50
```

**Results (LogInterval parameterization):**
| Seed | Status | Final A | Final λ₀ | Test EV |
|------|--------|---------|----------|---------|
| 123 | OK | **0.001** | -0.23 | 0.698 |
| 456 | OK | **0.001** | -0.23 | 0.689 |
| 42 | OK | **0.001** | -0.24 | 0.697 |

**Observation:**
- No collapse for any seed (including seed 42)
- But A collapsed to **lower bound** (0.001) for ALL seeds
- Test performance degraded: 0.70 vs 0.80-0.83

---

## 5. Analysis: Why A Collapses to Lower Bound

### 5.1 Gradient Through Sigmoid

The gradient of A with respect to raw is:
```
dA/d(raw) = A · (log(A_max) - log(A_min)) · σ(raw) · (1 - σ(raw))
```

At raw=0 (initialization):
- σ(0) = 0.5
- σ(0) · (1 - σ(0)) = 0.25
- dA/d(raw) = 0.0122 · 5.01 · 0.25 = 0.0153

Compare to original exponential:
```
A = exp(raw_A)
dA/d(raw_A) = A = 0.01
```

The gradients are similar in magnitude, so this isn't the primary issue.

### 5.2 Sigmoid Saturation Near Bounds

The problem is likely **sigmoid saturation**. When A approaches the bounds:
- Near A_min: σ(raw) → 0, gradient → 0
- Near A_max: σ(raw) → 1, gradient → 0

Once A gets pushed toward A_min (even slightly), the gradient diminishes and A gets "stuck".

### 5.3 F-step Dynamics

The F-step uses LBFGS to optimize A. The loss landscape in the transformed space may have:
1. A basin of attraction toward A_min
2. Flat regions near bounds due to sigmoid saturation
3. Different curvature than the original exp parameterization

### 5.4 Interaction with Analytical λ₀

The analytical formula for λ₀ is:
```
λ₀ = log(Σr) - log(Σ exp(A·λ_m + 0.5·A²·λ_var))
```

When A is small:
- The exponential term ≈ 1 for all samples
- λ₀ ≈ log(Σr) - log(N) = log(mean(r))
- All predictions become similar → low correlation

This creates a feedback loop: small A → poor predictions → optimizer reduces A further.

---

## 6. Possible Alternative Approaches

### 6.1 Clamped Exponential (Hard Bounds)
```python
A = exp(raw_A).clamp(min=A_min, max=A_max)
```
- Preserves original gradient dynamics in valid range
- Hard boundary may cause gradient discontinuity

### 6.2 Softplus with Shift
```python
A = A_min + softplus(raw_A)  # with optional clamping at A_max
```
- More similar to original exp behavior
- Only lower bound is "soft"

### 6.3 Scaled Sigmoid (Different Initialization)
```python
# Initialize raw so that A starts at desired value, not geometric mean
raw_init = logit((log(A_init) - log(A_min)) / (log(A_max) - log(A_min)))
```
- Keep LogInterval but fix initialization

### 6.4 Gradient Clipping
```python
torch.nn.utils.clip_grad_norm_(likelihood.parameters(), max_norm=10.0)
```
- Don't change parameterization
- Prevent large gradient steps that cause A explosion

### 6.5 Regularization on A
```python
loss = -ELL + KL + λ_reg * (A - A_target)²
```
- Soft constraint to keep A near reasonable values
- Doesn't prevent explosion but discourages it

---

## 7. Files Modified

| File | Change |
|------|--------|
| `likelihoods.py` | Added `LogInterval` class, updated `PoissonLikelihood` |
| `investigations/test_seed_stability.py` | Created diagnostic script |

---

## 8. Commands to Reproduce

### Run unit tests:
```bash
cd /home/idv-eqs8-pza/IDV_code/ClosedLoopProject/gaussian_processes/Spatial_GP_repo/scripts/gpytorch_porting
conda run -n pytorch_gpytorch python likelihoods.py
```

### Run stability test (with LogInterval):
```bash
conda run -n pytorch_gpytorch python investigations/test_seed_stability.py \
    --seeds 123 456 42 --ntilde 75 --n-iterations 50
```

### To revert to original parameterization:
```bash
git checkout HEAD~1 -- likelihoods.py
```

---

## 9. Conclusion

**What worked:**
- LogInterval constraint successfully prevented A from exploding
- Seed 42 no longer collapses at M=75
- lambda0 correctly changed from Parameter to Buffer

**What didn't work:**
- A collapsed to lower bound instead of finding optimal value
- Test performance degraded from ~0.80 to ~0.70
- Sigmoid saturation creates "sticky" bounds

**Recommendation:**
Try alternative approaches (Section 6), particularly:
1. Gradient clipping (simplest, no parameterization change)
2. Clamped exponential (preserves gradient dynamics)
3. Better initialization for LogInterval

---

*Document created by Claude, 2026-01-21*
