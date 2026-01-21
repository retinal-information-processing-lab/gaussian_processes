# Session Handoff: Amp Parameter Implementation

**Created**: 2026-01-21
**Branch**: `bugfix/amp-parameter-mismatch` (already checked out)
**Plan file**: `/home/idv-eqs8-pza/.claude/plans/quiet-wobbling-toast.md`

---

## TASK SUMMARY

Implement the `Amp` parameter in `ArcCosineKernel` to match the legacy varGP implementation exactly. This fixes a **mathematical mismatch** between the two implementations.

---

## THE PROBLEM (CRITICAL TO UNDERSTAND)

### Current GPyTorch Implementation (WRONG)
```python
# In kernels.py _compute_C_matrix():
C = alpha[:, None] * C_smooth * alpha[None, :]  # No Amp!

# Then ScaleKernel wraps the kernel:
kernel = gpytorch.kernels.ScaleKernel(base_kernel)
kernel.outputscale = 1e-4  # Linear output scaling
# Result: K_scaled = outputscale * K_base
```

### Legacy varGP Implementation (CORRECT)
```python
# In utils.py localker() line 3612:
C = theta['Amp'] * alpha_local[:, None] * C_smooth * alpha_local[None, :]
```

### WHY THESE ARE DIFFERENT

In the arc-cosine kernel (see `utils.py:acosker()` lines 3702-3714):
```python
X1 = sqrt(sum(x1 * (C @ x1)) + sigma_0²)  # Amp is INSIDE sqrt
X2 = sqrt(sum(x2 * (C @ x2)) + sigma_0²)
x1x2 = x1.T @ C @ x2 + sigma_0²           # Amp affects cross-term
cosdelta = x1x2 / (X1 * X2)               # Amp affects angle
K = X1 * X2 * J(arccos(cosdelta))         # Non-linear dependency
```

With `C = Amp * C_base`:
- `v_x = Amp * (x^T C_base x) + σ₀²` — Amp is inside the sqrt and arccos
- The effect is **non-linear** through the angle computation

With `ScaleKernel`:
- `K_scaled = outputscale * K_base` — Linear output scaling only
- Does NOT affect the internal angle computation

**These are mathematically different. The fix is to add Amp directly to C.**

---

## GRADIENT FORMULA

From `utils.py:localker()` line 3619:
```python
dC_Amp = C / theta['Amp']
```

Since `C = Amp * C_base`, then `dC/dAmp = C_base = C/Amp`.

For the VJP backward pass:
```python
grad_Amp = (dL_dC * C).sum() / Amp_val
```

---

## IMPLEMENTATION REQUIREMENTS

### 1. kernels.py

**Add Amp parameter with Positive() constraint** (same pattern as sigma_0):
- Register `raw_Amp` parameter
- Add `Positive()` constraint
- Add property getter/setter
- Clamp max at 1000.0 in `clamp_hyperparameters()`
- Multiply C by Amp in `_compute_C_matrix()`
- Pass Amp to analytical gradient functions in `forward()`

**Key code pattern to follow** (from existing sigma_0):
```python
# Registration (~line 140)
self.register_parameter('raw_sigma_0', ...)
self.register_constraint('raw_sigma_0', Positive())

# Property (~line 190)
@property
def sigma_0(self):
    return self.raw_sigma_0_constraint.transform(self.raw_sigma_0)
```

### 2. analytical_gradients_vjp.py

**Update forward signature** to include Amp:
```python
def forward(ctx, x1, x2, sigma_0, Amp, eps_0x, eps_0y, raw_m2log2beta, raw_mlog2rho2,
            n_px_side, use_mask, diag):
```

**Update C computation** (~line 111):
```python
C = Amp_val * alpha[:, None] * S * alpha[None, :]
```

**Add backward gradient** (~after line 310):
```python
grad_Amp = (dL_dC * C).sum() / Amp_val
```

**Update return tuple** to include grad_Amp.

### 3. analytical_gradients.py

Same changes as VJP for Jacobian mode consistency.

### 4. run_single_mode.py

**Remove ScaleKernel** (~lines 359-360):
```python
# BEFORE:
kernel = gpytorch.kernels.ScaleKernel(base_kernel)
kernel.outputscale = 1e-4

# AFTER:
kernel = base_kernel
kernel.Amp = 1e-4
```

**Add final_Amp to JSON output** (~line 618).

### 5. run_benchmark.py

Same ScaleKernel removal at lines ~293, ~390, ~497.

### 6. estep.py

Remove ScaleKernel wrapper handling (~line 1170, 1217).

### 7. Test files

Update all files using ScaleKernel:
- `tests/test_kernel_cache.py`
- `tests/test_m_whitening.py`
- `tests/test_mask_validation.py`
- `tests/test_reference_comparison.py`
- `tests/test_analytical_gradients.py`
- `tests/diagnose_unwhitened_performance.py`
- `benchmark_whitening_modes.py`
- `investigations/*.py`

---

## TEST PLAN

### Phase 1: Unit Tests
1. **Kernel matrix comparison**: GPyTorch K vs legacy `acosker()` with identical params → max diff < 1e-10
2. **Amp gradient test**: VJP vs autograd vs finite differences → rel error < 1e-5
3. **Amp clamping**: Set Amp=5000, clamp → verify Amp=1000

### Phase 2: Integration
4. **Single run**: `python run_single_mode.py --mode vargp_style --ntilde 50 --seed 123 --json-append results/amp_validation.jsonl`
   - Verify JSON contains `final_Amp`
   - Verify `test_r > 0.7`

5. **Legacy comparison**: `python run_benchmark.py --ntilde 50 --seed 123`
   - GPyTorch vs varGP difference < 0.05

### Phase 3: Regression
6. **Canonical matrix**: `python run_canonical_tests.py --seed 123 --output results/amp_regression.jsonl`
   - All 12 configs complete without error

---

## KEY FILES TO READ

| File | What to look at |
|------|-----------------|
| `kernels.py` | sigma_0 registration pattern (lines 140-200), `_compute_C_matrix()` (line 346), `forward()` gradient call (line 378) |
| `analytical_gradients_vjp.py` | Full file - forward and backward structure |
| `utils.py:localker()` | Lines 3612 (C formula), 3619 (dC/dAmp) |
| `utils.py:acosker()` | Lines 3702-3714 (kernel computation using C) |

---

## VERIFICATION CHECKLIST

After implementation:
1. [ ] `python -c "from kernels import ArcCosineKernel; k = ArcCosineKernel(Amp=1e-4, n_px_side=108); print(k.Amp)"` → prints `0.0001`
2. [ ] `python tests/test_analytical_gradients.py` passes
3. [ ] `python run_single_mode.py --mode vargp_style --ntilde 50 --seed 123` → test_r > 0.7
4. [ ] JSON output contains `final_Amp` field
5. [ ] `python run_benchmark.py --ntilde 50` → GPyTorch comparable to varGP

---

## IMPORTANT NOTES

- **NO log-space parameterization** for Amp. Use `Positive()` constraint directly (like sigma_0).
- **Clamp Amp at max 1000.0** in `clamp_hyperparameters()`.
- **Default Amp = 1.0** for backward compatibility.
- **Initialize Amp = 1e-4** in scripts (matching previous outputscale).
- **Activate conda environment**: `pytorch_gpytorch`
- **GPU required** for reasonable performance.

---

## CURRENT STATE

- [x] Branch created: `bugfix/amp-parameter-mismatch`
- [x] Plan written: `/home/idv-eqs8-pza/.claude/plans/quiet-wobbling-toast.md`
- [x] Mathematical analysis complete
- [ ] Implementation pending

**Start implementation with `kernels.py`**, then propagate changes outward.
