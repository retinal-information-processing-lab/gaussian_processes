# Investigation: LBFGS Closure NaN Guard for default_gpy Training

**Branch**: `pietro/arcsine-kernel`
**Date**: 2026-02-10
**Status**: Continuing
**Location**: `investigations/arcsine_kernel/` (the bug was found during arc-sine kernel work)

---

## Problem Statement

When running bounded-diagonal kernels (ArcSineKernel, ArcCosineKernelNormalized) with M=50, n_train=50, the entire K_uu matrix becomes NaN during the first LBFGS iteration:

```
NanError: cholesky_cpu: 2500 of 2500 elements of the torch.Size([50, 50]) tensor are NaN.
```

This is NOT the usual "Cholesky failed after adding jitter." The error is `NanError` — the kernel itself returns all NaN. The arc-cosine kernel at the same settings works fine (test_r=0.6819). The normalized kernel had the same failure (SESSION_LOG 2026-02-09: "M=50 causes all-NaN Cholesky with normalized kernel").

The root cause is that `gpy_training.py`'s LBFGS closure had zero protection against NaN or parameter overflow, unlike `eigenspace_mstep.py` which has multiple guardrails (lines 78-82, 98-99, 115-116, 304-306) plus `clamp_hyperparameters()` after each step (lines 132, 337).

## What Was Tried

### 1. NaN diagnostic in closure (catch exception from model call)
- **What**: Added try/except around `model(train_x)` in the closure to dump parameter values when NanError is thrown.
- **Result**: ALL parameters were already NaN (raw_sigma_0=nan, raw_Amp=nan, etc.), meaning the NaN entered BEFORE the failing closure call.
- **Interpretation**: The NaN propagates through gradients from a PREVIOUS closure call, then LBFGS applies NaN step, making all params NaN.
- **Verdict**: Confirmed the chain but didn't identify the root — needed to catch EARLIER.

### 2. Gradient NaN diagnostic after backward()
- **What**: Added NaN check on all parameter gradients after `loss.backward()`, printing parameter values when NaN gradients detected.
- **Result**: Clear progression visible. First problematic closure call (already a line search trial point, not initial params):
  ```
  loss = nan, ell = nan, kl = 3218745.5
  sigma_0=0.867452, Amp=11.7333
  raw_m2log2beta=186.837, raw_mlog2rho2=168.518
  eps_0x=-150.417, eps_0y=-66.4099
  ```
  Then exponential divergence: raw_m2log2beta goes 186 -> 1013 -> 5563 -> 30608 -> 168456 -> 927182 -> ... until all params become NaN.
- **Interpretation**: The first LBFGS line search step pushes parameters far from initial values. At `raw_m2log2beta=186.8`, `beta_factor = exp(186.8) = inf` in float32 (max is exp(88.7)). This collapses C to zero, making the kernel degenerate. The ELL becomes NaN (degenerate GP predictions under Poisson likelihood). LBFGS doesn't detect NaN loss and keeps escalating.
- **Verdict**: Root cause confirmed. LBFGS has no NaN awareness — it keeps calling the closure with increasingly extreme parameters.

### 3. Two-layer fix (matching eigenspace_mstep.py pattern)
- **What**: Added two protections to `gpy_training.py`:
  - **Layer 1 (inside closure)**: try/except around `model(train_x)` returning `float('inf')` on exception; NaN/Inf check on loss returning `float('inf')`.
  - **Layer 2 (after optimizer.step())**: `clamp_hyperparameters()` call with warning if any parameter is actually clamped.
- **Result**: Arc-sine at M=50 now completes (test_r=0.4623). Arc-cosine at M=50 shows no regression (test_r=0.6815 vs 0.6819 before). Normalized kernel at M=50 also works (test_r=0.4342).
- **Verdict**: Fix works. But needs further analysis (see below).

## Key Findings

1. **CONFIRMED**: `gpy_training.py` closure had ZERO NaN/error protection. No try/except, no loss check. Compare with `eigenspace_mstep.py` which has 4 separate guardrails returning `float('inf')` plus `clamp_hyperparameters()` after step.

2. **CONFIRMED**: The NaN chain is: initial gradients cause LBFGS to try extreme parameters during line search -> `raw_m2log2beta` overflows float32 (exp(186) = inf) -> C matrix collapses -> degenerate kernel -> NaN ELL -> NaN loss -> NaN gradients -> LBFGS propagates NaN to ALL parameters -> next closure call crashes with NanError in Cholesky.

3. **CONFIRMED**: Arc-cosine kernel is NOT immune to this bug — it just has gentler gradients so LBFGS doesn't push as hard on the first step. The vulnerability exists for all kernels.

4. **CONFIRMED**: The NaN guard inside the closure (return inf on NaN loss) is sufficient to prevent the crash. `clamp_hyperparameters()` after step did NOT fire its warning in any test — the inf guard catches everything before parameters are committed.

5. **HYPOTHESIS**: The reason bounded-diagonal kernels trigger this while arc-cosine doesn't is that their gradient landscape has steeper regions (arcsin gradient diverges near +/-1), causing larger initial gradients, causing LBFGS to try more extreme first steps.

6. **OPEN QUESTION**: `clamp_hyperparameters()` is called after `optimizer.step()` as a safety net. The user raised a valid concern: if the inf guard works, the clamp should never fire. If it does fire, that's a sign the guard is insufficient and we'd want to know via warning (now implemented). But: the clamp in `eigenspace_mstep.py` (lines 132, 337) has been there since the beginning — should it also get a warning? Or should it be removed entirely?

7. **OPEN QUESTION**: The current fix returns `float('inf')` without calling `backward()`. This means LBFGS gets stale gradients (from the previous successful closure call). Strong_wolfe should still work because inf violates sufficient decrease, but this hasn't been rigorously verified. The `eigenspace_mstep.py` closure does the same thing (returns inf without setting gradients), so it's at least consistent.

## Why This Was Stopped

Context ran out. The immediate fix is in place and verified. Next session should:
1. Decide on the final form of the fix (the current implementation is functional but the design choices around clamp_hyperparameters need discussion)
2. Run the arc-cosine kernel through the same test suite to verify no regression across all standard configurations
3. Potentially audit the eigenspace_mstep.py closure for consistency with the new gpy_training.py pattern

## Things Noticed But Not Acted Upon

1. `eigenspace_mstep.py` has a change (ArcSineKernel guard in analytical M-step) that is from a previous uncommitted session — not from this investigation.
2. The arc-sine kernel at M=50 gives test_r=0.4623, significantly worse than M=100 (0.7653). All images are in the saturated regime (K_sat > 0.93). M=100 is the minimum useful size for bounded kernels.
3. `.claude/rules/working_guidelines.md` and `.claude/settings.json` have uncommitted changes from a previous session — not related to this investigation.
4. The normalized kernel is superseded by the arc-sine kernel and can be ignored going forward.

## Uncommitted Changes

**Modified (this session's work):**
| File | Change |
|------|--------|
| `gpy_training.py` | NaN guard in closure + clamp_hyperparameters() after step |
| `kernels.py` | ArcSineKernel class (~75 lines) + clamp_hyperparameters() with warnings |

**Modified (from previous sessions, not committed):**
| File | Change |
|------|--------|
| `eigenspace_mstep.py` | ArcSineKernel guard in analytical M-step |
| `.claude/rules/working_guidelines.md` | Unknown changes from previous session |
| `.claude/settings.json` | Unknown changes from previous session |
| `imgs/default_gpy_M50.png` | Regenerated during verification |

**Untracked (this session):**
| File | Purpose |
|------|---------|
| `investigations/arcsine_kernel/run_arcsine.py` | Sandboxed training script using ArcSineKernel |
| `investigations/arcsine_kernel/imgs/` | Output plots from arc-sine training runs |

## Files Created

| File | Purpose | Keep/Delete |
|------|---------|-------------|
| `investigations/arcsine_kernel/run_arcsine.py` | Sandboxed runner for arc-sine kernel | Keep |
| `investigations/arcsine_kernel/imgs/arcsine_default_gpy_M50.png` | Arc-sine M=50 result plot | Keep |
| `investigations/arcsine_kernel/imgs/arcsine_default_gpy_M100.png` | Arc-sine M=100 result plot (if exists) | Keep |
| `investigations/arcsine_kernel/HANDOFF.md` | This file | Keep |

## If Someone Revisits This

**Most important next step**: Decide on the final form of the NaN guard fix in `gpy_training.py`. The current implementation works but needs review:

1. The closure returns `float('inf')` without `backward()` — LBFGS gets stale gradients. This matches `eigenspace_mstep.py`'s pattern, but verify LBFGS strong_wolfe handles this correctly (it should, since inf violates sufficient decrease).

2. `clamp_hyperparameters()` after `optimizer.step()` now warns if it actually clamps. Decide: should the same warning be added to the `eigenspace_mstep.py` calls (lines 132, 337)?

3. Run a full verification: arc-cosine at M=50/100/200 with default_gpy to confirm no regression from the closure change. The arc-cosine at M=50 showed test_r=0.6815 vs 0.6819 before — likely noise, but verify at more configurations.

4. The arc-sine kernel implementation (ArcSineKernel class + run_arcsine.py) is complete and tested but not committed. Commit both the kernel and the closure fix together, or separately.

**Do NOT retry**: The NaN diagnostic approach (printing params on NaN) was useful for diagnosis but should be removed — the production fix is the inf guard, not diagnostics.

**Key files to read first**:
- `gpy_training.py` lines 96-118 (the closure with NaN guard)
- `gpy_training.py` lines 151-156 (clamp after step)
- `kernels.py` clamp_hyperparameters() (with warning logic)
- `eigenspace_mstep.py` lines 78-132 (vargp_direct closure for comparison)

---

## Continuation Prompt

```
I'm continuing an investigation into the LBFGS closure NaN guard for default_gpy training.

Read these files first:
1. `investigations/arcsine_kernel/HANDOFF.md` (this handoff — findings, current state, open questions)
2. `gpy_training.py` lines 96-118 (closure with NaN guard) and 151-156 (clamp after step)
3. `kernels.py` clamp_hyperparameters() method (with warning logic)
4. `eigenspace_mstep.py` lines 78-132 (vargp_direct closure for comparison)

Current state: NaN guard is implemented and working. Arc-sine at M=50 no longer crashes.
Open questions:
- Should eigenspace_mstep.py's clamp_hyperparameters() calls get the same warning?
- Run full regression test: arc-cosine at M=50/100/200 default_gpy
- Commit strategy: arc-sine kernel + closure fix together or separate?

The normalized kernel is superseded — ignore it.
Check git status and git branch before starting.
```
