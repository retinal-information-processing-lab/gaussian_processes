# Investigation: Arc-Sine Kernel Implementation + LBFGS Stability Fixes

**Branch**: `pietro/arcsine-kernel`
**Date**: 2026-02-10
**Status**: Continuing
**Location**: `investigations/arcsine_kernel/`

---

## Problem Statement

Implementing the arc-sine kernel (Williams 1998) as an alternative to the arc-cosine kernel. During implementation, multiple bugs and design issues were uncovered in the LBFGS training pipeline and the initial hyperparameter configuration. This handoff covers the arc-sine kernel implementation, the LBFGS closure stability fixes, and two newly discovered issues with initial hyperparameters (beta and RF center).

## What Was Done

### Arc-Sine Kernel (committed in 65dca99)
- `ArcSineKernel` class in `kernels.py:593-666`, subclasses `ArcCosineKernel`
- K_sat(x,x') = (2/pi) * arcsin((x^T C x' + sigma_0^2) / sqrt((1+v_x)(1+v_x')))
- Saturates at 1.0 (prevents quadratic growth of arc-cosine)
- Sandboxed runner: `investigations/arcsine_kernel/run_arcsine.py`
- M=100 default_gpy: test_r=0.765 (previous session) or 0.545 (this session — see note below)
- M=50 default_gpy: test_r=0.462

### LBFGS Closure NaN Guard (committed in 62d8172)
- Added try/except + NaN/Inf loss check in `gpy_training.py` closure, returning `float('inf')` to reject bad LBFGS trial steps
- Added `clamp_hyperparameters()` after `optimizer.step()` as safety net

### params_in_bounds() (committed in 17a0c57)
- Added `params_in_bounds()` method to both `ArcCosineKernel` (kernels.py) and `PoissonLikelihood` (likelihoods.py)
- Added `clamp_params()` to `PoissonLikelihood` (bounds: A_MAX=10, lambda0 in [-50, 50])
- Updated ALL 4 LBFGS closures to call the relevant check at the top, before any computation:
  - `gpy_training.py`: kernel + likelihood
  - `eigenspace_mstep.py` autograd: kernel
  - `eigenspace_mstep.py` analytical: kernel (replaced inline Guardrail 1)
  - `eigenspace_fstep.py`: likelihood
- Added `clamp_params()` call after step in `eigenspace_fstep.py` (was missing) and `gpy_training.py`

### STA + RF Visualization (uncommitted)
- Extended `run_arcsine.py` `plot_fit()` from 1x2 to 2x2 figure
- Bottom-left: Initial STA + initial RF (center dot + 1/2-sigma circles)
- Bottom-right: Training-set STA + trained RF (center dot + 1/2-sigma circles)

## Key Findings

### LBFGS Stuck Pattern at M=50 (both kernels)

1. **CONFIRMED**: After ~10 iterations (arc-cosine) or ~3 iterations (arc-sine), LBFGS enters a stuck loop. Every iteration: 27 closure calls, 25 throw Cholesky exceptions, 1 kernel_oob, 1 ok. The optimizer cannot take any step.

2. **CONFIRMED**: The kernel_oob values are IDENTICAL across stuck iterations (e.g., arc-sine: Amp=11.73, raw_beta=186.8, raw_rho=168.5, eps=(-150.4, -66.41)). This proves LBFGS proposes the same trial point every iteration — no progress is made, no new gradient information is gained, the search direction never changes.

3. **CONFIRMED**: The `params_in_bounds()` check is NOT causing the problem — it catches 1 trial point per iteration. The 25 exceptions (Cholesky failures on the line search) are the real bottleneck.

4. **CONFIRMED**: Arc-cosine kernel gets 10 productive iterations at M=50 (loss drops from ~50 to 32, learns A and lambda0), then gets stuck. Arc-sine gets only 3 productive iterations.

### ~~Initial Beta Is Absurdly Broad~~ CORRECTED: Beta is fine

5. **CORRECTED** (2026-02-10, follow-up session): The analysis below was **WRONG**. The locality mask in the code is `alpha = exp(-exp(raw) * dist_sq)` where `exp(raw) = 1/(4*beta_nat^2)`. For beta_nat=0.1, `exp(raw) = 25.0`, giving sigma = beta_nat * sqrt(2) = **0.1414 normalized = 7.6 pixels**. The previous session confused natural beta (0.1) with the Gaussian coefficient (25.0). The initial RF is reasonable (~15px diameter at 2-sigma), not 120px. The visualization code had the same bug (used natural beta in `sqrt(1/(2*beta))`) — fixed in 17a1630.

6. ~~HYPOTHESIS~~: Invalidated — the initial RF is not absurdly broad.

### STA Center-of-Mass Is a Poor RF Center Estimate — FIXED

7. **CONFIRMED**: For cell 8, the STA center-of-mass (all 3160 images, z-scored) was at pixel (55.9, 42.0), but the STA peak (argmax of |STA|) is at pixel (64, 52). The distance was 12.9 pixels.

8. **CONFIRMED**: Only 15.4% of total |STA| mass was within 15px of the center-of-mass. The z-scored STA is very diffuse — background noise contributes significant mass that pulls the CoM away from the actual RF peak.

9. **FIXED** (17a1630): `compute_rf_center_from_sta()` in `utils.py` now uses Gaussian-smoothed argmax (blur_sigma=3px) instead of center-of-mass. New center is 0px from raw argmax for cell 8.

## Debug Logs — REMOVED

Debug closure counters were removed from `gpy_training.py` (they were never committed). The closure guards (params_in_bounds, try/except, NaN check) remain.

## Why This Was Stopped

Context ran out. The arc-sine kernel is implemented, the LBFGS stability fixes are committed, and the STA visualization is working. Two new issues were discovered (initial beta too broad, STA CoM inaccurate) that need to be addressed before the arc-sine kernel can be properly evaluated.

## Things Noticed But Not Acted Upon

1. Arc-sine M=100 test_r dropped from 0.765 (previous session) to 0.545 in this session. Cause unclear — could be the params_in_bounds() checks rejecting some LBFGS steps that previously made progress. Needs investigation.

2. The Pylance diagnostic "Code is structurally unreachable" at line ~710 of run_arcsine.py is a pre-existing issue, not related to this work.

3. `eigenspace_mstep.py` has an uncommitted ArcSineKernel guard in the analytical M-step (from a previous session). Should be committed alongside the arc-sine kernel work.

4. Several untracked files from previous sessions (experiments/exploratory/, normalized_kernel investigation images) clutter git status.

## Uncommitted Changes

**Modified (this session):**
| File | Change |
|------|--------|
| `gpy_training.py` | DEBUG closure counters + kernel_oob param detail prints (TEMPORARY — remove before commit) |
| `investigations/arcsine_kernel/run_arcsine.py` | STA + RF visualization (plot_fit expanded to 2x2, _compute_sta_2d helper, initial/trained RF overlay) |
| `investigations/arcsine_kernel/imgs/arcsine_default_gpy_M100.png` | Updated with 2x2 visualization |
| `investigations/arcsine_kernel/imgs/arcsine_default_gpy_M50.png` | Updated with 2x2 visualization |

**Modified (from previous sessions, not this work):**
| File | Change |
|------|--------|
| `imgs/default_gpy_M50.png`, `imgs/default_gpy_M100.png`, `imgs/vargp_direct_M100.png` | Regenerated during earlier verification |

## Files Created

| File | Purpose | Keep/Delete |
|------|---------|-------------|
| `investigations/arcsine_kernel/HANDOFF.md` | This file | Keep |

## If Someone Revisits This

**Completed (follow-up session, 2026-02-10):**

1. ~~Remove DEBUG closure counters~~ — DONE (were never committed, removed to restore clean state)
2. ~~Investigate initial beta~~ — RESOLVED: beta=0.1 gives sigma=7.6px, not 120px. Previous analysis was wrong. No change needed.
3. ~~Investigate STA center accuracy~~ — DONE: switched to smoothed argmax in `utils.py` (17a1630). 0px error vs 12.9px with old CoM.
4. ~~Commit remaining work~~ — DONE: run_arcsine.py viz (17a1630), ported to run_single_mode.py (5869c75).

**Remaining next steps:**

1. **Investigate test_r regression**: Arc-sine M=100 dropped from 0.765 to 0.545 between sessions. Cause still unclear.

2. **Investigate LBFGS stuck pattern at M=50**: Both kernels get stuck after a few productive iterations. The initial beta is NOT the cause (it's reasonable). Root cause is likely the Cholesky failures during line search with only 50 inducing points.

3. **Commit eigenspace_mstep.py ArcSineKernel guard** (from a previous session, still uncommitted).

**Do NOT retry**: The LBFGS closure diagnostic approach (printing all params on kernel_oob) was useful for understanding the stuck pattern but should not be kept in production code.

**Key files to read first**:
- `investigations/arcsine_kernel/HANDOFF.md` (this file)
- `gpy_training.py` lines 96-140 (closure with all guards + DEBUG counters to remove)
- `kernels.py:276-308` (params_in_bounds), `kernels.py:310-356` (clamp_hyperparameters)
- `likelihoods.py:81-131` (params_in_bounds + clamp_params)
- `default_params.json` kernel.beta (the initial value under question)
- `utils.py:20-91` (compute_rf_center_from_sta — the CoM approach)

---

## Continuation Prompt

```
I'm continuing the arc-sine kernel investigation on branch pietro/arcsine-kernel.

Read: `investigations/arcsine_kernel/HANDOFF.md`

Key context:
- Arc-sine kernel, NaN guard, params_in_bounds all committed
- STA+RF visualization committed in both run_arcsine.py and run_single_mode.py
- RF center now uses smoothed argmax (was CoM, 12.9px off — fixed)
- Initial beta=0.1 is FINE (sigma=7.6px, previous "120px" claim was a math error)
- gpy_training.py debug counters removed (clean)

Open issues:
1. Arc-sine M=100 test_r regression (0.765 → 0.545 between sessions)
2. LBFGS stuck pattern at M=50 (both kernels, root cause unclear)
3. eigenspace_mstep.py has uncommitted ArcSineKernel guard
```
