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

### Initial Beta Is Absurdly Broad

5. **CONFIRMED**: `default_params.json` sets `kernel.beta = 0.1`. The locality mask is `alpha = exp(-beta * ||x - xi_0||^2)` in normalized [-1,1] coords. With beta=0.1:
   - sigma = sqrt(1/(2*0.1)) = 2.24 in normalized coords (image spans 2.0)
   - sigma = 119.6 pixels on a 108x108 image
   - The initial RF covers MORE than the entire image at 1-sigma
   - This means the initial kernel treats every pixel with nearly equal weight — effectively no spatial structure
   - For reference: a ~30px RF needs beta=1.59, a ~20px RF needs beta=3.58

6. **HYPOTHESIS**: This is a contributing factor to the LBFGS stuck pattern. The optimizer has to simultaneously shrink beta from "entire image" to "localized RF" while learning A, lambda0, and other params. The gradient landscape around beta=0.1 may have a steep cliff that causes LBFGS to overshoot on its first line search step.

### STA Center-of-Mass Is a Poor RF Center Estimate

7. **CONFIRMED**: For cell 8, the STA center-of-mass (all 3160 images, z-scored) is at pixel (55.9, 42.0), but the STA peak (argmax of |STA|) is at pixel (64, 52). The distance is 12.9 pixels.

8. **CONFIRMED**: Only 15.4% of total |STA| mass is within 15px of the center-of-mass. The z-scored STA is very diffuse — background noise contributes significant mass that pulls the CoM away from the actual RF peak.

9. **HYPOTHESIS**: Using argmax or thresholded center-of-mass (e.g., only pixels above 90th percentile of |STA|) would give a more accurate initial RF center. This matters because the initial center affects inducing point selection via pivoted Cholesky.

## Debug Logs (Temporary)

`gpy_training.py` contains temporary DEBUG closure counters (total/kernel_oob/lik_oob/exception/nan_inf/ok) and per-iteration summary prints. These are NOT committed. They must be **removed before committing** — they are purely diagnostic and add noise to training output.

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

**Most promising next steps (in order):**

1. **Remove DEBUG closure counters** from `gpy_training.py` before committing anything else. They are temporary diagnostic code.

2. **Investigate initial beta**: Consider changing `default_params.json` kernel.beta from 0.1 to something reasonable (e.g., 1.0 gives sigma~50px, a moderate initial RF). Test both kernels at M=50 and M=100 to see if a tighter initial beta helps LBFGS make more progress. This is likely the single biggest improvement for training stability.

3. **Investigate STA center accuracy**: Try argmax instead of center-of-mass in `compute_rf_center_from_sta()`, or use a thresholded CoM (only pixels above some |STA| percentile). Compare the initial RF center to the trained one.

4. **Investigate test_r regression**: Arc-sine M=100 dropped from 0.765 to 0.545 between sessions. Run with the same parameters and check whether the params_in_bounds() checks are interfering with optimization.

5. **Commit remaining work**: After removing debug code, commit run_arcsine.py visualization changes and the eigenspace_mstep.py ArcSineKernel guard.

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
I'm continuing the arc-sine kernel implementation on branch pietro/arcsine-kernel.

Read these files first:
1. `investigations/arcsine_kernel/HANDOFF.md` — full context, findings, next steps
2. `gpy_training.py` — has TEMPORARY DEBUG closure counters that MUST be removed
3. `default_params.json` kernel section — initial beta=0.1 is too broad (see handoff)
4. `utils.py:20-91` — compute_rf_center_from_sta uses CoM (12.9px off from STA peak)

Key context from previous session:
- params_in_bounds() and clamp_params() are committed (17a0c57)
- Arc-sine kernel and NaN guard are committed (65dca99, 62d8172)
- run_arcsine.py has STA+RF visualization (uncommitted)
- Two issues found: (1) initial beta=0.1 gives sigma=120px (entire image),
  (2) STA center-of-mass is 12.9px away from STA peak for cell 8

Priority: remove debug code, then investigate initial beta impact on training.
Check git status and git branch before starting.
```
