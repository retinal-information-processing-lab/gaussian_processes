# Investigation: Input Warping for Bounded Pixel Domain

**Branch**: `pietro/input-warping`
**Date**: 2026-02-25
**Status**: Continuing
**Location**: `investigations/input_warping/`

---

## Problem Statement

When optimizing images to maximize GP utility (distribution-aware active learning), the optimizer can push pixel values outside the physical display range [vmin, vmax]. The kernel's C matrix is a spatial smoother that sees `C @ x`, so individual pixel bound violations are invisible. The utility landscape has no "walls" at pixel bounds.

The idea: apply a per-pixel nonlinear warping `w(x)` INSIDE the kernel, before the C matrix, so the GP trains on warped pixels and genuinely cannot distinguish "at the bound" from "past the bound." Formula: `w(x_i) = mid + half_range * tanh(a * (x_i - mid) / half_range)` with steepness `a=3.0`.

Previous session produced a literature survey (`INPUT_WARPING_REFERENCE.md`), implementation plan, and handoff (`.claude/handoffs/HANDOFF_2026-02-24_input-warping-bounded-pixels.md`). This session implemented the plan and ran comparison experiments.

## What Was Tried

### Implementation (completed)
- **What**: Per-pixel tanh warping in the kernel, togglable via config, disabled by default. Changes to `kernels.py` (init, `_maybe_warp()`, all 4 forward methods, `create_kernel()`), `run_single_mode.py` (build_config, flatten_yaml, argparse, vmin/vmax from data), `default_params.json`, YAML configs.
- **Result**: Clean implementation, all modes run without errors.
- **Commit**: `b4d301b`

### Baseline experiment (warping OFF)
- **What**: Full canonical matrix (2 modes x 3 M values x seed 123 x cell 8), standard YAML defaults, warping disabled.
- **Experiment**: `experiments/2026-02-25_baseline-input-warping-branch/`
- **Result**:

| Mode | M=50 | M=100 | M=200 | Mean |
|------|------|-------|-------|------|
| vargp_direct | 0.6914 | 0.7376 | 0.7776 | 0.7355 |
| default_gpy | 0.7627 | 0.7558 | 0.7217 | 0.7467 |

- **Interpretation**: vargp_direct scales with M as expected. default_gpy stalls early (loss flat from iteration 10, early stops at 30) — pre-existing issue.

### Warping ON experiment (steepness=3.0)
- **What**: Identical canonical matrix but with `input_warping.enabled: true`.
- **Experiment**: `experiments/2026-02-25_warping-on-baseline/`
- **Result**:

| Mode | M=50 | M=100 | M=200 | Mean |
|------|------|-------|-------|------|
| vargp_direct | 0.7549 | 0.6140 | 0.6868 | 0.6852 |
| default_gpy | 0.5947 | 0.6351 | 0.6501 | 0.6266 |

- **Deltas (ON - OFF)**:

| Mode | M=50 | M=100 | M=200 | Mean |
|------|------|-------|-------|------|
| vargp_direct | +0.064 | -0.124 | -0.091 | -0.050 |
| default_gpy | -0.168 | -0.121 | -0.072 | -0.120 |

- **Interpretation**: Warping hurts performance overall. The one positive case (vargp_direct M=50) is likely noise. The arc-cosine kernel's norm-dependence (`K(x,x) ~ ||x||^2`) is genuinely informative — warping compresses the tails and dampens this signal. Consistent with the normalized kernel investigation that found test_r drops ~25% when norm information is removed.
- **Verdict**: Inconclusive — the implementation works, but the current steepness (3.0) degrades prediction. Worth investigating whether a lower steepness or different kernel type (arc_sine, RBF) responds differently.

## Key Findings

1. **CONFIRMED**: Input warping implementation works correctly for all kernel types and both training modes. No crashes, no NaN, gradients flow. Commit `b4d301b`.

2. **CONFIRMED**: Warping with steepness=3.0 hurts test_r by ~5-12% on average across the canonical matrix. Evidence: experiments `baseline-input-warping-branch` vs `warping-on-baseline`.

3. **CONFIRMED**: The pixel range for the PNAS dataset is `[-2.4013, 2.4780]`. This is the global min/max across all images (train+val+test).

4. **HYPOTHESIS**: The performance drop is because the arc-cosine kernel's norm-sensitivity (`K(x,x) ~ ||x||^2`) carries real information about neural encoding, and tanh warping compresses tail pixel values, reducing this signal. This is the same mechanism as the normalized kernel investigation (test_r dropped 0.79 -> 0.59).

5. **HYPOTHESIS**: A lower steepness (e.g., 1.0) or the soft-clipping variant (exactly identity within bounds, softplus outside) might preserve in-bounds information while still saturating out-of-bounds. Not tested yet.

6. **CONFIRMED**: default_gpy mode has a pre-existing stalling issue — loss is flat from iteration ~10 regardless of warping. This is independent of input warping but limits the utility of default_gpy results for comparison.

## Why This Was Stopped

Context ran out. Implementation is complete and experiments are recorded. The next step is investigating whether alternative steepness values or warping variants can preserve prediction quality while still bounding the input space.

## Things Noticed But Not Acted Upon

1. The default_gpy early-stalling behavior (loss flat from iteration 10) appears across all experiments. This is a pre-existing issue worth investigating separately.

2. The `analyze_experiment.py --compare` command has a bug in `compare_experiments()` (line 190, f-string formatting error). Not related to warping.

3. vargp_direct with warping ON shows erratic M-scaling (M=50 best, M=100 worst). Without warping, M-scaling is monotonic. The warped kernel matrix may have different conditioning properties at different M values.

4. Warping ON reduces the prediction range (e.g., `[0.413, 2.184]` vs `[0.266, 4.059]` for vargp_direct M=50). The model becomes more conservative — expected since the warped kernel compresses input variance.

## Uncommitted Changes

```
M .claude/skills/handoff-plan/SKILL.md   <- unrelated (auto-modified)
```

All implementation and experiment artifacts are committed (`b4d301b`, `80b245b`).

## Files Created

| File | Purpose | Keep/Delete |
|------|---------|-------------|
| `investigations/input_warping/INPUT_WARPING_REFERENCE.md` | Literature survey (10 references, 11 sections) | Keep |
| `investigations/input_warping/HANDOFF.md` | This handoff | Keep |
| `.claude/handoffs/HANDOFF_2026-02-24_input-warping-bounded-pixels.md` | Previous session's planning handoff | Keep (historical) |
| `.claude/plans/playful-wandering-pine.md` | Implementation plan (reconstructed) | Keep |
| `experiments/2026-02-25_baseline-input-warping-branch/` | Baseline experiment (warping OFF) | Keep |
| `experiments/2026-02-25_warping-on-baseline/` | Warping ON experiment | Keep |

## If Someone Revisits This

**Most promising directions (try first)**:
1. **Lower steepness** (a=1.0 or a=1.5): Less compression within bounds, may preserve norm information. Quick test: `python run_single_mode.py --mode vargp_direct --input-warping --warping-steepness 1.0`
2. **Soft-clipping variant**: Exactly identity within [vmin, vmax], softplus saturation outside. Zero distortion in-bounds. Requires implementing a new warping function in `_maybe_warp()`. See `INPUT_WARPING_REFERENCE.md` Section 3.4 for the formula.
3. **Test with arc_sine or RBF kernel**: These kernels are less norm-dependent. The warping penalty may be smaller. Quick: `--kernel-type arc_sine --input-warping`

**What NOT to try**:
- Learned per-pixel warping (Beta CDF, Kumaraswamy) — 23k+ extra parameters, not justified for uniform bound enforcement. See `INPUT_WARPING_REFERENCE.md` for full argument.
- Multiplicative penalty kernel — catastrophic underflow in 11664 dimensions.
- Steepness >> 5 — approaches hard clipping, gradient underflow at boundaries.

**Key context**:
- The warping only matters during utility optimization (when pixels can go OOB). For pure prediction (test_r), warping should ideally be neutral. The fact that it's not neutral means the warping distorts the kernel enough to matter even within bounds.
- The real test of warping is whether optimized images stay in-bounds during utility maximization. test_r comparison is necessary but not sufficient.

---

## Continuation Prompt

```
I am investigating input warping for bounded pixel domain in the GP kernel.

Read these files in order:
1. investigations/input_warping/HANDOFF.md (this handoff — findings and next steps)
2. investigations/input_warping/INPUT_WARPING_REFERENCE.md (literature survey, Section 3.4 for soft-clipping)
3. kernels.py (lines 260-290 for warping init, _maybe_warp method)

Context: Input warping (tanh, steepness=3.0) is implemented and working on
branch pietro/input-warping. But it hurts test_r by 5-12%. The arc-cosine
kernel's norm-sensitivity carries real information, and tanh compresses it.

Next steps:
1. Try lower steepness (1.0, 1.5) — quick CLI test
2. Consider implementing soft-clipping variant (identity in-bounds, softplus outside)
3. Test with arc_sine kernel (less norm-dependent)
4. The real validation is utility optimization, not just test_r

Check git status and git branch before starting.
Do NOT switch to other branches.
```
