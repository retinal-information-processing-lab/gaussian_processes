# Investigation: LocalRBFKernel with C Matrix

**Branch**: `pietro/rbf-kernel`
**Date**: 2026-02-11
**Status**: Continuing
**Location**: `investigations/rbf_kernel/`
**Worktree**: `/home/idv-eqs8-pza/IDV_code/ClosedLoopProject/gaussian_processes/Spatial_GP_repo/gpytorch_porting_rbf_kernel`

---

## Problem Statement

Compare a stationary RBF kernel against the existing non-stationary arc-cosine kernel for modeling neural responses to visual stimuli. The RBF kernel uses the **same C matrix** (receptive field structure) as the arc-cosine kernel, enabling an apples-to-apples comparison of kernel families while keeping the spatial structure identical.

The scientific question: how much does stationarity (constant k(x,x) = 1) hurt, given that the arc-cosine kernel's norm-dependent diagonal carries genuine signal about stimulus contrast?

## Implementation — Complete

### Kernel class: `LocalRBFKernel` in `kernels.py`

**Formula**:
```
k(x, x') = exp(-(x - x')^T C_base (x - x') / (2 l^2))
```

**C_base** (RF structure, NO Amp):
```
C_base = diag(alpha) @ C_smooth @ diag(alpha)

alpha_i     = exp(-beta_raw * ||pixel_i - center||^2)
C_smooth_ij = exp(-rho_raw * ||pixel_i - pixel_j||^2)
```

where `beta_raw = exp(raw_m2log2beta)`, `rho_raw = exp(raw_mlog2rho2)`.

**Design**: Subclasses `ArcCosineKernel` (same pattern as `ArcSineKernel`). Overrides:
- `_compute_C_matrix()` — drops the `self.Amp *` multiplication
- `forward()` — RBF formula with `/ self.lengthscale**2`
- `params_in_bounds()` — adds lengthscale bounds check
- `clamp_hyperparameters()` — adds lengthscale clamping

### Hyperparameter table

| Param | Role | Raw storage | Transform | Bounds | Optimized? |
|-------|------|-------------|-----------|--------|------------|
| l | Distance sensitivity | `raw_log_lengthscale` | l = exp(raw) | [0.1, 100000] | Yes (LBFGS) |
| beta | RF size | `raw_m2log2beta` | beta = exp(-raw/2)/2 | [0.01, 1.0] | Yes (LBFGS) |
| rho | Pixel smoothness | `raw_mlog2rho2` | rho = exp(-raw/2)/sqrt(2) | [0.01, 0.5] | Yes (LBFGS) |
| eps_0x/y | RF center | direct | none | [-1, 1] | Yes (LBFGS) |
| sigma_0 | (inherited, unused) | `raw_sigma_0` | softplus | >0 | No (zero grad) |
| Amp | (inherited, unused) | `raw_Amp` | softplus | (0, 1000] | No (zero grad) |

### Investigation script: `investigations/rbf_kernel/run_rbf.py`

Copy of `run_arcsine.py` with kernel swapped to `LocalRBFKernel`. Supports `--lengthscale` CLI arg (default 100.0, RBF-specific, not in `default_params.json`). Uses absolute path to PNAS data (worktree doesn't have notebooks/ symlink).

## What Was Tried

### Attempt 1: Amp as lengthscale (initial commit 9ae87b7)
- **What**: Used inherited Amp parameter directly in `(x-y)^T C (x-y)` (C includes Amp)
- **Result**: Amp=1.0 (arc-cosine default) gives dist_sq ~ O(10^4), all kernel values = 0. Manual Amp=0.001 gave test_r=0.554, Amp=0.0001 gave test_r=0.580.
- **Interpretation**: Amp controls distance scale exponentially in the RBF exponent (vs angular structure in arc-cosine). Default initialization is catastrophically wrong. Manual tuning required.
- **Verdict**: Dead end — replaced by lengthscale reparametrization.

### Attempt 2: Log-space lengthscale (commit 565717b)
- **What**: Removed Amp from C_base, added new `raw_log_lengthscale` parameter with l = exp(raw). Default l=100.
- **Result**: test_r=0.7785 (M=50, cell 8, seed 42). Lengthscale learned from 100 -> 17.1. Loss 391 (best so far). A=1.123, lambda0=0.236.
- **Interpretation**: Log-space lets LBFGS navigate scale freely. Major improvement over manual Amp tuning.
- **Verdict**: Promising — current implementation. Continue from here.

## Key Findings

1. CONFIRMED: The RBF kernel with C matrix works and trains successfully with `default_gpy` mode. test_r=0.7785 on PNAS cell 8, M=50, seed=42.

2. CONFIRMED: Stationarity hurts but less than expected. test_r=0.78 vs arc-cosine ~0.84 is a ~7% drop. The normalized arc-cosine kernel (also constant diagonal) showed a ~25% drop (0.79 -> 0.59), so the RBF kernel recovers much of the performance through its different distance metric.

3. CONFIRMED: Amp=1.0 is catastrophically wrong for the RBF kernel. C-weighted pairwise distances with Amp=1 are O(10^4), giving exp(-5000) = 0 for all pairs. The lengthscale reparametrization completely solves this.

4. CONFIRMED: sigma_0 has no effect on the RBF kernel (bias cancels in x-y differences for stationary kernels). Amp also has no effect after the reparametrization (excluded from C_base). Both inherited, both have zero gradient.

5. CONFIRMED: Lengthscale gradient is non-zero (tested: grad=1.44 on synthetic data). LBFGS successfully optimizes it from init=100 to final=17.1 on real data.

6. HYPOTHESIS: M=100 with the RBF kernel may have K_uu conditioning issues. One test (Amp=0.001, M=100) showed worse results than M=50 (test_r=0.485 vs 0.554, stuck at loss 455). Not investigated further after the reparametrization. Worth retesting with the lengthscale version.

## Why This Was Stopped

Context running out. Implementation is solid. Next steps are further evaluation and comparison.

## Things Noticed But Not Acted Upon

1. The `default_params.json` in this worktree was missing `f_max` key (added in a later arcsine branch commit). Fixed by adding it manually. The worktree may be behind on other `default_params.json` changes too.

2. The data path in `run_rbf.py` uses an absolute path (`/home/.../Spatial_GP_repo/notebooks/PNAS_paper_sorted_data.npz`) because the worktree lacks the notebooks directory. This is fragile if the main repo moves.

3. The kernel early-stops quickly (30 iterations). Might benefit from longer runs or disabling early stopping to see if performance improves.

4. Only tested on cell 8, M=50, seed=42. No multi-cell or multi-seed validation yet.

5. The `kernel.Amp = config['Amp']` line was removed from the default_gpy kernel creation in `run_rbf.py` (since Amp is unused). If someone compares parameters between run_rbf.py and run_arcsine.py, this difference could be confusing.

## Uncommitted Changes

Working tree is clean. Only untracked files are generated plots:
```
Untracked: investigations/rbf_kernel/imgs/  (generated plots, safe to ignore)
```

## Files Created

| File | Purpose | Keep/Delete |
|------|---------|-------------|
| `kernels.py` (modified) | Added `LocalRBFKernel` class (~140 lines) after `ArcSineKernel` | Keep |
| `investigations/rbf_kernel/run_rbf.py` | Investigation script, copy of run_arcsine.py with RBF kernel | Keep |
| `investigations/rbf_kernel/imgs/` | Generated plots from training runs | Optional (regenerable) |
| `default_params.json` (modified) | Added `f_max: 100` to utility section | Keep |
| `investigations/rbf_kernel/HANDOFF.md` | This file | Keep |

## If Someone Revisits This

**Most promising next steps (in order):**
1. Run M=100 with the new lengthscale reparametrization (the M=100 failure was with old Amp approach)
2. Test on multiple cells (cell 10 is the other standard) and seeds for robustness
3. Compare training dynamics: does the RBF kernel converge to a different RF (beta, rho, center) than arc-cosine?
4. Try wrapping in `ScaleKernel` for a learnable output variance (currently fixed at 1)

**Do NOT retry:**
- Using Amp directly as the distance scale parameter (commits before 565717b). Dead end — log-space lengthscale is strictly better.
- Disabling masking (use_mask=False). C matrix would be 11664x11664, too large.

**Context to keep in mind:**
- The kernel is `default_gpy` mode only. `vargp_direct` would need augmented matrix work.
- The lengthscale default (100.0) is hardcoded in `run_rbf.py` and the kernel class, NOT in `default_params.json`. This is intentional (RBF-specific param).

---

## Continuation Prompt

```
I'm continuing work on the LocalRBFKernel investigation on branch
pietro/rbf-kernel in the worktree at:
/home/idv-eqs8-pza/IDV_code/ClosedLoopProject/gaussian_processes/Spatial_GP_repo/gpytorch_porting_rbf_kernel

Read the handoff at:
investigations/rbf_kernel/HANDOFF.md

The kernel implementation is complete (2 commits: 9ae87b7, 565717b).
It uses k(x,y) = exp(-(x-y)^T C_base (x-y) / (2l^2)) with log-space
lengthscale and C_base excluding Amp. test_r=0.7785 on cell 8, M=50.

Next steps: test with M=100, multi-cell validation, compare RF params
with arc-cosine, potentially add ScaleKernel wrapping.

Check git status and git branch before starting.
```
