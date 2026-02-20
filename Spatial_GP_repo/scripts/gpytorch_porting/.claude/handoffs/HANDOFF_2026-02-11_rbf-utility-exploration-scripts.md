# Handoff: RBF Kernel Utility Exploration Scripts

**Branch**: `pietro/rbf-kernel`
**Date**: 2026-02-11
**Status**: Ready for implementation
**Plan file**: `.claude/plans/sorted-sniffing-tiger.md`

---

## Motivation

The LocalRBFKernel (`kernels.py:702-863`) was implemented in two earlier commits (9ae87b7, 565717b) and achieves test_r=0.7785 on PNAS cell 8 (M=50) — a 7% drop vs arc-cosine's ~0.84. Training and basic validation (`investigations/rbf_kernel/run_rbf.py`) are complete.

The next step is understanding the RBF kernel's utility behavior: how standard and distribution-aware utility functions behave with a stationary kernel that has constant diagonal K(x,x)=1. Two existing scripts for the unnormalized arc-cosine kernel serve as templates:
- `investigations/understanding_utility/explore_utility.py` — utility workbench with landscape plots
- `investigations/understanding_utility/gradient_unnormalized.py` — LBFGS gradient ascent on DA utility

These need RBF-adapted copies in `investigations/rbf_kernel/`.

## Decisions and Rationale

### 1. Right panel x-axis: K(x*, x_cond) instead of ||x||_C

The original `explore_utility.py` plots "U_DA vs ||x||_C colored by angle" because the arc-cosine kernel factorizes as K(x,y) = magnitude * J(angle). For RBF, ||x||_C = sqrt(K(x,x)) = 1 always — there is no norm/angle decomposition.

**Decision**: Use K(x*, x_cond) as the x-axis. This directly captures how "close" the candidate is to the conditioning image in kernel space, which is the single axis of variation for a stationary kernel. No coloring by angle needed since K already encodes the full relationship.

**Rejected**: d_C distance (same information as K but less intuitive, requires knowing l). Also rejected: keeping the original norm+angle axes for comparison (would just show a vertical line at norm=1, uninformative).

### 2. Import from run_rbf.py, not run_single_mode.py

`run_single_mode.py:run_single_config()` creates an `ArcCosineKernel`. `run_rbf.py` has its own `build_config_from_defaults()` (adds `lengthscale=100.0`) and `run_single_config()` (creates `LocalRBFKernel`).

**Decision**: Import from `run_rbf.py`. This keeps the RBF exploration self-contained within `investigations/rbf_kernel/`. The user confirmed that duplicating function definitions is acceptable for this playground.

### 3. Remove sigma_0 override logic

The original scripts have `SIGMA_0 = None` with a post-training override option. For LocalRBFKernel, sigma_0 is inherited from ArcCosineKernel but has zero gradient and no effect on forward() — the bias term cancels in the (x-y) difference.

**Decision**: Remove the sigma_0 override blocks entirely. No equivalent needed for RBF.

### 4. Keep kernel_norm() and kernel_angle() helpers

Even though kernel_norm() always returns ~1.0 for RBF, it serves as a verification check. kernel_angle() = arccos(K(x1,x2)) is still a meaningful dissimilarity measure for RBF (it equals the "angle" in the sense that K = cos(angle) for values in [0,1]).

**Decision**: Keep both, plus add a new `kernel_distance()` helper that computes `sqrt(-2*log(K))` (the scaled C-distance d_C/l).

### 5. Scaling experiment still relevant

For arc-cosine, scaling x by 5 dramatically increases ||x||_C and utility. For RBF, K(cx, cx) = 1 still, but K(cx, y) != K(x, y) because the distance (cx-y)^T C (cx-y) changes. So the scaling comparison is still informative — it shows whether RBF utility is immune to amplitude manipulation (it should be less sensitive but not immune since distance changes).

## Critical Subtleties

1. **run_rbf.py has its own build_config_from_defaults()**: Do NOT use the one from `run_single_mode.py`. The RBF version adds `lengthscale=100.0` to the config and its `run_single_config()` creates `LocalRBFKernel`. If you accidentally import from `run_single_mode`, you get an ArcCosineKernel and the entire investigation is wrong with no error message.

2. **data_path is hardcoded in run_rbf.py**: `run_rbf.py:554` uses an absolute path to PNAS data. The explore/gradient scripts that import from it will inherit this. This is fine for the investigation but worth noting.

3. **acquisition.py is kernel-agnostic**: `standard_utility()` and `distribution_aware_utility()` work via `model(X)` which dispatches to whatever kernel the model has. No kernel-specific code needed in the utility calls.

4. **RBF diagonal is exactly 1.0 by construction**: `forward()` returns `exp(-0.5 * dist_sq / ls_sq)` and dist_sq of x with itself is 0 (clamped). So `kernel_value(model, x, x)` should be exactly 1.0, not approximately. If it's not, something is wrong.

5. **The existing `run_rbf.py` N_TRAIN and M**: The explore_utility scripts hardcode `N_TRAIN=70, M=50`. For RBF, keep the same values for comparability with the arc-cosine exploration scripts.

## Uncommitted Changes

Working tree is clean. No uncommitted changes.

## Files to Read First

1. **`.claude/plans/sorted-sniffing-tiger.md`** — The implementation plan with per-file change lists
2. **`investigations/understanding_utility/explore_utility.py`** — Template for explore_utility_rbf.py (654 lines)
3. **`investigations/understanding_utility/gradient_unnormalized.py`** — Template for gradient_rbf.py (527 lines)
4. **`investigations/rbf_kernel/run_rbf.py`** — RBF training script; provides `build_config_from_defaults()` and `run_single_config()` to import
5. **`kernels.py:702-863`** — LocalRBFKernel class, to understand K(x,x)=1, sigma_0 irrelevance, lengthscale parameterization

---

## Continuation Prompt

```
I'm continuing work on the RBF kernel utility exploration scripts.

Read the handoff: .claude/handoffs/HANDOFF_2026-02-11_rbf-utility-exploration-scripts.md
Read the plan: .claude/plans/sorted-sniffing-tiger.md

Task: Create two scripts in investigations/rbf_kernel/:
1. explore_utility_rbf.py — adapted from investigations/understanding_utility/explore_utility.py
2. gradient_rbf.py — adapted from investigations/understanding_utility/gradient_unnormalized.py

Key decisions: Import from run_rbf.py (not run_single_mode.py), right panel uses
K(x*, x_cond) as x-axis, no sigma_0 override, keep kernel_norm/angle helpers plus
add kernel_distance.

Check git status and git branch before starting. Branch should be pietro/rbf-kernel.
```
