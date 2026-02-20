# Investigation: RBF Kernel - Multi-Image DA Utility Optimization

**Branch**: `pietro/rbf-kernel`
**Date**: 2026-02-12
**Status**: Continuing
**Location**: `investigations/rbf_kernel/`
**Worktree**: `/home/idv-eqs8-pza/IDV_code/ClosedLoopProject/gaussian_processes/Spatial_GP_repo/scripts/gpytorch_porting_rbf/Spatial_GP_repo/scripts/gpytorch_porting`

---

## Problem Statement

Gradient ascent on DA utility with the RBF kernel causes pixel saturation — even with sigmoid bounds (dataset range), pixels hit the limits. Two approaches being explored:

1. **Multi-image conditioning**: Condition on N_SAMPLE (~50-200) pool images instead of 1 target. The DA utility itself becomes the regularizer.
2. **Tighter pixel bounds**: Constrain pixels to target image range or pool percentiles.

Additionally, when using multiple lambda samples (`SAMPLE_LAMBDA=True, N_LAMBDA_SAMPLES=250`), the original code OOMs due to computation graph accumulation. A per-sample backward approach was implemented but has a remaining bug.

## What Was Done This Session

### 1. Created distribution_gradient.py (committed c1b19c0)
- **What**: New script for LBFGS gradient ascent on DA utility conditioned on N_SAMPLE pool images
- **Key design**: Imports `setup()` from `explore_utility_rbf`, `_reconstruct_image()` from `gradient_rbf`. Starting image = uniform gray at mean RF intensity of sampled images + small noise.
- **Result**: With N_SAMPLE=50, the optimizer converges immediately (U_DA: 0.0693 -> 0.0694, then flat). Gradient norm ~1.3e-5. The 50 conditioning images' gradients cancel each other out.
- **Verdict**: Inconclusive — the flat landscape could be due to genuinely canceling gradients or due to the model's low test_r (0.25).

### 2. Added BOUNDS_MODE to gradient_rbf.py (committed c1b19c0)
- **What**: Added `BOUNDS_MODE` parameter ('dataset', 'target', 'percentile') with `PERCENTILE_LO`/`PERCENTILE_HI`. Overrides vmin/vmax after x_target is defined.
- **Status**: Complete, ~15 lines of code.

### 3. Fixed OOM in acquisition.py (committed c1b19c0)
- **What**: Wrapped `model(x_i.unsqueeze(0))` calls inside `distribution_aware_utility()`'s MC loop in `torch.no_grad()`.
- **Why**: Each `model(x_i)` built a computation graph for the conditioning image. With N_SAMPLE=100+, all graphs accumulated in memory before `backward()`. These graphs contributed zero gradient to x* (the query being optimized).
- **Status**: Complete. Memory for conditioning images is now O(1) instead of O(N_SAMPLE).

### 4. Wired adaptive r_max into run_rbf.py (committed d24637b)
- **What**: Added `adaptive_r_max`, `adaptive_safety_k`, `adaptive_max_rmax`, `adaptive_min_rmax` from `default_params.json` into `run_rbf.build_config_from_defaults()`. Was missing, causing KeyError.

### 5. Per-sample backward approach (UNCOMMITTED — HAS BUG)
- **What**: Replaced the single big `loss.backward()` with per-sample backward in both `gradient_rbf.py` and `distribution_gradient.py`. Since loss = -utility = -H_marg + E[H_cond], we accumulate d(loss)/dx via `(-H_marg).backward()` then `(H_cond_i / n_mc).backward()` per sample, freeing each graph immediately. Memory O(1) per sample.
- **Bug identified**: `A_val = likelihood.A.squeeze()` and `lam0_val = likelihood.lambda0.squeeze()` have `grad_fn` (they're derived from Parameters). `H_marg.backward()` traverses through A_val/lam0_val's grad_fn and frees their saved tensors. Then `H_cond_i.backward()` tries to traverse the same A_val/lam0_val nodes -> RuntimeError "backward through the graph a second time".
- **Fix identified but not applied**: `.detach()` on A_val and lam0_val (we're not optimizing likelihood params, only image pixels). This was about to be applied when context ran out.
- **Verdict**: The per-sample backward approach is correct in principle. Two fixes needed: (a) detach A_val/lam0_val, (b) recompute x_query fresh per MC sample (already done). After these two fixes, both scripts should work with large N_LAMBDA_SAMPLES.

## Key Findings

1. CONFIRMED: With 50 conditioning images, DA utility gradient is essentially zero at natural images. Utility barely moves (0.0693 -> 0.0694). The averaging across diverse conditioning images cancels out directional gradients. The landscape is flat.

2. CONFIRMED: `distribution_aware_utility()` accumulated O(N_SAMPLE) computation graphs in the MC loop, causing OOM with N>=100. Fixed by wrapping conditioning-image model() calls in `torch.no_grad()` (acquisition.py).

3. CONFIRMED: Per-sample backward requires fresh `x_query` per MC iteration because `H_marg.backward()` frees the graph from `opt_var` -> `x_query`. Fixed by recomputing `_to_pixel(opt_var)` + `_reconstruct_image()` each iteration (cheap ops).

4. CONFIRMED (BUG): `A_val = likelihood.A.squeeze()` retains `grad_fn` (SqueezeBackward). When shared across multiple `backward()` calls, the second backward fails because the first freed A_val's saved tensors. Fix: `.detach()` since we only optimize image pixels, not likelihood parameters.

5. HYPOTHESIS: The per-sample backward approach should scale to N_LAMBDA_SAMPLES=250+ once both fixes (detach A_val/lam0_val + fresh x_query) are applied.

## Why This Was Stopped

Context ran out. The A_val/lam0_val detach fix was identified and about to be applied to both scripts when the session ended.

## Things Noticed But Not Acted Upon

1. `explore_utility_rbf.py` has N_TRAIN changed from 50 to 150 by the user (uncommitted in a previous session, now committed). The gradient scripts still use setup() which reads N_TRAIN from explore_utility_rbf.py.

2. The model's test_r is only 0.2531 with RBF kernel (M=50, N_TRAIN=50). This is much lower than arc-cosine (~0.78). Could contribute to the flat utility landscape with multi-image conditioning.

3. User has been experimenting with many parameter combinations across both scripts. Current uncommitted state reflects latest experiments (see Uncommitted Changes).

## Uncommitted Changes

```
On branch pietro/rbf-kernel

Modified (not staged):
  .claude/rules/working_guidelines.md          # Minor additions
  SESSION_LOG.md                                # Updated log entry
  investigations/rbf_kernel/HANDOFF.md          # This file
  investigations/rbf_kernel/distribution_gradient.py  # Per-sample backward + user param tweaks
  investigations/rbf_kernel/gradient_rbf.py     # Per-sample backward + BOUNDS_MODE + user param tweaks

Untracked:
  investigations/rbf_kernel/distribution_gradient.png  # Generated plot (regenerable)
  investigations/rbf_kernel/explore_utility_rbf.png    # Generated plot (regenerable)
  investigations/rbf_kernel/imgs/                       # User-generated images
```

Key uncommitted code changes in gradient_rbf.py and distribution_gradient.py:
- Per-sample backward closures (replacing single loss.backward())
- Fresh x_query recomputation per MC sample
- **BUG**: A_val/lam0_val NOT yet detached (fix identified, not applied)
- User parameter tweaks (USE_SYNTHETIC=True, BOUNDS_MODE='percentile', LR=0.1, N_LAMBDA_SAMPLES=250, etc.)

## Files Created This Session

| File | Purpose | Keep/Delete |
|------|---------|-------------|
| `investigations/rbf_kernel/distribution_gradient.py` | Multi-image DA utility LBFGS optimization | Keep |
| `investigations/rbf_kernel/distribution_gradient.png` | Generated plot from first run (N=50, flat result) | Optional (regenerable) |

## If Someone Revisits This

**Immediate next step (do this first):**
1. Apply the `.detach()` fix to A_val/lam0_val in BOTH `gradient_rbf.py` and `distribution_gradient.py`. Lines are:
   ```python
   A_val = likelihood.A.squeeze().detach()
   lam0_val = likelihood.lambda0.squeeze().detach()
   ```
2. Test with `SAMPLE_LAMBDA=False` (n_lambda_samples=1) first — should run without error.
3. Then test with `SAMPLE_LAMBDA=True, N_LAMBDA_SAMPLES=250` — should not OOM.

**Then continue exploration:**
- Run distribution_gradient.py with the gray-start and see if optimizer finds anything interesting
- Try different N_SAMPLE values (5, 10, 50, 200) to see how the landscape changes
- Compare single-target (gradient_rbf.py) vs multi-image (distribution_gradient.py) results

**Do NOT retry:**
- Single big `loss.backward()` with large N_SAMPLE/N_LAMBDA_SAMPLES — guaranteed OOM.
- Using `distribution_aware_utility()` directly in the gradient closure — it builds O(N) graphs.

**Key architecture of the per-sample backward:**
```
closure():
    # loss = -utility = -H_marg + E[H_cond]
    # Accumulate d(loss)/dx = -d(H_marg)/dx + d(H_cond)/dx

    x_query = build from opt_var (one graph)
    H_marg = compute_H(get_gp_marginal_moments(model, x_query))
    (-H_marg).backward()  # -d(H_marg)/dx into grad, frees graph

    for each MC sample:
        x_query_i = rebuild from opt_var (fresh graph)
        H_cond_i = compute_H(get_gp_conditional_moments(model, x_query_i, x_i, lambda_i))
        (H_cond_i / n_mc).backward()  # +d(H_cond)/dx into grad, frees graph

    return scalar loss (for LBFGS line search)
```

---

## Continuation Prompt

```
I'm continuing the RBF kernel utility exploration on branch
pietro/rbf-kernel in the worktree at:
/home/idv-eqs8-pza/IDV_code/ClosedLoopProject/gaussian_processes/Spatial_GP_repo/scripts/gpytorch_porting_rbf/Spatial_GP_repo/scripts/gpytorch_porting

Read the handoff at:
investigations/rbf_kernel/HANDOFF.md

IMMEDIATE FIX NEEDED: The per-sample backward approach in both
gradient_rbf.py and distribution_gradient.py has a bug —
A_val/lam0_val are not detached, causing "backward through graph
a second time" error. Fix: add .detach() to both.

Three scripts in investigations/rbf_kernel/:
- explore_utility_rbf.py (utility workbench, complete)
- gradient_rbf.py (single-target LBFGS, per-sample backward needs fix)
- distribution_gradient.py (multi-image LBFGS, per-sample backward needs fix)

Focus: fix the detach bug, then continue image optimization experiments.

Check git status and git branch before starting.

Uncommitted changes include per-sample backward code + user param tweaks.
```
