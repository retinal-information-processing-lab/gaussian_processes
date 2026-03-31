# Investigation: Diffusion Model for Natural Image Generation

**Branch**: `pietro/diffusion-investigation`
**Date**: 2026-03-03
**Status**: Continuing
**Location**: `investigations/diffusion/`
**Worktree**: `Spatial_GP_repo/scripts/gpytorch_porting_diffusion/Spatial_GP_repo/scripts/gpytorch_porting/`
**Reference**: `investigations/diffusion/REFERENCE.md` -- single entry point for all architecture, pipeline, CLI, and decision details.

---

## Problem Statement

Train a DDPM to generate 64x64 natural image patches matching PNAS dataset statistics. This is the first step toward diffusion-guided optimization for GP utility maximization. This investigation covers unconditional generation only -- no GP integration yet.

Motivation: unconstrained gradient ascent on GP utility produces smooth, unnatural images (norm exploitation, ~c^1.9 scaling). A trained diffusion model provides a score function encoding full distributional structure of natural images.

## What Was Done

### Session 1: Code Implementation (Steps 0-3 of Plan)
- **What**: Wrote all code from scratch (diffusion_model.py, train.py, sample.py), smoke-tested
- **Result**: 2.16M param U-Net, ~1.2s/epoch on CUDA, pipeline works end-to-end
- **Verdict**: Complete

### Session 2: Training, Bug Fix, Evaluation
- **What**: Ran 500-epoch and 1000-epoch training. Discovered generated images were all black.
- **Root cause**: Cosine schedule at t=T=1000 has beta=0.999, making 1/sqrt(alpha) = 31.6. The reverse formula amplifies any noise prediction error by 31x at the first step, causing cascading divergence. Generated pixels ended up in [-1200, +200] instead of [-2.5, +2.5].
- **Fix**: Start reverse sampling loop from t=T-1=999 instead of t=T=1000. At t=999, the amplification is 2.0x (stable). No retraining needed -- model weights were fine.
- **Result after fix**: Generated images in [-3.1, 2.9] (real: [-2.4, 2.5]). Pixel histogram closely matches real data. Loss plateau at ~0.12 (MSE).
- **Verdict**: Unconditional generation works. Ready for GP integration.

## Key Findings

1. CONFIRMED: The T-1 sampling fix resolves divergence completely. Model predictions at t=1000 are accurate (error std=0.01) but 31.6x amplification makes it numerically unstable. See REFERENCE.md "Known Issue: T-1 Sampling Fix".
2. CONFIRMED: No clipping or renormalization anywhere in the pipeline. The pixel distribution match is genuine model quality. Verified by tracing the full path from model output to plot.
3. CONFIRMED: Training loss plateaus around 0.12 for both 500 and 1000 epochs (min 0.1224 at epoch 253, min 0.1168 at epoch 909). Slight oscillation is normal with random augmentation.
4. CONFIRMED: imshow with fixed vmin/vmax silently clips outlier pixels to colormap endpoints. The histogram is the honest view of the pixel distribution.
5. CONFIRMED: Generated images have slightly fatter tails and a small negative mean bias (-0.79 vs -0.07 at 500 epochs) compared to real data. Improves with more training.

## Why This Was Stopped

Context limit approaching. Unconditional generation is working. Next phase is integrating the diffusion model's score function with GP utility optimization -- a separate investigation scope.

## Things Noticed But Not Acted Upon

1. The existing docs on `pietro/workingbranch` (`diffusion_models_introduction.tex`, `motivation_diffusion_guided_optimization.md`) reference "30x30 images (~900 pixels)" and "~10,000 images" -- these are WRONG (actual: 108x108, 3,160 images). Should be corrected.
2. Training ran with lr=2e-4 and batch_size=64 (not the defaults of lr=1e-4 and batch_size=32 from the plan). The user likely set these via CLI. The checkpoints record the actual values used.
3. The parent-level worktree at `/home/idv-eqs8-pza/IDV_code/ClosedLoopProject/.claude/worktrees/diffusion-investigation/` is still lingering. Harmless but should be cleaned up.

## Uncommitted Changes

```
modified:   diffusion_model.py        # T-1 sampling fix (6 lines changed)
untracked:  PLAN_diffusion_model_training.md   # Copied from main worktree
untracked:  REFERENCE.md              # New -- comprehensive reference doc
```

All three should be committed.

## Files Created (This Session)

| File | Purpose | Keep/Delete |
|------|---------|-------------|
| `REFERENCE.md` | Comprehensive reference doc -- single entry point for future sessions | Keep |
| `PLAN_diffusion_model_training.md` | Copied from main worktree (was missing in this worktree) | Keep |

## If Someone Revisits This

**Next step**: Integrate the trained diffusion model with GP utility optimization. The diffusion model provides a score function (gradient of log p(x)) that can act as a regularizer during gradient ascent on utility, pushing optimized images toward the natural image manifold.

**What to try**:
1. Use the trained model's score function to guide image optimization in acquisition.py
2. The score at a given image x and noise level t is: score = -eps_theta(x_t, t) / sqrt(1 - alpha_bar_t)
3. Start with the 1000-epoch checkpoint (`checkpoints/ddpm_epoch1000.pt`)

**What NOT to try**:
- Don't increase model size to improve generation quality -- train longer first
- Don't switch to DDIM sampling -- DDPM is fast enough at 64x64
- Don't try to fix the t=T step with tighter beta clipping -- the T-1 fix is clean and standard

**Prerequisites for GP integration**:
- The diffusion model operates on 64x64 crops. GP utility operates on 108x108 images (or their projections). Resolution bridging will be needed.
- The diffusion model normalizes to [-1,1] via scale_factor=2.4780. The GP codebase uses raw pixel values. Keep normalization/denormalization explicit.

---

## Continuation Prompt

```
I am continuing the diffusion model investigation, moving to GP utility integration.
This session works in a git worktree on branch pietro/diffusion-investigation.

Read the reference first:
  investigations/diffusion/REFERENCE.md

Key context:
- Unconditional 64x64 DDPM generation is WORKING (1000-epoch checkpoint)
- T-1 sampling fix applied (diffusion_model.py), not yet committed
- Files: diffusion_model.py (UNet + schedule), train.py (data pipeline), sample.py (eval)
- Model: 2.16M params, cosine schedule T=1000, scale_factor=2.4780
- No GP integration yet -- that is the next step
- Resolution mismatch: diffusion is 64x64, GP utility is 108x108

Uncommitted changes: diffusion_model.py (T-1 fix), REFERENCE.md (new), PLAN copy.
Commit these first, then proceed with GP integration planning.

Check git status and git branch before starting.
```
