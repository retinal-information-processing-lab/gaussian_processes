# Investigation: Diffusion Model for Natural Image Generation

**Branch**: `pietro/diffusion-investigation`
**Date**: 2026-03-03
**Status**: Continuing
**Location**: `investigations/diffusion/`
**Worktree**: `/home/idv-eqs8-pza/IDV_code/ClosedLoopProject/gaussian_processes/Spatial_GP_repo/scripts/gpytorch_porting_diffusion/Spatial_GP_repo/scripts/gpytorch_porting/`
**Plan**: `investigations/diffusion/PLAN_diffusion_model_training.md` (also saved on `pietro/workingbranch`)

---

## Problem Statement

We want to train a DDPM (Denoising Diffusion Probabilistic Model) to generate natural image patches that match our PNAS dataset statistics. This is the first step toward using diffusion-guided optimization for GP utility maximization -- but this investigation is purely about getting unconditional generation working. No GP integration yet.

**Motivation**: Unconstrained gradient ascent on GP utility produces smooth, unnatural images (norm exploitation, ~c^1.9 scaling). Subspace approaches (PCA, Fourier, C-eigenspace) only capture second-order statistics. A trained diffusion model provides a score function that encodes the full distributional structure of natural images, acting as a "restoring force" during guided generation.

## What Was Done

### Step 0: Planning and Setup (COMPLETE)
- **What**: Read motivation docs, existing LaTeX reference, utility REFERENCE.md, subspace analysis. Discussed architecture decisions. Created worktree.
- **Key decisions**:
  - 64x64 resolution (not 108x108): power-of-2 for clean U-Net downsampling, massive crop augmentation, faster training
  - Random crops + D4 symmetry (8 variants): effectively unlimited training data from 3,160 base images
  - Pure PyTorch, no HuggingFace/diffusers (no pretrained models exist for our distribution)
  - Cosine noise schedule, T=1000
- **Created**: `investigation_log.md`, `unet_and_ddpm_introduction.tex` (LaTeX reference for U-Net architecture and DDPM practical mechanics)

### Step 1: diffusion_model.py (COMPLETE)
- **What**: Core module with U-Net, noise schedule, forward/reverse process
- **Architecture**: Tiny U-Net, channels (32, 64, 128), 3 downsampling levels (64->32->16->8), sinusoidal time embedding (d=128), GroupNorm, SiLU, skip connections via concatenation
- **Parameters**: 2.16M (slightly above original 0.5-1M estimate, but fine with unlimited augmented data)
- **Verified**: Forward pass, shape correctness, schedule constants (alpha_bar goes from 1.0 to ~0)

### Step 2: train.py (COMPLETE)
- **What**: Training script with data pipeline (random 64x64 crop, D4 augmentation, normalize to [-1,1])
- **Data**: 3,160 images (2,910 train + 250 val combined). Scale factor = 2.4780 (max abs pixel value)
- **Smoke test**: 2 epochs in 2.4s (~1.2s/epoch), loss dropped 0.304 -> 0.169
- **CLI**: `--epochs`, `--batch-size`, `--lr`, `--T`, `--resume`, `--data`, `--print-every`, `--save-every`

### Step 3: sample.py (COMPLETE)
- **What**: Generation and evaluation script with 6 plot types
- **Plots**: generated grid, real grid, side-by-side comparison, pixel histogram, power spectrum, loss curve
- **Smoke test**: Generated 16 images in 1.6s (0.1s/image) from 2-epoch checkpoint. Pipeline works end-to-end.
- **Note**: 2-epoch images are garbage (expected) -- pixel values in [-40k, 76k]. Real training needed.

## Key Findings

1. CONFIRMED: 64x64 U-Net with (32, 64, 128) channels has 2.16M parameters. Forward pass shape-correct.
2. CONFIRMED: Training runs at ~1.2s/epoch on CUDA with batch_size=32. 500 epochs would take ~10 minutes.
3. CONFIRMED: Scale factor for [-1,1] normalization is 2.4780 (max abs value across all 3,160 images).
4. CONFIRMED: Random crop augmentation works -- 10 calls to dataset[0] produce 10 different images.
5. CONFIRMED: Sampling takes ~0.1s per image at 64x64 (1000 reverse steps).
6. CONFIRMED: Dataset has 2,910 train + 250 val = 3,160 total images, all 108x108x1, float32.

## Why This Was Stopped

Context ran out. All code is written and smoke-tested but no real training has been run yet. The next session should run full training (500 epochs) and evaluate the generated images.

## Things Noticed But Not Acted Upon

1. The existing `diffusion_models_introduction.tex` and `motivation_diffusion_guided_optimization.md` reference "30x30 images (~900 pixels)" and "~10,000 images" -- these are WRONG (actual: 108x108, 3,160 images). The docs on `pietro/workingbranch` should be corrected at some point.
2. The parent-level worktree created by `EnterWorktree` tool is still lingering at `/home/idv-eqs8-pza/IDV_code/ClosedLoopProject/.claude/worktrees/diffusion-investigation/`. It has a branch `pietro/diffusion-investigation` in the PARENT repo (not the submodule). This is harmless but should be cleaned up eventually.
3. The `__pycache__/` directory in the diffusion folder is not gitignored (only `checkpoints/` and `samples/` are). Minor -- `.pyc` files are gitignored at the repo root level.

## Uncommitted Changes

None. Working tree clean after commit `bdbba81`.

## Files Created

| File | Purpose | Keep/Delete |
|------|---------|-------------|
| `investigations/diffusion/.gitignore` | Excludes checkpoints/ and samples/ | Keep |
| `investigations/diffusion/investigation_log.md` | Decision log | Keep |
| `investigations/diffusion/unet_and_ddpm_introduction.tex` | LaTeX reference: U-Net architecture + DDPM mechanics | Keep |
| `investigations/diffusion/diffusion_model.py` | Core module: UNet, cosine_schedule, q_sample, p_sample, sample | Keep |
| `investigations/diffusion/train.py` | Training script with data pipeline and augmentation | Keep |
| `investigations/diffusion/sample.py` | Generation, evaluation, and plotting | Keep |
| `investigations/diffusion/PLAN_diffusion_model_training.md` | Full plan (on workingbranch, not in worktree) | Keep |
| `investigations/diffusion/HANDOFF.md` | This file | Keep |

## If Someone Revisits This

**What to do next (in order)**:
1. Run full training: `python investigations/diffusion/train.py --epochs 500`
2. Evaluate: `python investigations/diffusion/sample.py --checkpoint investigations/diffusion/checkpoints/ddpm_epoch0500.pt`
3. Look at the generated grid -- do images look like natural patches? Check power spectrum for 1/f^2 falloff.
4. If quality is poor: try more epochs (1000), or slightly larger batch size (64), or lr=2e-4. Do NOT increase model size first.
5. If quality is good: commit checkpoint reference, update investigation_log.md, and this investigation phase is done. Next phase is GP utility guidance (separate investigation).

**What NOT to try**:
- Don't increase model size to fix poor generation -- train longer first (effectively unlimited data means overfitting is unlikely)
- Don't switch to DDIM or other fast samplers yet -- DDPM at 64x64 is fast enough (<1s/image)
- Don't try MLP architecture -- too many parameters for 64x64 (would need ~50M)

---

## Continuation Prompt

```
I am continuing a diffusion model investigation for natural image generation.
This session works in a git worktree on branch pietro/diffusion-investigation.

Read the handoff first:
  investigations/diffusion/HANDOFF.md

Then read the plan:
  investigations/diffusion/PLAN_diffusion_model_training.md

Key context:
- All code is written and smoke-tested (Steps 0-3 of the plan are COMPLETE)
- NO real training has been run yet -- that is the immediate next step
- Files: diffusion_model.py (UNet + schedule), train.py (data pipeline), sample.py (eval)
- 64x64 grayscale, cosine schedule T=1000, random crop + D4 augmentation from 3,160 PNAS images
- Model: 2.16M params, ~1.2s/epoch on CUDA, ~0.1s/image sampling
- Data path uses absolute path (npz is gitignored, not in worktrees)

Next steps:
1. Run: python investigations/diffusion/train.py --epochs 500
2. Evaluate: python investigations/diffusion/sample.py --checkpoint investigations/diffusion/checkpoints/ddpm_epoch0500.pt
3. Assess generation quality (visual, pixel histogram, power spectrum)

Check git status and git branch before starting.
```
