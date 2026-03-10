# Investigation: Synthetic Image Generator Package for External User

**Branch**: pietro/workingbranch
**Date**: 2026-03-10 (scoping started 2026-03-09)
**Status**: Continuing (in new session, different worktree)
**Location**: Work continues in `../gpytorch_imagenet_diffusion/` worktree

---

## Problem Statement

An external collaborator already has a GP inference package (`~/gp_neural_fitting.tar.gz`, validated). They also need to generate large batches of synthetic 64x64 grayscale natural images matching van Hateren/PNAS statistics, to pretrain their own neural network. A fine-tuned DDPM diffusion model exists and works, but it's embedded in a development repo with hardcoded paths and machine-specific features. The task is to package it as a clean standalone tarball.

## What Was Tried

### Approach 1: Assess whether to merge into GP package
- **What**: Evaluated merging diffusion model into the existing GP tarball
- **Result**: Rejected. Different purpose (image generation vs GP inference), different deps (diffusers vs gpytorch), no shared code. Would create a 700+ MB blob.
- **Interpretation**: Separate tarballs is the clean approach.
- **Verdict**: Decision made — separate tarball.

### Approach 2: Assess where to work
- **What**: Evaluated working from gpytorch_porting vs starting a new session in the diffusion worktree (`../gpytorch_imagenet_diffusion/`)
- **Result**: New session in diffusion worktree is better. All source material (generate_samples.py, model weights, docs) is local there. This session's context is loaded with GP internals (4000+ lines of rules about eigenspace, jitter, acquisition functions) that are pure noise for diffusion packaging.
- **Interpretation**: Clean context = faster, less error-prone work.
- **Verdict**: New session in diffusion worktree. Comprehensive handoff written there.

### Approach 3: Scope pixel range output
- **What**: Asked user whether NPZ should use raw uint8 [0,255], PNAS-normalized [-2.478, 2.478], or model space [-1,1]
- **Result**: User chose **selectable via CLI flag** — `--pixel-range {uint8, pnas}`. For PNAS mode, use exact dataset range (`PNAS_ABS_MAX = 2.478047`). README must state GP expects PNAS-normalized inputs.
- **Interpretation**: Flexibility without complexity. Two clear modes.
- **Verdict**: Decision made.

### Approach 4: Scope synthetic dataset script
- **What**: Asked whether a script to run generated images through GP (producing images + predicted spike counts) is in scope
- **Result**: User chose **deferred**. Just the image generator.
- **Interpretation**: Keep deliverable focused. GP inference on synthetic images is a future task.
- **Verdict**: Decision made — out of scope.

## Key Findings

1. **CONFIRMED**: The diffusion worktree (`../gpytorch_imagenet_diffusion/`) is a git worktree (not a separate repo). Branch: `gpytorch_imagenet_diffusion`. Contains all source material: `generate_samples.py` (~200 lines), model weights (~380 MB safetensors), docs (DDPM_MODEL_REFERENCE.md, PLAN_finetune_and_generate.md).

2. **CONFIRMED**: `generate_samples.py` has two things to strip: `PNAS_DATA_PATH` (hardcoded absolute path to this machine) and `load_pnas_comparison()` (loads real PNAS images for side-by-side grid — external user won't have this data).

3. **CONFIRMED**: The conda+pip pattern for environment.yml works (validated for GP package this session). Conda-only for PyTorch+CUDA fails with `LibMambaUnsatisfiableError`. The diffusion env should use the same pattern: conda for basics, pip with `--extra-index-url https://download.pytorch.org/whl/cu121` for torch, diffusers, safetensors, accelerate.

4. **CONFIRMED**: Model weights to copy are only: `ddpm-pnas-finetuned/model_index.json`, `unet/` (config.json + diffusion_pytorch_model.safetensors), `scheduler/` (scheduler_config.json). Training checkpoints (`checkpoint_epoch_010` through `_050`) are NOT needed.

5. **CONFIRMED**: Pixel conversion chain: diffusion model [-1,1] -> pipeline postprocess [0,1] -> PIL [0,255]. To get PNAS range: `x_pnas = (2 * pixel_01 - 1) * PNAS_ABS_MAX` where `PNAS_ABS_MAX = 2.478047`.

6. **CONFIRMED**: The fine-tuned model's sample quality was visually verified in a prior session (sample grid PNGs exist at `ddpm-pnas-finetuned/samples_epoch_*.png`). No FID metric available.

## Why This Was Stopped

Context ran out (session started with GP package validation, compacted, then scoped this deliverable). All decisions are finalized. The work is ready to implement in a new session in the diffusion worktree.

## Things Noticed But Not Acted Upon

1. The CLAUDE.md in the diffusion worktree is identical to the GP porting CLAUDE.md (same branch content, different worktree). The new session should ignore all GP-specific rules — they're irrelevant for this task.

2. `diffusers` version in the working env is 0.36.0 (installed during a prior session). The environment.yml should pin this version for reproducibility.

3. The existing `generate_samples_finetuned.py` is a thin wrapper (~20 lines) that imports from `generate_samples.py` and changes defaults. It can be used as a reference but won't be included in the tarball.

4. There are uncommitted changes in gpytorch_porting from multiple prior sessions (analyze_experiment.py, utility investigation scripts). These are unrelated to the diffusion packaging task.

## Uncommitted Changes

```
 D .claude/handoffs/HANDOFF_2026-03-09_inference-package.md    # Deleted this session (user requested)
 D .claude/skills/handoff-investigation/SKILL.md               # Prior session deletion
 D .claude/skills/handoff-plan/SKILL.md                        # Prior session deletion
 M SESSION_LOG.md                                              # Updated this session
 M analyze_experiment.py                                       # Prior session changes
 M investigations/utility/gradient_ascent.py                   # Prior session changes
 M investigations/utility/subspace_optimization.py             # Prior session changes
 M investigations/utility/workbench.py                         # Prior session changes
?? .claude/handoffs/HANDOFF_2026-03-09_synthetic-image-generator-package.md  # This file
?? investigations/diffusion/                                   # Prior session (background docs)
```

## Files Created

| File | Purpose | Keep/Delete |
|------|---------|-------------|
| `.claude/handoffs/HANDOFF_2026-03-09_synthetic-image-generator-package.md` (this repo) | This handoff — session wrap-up for gpytorch_porting | Keep |
| `.claude/plans/giggly-riding-cook.md` (this repo) | Plan file with finalized decisions | Keep (reference) |
| `.claude/handoffs/HANDOFF_2026-03-10_synthetic-image-generator-package.md` (diffusion worktree) | Comprehensive technical handoff for new session | Keep (primary working doc) |

## If Someone Revisits This

**What to do**: Start a new Claude Code session in `../gpytorch_imagenet_diffusion/Spatial_GP_repo/scripts/gpytorch_porting/`. Read the handoff there: `.claude/handoffs/HANDOFF_2026-03-10_synthetic-image-generator-package.md`. It contains all decisions, source file locations, pixel conventions, environment.yml pattern, draft implementation steps, and verification checklist.

**What NOT to do**: Do not work from gpytorch_porting — all diffusion source material is in the other worktree. Do not implement a synthetic dataset script (GP inference on generated images) — that's deferred. Do not try conda-only for PyTorch — it fails.

**The task is small**: adapt one script (~200 lines), create environment.yml + README, copy ~380 MB model weights, tar it up. Should complete in one session.

---

## Continuation Prompt

```
I'm packaging a fine-tuned diffusion model as a standalone tarball for an external user.

Read the handoff: .claude/handoffs/HANDOFF_2026-03-10_synthetic-image-generator-package.md

Summary: Adapt the existing generate_samples.py into a standalone generate_images.py,
strip machine-specific paths, add --pixel-range {uint8, pnas} flag with NPZ output,
create environment.yml + README, copy model weights, package as ~/synthetic_image_generator.tar.gz.

All decisions are finalized in the handoff. The draft implementation steps are there too.
Read them, refine if needed, then implement.

IMPORTANT: The CLAUDE.md in this worktree is about the GP porting project — ignore all GP-specific
rules. This task is purely about packaging a diffusion model.

Check git branch before starting. Do NOT commit to main.
```
