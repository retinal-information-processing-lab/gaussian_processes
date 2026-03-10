# Handoff: Diffusion Model Training for Natural Image Generation

**Branch**: pietro/workingbranch (need to create `pietro/diffusion-investigation` before implementation)
**Date**: 2026-03-03
**Status**: Ready for implementation
**Plan file**: `investigations/diffusion/PLAN_diffusion_model_training.md`

---

## Motivation

We optimize images x* to maximize a GP-based utility function U(x*) via gradient ascent. The kernel's C matrix acts as a low-pass filter, so optimized images lack high-frequency natural structure (edges, textures). Previous analysis (in `investigations/utility/subspace_optimization_pca_ceigen_fourier.md`) showed that subspace approaches (PCA, C-eigenspace, Fourier) only capture second-order statistics and cannot enforce distributional membership.

The proposed solution: train a diffusion model that learns the full distribution of natural images. The score function provides the "naturalness force" that gradient ascent currently lacks. Eventually, guided diffusion generation will combine the score (naturalness) with the GP utility gradient (informativeness). But first, we need to build and validate the diffusion model itself as a standalone tool.

The user is not a diffusion model expert and explicitly requested that this first step be self-contained — no GP integration, no utility optimization. The goal is to understand the tool deeply before connecting it to the existing codebase.

## Decisions and Rationale

### 1. Train on full 108x108 images (not masked RF region)

**Decided**: Use full 108x108 images for diffusion training.

**Why**: Makes the diffusion model cell-agnostic — train once, reuse across all 41 cells. The RF mask depends on the trained GP model (beta is learned), so training on masked pixels would couple the diffusion model to a specific GP fit. During guided generation, the utility gradient is naturally zero outside the RF, so guidance self-focuses without needing a mask.

**Rejected**: Training on ~2,480 masked RF pixels. Better data/dim ratio and MLP would suffice, but requires retraining per cell and couples to a specific GP model. The motivation doc incorrectly states "30x30 (~900 pixels)" — actual mask has ~2,480 pixels (see `gradient.py:179`).

### 2. Tiny U-Net architecture (not MLP)

**Decided**: Small CNN encoder-decoder with skip connections (~1-2M parameters).

**Why**: At 11,664 dims (108x108), an MLP with 2048-wide hidden layers would have ~50M parameters for only 3,160 training images — severe overfitting risk. A CNN exploits spatial locality with ~30x fewer parameters. The architecture is: 3 down-blocks [1->32->64->128 channels], middle block, 3 up-blocks with skip connections, sinusoidal time embedding.

**Rejected**: MLP (too many parameters for the dataset size). Full U-Net with attention (overkill for 108x108 grayscale).

### 3. Pure PyTorch (no diffusers, no HuggingFace)

**Decided**: Implement from scratch in PyTorch.

**Why**: No pretrained models exist for our specific image distribution (108x108 grayscale, PNAS natural scene patches, specific normalization). The `diffusers` library is designed for large-scale generation (256x256+ RGB, millions of images) and would add dependency complexity without benefit. At this scale, the full DDPM is ~300-400 lines across 3 files. The conda environment has no `diffusers` or `einops` installed.

### 4. Both horizontal and vertical flip augmentation (4x data)

**Decided**: Apply H+V flips -> 12,640 effective training images.

**Why**: User confirmed both flips are acceptable. This substantially improves the data/dim ratio for the small CNN. Natural scenes at 108x108 patch scale are approximately symmetric under both flips.

### 5. Cosine noise schedule, T=1000, DDPM sampling

**Decided**: Standard DDPM with cosine schedule.

**Why**: Cosine schedule is only marginally more complex than linear but produces better results at low noise levels, where fine-grained natural structure lives. DDPM sampling is the simplest approach and fast enough at this scale (~1-2s per image). DDIM deferred until speed matters.

### 6. Self-contained investigation folder (not a separate repo)

**Decided**: Everything lives in `investigations/diffusion/` with 3 Python files.

**Why**: The user asked explicitly — is this a beast needing its own repo? No. At this scale it's 3 files (~300-400 lines total). The existing `investigations/` pattern works. If the approach succeeds, it can graduate to the main codebase later.

## Critical Subtleties

1. **Dataset size is 3,160 images, not 10,000**: The motivation doc (`motivation_diffusion_guided_optimization.md`) states "~10,000 natural images." Actual: 2,910 train + 250 val = 3,160. With 4x augmentation: 12,640. This is workable for a 1-2M param model but is the main quality risk. If generation quality is poor, the symptom will be blurry/noisy samples in Step 3. Mitigation: keep the model small.

2. **Images are 108x108, not 30x30**: Same doc states "30x30 grayscale (~900 pixels)." The 30x30 figure refers to the approximate RF mask region, not the image resolution. The images are 108x108 = 11,664 pixels. This changes the architecture decision (CNN over MLP).

3. **Image normalization**: PNAS images are already normalized (mean~0, range [-2.4, 2.5]). The diffusion forward process adds Gaussian noise assuming input is approximately unit-scale. May need to verify whether the [-2.4, 2.5] range is compatible with standard DDPM assumptions, or whether additional rescaling is needed.

4. **No imports from GP codebase in diffusion code**: The user specifically wants the diffusion model to be a standalone module. The only shared dependency is the `.npz` data file for image loading. Do not import from `run_single_mode.py`, `acquisition.py`, or any GP-related code in the diffusion training/sampling scripts.

5. **LaTeX doc requested**: The user asked for a `unet_and_ddpm_introduction.tex` covering the architecture and practical mechanics of U-Nets and DDPM. This complements the existing `diffusion_models_introduction.tex` (which covers the mathematical theory). Write this as part of Step 0.

## Uncommitted Changes

```
Untracked files:
  investigations/diffusion/   <-- the investigation folder (motivation doc, LaTeX intro, and now the plan file)
  investigations/utility/subspace_optimization_pca_ceigen_fourier.md  <-- analysis that motivated diffusion approach
  experiments/exploratory/2026-02-07_jitter_*  <-- old jitter experiments (unrelated)
  imgs/*.png  <-- plot images (unrelated)
```

No staged changes. No modified tracked files (the `subspace_optimization.py` diff shown in git status is from a prior session).

The `investigations/diffusion/PLAN_diffusion_model_training.md` is an untracked file that should be committed with the new branch.

## Files to Read First

1. **`investigations/diffusion/PLAN_diffusion_model_training.md`** — The implementation plan. Steps 0-3 with folder layout, architecture specs, training params, and verification criteria.
2. **`investigations/diffusion/diffusion_models_introduction.tex`** — Mathematical reference for DDPM: forward process, score matching, reverse sampling, Tweedie estimate, guided generation formulas. Read this before writing the U-Net/DDPM LaTeX doc.
3. **`investigations/diffusion/motivation_diffusion_guided_optimization.md`** — Why diffusion models, what failed before. Note: contains incorrect numbers for image resolution (30x30) and dataset size (10,000) — see Critical Subtleties above.
4. **`investigations/utility/subspace_optimization_pca_ceigen_fourier.md`** — Analysis that motivated the diffusion approach. Explains why PCA/Fourier/C-eigenspace are insufficient (second-order statistics only).

## Caveats and Open Questions

1. **Data sufficiency is uncertain**: 12,640 images (with augmentation) at 108x108 is workable but on the tighter side for diffusion models. The model may produce mediocre generations. This will be immediately visible in Step 3 (sample.py). If quality is poor, options include: smaller architecture, more aggressive augmentation (random crops, brightness jitter), or accepting lower quality as "good enough" for the guidance signal.

2. **Image normalization compatibility with DDPM**: Standard DDPM tutorials assume data in [-1, 1]. Our data range is [-2.4, 2.5]. May need a simple affine rescale to [-1, 1] before training and inverse after sampling. This is a one-line fix but needs to be noticed.

3. **The 108x108 resolution is not a power of 2**: Standard U-Nets often assume power-of-2 resolutions (64, 128, 256) for clean downsampling/upsampling. 108 does not divide evenly by 2 three times (108->54->27->13.5). The architecture will need padding or careful handling of odd dimensions. This is a minor implementation detail but worth noting.

4. **Diffusion model hyperparameters are NOT GP parameters**: The plan specifies T=1000, lr=1e-4, batch_size=32, epochs=500. These are standard diffusion model defaults, not values from `default_params.json`. The diffusion model is a separate system with its own parameter regime. Do not wire these through the GP config system.

5. **No guidance in this phase**: The plan explicitly defers GP utility guidance to a future investigation. The next session should resist the temptation to add guidance prematurely — unconditional generation must work first.

---

## Continuation Prompt

```
I am continuing work on a diffusion model investigation for natural image
generation. This is a self-contained module — no GP integration yet.

Read the handoff first:
  .claude/handoffs/HANDOFF_2026-03-03_diffusion-model-training-investigation.md

Then read the plan:
  investigations/diffusion/PLAN_diffusion_model_training.md

Key context:
- Train a DDPM on 108x108 grayscale PNAS images (3,160 images, 4x with flips)
- Tiny U-Net (~1-2M params), cosine schedule, pure PyTorch, no external libs
- Self-contained in investigations/diffusion/ (3 Python files)
- First task: create branch pietro/diffusion-investigation from pietro/workingbranch
- Then Step 0: folder setup, investigation_log.md, write unet_and_ddpm_introduction.tex
- Check git status and git branch before starting
- Note: images are NOT 30x30 as the motivation doc says — they are 108x108
- Note: 108 is not a power of 2 — handle odd dimensions in U-Net downsampling
```
