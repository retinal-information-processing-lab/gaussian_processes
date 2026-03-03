# Handoff: DDPM Fine-tuning and Generation Scripts

**Branch**: `gpytorch_imagenet_diffusion`
**Date**: 2026-03-03
**Status**: Ready for implementation
**Plan file**: `ddpm-imagenet-grayscale/PLAN_finetune_and_generate.md`

---

## Motivation

A pretrained DDPM model (64x64 grayscale, trained on ImageNet by Simone/azeglio on a DGX cluster) was uploaded to `gpytorch_porting/ddpm-imagenet-grayscale/`. The broader goal is to use diffusion models as learned priors over natural images for GP acquisition function optimization -- keeping optimized stimuli on the natural image manifold.

Before any integration with the GP pipeline, we need basic infrastructure: scripts to generate samples from the pretrained model, fine-tune it on the actual experimental images (PNAS dataset, ~3190 images of 108x108 grayscale), and generate from the fine-tuned model. This session explored the model, installed `diffusers`, verified loading works, and produced a detailed implementation plan.

## Decisions and Rationale

### 1. Normalization: divide-by-max, not linear min-max rescaling

PNAS images are z-scored (mean=0, std=1, range [-2.40, 2.48]). The DDPM works in [-1, 1].

**Chosen**: `x_model = x_pnas / 2.478047` (PNAS_ABS_MAX).

**Why**: Preserves zero-mean structure (0 maps to 0). PNAS images are z-scored with mean exactly 0 -- linear rescaling `2*(x-min)/(max-min)-1` would shift the mean to a nonzero value in model space. Dividing by the absolute max maps to [-0.969, 1.0], which is nearly symmetric and simple to invert.

**Rejected**: Linear min-max rescaling (shifts zero-mean), clipping before rescaling (unnecessary since max(abs) covers the range).

### 2. 108x108 -> 64x64 via random crop, not resize

**Chosen**: Random crop during training.

**Why**: Preserves pixel scale (the neuron sees these images at this resolution -- pixel statistics matter). Provides massive data augmentation: (108-64+1)^2 = 2025 possible crops per image, so 3190 images become ~6.5M effective patches. Resize would distort spatial frequencies.

### 3. Three separate scripts, not one unified script

**Chosen**: `generate_samples.py`, `finetune.py`, `generate_samples_finetuned.py` as requested. Script 3 is a thin wrapper (~20 lines) importing from Script 1 with different defaults.

**Why**: User explicitly requested 3 scripts. Script 3 imports `create_parser()` and `main()` from Script 1 to avoid code duplication.

### 4. Fine-tuning hyperparameters

**Chosen**: Adam lr=1e-5, batch_size=16, 50 epochs.

**Why**: The pretrained model was trained on 1.28M images. Fine-tuning on 3K images with a high learning rate risks catastrophic forgetting. 1e-5 is conservative. 50 epochs x ~200 steps/epoch = 10K steps, enough for a small dataset to influence the model without destroying pretrained features. All values are CLI arguments, easily adjusted.

### 5. Use all PNAS splits for fine-tuning (train + val + test)

**Chosen**: Concatenate all 3190 images (2910 train + 250 val + 30 test).

**Why**: The fine-tuning objective is learning the distribution of natural images shown to the neuron, not supervised prediction. Every image helps. With only 3190 total, we need all of them. This is distribution learning, not a train/test split scenario.

### 6. Augmentation: H-flip + V-flip, no rotation

**Chosen**: Random horizontal flip, random vertical flip, random crop.

**Why**: Natural scenes have approximate horizontal symmetry. Vertical flip is debatable (sky vs ground) but at 64x64 patch scale, vertical structure is less dominant. 90/180/270 rotation would break natural scene statistics. The key augmentation is random cropping.

### 7. diffusers installed into pytorch_gpytorch conda env

**Done this session**: `pip install diffusers safetensors accelerate`. Installed diffusers 0.36.0, safetensors 0.7.0, accelerate 1.12.0. The pretrained model was saved with diffusers 0.29.1 but loads fine in 0.36.0 (backward compatible).

## Critical Subtleties

1. **Pipeline pixel convention mismatch**: The DDPMPipeline internally converts model space [-1,1] to display space [0,1] via `(x/2+0.5).clamp(0,1)`. When comparing generated images with real PNAS images, the PNAS images must also be mapped to [0,1]: `x_display = (x_pnas / PNAS_ABS_MAX + 1) / 2`. Getting this wrong makes side-by-side comparisons meaningless (brightness/contrast mismatch).

2. **Training uses UNet+scheduler directly, not the pipeline**: The `DDPMPipeline.__call__()` is for inference only. Training requires calling `scheduler.add_noise()` and `unet()` directly. The pipeline is only used for loading, saving, and generating.

3. **Scheduler clip_sample=True clips the denoising trajectory**: During generation, intermediate samples are clipped to [-1,1]. During training, `add_noise()` does NOT clip -- noisy x_t can be outside [-1,1], which is correct (it's not a clean image). Do not add manual clipping during training.

4. **The training script is NOT present locally**: It was at `/raid/home/azeglio/Simone/DiffusionInformation/NeuronConditionedDM/ImagenetPatchesDMTraining.py` on the training cluster. We cannot inspect the exact training configuration (lr, optimizer, augmentation). The logs reveal batch_size=256*7=1792, 200 epochs, loss plateau at ~0.031.

5. **FID was never computed**: The training script's FID computation failed (feature size 32 vs minimum 64 required by InceptionV3). `fid_curve.png` is blank. There is no quantitative quality metric for this model.

6. **PNAS_ABS_MAX = 2.478047**: This specific value comes from `max(abs(images_train.min()), abs(images_train.max()))` across the full dataset. It must be hardcoded as a constant, not recomputed per-batch.

## Uncommitted Changes

```
Untracked files:
  ddpm-imagenet-grayscale/    # The entire pretrained model directory (uploaded by user)
```

This includes the model weights (~380MB safetensors + 1.1GB checkpoint), training logs, sample images, and the two reference docs created this session (`DDPM_MODEL_REFERENCE.md`, `PLAN_finetune_and_generate.md`).

Note: The large binary files (safetensors, .pt checkpoints) should probably be git-ignored or handled with git-lfs rather than committed directly.

## Files to Read First

1. `ddpm-imagenet-grayscale/PLAN_finetune_and_generate.md` -- the implementation plan with all script specifications, argparse interfaces, data flow diagrams, and implementation order
2. `ddpm-imagenet-grayscale/DDPM_MODEL_REFERENCE.md` -- model architecture, file structure, training history, pixel conventions, known issues
3. `ddpm-imagenet-grayscale/unet/config.json` -- UNet architecture config (confirm 1ch, 64x64, block structure)
4. `ddpm-imagenet-grayscale/scheduler/scheduler_config.json` -- scheduler config (confirm beta schedule, timesteps, clip_sample)

## Caveats and Open Questions

1. **Catastrophic forgetting risk**: With only 3K images and a 99.5M parameter model, fine-tuning could degrade ImageNet features. The low lr (1e-5) and short training (50 epochs) mitigate this, but it needs visual monitoring. The checkpoint sample generation is designed for this purpose.

2. **No FID baseline**: We cannot quantitatively compare pretrained vs fine-tuned generation quality. Evaluation will be visual (side-by-side grids). This is acceptable for exploratory work but should be noted.

3. **PNAS_ABS_MAX precision**: The value 2.478047 was extracted from a quick data inspection. The next session should verify this by loading the npz and computing `max(abs(all_images.min()), abs(all_images.max()))` before hardcoding it.

4. **Large files and git**: The `ddpm-imagenet-grayscale/` directory contains ~1.9GB of binary files. These should not be committed to git as regular files. The user needs to decide: gitignore them, use git-lfs, or keep them untracked.

5. **V-flip augmentation**: Including vertical flips for natural scene patches is debatable. If fine-tuned samples look wrong (inverted sky/ground patterns), removing V-flip is the first thing to try.

---

## Continuation Prompt

```
I'm continuing work on DDPM fine-tuning and generation scripts for a pretrained
diffusion model. This is NOT related to the GP model -- it's about the diffusion
model in ddpm-imagenet-grayscale/.

Read these files first:
1. ddpm-imagenet-grayscale/PLAN_finetune_and_generate.md (implementation plan)
2. ddpm-imagenet-grayscale/DDPM_MODEL_REFERENCE.md (model reference)
3. .claude/handoffs/HANDOFF_2026-03-03_ddpm-finetune-generate-scripts.md (decisions/rationale)

Task: Implement 3 scripts per the plan:
1. generate_samples.py -- generate from pretrained model
2. finetune.py -- fine-tune on PNAS dataset
3. generate_samples_finetuned.py -- generate from fine-tuned model

diffusers is already installed (0.36.0). Model loads with DDPMPipeline.from_pretrained().
Verify PNAS_ABS_MAX by loading the npz before hardcoding.
Check git status and git branch before starting.
Implementation order: Script 1 first, test it, then Script 2, then Script 3.
```
