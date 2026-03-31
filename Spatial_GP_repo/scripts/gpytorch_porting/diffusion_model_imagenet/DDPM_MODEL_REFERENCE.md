# DDPM ImageNet Grayscale - Model Reference

**Purpose**: Introductory document for Claude Code sessions working with this pretrained diffusion model. Read this before doing anything with the model.

**Location**: `gpytorch_porting/diffusion_model_imagenet/`

---

## What This Is

A **pretrained DDPM** (Denoising Diffusion Probabilistic Model) that generates 64x64 grayscale image patches. It was trained on the full ImageNet dataset (1.28M images), converted to grayscale. The model uses HuggingFace `diffusers` library format (`DDPMPipeline`).

**Trained by**: Simone (azeglio), on a multi-GPU cluster (DGX, 7 GPUs).
**Training script** (NOT present locally): `/raid/home/azeglio/Simone/DiffusionInformation/NeuronConditionedDM/ImagenetPatchesDMTraining.py`

---

## Quick Reference

| Property | Value |
|----------|-------|
| Framework | `diffusers` 0.29.1 (`DDPMPipeline`) |
| Architecture | `UNet2DModel` with self-attention |
| Image size | **64x64, 1 channel (grayscale)** |
| Diffusion timesteps | 1000 |
| Prediction type | Epsilon (noise prediction) |
| Best training loss | 0.0312 (epoch 187) |
| UNet parameter count | ~99M (estimated from 380MB safetensors) |
| Total training time | ~12.5 hours across 200 epochs on 7 GPUs |

---

## How to Load and Generate

```python
from diffusers import DDPMPipeline
import torch

# Load from local directory
pipeline = DDPMPipeline.from_pretrained('diffusion_model_imagenet')
pipeline = pipeline.to('cuda')  # GPU recommended

# Generate images
output = pipeline(batch_size=4)
images = output.images  # List of PIL Images, 64x64 grayscale
```

**IMPORTANT**: `diffusers` is NOT installed in the `pytorch_gpytorch` conda environment. Install with:
```bash
pip install diffusers safetensors accelerate
```

---

## UNet Architecture

### Channel Progression
```
Input (1ch, 64x64)
  -> DownBlock2D     (64ch,  32x32)
  -> DownBlock2D     (128ch, 16x16)
  -> DownBlock2D     (256ch, 8x8)
  -> AttnDownBlock2D (512ch, 4x4)   <-- SELF-ATTENTION HERE
  -> DownBlock2D     (512ch, 2x2)

  [Mid block at 512ch, 2x2]

  -> UpBlock2D       (512ch, 4x4)
  -> AttnUpBlock2D   (512ch, 8x8)   <-- SELF-ATTENTION HERE
  -> UpBlock2D       (256ch, 16x16)
  -> UpBlock2D       (128ch, 32x32)
  -> UpBlock2D       (64ch,  64x64)
Output (1ch, 64x64)
```

### Architecture Details

| Component | Value |
|-----------|-------|
| Block channels | [64, 128, 256, 512, 512] |
| Layers per block | 2 ResNet layers |
| Activation | SiLU (Swish) |
| Normalization | GroupNorm (32 groups, eps=1e-5) |
| Attention heads | 8 (at 512ch: 8 heads x 64ch each) |
| Attention location | Level 3 only (512 channels, both down and up) |
| Downsampling | Strided convolutions |
| Upsampling | Transposed convolutions |
| Time embedding | Sinusoidal positional (flip_sin_to_cos=true) |
| Dropout | 0.0 (none) |
| Class conditioning | None (unconditional generation) |

---

## Scheduler (Noise Schedule)

| Parameter | Value |
|-----------|-------|
| Type | DDPMScheduler (linear beta schedule) |
| Beta range | [0.0001, 0.02] |
| Timesteps | 1000 |
| Prediction | Epsilon (noise) |
| Variance | Fixed small (not learned) |
| Clip sample | Yes, to [-1.0, 1.0] |
| Thresholding | Disabled |

---

## File Structure

```
diffusion_model_imagenet/
|
|-- model_index.json              # Pipeline config (DDPMPipeline)
|
|-- unet/
|   |-- config.json               # UNet architecture config
|   |-- diffusion_pytorch_model.safetensors  # UNet weights (380 MB)
|
|-- scheduler/
|   |-- scheduler_config.json     # DDPMScheduler config
|
|-- best_model/                   # Identical architecture, best-loss weights
|   |-- model_index.json
|   |-- model.pt                  # Full pipeline state dict (380 MB)
|   |-- unet/
|   |   |-- config.json           # Same as top-level
|   |   |-- diffusion_pytorch_model.safetensors  # Same weights
|   |-- scheduler/
|       |-- scheduler_config.json # Same as top-level
|
|-- checkpoints/
|   |-- final_checkpoint.pt       # Full training state: model + optimizer + scheduler (1.1 GB)
|
|-- logs/
|   |-- training_log_rank{0-6}_*.log  # Multi-GPU training logs (4 runs)
|   |-- train_example/            # TensorBoard event files
|
|-- samples/                      # Individual sample PNGs per epoch (0009.png - 0199.png)
|-- samples_epoch_*.png           # Grid visualizations per epoch
|-- fid_curve.png                 # FID plot (empty -- FID computation failed)
```

### Top-level vs best_model

The configs are **identical**. Both contain the same UNet architecture and scheduler settings. The top-level `unet/` weights and `best_model/` weights are the same size (380 MB). In practice, `best_model/` was saved at the epoch with lowest loss.

### Which weights to load

- **For inference**: Use the top-level directory (`DDPMPipeline.from_pretrained('diffusion_model_imagenet')`)
- **For resuming training**: Use `checkpoints/final_checkpoint.pt` (contains optimizer state)
- **For best-loss model specifically**: Use `best_model/` as a standalone pipeline

---

## Training History

### 4 Training Runs (all on 2025-04-09)

| Run | Epochs | Duration | Notes |
|-----|--------|----------|-------|
| 1 | 0-9 | ~34 min | Fresh start |
| 2 | 10-74 | 4h 6m | Resumed; stopped early (FID error triggered early stopping) |
| 3 | 75-149 | 4h 55m | Resumed; stopped early (same FID error) |
| 4 | 150-199 | 3h 7m | Resumed; completed all 200 epochs |

**Total wall time**: ~12.5 hours

### Training Configuration

| Setting | Value |
|---------|-------|
| Dataset | ImageNet from HuggingFace (1,281,167 train / 50,000 val) |
| Preprocessing | Grayscale conversion, resize/crop to 64x64 |
| GPUs | 7 (distributed, MULTI_GPU) |
| Batch size per GPU | 256 |
| **Total batch size** | **1,792** |
| Steps per epoch | 714 |
| Total epochs | 200 |

### Loss Progression

```
Epoch   0:  0.0867  (initial)
Epoch   1:  0.0368  (big drop)
Epoch   5:  0.0341
Epoch  10:  0.0332
Epoch  50:  0.0321  (plateau begins)
Epoch 100:  0.0317
Epoch 150:  0.0316
Epoch 187:  0.0312  (BEST)
Epoch 199:  0.0317  (final)
```

Loss converges quickly (epoch 0 -> 1 drops from 0.087 to 0.037), then slowly improves from ~0.032 to ~0.031 over the remaining 190 epochs.

### FID Score

**Not available.** FID computation failed at every attempt with:
```
ValueError: Integer input to argument `feature` must be one of (64, 192, 768, 2048), but got 32.
```
The training script used `feature=32` for the FID metric, but `torchmetrics.FrechetInceptionDistance` only supports InceptionV3 feature sizes >= 64. The `fid_curve.png` is blank.

### Sample Quality

Epoch 9 samples: noisy, mostly incoherent blobs.
Epoch 199 samples: recognizable natural scene structure -- textures, edges, objects, spatial coherence. Good quality for 64x64 grayscale.

---

## Known Issues

1. **`diffusers` not installed** in `pytorch_gpytorch` conda env. Needs `pip install diffusers safetensors accelerate`.

2. **Checkpoint symlink bug**: The training script had a `FileExistsError` when creating `latest_checkpoint.pt` symlink. Non-fatal -- weights saved correctly, just the convenience symlink failed.

3. **FID never computed**: Due to feature size mismatch (32 vs minimum 64). No quantitative generation quality metric available.

4. **Training script not local**: The script that trained this model lives at `/raid/home/azeglio/Simone/DiffusionInformation/NeuronConditionedDM/ImagenetPatchesDMTraining.py` on the training cluster. Not present in this directory.

---

## Pixel Value Convention

The scheduler config specifies:
- `clip_sample: true`, `clip_sample_range: 1.0`
- This means the model works in **[-1, 1] pixel space**

When using generated images with other code that expects different ranges (e.g., [0, 1] or the PNAS dataset range), rescaling will be needed.

---

## Connection to the GP Project

This model sits inside `gpytorch_porting/` but is **independent** of the GP codebase. There are no imports, no shared code, no integration yet.

**Intended use** (context from the broader project): The diffusion model can serve as a learned prior over natural images, potentially useful for:
- Guided optimization of GP acquisition functions (keep optimized stimuli on the natural image manifold)
- Score-function-based constraints during utility maximization
- Generating synthetic training images that look natural

None of this is implemented yet -- the model is a standalone artifact ready for future integration.

---

*Document created: 2026-03-03*
*Based on: training logs, config files, and model artifacts in diffusion_model_imagenet/*
