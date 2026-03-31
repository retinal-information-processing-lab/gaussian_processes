# Diffusion Model Investigation Reference

Single entry point for any session working in `investigations/diffusion/`.
Self-contained DDPM for 64x64 grayscale natural image generation. No GP integration -- purely unconditional generation from the PNAS dataset.

**Branch**: `pietro/diffusion-investigation` (git worktree)
**Worktree root**: `Spatial_GP_repo/scripts/gpytorch_porting_diffusion/Spatial_GP_repo/scripts/gpytorch_porting/`
**LaTeX reference**: `unet_and_ddpm_introduction.tex` (U-Net architecture + DDPM mechanics, complements the math in `diffusion_models_introduction.tex`)

---

## Quick Start

```bash
# Train (saves checkpoints every 100 epochs)
python train.py --epochs 1000

# Generate + evaluate (produces 6 plots in samples/)
python sample.py --checkpoint checkpoints/ddpm_epoch1000.pt

# Resume training from checkpoint
python train.py --epochs 2000 --resume checkpoints/ddpm_epoch1000.pt
```

Environment: `pytorch_gpytorch` conda env (already active). GPU required.

---

## Files

| File | Lines | Purpose |
|------|-------|---------|
| `diffusion_model.py` | 384 | Core module: UNet, cosine_schedule, q_sample, p_sample, sample |
| `train.py` | 284 | Training: data pipeline, augmentation, training loop, checkpointing |
| `sample.py` | 338 | Generation + evaluation: image grids, pixel histogram, power spectrum, loss curve |
| `unet_and_ddpm_introduction.tex` | 374 | LaTeX reference for U-Net architecture and DDPM practical mechanics |
| `investigation_log.md` | 35 | Architecture decision log |
| `PLAN_diffusion_model_training.md` | 155 | Original investigation plan (Steps 0-3) |
| `HANDOFF.md` | 126 | First session handoff (outdated -- pre-fix) |
| `.gitignore` | | Excludes `checkpoints/` and `samples/` |

No imports from the GP codebase. Completely self-contained.

---

## Architecture

Tiny U-Net for DDPM noise prediction:

```
Input:  (B, 1, 64, 64)  grayscale image + timestep
Encoder: 3 levels [32, 64, 128 channels], spatial 64->32->16->8
Middle:  128 channels at 8x8
Decoder: 3 levels [128, 64, 32], spatial 8->16->32->64, skip connections via concat
Output: (B, 1, 64, 64)  predicted noise
```

- Parameters: 2.16M
- Time conditioning: sinusoidal embedding (d=128) -> MLP -> added to feature maps
- Normalization: GroupNorm (8 groups), SiLU activation
- Noise schedule: cosine (Nichol & Dhariwal 2021), T=1000

---

## Data Pipeline

**Source**: PNAS dataset, 3160 images (2910 train + 250 val), 108x108x1, float32.
Path: `~/IDV_code/ClosedLoopProject/gaussian_processes/Spatial_GP_repo/notebooks/PNAS_paper_sorted_data.npz`

**Augmentation** (applied on-the-fly, different each epoch):
1. Random 64x64 crop from 108x108 (44 possible offsets per axis)
2. Random D4 symmetry transform (4 rotations x 2 flips = 8 variants)

This gives effectively unlimited unique training samples from 3160 base images.

**Normalization to [-1, 1]**:
```
x_norm = x_raw / scale_factor
scale_factor = max(abs(all_pixels)) = 2.4780
```
The scale_factor is computed once from all 3160 images and saved in every checkpoint.

**Denormalization after sampling**:
```
x_raw = x_generated * scale_factor
```
No clipping, no sigmoid, no renormalization. The model's raw output (after denormalization) goes directly to plots and analysis.

---

## Normalization Pipeline (Important -- No Hidden Clipping)

The full path from model output to plot:

1. `sample()` in `diffusion_model.py`: reverse process produces raw tensor values. **No clamp.**
2. `generate_samples()` in `sample.py`: multiplies by `scale_factor`. **No clamp.**
3. `plot_pixel_histogram()`: plots raw values. **No clipping.**
4. `plot_image_grid()` / `plot_comparison()`: uses `imshow(vmin=real.min(), vmax=real.max())`.

**Caveat on imshow**: matplotlib's `imshow` with fixed vmin/vmax **silently clips** pixels outside that range to the colormap endpoints (pure black / pure white). This is visual clipping only -- the underlying data is unchanged. Generated images sometimes have a small fraction of pixels outside the real data range (slightly fatter tails). These appear as black/white in the grid plots but are visible in the histogram.

The histogram is the honest view -- always check it for the true pixel distribution.

---

## Known Issue: T-1 Sampling Fix

**Problem**: The original sampling loop started at t=T=1000. The cosine schedule at t=1000 has:
- beta_1000 = 0.999 (clipped maximum)
- alpha_1000 = 0.001
- The reverse formula prefactor 1/sqrt(alpha) = 1/sqrt(0.001) = 31.6

This 31.6x amplification at the very first reverse step causes divergence: even tiny noise prediction errors get blown up and compound through all 999 remaining steps. Result: generated pixel values in [-1200, +200] instead of [-2.5, +2.5].

**Fix** (applied): Start the reverse loop from t=T-1=999 instead of t=T=1000. At t=999, the prefactor is 1/sqrt(0.25) = 2.0, which is stable. Physically, alpha_bar_T ~ 0 means the image at t=T is already pure noise -- there's nothing meaningful to denoise at that step.

**Location**: `diffusion_model.py:sample()`, the `for t in range(T - 1, 0, -1)` loop with explanatory comment.

No retraining needed. The model weights are fine; this was purely a numerical issue in the sampling formula.

---

## Training Results

**Hyperparameters** (from checkpoints):
- T=1000, lr=2e-4, batch_size=64, crop_size=64
- Optimizer: Adam
- Checkpoints saved every 100 epochs

**Loss progression** (MSE between predicted and true noise):

| Epoch | Loss | Notes |
|-------|------|-------|
| 1 | 0.2990 | Initial |
| 100 | 0.1364 | Rapid descent phase |
| 253 | 0.1224 | Min for 500-epoch run |
| 500 | 0.1295 | Slight increase (noise floor) |
| 909 | 0.1168 | Min for 1000-epoch run |
| 1000 | 0.1249 | Final |

Loss plateaus around 0.12. The slight oscillation is normal -- each epoch sees different random crops and augmentations.

**Generation quality at 1000 epochs**:
- Pixel histogram closely matches real data distribution
- Generated range slightly wider than real (fatter tails, ~5% of pixels OOB)
- Generated mean slightly more negative than real (small dark bias)
- Images show natural-looking texture and spatial structure

---

## CLI Reference

### train.py

```
--epochs N          Number of training epochs (default: 500)
--batch-size N      Batch size (default: 32)
--lr FLOAT          Learning rate (default: 1e-4)
--T N               Diffusion timesteps (default: 1000)
--print-every N     Print loss every N epochs (default: 10)
--save-every N      Save checkpoint every N epochs (default: 100)
--resume PATH       Resume from checkpoint
--data PATH         Path to PNAS .npz file
```

### sample.py

```
--checkpoint PATH   Path to trained checkpoint (required)
--n-samples N       Number of images to generate (default: 64)
--save-dir PATH     Output directory (default: samples/)
--data PATH         Path to PNAS .npz file
--seed N            Random seed (default: 42)
```

Output plots (saved to `samples/` with epoch prefix):
- `*_generated_grid.png` -- 8x8 grid of generated images
- `*_real_grid.png` -- 8x8 grid of real images (center-cropped 64x64)
- `*_comparison.png` -- generated (top 2 rows) vs real (bottom 2 rows)
- `*_pixel_histogram.png` -- pixel value distributions overlaid
- `*_power_spectrum.png` -- azimuthally-averaged power spectrum (should show 1/f^2)
- `*_loss_curve.png` -- training loss over epochs

---

## Checkpoint Contents

Each `.pt` file contains:
```python
{
    'epoch': int,              # 0-indexed epoch number
    'model': state_dict,       # UNet weights
    'optimizer': state_dict,   # Adam state (for resuming)
    'loss_history': list,      # Per-epoch average MSE loss
    'scale_factor': float,     # 2.4780 (for denormalization)
    'config': {
        'T': 1000,
        'lr': float,
        'batch_size': int,
        'crop_size': 64,
        'n_images': 3160,
    },
}
```

---

## Decisions and Rationale

| Decision | Choice | Why |
|----------|--------|-----|
| Resolution | 64x64 (not 108x108) | Power-of-2 for clean U-Net downsampling, random crop augmentation, faster training |
| Augmentation | Random crop + D4 | Effectively unlimited data from 3160 images |
| Framework | Pure PyTorch | No pretrained models exist for our distribution; diffusers is overkill |
| Schedule | Cosine, T=1000 | Preserves more structure at low noise; standard choice |
| Sampler | DDPM (not DDIM) | Simpler, fast enough at 64x64 (<0.1s/image) |
| Model size | 2.16M params | Small but sufficient with unlimited augmented data |

---

## Deferred (Not For This Investigation)

| Item | Why deferred |
|------|-------------|
| GP utility guidance | Separate investigation after unconditional generation works |
| Integration with acquisition.py | Not needed until guidance step |
| DDIM fast sampling | DDPM is fast enough at this scale |
| Classifier-free guidance | Different training regime |
| EMA, attention layers | Production features, overkill for first version |
| Pixel clipping flags on imshow | Minor -- histogram provides honest view |

---

## If Generation Quality Is Poor

Try these in order:
1. Train longer (2000, 5000 epochs) -- overfitting is unlikely with augmentation
2. Increase batch size (64 -> 128)
3. Try lr=2e-4 (slightly higher)
4. Do NOT increase model size first -- train longer instead

---

## Schedule Diagnostics (For Debugging)

Key schedule values at boundary timesteps:

| t | alpha_bar | beta | 1/sqrt(alpha) | beta/sqrt(1-alpha_bar) |
|---|-----------|------|---------------|----------------------|
| 1 | 0.9999 | 0.0000 | 1.000 | 0.006 |
| 100 | 0.9721 | 0.0005 | 1.000 | 0.003 |
| 500 | 0.4938 | 0.0031 | 1.002 | 0.004 |
| 999 | 0.000002 | 0.7500 | 2.000 | 0.750 |
| 1000 | 0.000000 | 0.9990 | **31.623** | 0.999 |

The 31.6x amplification at t=1000 is why we skip it during sampling.
