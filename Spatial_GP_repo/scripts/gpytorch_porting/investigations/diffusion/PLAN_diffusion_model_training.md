# Diffusion Model Training: Investigation Plan

## Context

We want to explore diffusion models as a tool for generating natural images. Before integrating with GP utility optimization, we need to **understand the tool itself**: train a basic diffusion model, generate images, and verify they look like natural images. This is a self-contained investigation — no GP, no utility, no acquisition functions yet.

---

## What a diffusion model training actually involves

A minimal DDPM has 3 components:

1. **Noise schedule**: Precomputed constants (alpha_bar_t for t=1..T). ~15 lines of code.
2. **Score network**: A neural network that takes (noisy_image, timestep) -> predicted_noise. One file.
3. **Training loop**: Sample image, add noise, predict noise, MSE loss, gradient step. Standard PyTorch.
4. **Sampling loop**: Start from pure noise, iteratively denoise using the trained network. ~30 lines.

No pretrained models to download. No HuggingFace. We train from scratch on our dataset because no pretrained model exists for our specific image distribution. Training time: minutes on RTX 4090.

The whole thing fits in 2-3 files. No separate repo, no complex framework.

---

## Folder organization

```
investigations/diffusion/
    diffusion_model.py          # Network architecture + noise schedule + sampling
    train.py                    # Training script: load images, train, save checkpoint
    sample.py                   # Load checkpoint, generate images, plot grid
    investigation_log.md        # Decisions log (concise)
    checkpoints/                # Saved .pt files (gitignored)
    motivation_diffusion_guided_optimization.md   # (already exists)
    diffusion_models_introduction.tex             # (already exists)
```

Three Python files. Self-contained, no imports from the GP codebase.

---

## Step 0: Branch, folder setup, and LaTeX reference

- Branch `pietro/diffusion-investigation` from `pietro/workingbranch`
- Create folder structure above
- Add `checkpoints/` to `.gitignore`
- Create `investigation_log.md`
- Write `unet_and_ddpm_introduction.tex`: concise LaTeX document explaining:
  - **U-Net architecture**: encoder-decoder with skip connections, why skip connections matter for denoising (preserve spatial detail), how time conditioning is injected, channel progression
  - **DDPM**: forward process (noise addition), reverse process (learned denoising), the training objective (predict noise), the sampling algorithm
  - Figures/diagrams as ASCII or tikz where helpful
  - Complements the existing `diffusion_models_introduction.tex` (which covers the math/theory) by focusing on the **architecture and practical mechanics**

---

## Step 1: `diffusion_model.py` — the core module

Contains three things:

### 1a. Noise schedule
Precompute cosine schedule: alpha_bar_t for t=0..T (T=1000).
From alpha_bar, derive beta_t, sqrt_alpha_bar, sqrt_one_minus_alpha_bar, etc.
This is pure math, no neural network.

### 1b. Score network (tiny U-Net)
A small convolutional encoder-decoder:
- Input: (B, 1, 108, 108) image + timestep embedding
- Encoder: 3 downsampling blocks [1->32->64->128 channels]
- Decoder: 3 upsampling blocks with skip connections [128->64->32->1]
- Time conditioning: sinusoidal embedding -> small MLP -> added to feature maps
- Output: (B, 1, 108, 108) predicted noise
- ~1-2M parameters total

Why CNN and not MLP: at 108x108 (11,664 pixels), an MLP would need ~50M parameters. A CNN exploits spatial structure with ~30x fewer parameters. Critical when we only have ~3,000 training images.

### 1c. Forward process + sampling functions
- `q_sample(x_0, t, noise)`: add noise to clean image at timestep t (one line using alpha_bar)
- `p_sample(model, x_t, t)`: one reverse denoising step
- `sample(model, shape, T_steps)`: full reverse process from noise to image

---

## Step 2: `train.py` — training script

1. Load PNAS images (108x108 grayscale, from .npz file)
2. Apply augmentation (horizontal + vertical flips -> 4x data = ~12,640 images)
3. Training loop:
   - Sample random image x_0 from dataset
   - Sample random timestep t ~ Uniform(1, T)
   - Sample random noise eps ~ N(0, I)
   - Compute x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1-alpha_bar_t) * eps
   - Predict: eps_hat = model(x_t, t)
   - Loss = ||eps_hat - eps||^2
   - Adam step
4. Save checkpoint every N epochs
5. Plot loss curve

**Key parameters**: T=1000, lr=1e-4, batch_size=32, epochs=500 (adjust based on convergence).

---

## Step 3: `sample.py` — generate and evaluate

1. Load trained checkpoint
2. Generate N images (e.g., 64) using the reverse process
3. Display as a grid
4. Compare to real training images (side-by-side grid)
5. Basic quality checks:
   - Pixel value histogram (generated vs real)
   - Power spectrum comparison (should show 1/f^2 falloff)
   - Visual: do they look like natural image patches?

This is where we find out if the model works before even thinking about GP utility.

---

## Deferred (not for this investigation step)

| Item | Why deferred |
|------|-------------|
| GP utility guidance | Separate investigation after unconditional generation works |
| Integration with acquisition.py | Not needed until guidance step |
| DDIM fast sampling | DDPM is simpler and fast enough at this scale |
| Masked RF region training | Full images are simpler and cell-agnostic |
| Classifier-free guidance | Different training regime, not needed for basic generation |
| EMA, attention layers | Production features, overkill for first version |

---

## What success looks like

After these 3 steps, we should be able to:
1. Run `python train.py` and get a trained model in minutes
2. Run `python sample.py` and see a grid of generated images that look like plausible natural image patches
3. Understand concretely how the diffusion model works — noise schedule, denoising, sampling — before connecting it to anything else

---

## Feasibility

| Factor | Assessment |
|--------|-----------|
| Data | 3,160 images -> 12,640 with flips. Sufficient for ~1-2M param model |
| Compute | Training: ~5-15 min on RTX 4090. Sampling: ~1-2s per image |
| Code | ~300-400 lines total across 3 files. Pure PyTorch, no new dependencies |
| Risk | Main risk: generation quality with 12,640 images. Detectable immediately in Step 3 |

---

## Verification

1. Training loss should decrease and plateau
2. Generated images should visually resemble training images (not noise, not memorized copies)
3. Pixel statistics (mean, std, range) of generated images should match training set
4. Power spectrum of generated images should follow 1/f^2 (same as natural images)
