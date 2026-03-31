# Diffusion Investigation Reference

Single entry point for any session working in `investigations/diffusion/`.

**Branch**: `pietro/diffusion-investigation` (git worktree)
**Worktree root**: `Spatial_GP_repo/scripts/gpytorch_porting_diffusion/Spatial_GP_repo/scripts/gpytorch_porting/`

---

## What This Investigation Contains

Two components:

1. **Unconditional DDPM** (complete): Self-contained 64x64 grayscale image generation trained on the PNAS dataset. Files: `diffusion_model.py`, `train.py`, `sample.py`.

2. **Diffusion-guided GP utility optimization** (in progress): Combines the trained DDPM with the GP utility pipeline. Maximizes distribution-aware utility while regularizing toward the natural image manifold using the diffusion model's learned score. File: `guided_optimization.py`.

---

## Quick Start

```bash
# Validate GP at 64x64 (no diffusion model needed):
python investigations/diffusion/guided_optimization.py --validate-only

# Full run: GP training + utility-only baseline + diffusion-guided optimization:
python investigations/diffusion/guided_optimization.py --crop-size 64 --lambda-diff 1.0

# Utility-only (no diffusion term):
python investigations/diffusion/guided_optimization.py --no-diffusion

# Train DDPM from scratch (saves checkpoints every 100 epochs):
python investigations/diffusion/train.py --epochs 1000

# Generate unconditional samples + evaluation plots:
python investigations/diffusion/sample.py --checkpoint checkpoints/ddpm_epoch1000.pt
```

Environment: `pytorch_gpytorch` conda env (already active). GPU required.

---

## Files

| File | Lines | Purpose |
|------|-------|---------|
| **guided_optimization.py** | 997 | GP-diffusion integration: `setup_gp()`, `compute_tweedie_denoised()`, `gradient_ascent_guided()`, `main()` |
| `diffusion_model.py` | 384 | Core DDPM module: UNet, cosine_schedule, q_sample, p_sample, sample |
| `train.py` | 284 | DDPM training: data pipeline, augmentation, checkpointing |
| `sample.py` | 338 | Unconditional generation + evaluation plots |
| `unet_and_ddpm_introduction.tex` | 374 | LaTeX reference for U-Net architecture and DDPM mechanics |
| `investigation_log.md` | -- | Architecture decision log (DDPM) |
| `.gitignore` | -- | Excludes `checkpoints/` and `samples/` |

---

## guided_optimization.py -- Key Code Locations

| Function/Section | Lines (approx) | Purpose |
|------------------|-----------------|---------|
| `center_crop_images()` | 110-134 | Numpy center-crop, any square size |
| `setup_gp(crop_size)` | 141-378 | COMPLETE. Trains GP on center-cropped PNAS images. Do not modify. |
| `load_diffusion_model()` | 385-418 | Loads UNet + cosine schedule from checkpoint |
| `compute_tweedie_denoised()` | 421-471 | Tweedie score: x_0_hat from noisy input. Detached (no UNet backprop). |
| `_reconstruct_image()` | 478-493 | Reconstructs full image from RF pixels + natural background |
| `gradient_ascent_guided()` | 518-715 | Optimization loop. LBFGS for utility-only, SGD for combined. |
| SGD optimizer setup | ~566 | `SGD(lr=0.01, momentum=0.9)` |
| Sigma scaling | ~918 | `sigma_scaled = SIGMA_SMOOTH * (PNAS_SIZE / n_px_side)` |

### Constants (investigation-specific, top of file)

| Constant | Value | Source |
|----------|-------|--------|
| N_TRAIN | 300 | Same as explore_utility.py (overrides default_params.json n_train=500) |
| M | 300 | Same as explore_utility.py (overrides default_params.json ntilde=100) |
| N_STEPS | 50 | Same as gradient.py |
| LR | 0.5 | LBFGS lr, same as gradient.py |
| TARGET_INDEX | 0 | First pool image as target |
| SIGMA_SMOOTH | 1.0 | Base smoothing sigma (scaled by PNAS_SIZE/crop_size at runtime) |
| T_SCORE | 50 | Timestep for Tweedie denoising |
| LAMBDA_DIFF | 0.01 | Default diffusion regularization weight (override via --lambda-diff) |

### CLI for guided_optimization.py

```
--crop-size N        Square crop size (default: 64, matching diffusion model)
--kernel-type TYPE   arc_cosine | arc_sine | rbf (default: from default_params.json)
--validate-only      Only train GP, report test_r, skip optimization
--no-diffusion       Run utility-only optimization (skip diffusion term)
--lambda-diff FLOAT  Diffusion regularization weight (default: 0.01)
--t-score N          Timestep for Tweedie denoising (default: 50)
--checkpoint PATH    Path to DDPM checkpoint (default: checkpoints/ddpm_epoch1000.pt)
```

---

## How the Combined Optimization Works

### Objective

```
loss = -U_DA(x) + lambda_diff * 0.5 * ||x_rf - x_0_hat_rf||^2
```

- `U_DA(x)`: distribution-aware utility from GP (differentiable through kernel)
- `x_0_hat_rf`: Tweedie-denoised image restricted to RF pixels (detached -- no UNet gradients)
- `x_rf`: the optimization variable (RF pixels only)

### SGD update per step

```
x_rf <-- x_rf + lr * grad(U_DA) + lr * lambda_diff * (x_0_hat_rf - x_rf)
                 ^^^^^^^^^^^^^^^   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                 utility ascent    pull toward diffusion denoised estimate
```

### How the diffusion regularization works (step by step)

The UNet was trained to denoise images: given x_t (a natural image corrupted by noise at level t), predict what noise was added. It cannot process a clean image directly — that doesn't correspond to any noise level it was trained on.

At each optimization step, `compute_tweedie_denoised()` does this:

1. **Take the current image x** (the optimization state, treated as a "clean" image x_0)
2. **Scale to diffusion space**: `x_scaled = x / scale_factor` (raw pixels -> [-1,1] range)
3. **Add artificial noise at timestep t=50**:
   ```
   eps ~ N(0, I)                                         (fresh random noise)
   x_t = sqrt(alpha_bar_t) * x_scaled + sqrt(1 - alpha_bar_t) * eps
   ```
   This is the forward diffusion formula (Eq. 3 in diffusion_models_introduction.tex). We fabricate x_t as if x were a clean natural image that was corrupted by noise.
4. **UNet predicts the noise**: `eps_hat = UNet(x_t, t)`
5. **Tweedie formula recovers the model's estimate of the clean image** (Eq. 8 in the LaTeX):
   ```
   x_0_hat = (x_t - sqrt(1 - alpha_bar_t) * eps_hat) / sqrt(alpha_bar_t)
   ```
   x_0_hat is the UNet's answer to: "given this noisy image x_t, what natural image do you think produced it?"
6. **Scale back to raw pixels**: `x_0_hat_raw = x_0_hat * scale_factor`

Everything runs under `torch.no_grad()` — no gradients flow through the UNet.

The L2 penalty is then:
```
loss_diff = 0.5 * ||x_rf - x_0_hat_rf||^2
```
where both vectors are restricted to RF pixels. The gradient w.r.t. x_rf is simply `(x_rf - x_0_hat_rf)`, pulling x toward the model's "naturalized" version.

**Important**: This is NOT the score function. The diffusion score at noise level t (Eq. 9 in the LaTeX) would be `s(x_t, t) = -eps_theta(x_t, t) / sqrt(1 - alpha_bar_t)`, which acts on the noisy image x_t with specific scaling. Our formulation acts on the clean image x with a simple L2 penalty toward the Tweedie estimate. See Open Question #1 for whether this distinction matters.

**Important**: This is NOT the guided sampling algorithm from Section 6.3 of the LaTeX. That algorithm iterates the full reverse diffusion process (T steps from noise to image), modifying the denoising direction at each step. Our approach is gradient-based optimization on a fixed loss function, using the Tweedie estimate as a regularization target — not as part of a sampling process.

**Fixed seed**: `seed=42` for all steps (deterministic for SGD consistency).

### Image reconstruction

The optimization variable is `x_rf` (RF pixels only, ~21% of 64x64 = ~860 pixels). The full 64x64 image is reconstructed using `x_start` (smoothed target) as background for non-RF pixels. This gives the UNet a natural full image. The GP kernel ignores non-RF pixels regardless (`kernels.py:587-589` masks them out before any computation).

### Optimizer selection

- **Utility-only**: LBFGS with strong_wolfe line search (matches gradient.py)
- **Utility + diffusion**: SGD with momentum=0.9 (see "Dead Ends" below for why not LBFGS or Adam)

---

## GP at 64x64: setup_gp()

`setup_gp(crop_size)` replicates the `run_single_config()` pipeline using importable building blocks, but with center-cropped images. The main GP pipeline hardcodes 108x108 data loading -- this function breaks that limitation.

**How it works**:
1. Load PNAS data (108x108)
2. Center-crop all images to crop_size x crop_size (offset = (108 - crop_size) // 2)
3. Build config via `build_config_from_defaults(mode='default_gpy', n_px_side=crop_size, M=300, n_train=300)`
4. Compute RF center from STA on cropped images
5. Create kernel, select inducing points, build model, train, evaluate

**Neural responses unchanged**: The neuron was shown full 108x108 images when spike counts were recorded. The GP sees a subset of the stimulus.

**Validated results** (cell 8, seed 42):

| crop_size | test_r | reliability | training time |
|-----------|--------|-------------|---------------|
| 64 | 0.7734 | 0.9317 | 3.9s |
| 108 | 0.7155 | 0.9317 | 4.0s |

64x64 actually outperforms 108x108 at M=300 because fewer pixels = easier learning. RF center (0.33, -0.05) is well within the 64x64 crop.

---

## Latest Results (after zero-padding bug fix)

Run: `python guided_optimization.py --crop-size 64 --lambda-diff 1.0`

**Utility-only (LBFGS)**:
- U: 0.00144 -> 0.0418 in 2 steps, then stalls
- proj_coeff: 0.32 -> 6.4 (6.4x norm amplification)
- 48% pixels OOB
- This is the known arc-cosine norm-driven divergence

**Utility + diffusion (SGD lr=0.01, momentum=0.9, lambda=1.0)**:
- U: 0.00144 -> 0.00169 over 50 steps (slow steady increase)
- Pearson r: 0.74 -> 0.65 (structure well-preserved)
- proj_coeff: 0.32 -> 0.44 (no norm amplification -- stays below 1.0)
- L_diff: stable around 0.19-0.26
- **0% OOB pixels**

The diffusion regularization successfully prevents norm amplification and keeps the image within physical pixel bounds.

---

## Fixed Bugs

### Zero-padded UNet input (fixed 2026-03-04)

**Bug**: `_reconstruct_image()` created a zero-padded 64x64 image (zeros outside RF mask, ~79% of pixels). The UNet was trained on full natural images, so this was out-of-distribution input. Convolutions propagated zero-region influence into the RF region's denoised estimate, producing a biased score direction.

**Fix**: `_reconstruct_image()` now accepts an optional `background` parameter. All call sites pass `x_start` (smoothed target), so the UNet always sees a natural full image. The GP kernel ignores non-RF pixels regardless (`kernels.py:587-589`), so utility values and gradients are unchanged.

---

## Known Fragilities

### 1. Tweedie does not prevent norm amplification on its own

The L2 pull `||x - x_0_hat||^2` preserves image structure but NOT amplitude. When x gets amplified (large pixel values), the UNet denoises the amplified-noisy image back to an amplified-clean image. So `x_0_hat` is also amplified, and the L2 penalty doesn't fight the norm growth.

**Current mitigation**: SGD with small lr=0.01 slows the amplification enough that 50 steps stay within bounds. But with more steps or higher lr, amplification will resume.

**Potential fixes not yet tried**:
- Sigmoid pixel bounds (gradient.py has BOUNDS_MODE='dataset' support)
- Normalize x_0_hat to match target norm before computing L2
- Use score direction as unit vector rather than L2 penalty

### 2. SGD learning rate is hand-tuned

`lr=0.01` with `momentum=0.9` works for lambda_diff=1.0 at 50 steps. Not validated for other lambda values, more steps, or different crop sizes. The handoff tested lr=0.1 (too aggressive, 52% OOB) but lr=0.01 was only tested once (this session, after the zero-padding fix).

### 3. GP early-stops at iteration 30

Training consistently early-stops at iteration 30 with loss plateauing at 247.85. This might indicate the GP is undertrained at M=300, n_train=300 on 64x64 images. test_r=0.7734 is good, but more training iterations or different early-stopping parameters might improve it. Not investigated.

### 4. Sigma scaling is a heuristic

Gaussian smoothing for the starting image uses `sigma_scaled = SIGMA_SMOOTH * (PNAS_SIZE / crop_size)`. At crop_size=64, this gives sigma=1.69. At crop_size=108, sigma=1.0 (matching gradient.py). Whether this produces perceptually equivalent perturbation at different crop sizes is unverified.

### 5. UNet is hardcoded for 64x64

The UNet architecture accepts arbitrary spatial dimensions (stride-2 convolutions), but the trained checkpoint is specific to 64x64. Using `compute_tweedie_denoised()` with other crop sizes would require retraining the UNet. The code does not guard against this mismatch -- `setup_gp(crop_size=80)` would work but the diffusion score from the 64x64 UNet would be meaningless.

### 6. tight_layout warning in plots

The subplot grid has incompatible column counts (top row: variable number of images, bottom row: 3 fixed plots). matplotlib's `tight_layout` warns but the plot still renders. Cosmetic issue.

---

## Open Questions (Not Yet Investigated)

### 1. Is the L2 formulation the right way to use the diffusion score?

Current: `loss_diff = 0.5 * ||x_rf - x_0_hat_rf||^2` gives gradient `(x_rf - x_0_hat_rf)`.

The actual diffusion score is `grad log p(x) ~ (x_0_hat - x) / sigma_t^2`. Our formulation omits the `1/sigma_t^2` scaling and uses it as an L2 penalty rather than a score. Whether this matters for the optimization dynamics is unknown.

Alternative formulations:
- Scale by `1/sigma_t^2` (amplifies the pull at low noise levels)
- Use the score direction as a unit vector: `(x_0_hat - x) / ||x_0_hat - x||`
- Langevin-style: add noise proportional to step size alongside the score

### 2. What is the right t_score?

t_score=50 was chosen heuristically. The valid range has hard constraints at both ends:
- **t near 0**: Almost no noise added, so the UNet has nothing to predict. x_0_hat ~ x (degenerate, no useful score signal). The score function becomes trivially zero.
- **t near T=1000**: alpha_bar ~ 0, so 1/sqrt(alpha_bar) diverges. Same amplification problem as in the T-1 sampling fix. Also, the image is mostly noise — the UNet predicts the dataset mean, not a useful direction.

Within that range: lower t = more faithful denoising but weaker regularization. Higher t = stronger regularization but less precise score direction. The optimal t likely depends on lambda_diff and the image content. Not swept.

### 3. Does the diffusion regularization actually buy meaningful naturalness?

Utility-only reaches U=0.042; diffusion-guided reaches U=0.0017. This is a ~25x sacrifice. The science question: does the constrained image (proj_coeff=0.44, 0% OOB) actually look more natural or produce better experimental outcomes than the unconstrained one (proj_coeff=6.4, 48% OOB) when clipped to physical bounds? No neuroscience evaluation done.

### 4. Should non-RF pixels also be optimized by the diffusion term?

Currently only RF pixels are optimized (x_rf). Non-RF pixels are fixed at x_start values. The diffusion model sees and scores the full image, but only RF pixels contribute to the gradient. An alternative: optimize all 4096 pixels, with utility gradient flowing only through RF (kernel masks) and diffusion gradient flowing through all pixels. This would let the diffusion model shape the entire image for naturalness. Not tried.

### 5. Is the fixed Tweedie seed (42) biasing the score direction?

Using `seed=42` for all steps means the same noise realization is used every time. The Tweedie estimate is a function of both the image and the specific noise sample. A fixed seed gives a deterministic but potentially biased score direction. Averaging over multiple seeds per step would give a less biased estimate but at higher cost.

---

## Dead Ends (Do Not Retry As-Is)

| Approach | Why it failed |
|----------|---------------|
| LBFGS with combined utility+diffusion objective | strong_wolfe line search cannot satisfy Armijo+curvature conditions with competing forces. Completely stuck (zero movement across 50 steps). |
| Adam for combined objective | Adaptive learning rate normalizes away lambda_diff. lambda=1.0 and lambda=10.0 produce identical trajectories. |
| Tweedie seed varying per step with LBFGS | Makes the objective stochastic. LBFGS Hessian becomes inconsistent, causes wild oscillation. |

**Caveat**: These were tested with the zero-padded UNet input bug (pre-fix) and with SGD lr=0.1 as baseline. The bug biased the diffusion score direction, and the optimization dynamics may behave differently now. LBFGS and Adam should be revisited with the fixed code before being permanently ruled out — the competing forces that stalled LBFGS may be better balanced now that the UNet sees a natural full image. Similarly, Adam's normalization issue might be workable with a different loss formulation (e.g., separate backward passes for utility and diffusion terms).

---

## DDPM Architecture

Tiny U-Net for noise prediction:

```
Input:  (B, 1, 64, 64)  grayscale image + timestep
Encoder: 3 levels [32, 64, 128 channels], spatial 64->32->16->8
Middle:  128 channels at 8x8
Decoder: 3 levels [128, 64, 32], spatial 8->16->32->64, skip connections via concat
Output: (B, 1, 64, 64)  predicted noise
```

- Constructor: `UNet(in_channels=1, channels=(32, 64, 128), d_emb=128)`
- Parameters: 2.16M
- Time conditioning: sinusoidal embedding (d=128) -> MLP -> added to feature maps
- Normalization: GroupNorm (8 groups), SiLU activation
- Noise schedule: cosine (Nichol & Dhariwal 2021), T=1000

---

## DDPM Data Pipeline

**Source**: PNAS dataset, 3160 images (2910 train + 250 val), 108x108x1, float32.
Path: `~/IDV_code/ClosedLoopProject/gaussian_processes/Spatial_GP_repo/notebooks/PNAS_paper_sorted_data.npz`

**Training augmentation** (applied on-the-fly, different each epoch):
1. Random 64x64 crop from 108x108 (44 possible offsets per axis)
2. Random D4 symmetry transform (4 rotations x 2 flips = 8 variants)

**Normalization to [-1, 1]**:
```
x_norm = x_raw / scale_factor
scale_factor = max(abs(all_pixels)) = 2.4780
```
The scale_factor is computed once from all 3160 images and saved in every checkpoint.

**Denormalization**: `x_raw = x_generated * scale_factor`. No clipping, no sigmoid.

---

## DDPM Training Results

**Hyperparameters**: T=1000, lr=2e-4, batch_size=64, crop_size=64, Adam optimizer.

**Loss progression** (MSE between predicted and true noise):

| Epoch | Loss |
|-------|------|
| 1 | 0.2990 |
| 100 | 0.1364 |
| 500 | 0.1295 |
| 909 | 0.1168 (minimum) |
| 1000 | 0.1249 |

Loss plateaus around 0.12. Slight oscillation is normal (different random crops/augmentations each epoch).

---

## Known Issue: T-1 Sampling Fix

**Problem**: The original sampling loop started at t=T=1000. The cosine schedule at t=1000 has 1/sqrt(alpha) = 31.6, causing divergence at the first reverse step.

**Fix** (applied): Start reverse loop from t=T-1=999. At t=999, 1/sqrt(alpha) = 2.0 (stable).

**Location**: `diffusion_model.py:sample()`, the `for t in range(T - 1, 0, -1)` loop.

No retraining needed.

---

## Normalization Pipeline (No Hidden Clipping)

Full path from model output to plot:

1. `sample()` in `diffusion_model.py`: reverse process produces raw tensor values. No clamp.
2. `generate_samples()` in `sample.py`: multiplies by `scale_factor`. No clamp.
3. `compute_tweedie_denoised()` in `guided_optimization.py`: converts raw -> diffusion space -> Tweedie -> raw. No clamp.

**Caveat on imshow**: matplotlib silently clips pixels outside `[vmin, vmax]` to colormap endpoints. All plotting in `guided_optimization.py` follows the project rule: every subplot checks OOB and flags with red title.

---

## Checkpoint Contents

Each `.pt` file contains:
```python
{
    'epoch': int,              # 0-indexed
    'model': state_dict,       # UNet weights
    'optimizer': state_dict,   # Adam state (for resuming)
    'loss_history': list,      # Per-epoch average MSE loss
    'scale_factor': float,     # 2.4780
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

## DDPM CLI Reference

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

---

## Decisions and Rationale

| Decision | Choice | Why |
|----------|--------|-----|
| Resolution | 64x64 | Power-of-2 for clean U-Net downsampling, random crop augmentation, faster training |
| Augmentation | Random crop + D4 | Effectively unlimited data from 3160 images |
| Framework | Pure PyTorch | No pretrained models for our distribution; diffusers is overkill |
| Schedule | Cosine, T=1000 | Preserves more structure at low noise; standard choice |
| Sampler | DDPM (not DDIM) | Simpler, fast enough at 64x64 |
| Combined optimizer | SGD with momentum | LBFGS fails with competing forces; Adam normalizes away lambda_diff |
| Image background | x_start (smoothed target) | UNet needs natural full image; x_start has smooth transition at RF boundary |
| GP mode | default_gpy | Required for distribution_aware_utility (needs full covariance matrix) |

---

## Deferred (Not For This Investigation)

| Item | Why deferred |
|------|-------------|
| DDIM fast sampling | DDPM is fast enough at 64x64 |
| Classifier-free guidance | Different training regime |
| EMA, attention layers | Overkill for first version |
| vargp_direct support for utility | Needs augmented matrix approach (see CLAUDE.md) |
| Sigmoid pixel bounds in SGD path | Most promising next step but not yet implemented |
| Multi-cell validation | Cell 8 validation sufficient for first version |

---

## Schedule Diagnostics (For Debugging)

Key cosine schedule values at boundary timesteps:

| t | alpha_bar | beta | 1/sqrt(alpha) | beta/sqrt(1-alpha_bar) |
|---|-----------|------|---------------|----------------------|
| 1 | 0.9999 | 0.0000 | 1.000 | 0.006 |
| 100 | 0.9721 | 0.0005 | 1.000 | 0.003 |
| 500 | 0.4938 | 0.0031 | 1.002 | 0.004 |
| 999 | 0.000002 | 0.7500 | 2.000 | 0.750 |
| 1000 | 0.000000 | 0.9990 | **31.623** | 0.999 |

The 31.6x amplification at t=1000 is why we skip it during sampling.
