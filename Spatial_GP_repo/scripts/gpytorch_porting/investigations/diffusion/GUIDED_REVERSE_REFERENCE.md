# Guided Reverse Diffusion (Approach D) -- Reference

Single entry point for any session working on `investigations/diffusion/guided_reverse.py`.

**Branch**: `pietro/approach-d-guided-sampling` (git worktree at `gpytorch_approach_d/`)
**Algorithm spec**: `approach_d_guided_reverse_diffusion.tex` (Algorithms 1-2, Sections 1-7)
**Handoff**: `.claude/handoffs/HANDOFF_2026-03-05_approach-d-guided-reverse-diffusion.md`

---

## What This Is

A script that **generates natural images from scratch** that are maximally informative for a target neuron, by combining a trained diffusion model (99.5M param UNet) with GP utility guidance.

This is **Approach D** from the LaTeX document. It is fundamentally different from Approach A (`guided_optimization.py` in the `gpytorch_porting_diffusion` worktree), which starts from an existing image and optimizes it while using the diffusion model as a regularizer. Approach D instead generates the image through the diffusion reverse process, nudging each denoising step toward high utility. The diffusion model is always in-distribution.

| Property | Approach A (guided_optimization.py) | Approach D (guided_reverse.py) |
|----------|-------------------------------------|-------------------------------|
| Paradigm | Modify existing image | Generate from noise |
| UNet role | Regularizer (post-hoc L2 penalty) | Primary generator (in-distribution) |
| Score evaluation | At fixed noise level t* | At correct noise level per step |
| In-distribution? | No (image drifts during optimization) | Yes (reverse process by construction) |
| Starting point | User-chosen image | Random noise |
| Output diversity | Deterministic (one start -> one result) | Stochastic (different seeds -> different images) |
| Location | `gpytorch_porting_diffusion` worktree | `gpytorch_approach_d` worktree |

The script also runs a **pure utility baseline** for comparison: plain LBFGS gradient ascent on pixel values to maximize utility, starting from a smoothed version of the target image. This baseline does not use the diffusion model at all -- it is simpler than Approach A.

---

## How It Works

### The two systems

1. **GP model** (arc-cosine or RBF kernel, trained on 64x64 center-cropped PNAS images): Given an image, computes distribution-aware utility U -- how informative that image would be for learning about the neuron.

2. **Diffusion model** (99.5M param UNet2DModel from HuggingFace diffusers, ImageNet-pretrained + PNAS-finetuned): Generates realistic 64x64 grayscale natural images from pure noise in ~1000 denoising steps.

### The guided reverse process

Starting from pure noise x_T, the diffusion model denoises step by step. At each step t:

1. **UNet predicts noise**: eps_hat = UNet(x_t, t)
2. **Tweedie estimate**: x_0_hat = (x_t - sqrt(1-abar_t) * eps_hat) / sqrt(abar_t)  -- "what would the clean image look like?"
3. **Evaluate utility**: U(x_0_hat) via the GP (after converting to raw pixel space)
4. **Compute guidance gradient**: g_t = grad_{x_t} U(x_0_hat) via backprop
5. **Guided Tweedie shift**: x_0_hat_guided = x_0_hat + w * (1-abar_t)/sqrt(abar_t) * g_t
6. **Take reverse step** from x_0_hat_guided (DDPM stochastic or DDIM deterministic)
7. **Detach** x_t from computation graph (each step is independent)

The ramp factor `(1-abar_t)/sqrt(abar_t)` naturally modulates guidance strength: large at high noise (where image structure is decided), small at low noise (where fine details are resolved).

### Two samplers

- **DDPM** (stochastic): All T-1=999 steps, adds random noise at each step. Produces sharp textured images. ~15s per image with 99.5M model.
- **DDIM** (deterministic, sigma=0): Subsequence of S steps (default 50), no random noise. Faster (~1s) and reproducible given seed. Requires a sufficiently capable model.

Both samplers share the same guidance logic (steps 1-5 above). Only the reverse step formula differs.

### Two gradient modes

- **Approximate** (default): UNet runs under `torch.no_grad()`, eps_hat is detached. Gradient flows only through the Tweedie formula: `d(x_0_hat)/d(x_t) = 1/sqrt(abar_t)`. Skips the UNet Jacobian entirely. Fast.
- **Full** (`--use-full-gradient`): Gradient flows through the UNet forward pass. Captures the full chain rule including `d(eps_hat)/d(x_t)`. More accurate but requires UNet backward pass.

---

## Code Structure

```
guided_reverse.py (~1230 lines)
|
|-- Imports + path setup (lines 36-85)
|     importlib pattern for utils.py and acquisition.py (avoids sys.modules shadowing)
|     Standard gpytorch_porting imports (kernels, model, training, metrics)
|
|-- Constants (lines 88-127)
|     N_TRAIN=50, M=50, TARGET_INDEX=5, DDIM_STEPS=50, GUIDANCE_SCALE=1.0
|     DEFAULT_MODEL_PATH -> ddpm-pnas-finetuned (99.5M param, diffusers format)
|     PNAS_ABS_MAX = 2.478047 (scale factor between raw pixels and model space)
|
|-- GP helpers (copied from guided_optimization.py)
|     center_crop_images()      -- crop 108x108 -> 64x64
|     setup_gp()                -- full GP pipeline: data, kernel, training, evaluation
|     load_diffusion_model()    -- load DDPMPipeline, build schedule dict
|     compute_tweedie_denoised()-- Tweedie denoising (used by baseline when diffusion_env set)
|     _reconstruct_image()      -- place RF pixels into full image
|     rf_pearson_r()            -- Pearson r in RF mask
|     rf_proj_coeff()           -- projection coefficient in RF mask
|
|-- Pure utility baseline
|     gradient_ascent_guided()  -- LBFGS on RF pixels to maximize U_DA
|                                  Optional diffusion L2 penalty (lambda_diff term)
|
|-- Approach D core
|     _guidance_step()          -- single timestep: UNet -> Tweedie -> utility -> gradient -> guided shift
|     guided_reverse_diffusion()-- full reverse loop (DDPM or DDIM), calls _guidance_step per step
|
|-- Visualization
|     plot_guided_results()     -- 2-row figure: images + convergence plots
|
|-- main()
|     CLI parsing -> setup_gp -> load_diffusion_model
|     -> guided_reverse_diffusion (multi-seed) -> select best
|     -> gradient_ascent_guided (pure utility baseline, optional)
|     -> plot_guided_results -> summary
```

### Key functions

**`setup_gp(crop_size, kernel_type)`** -- Trains a GP model on center-cropped PNAS images. Returns dict with model, likelihood, X_pool, X_train, test_r, config. Uses `build_config_from_defaults()` with overrides for n_px_side, M, n_train.

**`load_diffusion_model(model_path, device)`** -- Loads a HuggingFace `DDPMPipeline`, extracts the UNet (moved to device, eval mode), and constructs a schedule dict from the scheduler's precomputed arrays. Returns dict with unet, schedule, scale_factor.

**`_guidance_step(unet, x_t, t, ...)`** -- Core of the algorithm. At timestep t: runs UNet, computes Tweedie estimate, evaluates utility, computes gradient g_t, applies guided Tweedie shift. Returns guided x_0_hat and per-step metrics.

**`guided_reverse_diffusion(unet, schedule, ..., sampler, ...)`** -- Outer loop. Initializes from noise, iterates over timesteps (all T-1 for DDPM, subsequence for DDIM), calls `_guidance_step` at each step, then applies the sampler-specific reverse step (DDPM stochastic or DDIM deterministic). Detaches x_t after each step.

**`gradient_ascent_guided(..., diffusion_env=None)`** -- Pure utility baseline. LBFGS optimization on RF pixels. When `diffusion_env` is provided, adds an L2 penalty toward the Tweedie-denoised estimate (this is the Approach A mechanism, but in current usage `diffusion_env=None` so it runs as pure utility optimization).

---

## Diffusion Model

The script uses a **99.5M parameter UNet2DModel** (HuggingFace diffusers format):

- **Architecture**: 5-level U-Net with self-attention at 512 channels. Block channels: [64, 128, 256, 512, 512]. 2 ResNet layers per block, GroupNorm, SiLU activation.
- **Pretrained on**: ImageNet (1.28M images), grayscale 64x64
- **Finetuned on**: PNAS natural images (3190 images, 50 epochs, lr=1e-5)
- **Schedule**: Linear beta [0.0001, 0.02], T=1000 timesteps
- **Pixel space**: [-1, 1] (raw PNAS pixels divided by PNAS_ABS_MAX=2.478)
- **Location**: `ddpm-imagenet-grayscale/pietro/ddpm-pnas-finetuned/` (in `gpytorch_imagenet_diffusion` worktree)

The previous tiny model (2.16M params, cosine schedule, in `gpytorch_porting_diffusion` worktree) was replaced because it could not produce recognizable images with DDIM sampling. The 99.5M model produces sharp natural images with both DDPM and DDIM.

### Schedule dict interface

The `load_diffusion_model` function builds a schedule dict from the diffusers scheduler:

```python
schedule = {
    'T': 1000,                          # total timesteps
    'alpha_bar': tensor(1000,),         # cumulative product of (1-beta)
    'sqrt_alpha_bar': tensor(1000,),    # sqrt of above
    'sqrt_one_minus_alpha_bar': tensor(1000,),
    'beta': tensor(1000,),              # noise schedule
    'alpha': tensor(1000,),             # 1 - beta
}
```

All tensors are float32 on CPU. Indexing: `schedule['alpha_bar'][t]` for t in 0..999, where t=0 is low noise (abar~1) and t=999 is high noise (abar~0). Values are moved to device with `.to(device)` at each step.

### UNet forward call

```python
output = unet(x_t, t_batch)  # returns UNet2DOutput
eps_hat = output.sample       # (B, 1, 64, 64) predicted noise
```

The `.sample` attribute is specific to diffusers UNet2DModel output format.

### Space conversions

```
PNAS raw pixels  <-->  diffusion model space
x_raw / PNAS_ABS_MAX = x_model    (approximately [-1, 1])
x_model * PNAS_ABS_MAX = x_raw    (approximately [-2.5, 2.5])
```

The conversion `x_0_hat.squeeze() * scale_factor` in `_guidance_step` MUST be inside the computation graph for gradients to flow from utility back to x_t.

---

## CLI Reference

```
python investigations/diffusion/guided_reverse.py [OPTIONS]
```

### GP arguments
| Flag | Default | Description |
|------|---------|-------------|
| `--kernel-type` | arc_cosine (from config) | `arc_cosine`, `arc_sine`, or `rbf` |
| `--crop-size` | 64 | Center-crop resolution |
| `--target-index` | 5 | Pool image index for utility conditioning |
| `--validate-only` | off | Train GP only, report test_r |

### Diffusion arguments
| Flag | Default | Description |
|------|---------|-------------|
| `--sampler` | ddpm | `ddpm` (stochastic, 999 steps) or `ddim` (deterministic) |
| `--ddim-steps` | 50 | DDIM steps (only for `--sampler ddim`) |
| `--guidance-scale` | 1.0 | Guidance weight w |
| `--use-full-gradient` | off | Backprop through UNet Jacobian |
| `--n-samples` | 5 | Number of random seeds to try |
| `--model-path` | PNAS-finetuned model | Path to diffusers pipeline directory |

### Control arguments
| Flag | Default | Description |
|------|---------|-------------|
| `--no-guidance` | off | Set w=0, unconditional generation |
| `--no-baseline` | off | Skip pure utility LBFGS baseline |

### Output filename convention
```
guided_reverse_crop{size}_{kernel}_tgt{index}_{sampler}_w{scale}[_S{steps}]_n{seeds}[_fullgrad][_unconditional].png
```

### Example commands
```bash
# Validate GP
python guided_reverse.py --validate-only

# Unconditional generation (control -- no GP influence)
python guided_reverse.py --no-guidance --sampler ddim --n-samples 1

# Guided DDIM, arc_cosine kernel
python guided_reverse.py --kernel-type arc_cosine --sampler ddim --guidance-scale 50 --n-samples 10

# Guided DDIM, RBF kernel (needs lower w)
python guided_reverse.py --kernel-type rbf --sampler ddim --guidance-scale 5 --n-samples 10

# With pure utility baseline comparison
python guided_reverse.py --sampler ddim --guidance-scale 50 --n-samples 5

# Different target image
python guided_reverse.py --target-index 15 --sampler ddim --guidance-scale 50 --n-samples 5

# Full gradient mode
python guided_reverse.py --sampler ddim --guidance-scale 50 --use-full-gradient --n-samples 3
```

---

## Critical Implementation Details

### S1: Detach after every reverse step
`x_t = x_t.detach()` after each DDPM/DDIM step prevents the computation graph from growing across all steps. Without this, CUDA OOM after ~10 steps.

### S2: Space conversion in computation graph
`x_0_hat_raw_flat = (x_0_hat.squeeze() * scale_factor).reshape(-1)` -- the multiplication by scale_factor MUST NOT be under `torch.no_grad()` or detached. If it is, the gradient from utility back to x_t will be zero.

### S3: UNet weight freezing vs input gradient
`unet.requires_grad_(False)` freezes UNet parameters but still allows gradient flow through the forward computation with respect to input x_t. This is correct: we want d(eps_hat)/d(x_t) but not d(eps_hat)/d(theta).

### S4: f_max firing rate guard
At high noise (early steps), the Tweedie estimate amplifies by ~1/sqrt(abar_t), producing extreme pixel values and unrealistic firing rates. When `exp(mu_g) >= f_max`, the guidance gradient is zeroed. This prevents chasing impossible utility at early steps. Expected behavior: guard fires frequently at high t, rarely at low t.

### S5: Guidance scale calibration
The optimal w depends on the kernel. With the 99.5M model and approximate gradient:
- arc_cosine (test_r~0.73): effective guidance requires w~50
- RBF (test_r~0.28): effective guidance at w~5, overshoots at w~50+

The ramp factor `(1-abar_t)/sqrt(abar_t)` already modulates by noise level. At t=999 the ramp is ~158, at t=500 it's ~3.4, at t=100 it's ~0.34.

### S6: Seed dependence
Most seeds produce images in low-utility regions (U~0). A minority of seeds find high-utility images. This is expected -- the utility landscape is sparse. Use `--n-samples 10+` and select the best.

### S7: Schedule indexing (diffusers vs old model)
The old cosine schedule had shape (T+1,) with index 0 = alpha_bar_0 = 1.0. The diffusers linear schedule has shape (T,) with index 0 = alpha_bar_0 = 0.9999. Both index 0 means "low noise." The DDIM subsequence `linspace(T-1, 0, S)` works with both.

---

## Dependencies

- `diffusers` (HuggingFace, for DDPMPipeline and UNet2DModel)
- `safetensors` (for loading model weights)
- `accelerate` (diffusers dependency)
- All standard gpytorch_porting dependencies (torch, gpytorch, numpy, matplotlib, scipy)

Install: `pip install diffusers safetensors accelerate`

---

## Related Files

| File | Purpose |
|------|---------|
| `guided_reverse.py` | This script (Approach D implementation) |
| `approach_d_guided_reverse_diffusion.tex` | Mathematical specification (Algorithms 1-2) |
| `PLAN_guided_reverse.md` | Original implementation plan (partially outdated after model swap) |
| `diffusion_model.py` | Old tiny UNet + cosine schedule (not used by guided_reverse.py anymore) |
| `REFERENCE.md` | Old diffusion investigation reference (unconditional generation, tiny model) |
| `.gitignore` | Excludes checkpoints/, samples/, and guided_reverse_*.png |

### Other worktrees
| Worktree | Branch | Contains |
|----------|--------|----------|
| `gpytorch_porting_diffusion` | `pietro/diffusion-investigation` | Approach A (`guided_optimization.py`), old tiny DDPM, training scripts |
| `gpytorch_imagenet_diffusion` | (not a worktree, just a directory) | 99.5M ImageNet DDPM, finetuning scripts, DDPM_MODEL_REFERENCE.md |

---

*Created: 2026-03-05*
*Model swap from 2.16M to 99.5M UNet: 2026-03-05 (same session)*
