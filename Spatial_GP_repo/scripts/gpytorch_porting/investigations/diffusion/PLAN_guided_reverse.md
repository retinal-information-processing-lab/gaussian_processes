# Plan: Implement Guided Reverse Diffusion (Approach D)

## Context

We have a trained 64x64 DDPM and a GP utility optimization pipeline. Previous work (Approach A) modified existing images with L2 penalty toward Tweedie-denoised estimates. Approach D flips the paradigm: **generate** images through the diffusion reverse process, nudging each denoising step toward high utility via classifier guidance. The UNet always operates in-distribution.

**Branch**: `pietro/approach-d-guided-sampling` (git worktree, confirmed)

**Output**: Single new file `investigations/diffusion/guided_reverse.py`

---

## Implementation Stages

### Stage 1: Copy helper functions from guided_optimization.py
CHECKPOINT: confirm with user before continuing

Copy these functions verbatim from the other worktree's `guided_optimization.py`:
- Imports + path setup (lines 1-73): importlib pattern for `utils.py` and `acquisition.py`
- `center_crop_images()` (lines 109-133)
- `setup_gp()` (lines 140-377)
- `load_diffusion_model()` (lines 384-417)
- `compute_tweedie_denoised()` (lines 420-470) — needed for Approach A baseline
- `_reconstruct_image()` (lines 477-492)
- `rf_pearson_r()` (lines 495-505)
- `rf_proj_coeff()` (lines 508-515)
- `gradient_ascent_guided()` (lines 518-714) — Approach A LBFGS baseline

Constants section (matching guided_optimization.py):
```python
N_TRAIN = 50
M = 50
PNAS_SIZE = 108
TARGET_INDEX = 5
SIGMA_SMOOTH = 5.0
# LBFGS constants for Approach A baseline
N_STEPS = 50; LR = 0.5; LBFGS_MAX_ITER = 20; LBFGS_MAX_EVAL = 25; LBFGS_HISTORY_SIZE = 10
T_SCORE = 50; LAMBDA_DIFF = 0.01
# Guided reverse defaults
DDIM_STEPS = 50; GUIDANCE_SCALE = 1.0; N_SEEDS = 5
```

**Checkpoint path**: Absolute path to other worktree (gitignored):
`/home/idv-eqs8-pza/.../gpytorch_porting_diffusion/.../checkpoints/ddpm_epoch1000.pt`

---

### Stage 2: Implement `guided_reverse_diffusion()`
CHECKPOINT: confirm with user before continuing

Core new function. Signature:
```python
def guided_reverse_diffusion(
    unet, schedule, scale_factor,
    model, likelihood, x_target, rf_mask,
    r_max, f_max,
    guidance_scale=1.0, ddim_steps=50,
    use_full_gradient=False, seed=None, device='cuda',
):
```

**Algorithm (from tex doc, Algorithm 2):**

1. **Freeze UNet**: `unet.requires_grad_(False)`
2. **Build DDIM subsequence**: `tau = torch.linspace(T-1, 1, ddim_steps).long()` (descending)
3. **Init noise**: `x_t = randn(1, 1, 64, 64)` with seed
4. **Loop** over DDIM steps `i = 0, ..., S-1`:
   - `t = tau[i]`, `t_next = tau[i+1]` (or 0 at last step)
   - Get schedule values: `abar_t`, `sqrt_abar_t`, `sqrt_1m_abar_t` (index CPU, .to(device))
   - `x_t = x_t.detach().requires_grad_(True)`
   - **UNet forward**:
     - If approximate (default): `with torch.no_grad(): eps_hat = unet(x_t, t)`; `eps_hat = eps_hat.detach()`
     - If full: `eps_hat = unet(x_t, t)` (grad tracks through UNet)
   - **Tweedie**: `x_0_hat = (x_t - sqrt_1m_abar_t * eps_hat) / sqrt_abar_t`
   - **Convert to raw**: `x_0_hat_raw_flat = x_0_hat.squeeze() * scale_factor` then `.reshape(-1)`
   - **Evaluate utility**: `distribution_aware_utility(model, likelihood, x_0_hat_raw_flat.unsqueeze(0), x_target.unsqueeze(0), r_max=r_max, adaptive_r_max=False, sample_lambda=False)`
   - **f_max guard**: If `exp(mu_g_marg) >= f_max` → zero gradient, print warning
   - **Compute g_t**: `(-utility).backward()` then `g_t = -x_t.grad`
   - **Guided Tweedie**: `x_0_hat_guided = x_0_hat + w * (1-abar_t) / sqrt(abar_t) * g_t`
   - **DDIM step** (sigma=0):
     - `direction = (x_t - sqrt_abar_t * x_0_hat_guided) / sqrt_1m_abar_t`
     - `x_t_next = sqrt_abar_t' * x_0_hat_guided + sqrt_1m_abar_t' * direction`
     - Last step: `x_0 = x_0_hat_guided`
   - **Detach**: `x_t = x_t_next.detach()`
   - **Log**: timestep, abar_t, utility, H_marg, H_cond, mu_g_marg, grad_norm, f_max triggered

**Returns**: dict with `x_0_raw_flat` (n_pixels,), `history` dict, `seed`

**Key details**:
- Each step is independent (detach after DDIM step)
- Approximate gradient mode: UNet Jacobian skipped, only `dx_0_hat/dx_t = 1/sqrt(abar_t)` term
- f_max guard + try/except for utility eval failure at early steps (rough Tweedie)
- RF mask: gradient nonzero only at ~21% of pixels; rest shaped by unconditional score

---

### Stage 3: Implement `main()` with CLI
CHECKPOINT: confirm with user before continuing

Three modes:
1. `--validate-only`: Train GP, report test_r, exit
2. `--no-guidance`: Run w=0 (unconditional DDIM), evaluate utility, show images
3. Full guided (default): Run w>0, multi-seed, select best, compare with Approach A baseline

CLI arguments:
- GP: `--crop-size` (64), `--kernel-type`, `--validate-only`
- Diffusion: `--checkpoint`, `--ddim-steps` (50), `--guidance-scale` (1.0), `--use-full-gradient`, `--n-samples` (5)
- Control: `--no-guidance`, `--no-baseline`, `--lambda-diff` (0.01)

Multi-seed: loop `seeds = [42, 43, ...]`, select best final utility.

Approach A baseline: Create smoothed starting image, run `gradient_ascent_guided()`.

---

### Stage 4: Implement `plot_guided_results()`
CHECKPOINT: confirm with user before continuing

Layout:
- **Top row**: Target image | Best Approach D | Approach A result (if run) | Worst/median Approach D
- **Bottom row**: Utility trajectory (all seeds overlaid, best highlighted) | Gradient norm trajectory | Bar chart comparing final utilities

All images: fixed vmin/vmax from dataset global range, OOB pixel check with red title.

---

## Files Modified

| File | Action |
|------|--------|
| `investigations/diffusion/guided_reverse.py` | CREATE — single new file |

No other files modified.

---

## Verification Plan

1. `python investigations/diffusion/guided_reverse.py --validate-only` — GP trains, test_r reported
2. `python investigations/diffusion/guided_reverse.py --no-guidance --n-samples 1` — unconditional DDIM produces natural-looking 64x64 image
3. `python investigations/diffusion/guided_reverse.py --guidance-scale 1.0 --n-samples 3` — utility increases over steps, images still natural
4. `python investigations/diffusion/guided_reverse.py --guidance-scale 1.0 --use-full-gradient --n-samples 1` — compare with approximate mode
5. Default run (with Approach A baseline) — compare Approach D vs Approach A utility and naturalness
