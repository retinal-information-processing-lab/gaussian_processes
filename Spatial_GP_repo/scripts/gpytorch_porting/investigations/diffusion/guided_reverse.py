"""
Guided reverse diffusion for utility-constrained image generation.

Generates images through the diffusion reverse process (DDPM or DDIM sampling),
nudging each denoising step toward high GP utility via classifier guidance
(Dhariwal & Nichol, 2021). The UNet always operates in-distribution.

Uses a 99.5M parameter UNet2DModel (ImageNet-pretrained, PNAS-finetuned) from
the HuggingFace diffusers library. Supports both DDPM (stochastic, 999 steps)
and DDIM (deterministic, configurable steps) sampling.

Algorithm reference: investigations/diffusion/guided_reverse_diffusion.tex

Three modes:
    --validate-only    Train GP, report test_r, exit
    --no-guidance      Unconditional generation (w=0), evaluate utility
    (default)          Guided generation (w>0), multi-seed, compare with Pure utility

Usage:
    # Validate GP at 64x64:
    python investigations/diffusion/guided_reverse.py --validate-only

    # Unconditional DDPM (stochastic, 999 steps):
    python investigations/diffusion/guided_reverse.py --no-guidance --n-samples 1

    # Guided DDPM (default):
    python investigations/diffusion/guided_reverse.py --guidance-scale 1.0 --n-samples 5

    # Guided DDIM (50 steps, faster):
    python investigations/diffusion/guided_reverse.py --sampler ddim --ddim-steps 50

    # Skip Pure utility baseline:
    python investigations/diffusion/guided_reverse.py --no-baseline

Output PNG naming convention:
    guided_reverse_{kernel}_{utility}_ntrain={N}_{sampler}_w={scale}_seeds{n}[_steps{S}][_fullgrad][_unconditional].png

    Examples:
        guided_reverse_arc_cosine_standard_ntrain=50_ddim_w=50.0_seeds30_steps50.png
        guided_reverse_rbf_dist_aware_ntrain=300_ddpm_w=5.0_seeds10.png
"""

import sys
import time
import argparse
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.ndimage import gaussian_filter

# ---------------------------------------------------------------------------
# Path setup (same pattern as guided_optimization.py)
# ---------------------------------------------------------------------------
_script_dir = Path(__file__).resolve().parent
_gpytorch_dir = _script_dir.parent.parent
sys.path.insert(0, str(_gpytorch_dir))

import importlib.util

# Import from local utils.py via importlib to avoid sys.modules shadowing
_local_utils_path = _gpytorch_dir / 'utils.py'
_spec = importlib.util.spec_from_file_location(
    "gpytorch_porting_utils", str(_local_utils_path))
_local_utils = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_local_utils)

compute_rf_center_from_sta = _local_utils.compute_rf_center_from_sta
select_inducing_points_pivoted = _local_utils.select_inducing_points_pivoted
apply_rf_center_bounds = _local_utils.apply_rf_center_bounds
get_gp_marginal_moments = _local_utils.get_gp_marginal_moments
compute_H = _local_utils.compute_H

# Import from acquisition.py via importlib (same pattern)
_acquisition_path = _gpytorch_dir / 'acquisition.py'
_spec_acq = importlib.util.spec_from_file_location(
    "gpytorch_porting_acquisition", str(_acquisition_path))
_acquisition = importlib.util.module_from_spec(_spec_acq)
_spec_acq.loader.exec_module(_acquisition)

distribution_aware_utility = _acquisition.distribution_aware_utility
standard_utility = _acquisition.standard_utility

# Standard imports from gpytorch_porting (on sys.path)
from run_single_mode import build_config_from_defaults
from kernels import create_kernel
from gpy_model import VariationalGPModel
from likelihoods import PoissonLikelihood
from gpy_training import train_gpy_default, predict
from metrics import compute_pearson_correlation, compute_explained_variance
from tests.test_utils import set_reproducible_seed

# ---------------------------------------------------------------------------
# Investigation-specific constants
# ---------------------------------------------------------------------------
N_TRAIN = 50       # override default_params.json n_train=500
M = 50             # override default_params.json ntilde=100

# LBFGS optimization constants for Pure utility baseline (same as guided_optimization.py)
N_STEPS = 50
LR = 0.5
LBFGS_MAX_ITER = 20
LBFGS_MAX_EVAL = 25
LBFGS_HISTORY_SIZE = 10

# Image creation
TARGET_INDEX = 5   # User-confirmed choice (guided_optimization.py uses 9)
SIGMA_SMOOTH = 5.0

# Guided reverse (Guided diffusion) defaults
DDIM_STEPS = 50
GUIDANCE_SCALE = 1.0
N_SEEDS = 5

# Original PNAS image size
PNAS_SIZE = 108

# PNAS data path (canonical absolute path -- npz is gitignored)
PNAS_DATA_PATH = Path.home() / (
    'IDV_code/ClosedLoopProject/gaussian_processes/'
    'Spatial_GP_repo/notebooks/PNAS_paper_sorted_data.npz'
)

# PNAS-finetuned diffusion model (99.5M param UNet, diffusers format)
DEFAULT_MODEL_PATH = (
    '/home/idv-eqs8-pza/IDV_code/ClosedLoopProject/gaussian_processes/'
    'Spatial_GP_repo/scripts/gpytorch_imagenet_diffusion/'
    'Spatial_GP_repo/scripts/gpytorch_porting/'
    'ddpm-imagenet-grayscale/pietro/ddpm-pnas-finetuned'
)

# Normalization: PNAS pixel range / scale_factor -> model space ~[-1, 1]
PNAS_ABS_MAX = 2.478047  # max(abs(all_pixels)) across all 3190 PNAS images


def eval_utility(model, likelihood, x_candidates, x_target, r_max,
                 utility_type='da'):
    """Unified wrapper for utility evaluation.

    Calls either distribution_aware_utility or standard_utility and returns
    a dict with normalized keys so all call sites don't need if/else.

    Args:
        utility_type: 'da' (distribution-aware, needs x_target) or
                      'standard' (H_marg - H_noise, ignores x_target)

    Returns:
        dict with keys: 'utility', 'mu_g' (log-firing rate for f_max guard)
    """
    if utility_type == 'da':
        result = distribution_aware_utility(
            model, likelihood, x_candidates, x_target.unsqueeze(0),
            r_max=r_max, adaptive_r_max=False, sample_lambda=False,
        )
        return {'utility': result['utility'], 'mu_g': result['mu_g_marg']}
    elif utility_type == 'standard':
        result = standard_utility(
            model, likelihood, x_candidates,
            r_max=r_max, adaptive_r_max=False,
        )
        return {'utility': result['utility'], 'mu_g': result['mu_g']}
    else:
        raise ValueError(f"Unknown utility_type: {utility_type}")


# ============================================================================
# Stage 1: Helper functions (copied from guided_optimization.py)
# ============================================================================

def center_crop_images(images, crop_size):
    """Center-crop images from their original spatial size to crop_size x crop_size.

    Args:
        images: numpy array of shape (N, H, W, 1) or (N, H, W).
        crop_size: target square side length. Must be <= H.

    Returns:
        Cropped numpy array of same ndim, with spatial dims = crop_size.
    """
    H = images.shape[1]
    W = images.shape[2]
    assert H == W, f"Images must be square, got {H}x{W}"
    assert crop_size <= H, f"crop_size={crop_size} > image size={H}"

    if crop_size == H:
        return images

    offset = (H - crop_size) // 2
    if images.ndim == 4:
        return images[:, offset:offset + crop_size, offset:offset + crop_size, :]
    elif images.ndim == 3:
        return images[:, offset:offset + crop_size, offset:offset + crop_size]
    else:
        raise ValueError(f"Expected 3D or 4D array, got {images.ndim}D")


def setup_gp(crop_size=64, kernel_type=None, n_train=None, beta_override=None):
    """Train a GP model on center-cropped PNAS images at any square resolution.

    Replicates the run_single_config() pipeline but with center-cropped images.
    See guided_optimization.py for full docstring.

    Args:
        beta_override: If set, override kernel beta (RF mask size).
            Larger beta = smaller mask. Default ~0.1 from config.

    Returns:
        dict with keys: model, likelihood, X_pool, X_train, x_target, config,
        kernel_type, test_r, reliability, indices_train
    """
    if n_train is None:
        n_train = N_TRAIN

    print(f"\n{'=' * 60}")
    print(f"setup_gp: Training GP on {crop_size}x{crop_size} center-cropped images")
    print(f"{'=' * 60}")

    overrides = dict(
        mode='default_gpy',
        n_px_side=crop_size,
        M=M,
        n_train=n_train,
    )
    if kernel_type is not None:
        overrides['kernel_type'] = kernel_type
    config = build_config_from_defaults(**overrides)

    dtype = torch.float32
    device = torch.device(config['device'])
    cell = config['cell']
    seed = config['seed']

    set_reproducible_seed(seed, device=device)
    print(f"  Cell: {cell}, Seed: {seed}, Device: {device}")

    # Load PNAS data and center-crop
    print(f"  Loading data from: {PNAS_DATA_PATH}")
    data = np.load(PNAS_DATA_PATH)

    images_train = data['images_train']
    images_val = data['images_val']
    images_test = data['images_test']

    print(f"  Raw shapes: train={images_train.shape}, val={images_val.shape}, "
          f"test={images_test.shape}")

    if crop_size < PNAS_SIZE:
        offset = (PNAS_SIZE - crop_size) // 2
        print(f"  Center-cropping {PNAS_SIZE} -> {crop_size} (offset={offset})")
        images_train = center_crop_images(images_train, crop_size)
        images_val = center_crop_images(images_val, crop_size)
        images_test = center_crop_images(images_test, crop_size)
    elif crop_size == PNAS_SIZE:
        print(f"  No cropping needed (crop_size={crop_size} == PNAS_SIZE)")
    else:
        raise ValueError(f"crop_size={crop_size} > PNAS_SIZE={PNAS_SIZE}")

    X_all_np = np.concatenate([images_train, images_val], axis=0)
    X_all = torch.tensor(X_all_np, dtype=dtype).reshape(
        X_all_np.shape[0], -1).to(device)
    X_test = torch.tensor(images_test, dtype=dtype).reshape(
        images_test.shape[0], -1).to(device)

    R_all = torch.cat([
        torch.tensor(data['responses_train'], dtype=dtype),
        torch.tensor(data['responses_val'], dtype=dtype),
    ], dim=0).to(device)
    R_test = torch.tensor(data['responses_test'], dtype=dtype).to(device)

    r = R_all[:, cell]
    r_test = R_test[:, :, cell]

    print(f"  Cropped data: X_all={X_all.shape}, X_test={X_test.shape}")

    # Compute RF center from STA
    eps_0x, eps_0y = compute_rf_center_from_sta(
        X_all, r, crop_size, zscore=True)
    print(f"  RF center from STA: ({eps_0x:.4f}, {eps_0y:.4f})")

    # Create kernel with RF structure
    kernel = create_kernel(config, crop_size, eps_0x, eps_0y)
    apply_rf_center_bounds(kernel, eps_0x, eps_0y, config)
    if beta_override is not None:
        import math
        # raw_m2log2beta = -2 * log(2 * beta)
        # Larger beta -> wider Gaussian -> bigger mask
        with torch.no_grad():
            raw_val = -2.0 * math.log(2.0 * beta_override)
            kernel.raw_m2log2beta.fill_(raw_val)
        # Freeze beta so training doesn't change it
        kernel.raw_m2log2beta.requires_grad_(False)
        print(f"  Beta overridden and FROZEN at {beta_override} "
              f"(raw={raw_val:.3f}, default beta~0.1)")
    kernel = kernel.to(dtype=dtype, device=device)

    # Select inducing points
    M_actual = min(config['M'], X_all.shape[0])
    jitter = config['jitter']

    if config['ip_selection'] == 'pivoted':
        inducing_points, indices_inducing = select_inducing_points_pivoted(
            X_all, kernel, M_actual,
            n_candidates=config['n_candidates'],
            seed=seed,
            jitter=jitter,
        )
        print(f"  Inducing points: {M_actual} (pivoted Cholesky)")
    else:
        all_indices = torch.randperm(X_all.shape[0], device=device)
        indices_inducing = all_indices[:M_actual]
        inducing_points = X_all[indices_inducing].clone()
        print(f"  Inducing points: {M_actual} (random)")

    # Build training set
    n_train = min(config['n_train'], X_all.shape[0])
    inducing_set = set(indices_inducing.cpu().numpy().tolist())

    if n_train > M_actual:
        remaining = [i for i in range(X_all.shape[0]) if i not in inducing_set]
        remaining_t = torch.tensor(remaining, device=device)
        n_extra = n_train - M_actual
        perm = torch.randperm(remaining_t.shape[0], device=device)[:n_extra]
        extra_indices = remaining_t[perm]
        indices_train = torch.cat([indices_inducing, extra_indices])
    else:
        indices_train = indices_inducing
        n_train = M_actual

    X_train = X_all[indices_train]
    r_train = r[indices_train]
    print(f"  Training set: {X_train.shape[0]} images "
          f"({M_actual} inducing + {n_train - M_actual} extra)")

    # Create model + likelihood
    model = VariationalGPModel(
        inducing_points, kernel, jitter=jitter,
        standard_variational_distribution=True,
    )
    likelihood = PoissonLikelihood(
        A_init=config['A_init'],
        lambda0_init=config['lambda0_init'],
    )
    model = model.to(dtype=dtype, device=device)
    likelihood = likelihood.to(dtype=dtype, device=device)

    print(f"  A_init={config['A_init']}, lambda0_init={config['lambda0_init']}")

    # Train
    print_every = max(1, config['n_iterations'] // 5)
    start_time = time.time()

    with torch.enable_grad():
        train_result = train_gpy_default(
            model, likelihood, X_train, r_train,
            optimizer_name=config['optimizer'],
            lr=config['lr'],
            n_iterations=config['n_iterations'],
            print_every=print_every,
            device=device,
            early_stop=config['early_stop'],
            stop_window=config['stop_window'],
            stop_thresh=config['stop_thresh'],
            min_iterations=config['min_iterations'],
            lbfgs_max_iter=config['gpy_lbfgs_max_iter'],
            jitter=jitter,
            cholesky_max_tries=config['cholesky_max_tries'],
        )

    train_time = time.time() - start_time
    print(f"\n  Training time: {train_time:.1f}s")
    print(f"  Final A={likelihood.A.item():.4f}, "
          f"lambda0={likelihood.lambda0.item():.4f}")

    # Evaluate
    model.eval()
    predictions = predict(
        model, likelihood, X_test, device=device,
        jitter=jitter, cholesky_max_tries=config['cholesky_max_tries'])
    f_pred = predictions['f_pred']

    r_test_mean = r_test.mean(dim=0)
    test_r = float(compute_pearson_correlation(r_test_mean, f_pred))
    explained_var, reliability = compute_explained_variance(r_test, f_pred)
    reliability = float(reliability)

    print(f"\n  Test Pearson r: {test_r:.4f}")
    print(f"  Reliability:    {reliability:.4f}")
    print(f"  Explained var:  {float(explained_var):.4f}")

    # Build pool (non-training images)
    pool_indices = sorted(
        set(range(X_all.shape[0])) - set(indices_train.cpu().numpy().tolist()))
    X_pool = X_all[pool_indices]

    print(f"\n  Pool: {X_pool.shape[0]} images, Train: {X_train.shape[0]} images")

    return {
        'model': model,
        'likelihood': likelihood,
        'X_pool': X_pool,
        'X_train': X_train,
        'x_target': X_pool[0].clone(),
        'config': config,
        'kernel_type': kernel_type or config['kernel_type'],
        'test_r': test_r,
        'reliability': reliability,
        'indices_train': indices_train,
    }


def load_diffusion_model(model_path, device):
    """Load 99.5M UNet from HuggingFace diffusers pipeline.

    Builds a schedule dict compatible with our guided_reverse_diffusion() from
    the scheduler's precomputed arrays.

    Args:
        model_path: path to diffusers pipeline directory (DDPMPipeline format)
        device: torch device

    Returns:
        dict with keys: unet, schedule, scale_factor
    """
    from diffusers import DDPMPipeline

    pipeline = DDPMPipeline.from_pretrained(model_path)
    unet = pipeline.unet.to(device)
    unet.eval()
    scheduler = pipeline.scheduler

    T = scheduler.config.num_train_timesteps  # 1000
    # diffusers alphas_cumprod: shape (T,), indices 0..T-1
    # index 0 = low noise (abar ~ 1), index T-1 = high noise (abar ~ 0)
    alphas_cumprod = scheduler.alphas_cumprod  # CPU float32
    schedule = {
        'T': T,
        'alpha_bar': alphas_cumprod,
        'sqrt_alpha_bar': alphas_cumprod.sqrt(),
        'sqrt_one_minus_alpha_bar': (1 - alphas_cumprod).sqrt(),
        'beta': scheduler.betas,
        'alpha': scheduler.alphas,
    }

    n_params = sum(p.numel() for p in unet.parameters())
    print(f"  Loaded diffusion model: {n_params/1e6:.1f}M params, T={T}")
    print(f"  Schedule: linear beta [{scheduler.betas[0]:.6f}, {scheduler.betas[-1]:.4f}]")
    print(f"  scale_factor (PNAS_ABS_MAX): {PNAS_ABS_MAX:.4f}")

    return {
        'unet': unet,
        'schedule': schedule,
        'scale_factor': PNAS_ABS_MAX,
    }


def _reconstruct_image(x_rf, rf_mask, n_pixels, dtype, device, background=None):
    """Place RF pixel values into full image."""
    if background is not None:
        x_full = background.detach().clone()
    else:
        x_full = torch.zeros(n_pixels, dtype=dtype, device=device)
    x_full[rf_mask] = x_rf
    return x_full


def rf_pearson_r(x, target, rf_mask):
    """Pearson correlation between x and target within RF mask."""
    a = x[rf_mask]
    b = target[rf_mask]
    a_c = a - a.mean()
    b_c = b - b.mean()
    num = (a_c * b_c).sum()
    denom = a_c.norm() * b_c.norm()
    if denom < 1e-12:
        return 0.0
    return (num / denom).item()


def rf_proj_coeff(x, target, rf_mask):
    """Projection coefficient: s minimizing ||x - s*target|| within RF."""
    a = x[rf_mask]
    b = target[rf_mask]
    denom = (b * b).sum()
    if denom < 1e-12:
        return 0.0
    return ((a * b).sum() / denom).item()


def gradient_ascent_guided(model, likelihood, x_start, x_target, rf_mask,
                           r_max, f_max, n_px_side,
                           n_steps, lr, max_iter, max_eval, history_size,
                           x_background=None, utility_type='da'):
    """Pure utility baseline: LBFGS gradient ascent on image pixels.

    Maximizes GP utility by directly optimizing RF pixel values.
    Uses LBFGS with strong_wolfe line search.
    """
    n_pixels = x_start.shape[0]
    device = x_start.device
    dtype = x_start.dtype

    x_rf_init = x_start[rf_mask]
    x_rf = x_rf_init.clone().detach().requires_grad_(True)

    optimizer = torch.optim.LBFGS(
        [x_rf], lr=lr, max_iter=max_iter, max_eval=max_eval,
        history_size=history_size, line_search_fn='strong_wolfe',
    )
    print(f"  Optimizer: LBFGS (lr={lr}, max_iter={max_iter})")

    history = {
        'step': [], 'utility': [], 'grad_norm': [],
        'pearson_r': [], 'proj_coeff': [],
        'loss_utility': [],
    }

    # Record initial state
    with torch.no_grad():
        x_full_init = _reconstruct_image(
            x_rf, rf_mask, n_pixels, dtype, device, background=x_background)
        try:
            result_init = eval_utility(model, likelihood,
                                       x_full_init.unsqueeze(0), x_target,
                                       r_max, utility_type=utility_type)
            utility_init = result_init['utility'].item()
        except Exception:
            utility_init = float('nan')
        pr_init = rf_pearson_r(x_full_init, x_target, rf_mask)
        pc_init = rf_proj_coeff(x_full_init, x_target, rf_mask)

    history['step'].append(-1)
    history['utility'].append(utility_init)
    history['grad_norm'].append(0.0)
    history['pearson_r'].append(pr_init)
    history['proj_coeff'].append(pc_init)
    history['loss_utility'].append(-utility_init if not np.isnan(utility_init) else float('nan'))
    print(f"  init   : U={utility_init:.6f}  r={pr_init:.4f}  proj={pc_init:.4f}")

    for step in range(n_steps):
        def closure():
            optimizer.zero_grad()
            x_full = _reconstruct_image(
                x_rf, rf_mask, n_pixels, dtype, device,
                background=x_background)

            try:
                result = eval_utility(model, likelihood,
                                      x_full.unsqueeze(0), x_target,
                                      r_max, utility_type=utility_type)
            except Exception:
                return torch.tensor(float('inf'), device=device)

            mu_g = result['mu_g']
            if torch.exp(mu_g).item() > f_max:
                return torch.tensor(float('inf'), device=device)

            loss = -result['utility'].squeeze()
            if torch.isnan(loss):
                return torch.tensor(float('inf'), device=device)

            loss.backward()
            return loss

        optimizer.step(closure)

        if step == 0:
            if x_rf.grad is not None and x_rf.grad.norm() > 0:
                print(f"  Step 0: |grad|={x_rf.grad.norm().item():.4e}")
            else:
                print(f"  WARNING: Step 0 has no gradient (LBFGS line search may have failed)")

        with torch.no_grad():
            x_full = _reconstruct_image(
                x_rf, rf_mask, n_pixels, dtype, device, background=x_background)
            try:
                result = eval_utility(model, likelihood,
                                      x_full.unsqueeze(0), x_target,
                                      r_max, utility_type=utility_type)
                utility = result['utility'].item()
            except Exception:
                utility = float('nan')
            pr = rf_pearson_r(x_full, x_target, rf_mask)
            pc = rf_proj_coeff(x_full, x_target, rf_mask)

        grad_norm = x_rf.grad.norm().item() if x_rf.grad is not None else 0.0

        history['step'].append(step)
        history['utility'].append(utility)
        history['grad_norm'].append(grad_norm)
        history['pearson_r'].append(pr)
        history['proj_coeff'].append(pc)
        history['loss_utility'].append(-utility if not np.isnan(utility) else float('nan'))

        if np.isnan(utility) or np.isnan(grad_norm):
            print(f"  step {step}: NaN detected - stopping")
            break

        if step % 5 == 0 or step == n_steps - 1:
            print(f"  step {step:4d}: U={utility:.6f}  "
                  f"r={pr:.4f}  proj={pc:.4f}  |grad|={grad_norm:.4e}")

    with torch.no_grad():
        x_final = _reconstruct_image(
            x_rf, rf_mask, n_pixels, dtype, device, background=x_background)
    return x_final.detach(), history


# ============================================================================
# Stage 2: Guided reverse diffusion (Guided diffusion core)
# ============================================================================

def _guidance_step(unet, x_t, t, schedule, scale_factor, model, likelihood,
                    x_target, r_max, f_max, guidance_scale, use_full_gradient,
                    device, step_idx, total_steps, utility_type='da',
                    noisy_gradient=False):
    """Compute guidance gradient and guided Tweedie estimate at one timestep.

    Shared by both DDPM and DDIM samplers. Returns the guided x_0_hat and
    per-step metrics.

    Returns:
        x_0_hat_guided: (1, 1, 64, 64) guided clean image estimate (detached)
        eps_hat_det: (1, 1, 64, 64) noise prediction (detached)
        metrics: dict with utility, grad_norm, f_max_triggered, etc.
    """
    abar_t = schedule['alpha_bar'][t].to(device)
    sqrt_abar_t = schedule['sqrt_alpha_bar'][t].to(device)
    sqrt_1m_abar_t = schedule['sqrt_one_minus_alpha_bar'][t].to(device)

    # Prepare x_t as leaf tensor with gradient
    x_t = x_t.detach().requires_grad_(True)

    # UNet forward pass -- diffusers UNet returns .sample attribute
    t_batch = torch.tensor([t], device=device, dtype=torch.long)
    if use_full_gradient:
        eps_hat = unet(x_t, t_batch).sample
    else:
        with torch.no_grad():
            eps_hat = unet(x_t, t_batch).sample
        eps_hat = eps_hat.detach()

    # Tweedie estimate
    x_0_hat = (x_t - sqrt_1m_abar_t * eps_hat) / sqrt_abar_t

    # Convert to raw pixel space (must be in computation graph for gradient)
    if noisy_gradient:
        # Evaluate utility at the noisy x_t (direct gradient, no Tweedie chain rule)
        x_eval_flat = (x_t.squeeze() * scale_factor).reshape(-1)
    else:
        # Evaluate utility at the denoised Tweedie estimate (default)
        x_eval_flat = (x_0_hat.squeeze() * scale_factor).reshape(-1)

    # Evaluate utility and compute guidance gradient
    utility_val = float('nan')
    mu_g_val = float('nan')
    grad_norm_val = 0.0
    f_max_hit = False
    g_t = torch.zeros_like(x_t)

    try:
        result = eval_utility(model, likelihood,
                              x_eval_flat.unsqueeze(0), x_target,
                              r_max, utility_type=utility_type)
        utility_val = result['utility'].item()
        mu_g_val = result['mu_g'].item()

        if np.exp(mu_g_val) >= f_max:
            f_max_hit = True
        else:
            loss = -result['utility'].squeeze()
            loss.backward()
            g_t = -x_t.grad
            grad_norm_val = g_t.norm().item()

    except Exception as e:
        if step_idx < 5 or step_idx % 100 == 0:
            print(f"    step {step_idx} (t={t}): utility eval failed: {e}")

    # Guided Tweedie: x_0_hat + w * (1-abar_t) / sqrt(abar_t) * g_t
    x_0_hat_det = x_0_hat.detach()
    ramp = (1.0 - abar_t) / sqrt_abar_t
    x_0_hat_guided = x_0_hat_det + guidance_scale * ramp * g_t.detach()

    metrics = {
        'utility': utility_val, 'mu_g': mu_g_val, 'grad_norm': grad_norm_val,
        'f_max_triggered': f_max_hit, 'abar_t': abar_t.item(),
    }
    return x_0_hat_guided, eps_hat.detach(), metrics


def guided_reverse_diffusion(
    unet, schedule, scale_factor,
    model, likelihood, x_target, rf_mask,
    r_max, f_max,
    guidance_scale=1.0, sampler='ddpm', ddim_steps=50,
    use_full_gradient=False, seed=None, device='cuda',
    utility_type='da', noisy_gradient=False,
):
    """Generate an image via guided reverse diffusion (DDPM or DDIM).

    At each reverse step:
      1. Predict noise with UNet
      2. Compute Tweedie estimate x_0_hat
      3. Evaluate GP utility on x_0_hat (in raw pixel space)
      4. Compute guidance gradient g_t
      5. Shift Tweedie: x_0_hat_guided = x_0_hat + w*(1-abar_t)/sqrt(abar_t)*g_t
      6. Take reverse step (DDPM stochastic or DDIM deterministic)

    Args:
        unet: Trained UNet in eval mode (diffusers UNet2DModel)
        schedule: schedule dict with alpha_bar, beta, alpha, etc. (CPU tensors)
        scale_factor: PNAS_ABS_MAX (~2.478)
        model: Trained GP model (eval mode, default_gpy)
        likelihood: PoissonLikelihood
        x_target: (n_pixels,) conditioning target for DA utility
        rf_mask: boolean mask for RF pixels
        r_max, f_max: utility/firing rate bounds
        guidance_scale: w (classifier guidance weight)
        sampler: 'ddpm' (stochastic, T-1 steps) or 'ddim' (deterministic, ddim_steps)
        ddim_steps: S (number of DDIM steps, only used when sampler='ddim')
        use_full_gradient: If True, backprop through UNet Jacobian
        seed: Random seed for initial noise
        device: torch device

    Returns:
        dict with x_0_raw_flat, history, seed
    """
    T = schedule['T']  # 1000

    # Freeze UNet weights (allow input gradient flow)
    unet.requires_grad_(False)

    # Initialize from noise
    if seed is not None:
        torch.manual_seed(seed)
    x_t = torch.randn(1, 1, 64, 64, device=device)

    history = {
        'step_idx': [], 'timestep': [], 'abar_t': [],
        'utility': [], 'mu_g': [], 'grad_norm': [],
        'f_max_triggered': [],
    }

    if sampler == 'ddim':
        # DDIM: subsequence of ddim_steps timesteps, descending
        timesteps = torch.linspace(T - 1, 0, ddim_steps).round().long().tolist()
        total_steps = ddim_steps
        print(f"  Guided DDIM: S={ddim_steps}, w={guidance_scale}, "
              f"full_grad={use_full_gradient}, seed={seed}")
    elif sampler == 'ddpm':
        # DDPM: all timesteps T-1 down to 0
        timesteps = list(range(T - 1, -1, -1))
        total_steps = T
        print(f"  Guided DDPM: T={T}, w={guidance_scale}, "
              f"full_grad={use_full_gradient}, seed={seed}")
    else:
        raise ValueError(f"Unknown sampler: {sampler}")

    for i, t in enumerate(timesteps):
        # Compute guidance and guided Tweedie
        x_0_hat_guided, eps_hat, metrics = _guidance_step(
            unet, x_t, t, schedule, scale_factor,
            model, likelihood, x_target, r_max, f_max,
            guidance_scale, use_full_gradient, device,
            step_idx=i, total_steps=total_steps,
            utility_type=utility_type,
            noisy_gradient=noisy_gradient,
        )

        abar_t = schedule['alpha_bar'][t].to(device)
        sqrt_abar_t = schedule['sqrt_alpha_bar'][t].to(device)
        sqrt_1m_abar_t = schedule['sqrt_one_minus_alpha_bar'][t].to(device)

        if sampler == 'ddim':
            # DDIM step (sigma=0, deterministic)
            if i < len(timesteps) - 1:
                t_next = timesteps[i + 1]
                sqrt_abar_next = schedule['sqrt_alpha_bar'][t_next].to(device)
                sqrt_1m_next = schedule['sqrt_one_minus_alpha_bar'][t_next].to(device)
                direction = (x_t.detach() - sqrt_abar_t * x_0_hat_guided) / sqrt_1m_abar_t
                x_t = sqrt_abar_next * x_0_hat_guided + sqrt_1m_next * direction
            else:
                x_t = x_0_hat_guided

        elif sampler == 'ddpm':
            # DDPM step: mu_t + sqrt(beta_t) * z
            # Compute guided noise: eps_guided = eps_hat - w * sqrt(1-abar_t) * g_t
            # Then use guided Tweedie x_0_hat_guided in the DDPM formula:
            # mu = sqrt(1/alpha_t) * (x_t - beta_t/sqrt(1-abar_t) * eps_guided)
            # Equivalently: mu = sqrt(abar_{t-1}) * x_0_hat_guided * beta_t / (1-abar_t)
            #                  + sqrt(alpha_t) * (1-abar_{t-1}) / (1-abar_t) * x_t
            # But simplest: reconstruct eps_guided from x_0_hat_guided, use standard formula
            alpha_t = schedule['alpha'][t].to(device)
            beta_t = schedule['beta'][t].to(device)
            sqrt_recip_alpha = (1.0 / alpha_t).sqrt()

            # Reconstruct guided eps from guided x_0_hat
            eps_guided = (x_t.detach() - sqrt_abar_t * x_0_hat_guided) / sqrt_1m_abar_t

            # DDPM reverse mean
            mu_t = sqrt_recip_alpha * (x_t.detach() - (beta_t / sqrt_1m_abar_t) * eps_guided)

            if t > 0:
                noise = torch.randn_like(x_t)
                x_t = mu_t + beta_t.sqrt() * noise
            else:
                x_t = mu_t

        # Detach (prevents graph from growing across steps)
        x_t = x_t.detach()

        # Log
        history['step_idx'].append(i)
        history['timestep'].append(t)
        history['abar_t'].append(metrics['abar_t'])
        history['utility'].append(metrics['utility'])
        history['mu_g'].append(metrics['mu_g'])
        history['grad_norm'].append(metrics['grad_norm'])
        history['f_max_triggered'].append(metrics['f_max_triggered'])

        # Print progress (sparse for DDPM's 999 steps)
        if sampler == 'ddim':
            should_print = i < 3 or i % 10 == 0 or i == total_steps - 1
        else:
            should_print = i < 3 or i % 100 == 0 or i == total_steps - 1
        if should_print:
            fmax_str = " [f_max]" if metrics['f_max_triggered'] else ""
            print(f"    step {i:4d} t={t:4d} abar={metrics['abar_t']:.6f} "
                  f"U={metrics['utility']:+.4f} |g|={metrics['grad_norm']:.2e}{fmax_str}")

    # Final image in raw pixel space
    x_0_raw_flat = (x_t.squeeze() * scale_factor).reshape(-1).detach()

    return {
        'x_0_raw_flat': x_0_raw_flat,
        'history': history,
        'seed': seed,
    }


# ============================================================================
# Stage 4: Visualization
# ============================================================================

def plot_guided_results(results_list, baseline_result, x_target,
                        rf_mask, vmin, vmax, n_px_side, config, test_r,
                        out_path, target_utility=None,
                        utility_type='da', target_index=5,
                        dataset_utility_stats=None):
    """Summary figure for guided reverse diffusion results.

    Top row: Target | Best Guided diffusion | Pure utility (if run) | Worst/median D
    Bottom row: Utility trajectory (all seeds) | Gradient norm | Bar chart

    Args:
        results_list: list of dicts from guided_reverse_diffusion()
        baseline_result: dict with 'x_final' and 'history' from Pure utility, or None
        x_target: (n_pixels,) target image
        rf_mask: boolean RF mask
        vmin, vmax: dataset pixel range
        n_px_side: image side length (64)
        config: GP config dict
        test_r: GP test Pearson r
        out_path: output file path
    """
    from matplotlib.gridspec import GridSpec

    mask_2d = rf_mask.cpu().numpy().reshape(n_px_side, n_px_side)

    def to_image(x_flat):
        return x_flat.detach().cpu().numpy().reshape(n_px_side, n_px_side)

    def check_oob(x_flat, label):
        vals = x_flat.detach().cpu().numpy()
        below = (vals < vmin).sum()
        above = (vals > vmax).sum()
        total = vals.size
        if below + above > 0:
            pct = 100 * (below + above) / total
            return True, pct
        return False, 0.0

    # Sort results by final utility (descending), skip only inf/nan
    final_utilities = []
    for r in results_list:
        u_vals = [u for u in r['history']['utility'] if np.isfinite(u)]
        final_utilities.append(u_vals[-1] if u_vals else float('-inf'))
    # Best/worst among finite values (no secondary threshold)
    valid_indices = [i for i, u in enumerate(final_utilities) if np.isfinite(u)]
    if valid_indices:
        best_idx = max(valid_indices, key=lambda i: final_utilities[i])
        worst_idx = min(valid_indices, key=lambda i: final_utilities[i])
    else:
        best_idx, worst_idx = 0, len(results_list) - 1

    best = results_list[best_idx]
    worst = results_list[worst_idx]

    # Count image panels
    n_img = 2  # Target + Best D
    if baseline_result is not None:
        n_img += 1
    if len(results_list) > 1:
        n_img += 1  # Worst D

    n_cols = max(n_img, 3)  # At least 3 columns for bottom row
    fig = plt.figure(figsize=(4 * n_cols, 8))
    gs = GridSpec(2, n_cols, figure=fig, hspace=0.35, wspace=0.3)

    def show_image(ax, x_flat, label, utility_val=None):
        oob, pct = check_oob(x_flat, label)
        img = to_image(x_flat)
        ax.imshow(img, cmap='gray', vmin=vmin, vmax=vmax, aspect='equal')
        ax.contour(mask_2d, levels=[0.5], colors='cyan', linewidths=0.8)
        title_color = 'red' if oob else 'black'
        title_str = label
        if oob:
            title_str += f'\nOOB: {pct:.1f}%'
        if utility_val is not None and not np.isnan(utility_val):
            title_str += f'\nU={utility_val:.4f}'
        ax.set_title(title_str, fontsize=9, color=title_color)
        ax.axis('off')

    # Top row: images
    col = 0
    ax = fig.add_subplot(gs[0, col])
    show_image(ax, x_target, 'Target', utility_val=target_utility)
    col += 1

    ax = fig.add_subplot(gs[0, col])
    show_image(ax, best['x_0_raw_flat'], f'Best D (seed={best["seed"]})',
               utility_val=final_utilities[best_idx])
    col += 1

    if baseline_result is not None:
        ax = fig.add_subplot(gs[0, col])
        u_baseline = baseline_result['history']['utility'][-1]
        show_image(ax, baseline_result['x_final'], 'Pure utility',
                   utility_val=u_baseline)
        col += 1

    if len(results_list) > 1:
        ax = fig.add_subplot(gs[0, col])
        show_image(ax, worst['x_0_raw_flat'], f'Worst D (seed={worst["seed"]})',
                   utility_val=final_utilities[worst_idx])

    # Bottom row: 3 plots
    plot_width = n_cols / 3.0
    plot_starts = [round(i * plot_width) for i in range(3)]
    plot_ends = [round((i + 1) * plot_width) for i in range(3)]

    # Divergence threshold (shared across all subplots)
    DIV_THRESHOLD = 1e9

    # Utility trajectories (replace diverged values with NaN for display)
    ax_u = fig.add_subplot(gs[1, plot_starts[0]:plot_ends[0]])
    all_finite_utils = []
    for idx, r in enumerate(results_list):
        steps = r['history']['step_idx']
        utils = [u if (np.isfinite(u) and abs(u) < DIV_THRESHOLD) else np.nan
                 for u in r['history']['utility']]
        all_finite_utils.extend([u for u in utils if not np.isnan(u)])
        alpha = 1.0 if idx == best_idx else 0.3
        lw = 2.0 if idx == best_idx else 0.8
        label = f'seed={r["seed"]}' if idx == best_idx else None
        ax_u.plot(steps, utils, alpha=alpha, linewidth=lw, label=label,
                  color='tab:blue')
    if baseline_result is not None:
        ax_u.axhline(baseline_result['history']['utility'][-1],
                      color='red', linestyle='--', linewidth=1.5,
                      label='Pure utility')
    # Set y-limits from finite values only
    if all_finite_utils:
        ymin = min(all_finite_utils)
        ymax = max(all_finite_utils)
        margin = max(0.01, (ymax - ymin) * 0.1)
        ax_u.set_ylim(ymin - margin, ymax + margin)
    ax_u.set_xlabel('DDIM step')
    ax_u.set_ylabel('Utility')
    ax_u.set_title('Utility trajectory')
    ax_u.legend(fontsize=7)
    ax_u.grid(True, alpha=0.3)

    # Gradient norm trajectory
    ax_g = fig.add_subplot(gs[1, plot_starts[1]:plot_ends[1]])
    for idx, r in enumerate(results_list):
        steps = r['history']['step_idx']
        gnorms = r['history']['grad_norm']
        alpha = 1.0 if idx == best_idx else 0.3
        lw = 2.0 if idx == best_idx else 0.8
        ax_g.plot(steps, gnorms, alpha=alpha, linewidth=lw, color='tab:green')
    ax_g.set_xlabel('DDIM step')
    ax_g.set_ylabel('|g_t|')
    ax_g.set_title('Gradient norm')
    ax_g.set_yscale('log')
    ax_g.grid(True, alpha=0.3)

    # Utility distribution: strip + box for both generated and dataset
    ax_b = fig.add_subplot(gs[1, plot_starts[2]:plot_ends[2]])

    def _filter_valid(vals):
        return np.array([u for u in vals if np.isfinite(u) and abs(u) < DIV_THRESHOLD])

    def _plot_strip_box(ax, values, x_pos, color, label):
        """Plot strip (dots) + box at x_pos."""
        valid = _filter_valid(values)
        if len(valid) == 0:
            return valid
        jitter = np.random.default_rng(0).uniform(-0.15, 0.15, len(valid))
        ax.scatter(x_pos + jitter, valid, s=10, alpha=0.35, color=color, zorder=3)
        bp = ax.boxplot([valid], positions=[x_pos], widths=0.4,
                         patch_artist=True, zorder=2, showfliers=False)
        bp['boxes'][0].set_facecolor(color)
        bp['boxes'][0].set_alpha(0.15)
        bp['medians'][0].set_color('black')
        bp['medians'][0].set_linewidth(2)
        return valid

    # Left: Generated seeds
    gen_valid = _plot_strip_box(ax_b, final_utilities, 0, 'tab:blue', 'Generated')
    # Highlight best seed
    best_u = final_utilities[best_idx]
    if np.isfinite(best_u) and abs(best_u) < DIV_THRESHOLD:
        ax_b.scatter([0], [best_u], s=50, color='tab:orange', edgecolors='black',
                     linewidths=0.8, zorder=4)

    # Right: Dataset pool images (same strip + box)
    if dataset_utility_stats is not None:
        ds_values = dataset_utility_stats['values']
        ds_valid = _plot_strip_box(ax_b, ds_values, 1.0, 'gray', 'Dataset')

    # Reference lines
    if baseline_result is not None:
        ax_b.axhline(baseline_result['history']['utility'][-1],
                      color='red', linestyle='--', linewidth=1.5)
    if target_utility is not None and not np.isnan(target_utility):
        ax_b.axhline(target_utility, color='dimgray', linestyle=':',
                      linewidth=1.5)

    # Stats annotation for generated
    n_total = len(final_utilities)
    n_div = n_total - len(gen_valid)
    gen_mean = float(gen_valid.mean()) if len(gen_valid) > 0 else 0
    gen_median = float(np.median(gen_valid)) if len(gen_valid) > 0 else 0
    gen_best = float(gen_valid.max()) if len(gen_valid) > 0 else 0
    stats_lines = [
        f"n={len(gen_valid)}/{n_total}",
        f"median={gen_median:.4f}",
        f"mean={gen_mean:.4f}",
        f"best={gen_best:.4f}",
    ]
    if n_div > 0:
        stats_lines.append(f"DIVERGED: {n_div}")
    ax_b.text(0.98, 0.98, '\n'.join(stats_lines), transform=ax_b.transAxes,
              fontsize=7, verticalalignment='top', horizontalalignment='right',
              fontfamily='monospace',
              color='red' if n_div > 0 else 'black',
              bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

    ax_b.set_xticks([0, 1.0])
    ax_b.set_xticklabels(['Generated', 'Dataset'], fontsize=8)
    ax_b.set_ylabel('Utility')
    ax_b.set_title('Final utility distribution')
    ax_b.grid(True, alpha=0.3, axis='y')

    util_label = 'DA utility' if utility_type == 'da' else 'Standard utility'
    fig.suptitle(
        f'Guided Reverse Diffusion -- {util_label}, '
        f'target={target_index}, crop={n_px_side}, test_r={test_r:.3f}',
        fontsize=11)
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {out_path}")
    plt.close(fig)


# ============================================================================
# Stage 3: Main with CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Guided diffusion: Guided reverse diffusion for utility-constrained generation')

    # GP arguments
    parser.add_argument('--crop-size', type=int, default=64,
                        help='Square crop size for GP (default: 64)')
    parser.add_argument('--kernel-type', type=str, default=None,
                        choices=['arc_cosine', 'arc_sine', 'rbf'],
                        help='Kernel type (default: from default_params.json)')
    parser.add_argument('--validate-only', action='store_true',
                        help='Only train GP and report test_r, skip generation')
    parser.add_argument('--n-train', type=int, default=N_TRAIN,
                        help=f'Number of training images for GP (default: {N_TRAIN})')
    parser.add_argument('--target-index', type=int, default=TARGET_INDEX,
                        help=f'Pool image index for conditioning target (default: {TARGET_INDEX})')

    # Diffusion arguments
    parser.add_argument('--model-path', type=str, default=DEFAULT_MODEL_PATH,
                        help='Path to diffusers pipeline directory')
    parser.add_argument('--sampler', type=str, default='ddpm',
                        choices=['ddpm', 'ddim'],
                        help='Sampling method: ddpm (stochastic, 999 steps) or ddim (deterministic)')
    parser.add_argument('--ddim-steps', type=int, default=DDIM_STEPS,
                        help=f'Number of DDIM steps (only for --sampler ddim, default: {DDIM_STEPS})')
    parser.add_argument('--guidance-scale', type=float, default=GUIDANCE_SCALE,
                        help=f'Guidance weight w (default: {GUIDANCE_SCALE})')
    parser.add_argument('--use-full-gradient', action='store_true',
                        help='Backprop through UNet Jacobian (slower, more accurate)')
    parser.add_argument('--noisy-gradient', action='store_true',
                        help='Compute utility gradient at noisy x_t instead of denoised x_0_hat')
    parser.add_argument('--n-samples', type=int, default=N_SEEDS,
                        help=f'Number of seeds to try (default: {N_SEEDS})')

    # Control arguments
    parser.add_argument('--no-guidance', action='store_true',
                        help='Run w=0 unconditional generation (evaluate utility only)')
    parser.add_argument('--no-baseline', action='store_true',
                        help='Skip Pure utility baseline comparison')
    parser.add_argument('--utility', type=str, default='da',
                        choices=['da', 'standard'],
                        help='Utility type: da (distribution-aware) or standard (H_marg - H_noise)')
    parser.add_argument('--lambda-diff', type=float, default=0.01,
                        help='Diffusion reg weight for Pure utility (default: 0.01)')
    parser.add_argument('--beta', type=float, default=None,
                        help='Override kernel beta (RF mask size). Larger = smaller mask. Default from config (~0.1)')

    args = parser.parse_args()

    # Override guidance scale if --no-guidance
    if args.no_guidance:
        args.guidance_scale = 0.0

    # === Stage 1: Train GP ===
    env = setup_gp(crop_size=args.crop_size, kernel_type=args.kernel_type,
                    n_train=args.n_train, beta_override=args.beta)
    model = env['model']
    likelihood = env['likelihood']
    X_pool = env['X_pool']
    config = env['config']
    test_r = env['test_r']

    if args.validate_only:
        print(f"\n{'=' * 60}")
        print(f"Validation complete: crop_size={args.crop_size}, "
              f"test_r={test_r:.4f}")
        print(f"{'=' * 60}")
        if test_r < 0.5:
            print("WARNING: test_r < 0.5 -- RF may be outside cropped region.")
        return

    if test_r < 0.5:
        print(f"\nWARNING: test_r={test_r:.4f} < 0.5. RF may be outside "
              f"cropped region. Proceeding anyway.")

    # === Setup ===
    n_px_side = config['n_px_side']
    n_pixels = n_px_side ** 2
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    r_max = config['r_max']
    f_max = config['f_max']

    # Get RF mask
    kernel = getattr(model, 'covar_module', None) or getattr(model, 'kernel', None)
    if not (hasattr(kernel, '_cached_mask') and kernel._cached_mask is not None):
        with torch.no_grad():
            _ = model(X_pool[0].unsqueeze(0))
    rf_mask = kernel._cached_mask.squeeze()

    # Dataset pixel bounds
    rf_mask_np = rf_mask.cpu().numpy()
    X_all = torch.cat([X_pool, env['X_train']], dim=0)
    all_rf_vals = X_all.cpu().numpy()[:, rf_mask_np]
    vmin = float(all_rf_vals.min())
    vmax = float(all_rf_vals.max())
    print(f"\nDataset RF pixel range: [{vmin:.3f}, {vmax:.3f}]")

    # Target image and its utility
    x_target = X_pool[args.target_index]
    with torch.no_grad():
        try:
            res_tgt = eval_utility(model, likelihood,
                                   x_target.unsqueeze(0), x_target,
                                   r_max, utility_type=args.utility)
            target_utility = res_tgt['utility'].item()
        except Exception:
            target_utility = float('nan')
    print(f"Target image: pool[{args.target_index}], U={target_utility:.6f}")

    # Compute dataset baseline: utility of random pool images
    n_baseline = min(200, X_pool.shape[0])
    torch.manual_seed(0)
    baseline_indices = torch.randperm(X_pool.shape[0])[:n_baseline]
    dataset_utilities = []
    with torch.no_grad():
        for idx in baseline_indices:
            try:
                res = eval_utility(model, likelihood,
                                   X_pool[idx].unsqueeze(0), x_target,
                                   r_max, utility_type=args.utility)
                dataset_utilities.append(res['utility'].item())
            except Exception:
                pass
    dataset_utilities = np.array(dataset_utilities)
    dataset_utility_stats = {
        'values': dataset_utilities,  # raw array for strip+box plot
        'mean': float(dataset_utilities.mean()),
        'std': float(dataset_utilities.std()),
        'median': float(np.median(dataset_utilities)),
    }
    print(f"Dataset baseline ({n_baseline} pool images): "
          f"mean={dataset_utility_stats['mean']:.6f} +/- {dataset_utility_stats['std']:.6f}, "
          f"median={dataset_utility_stats['median']:.6f}")

    # === Load diffusion model ===
    print(f"\n{'=' * 60}")
    print("Loading diffusion model")
    print(f"{'=' * 60}")
    diffusion_env = load_diffusion_model(args.model_path, device)
    unet = diffusion_env['unet']
    schedule = diffusion_env['schedule']
    scale_factor = diffusion_env['scale_factor']

    # === Guided diffusion: Guided reverse diffusion ===
    total_steps = args.ddim_steps if args.sampler == 'ddim' else schedule['T']
    print(f"\n{'=' * 60}")
    print(f"Guided diffusion: Guided {args.sampler.upper()} (w={args.guidance_scale}, "
          f"steps={total_steps}, seeds={args.n_samples})")
    print(f"{'=' * 60}")

    results_list = []
    seeds = list(range(42, 42 + args.n_samples))

    for seed in seeds:
        print(f"\n--- Seed {seed} ---")
        t0 = time.time()
        result = guided_reverse_diffusion(
            unet, schedule, scale_factor,
            model, likelihood, x_target, rf_mask,
            r_max, f_max,
            guidance_scale=args.guidance_scale,
            sampler=args.sampler,
            ddim_steps=args.ddim_steps,
            use_full_gradient=args.use_full_gradient,
            seed=seed,
            device=device,
            utility_type=args.utility,
            noisy_gradient=args.noisy_gradient,
        )
        elapsed = time.time() - t0

        # Report final utility
        u_vals = [u for u in result['history']['utility'] if not np.isnan(u)]
        final_u = u_vals[-1] if u_vals else float('nan')
        n_fmax = sum(result['history']['f_max_triggered'])
        print(f"  Done in {elapsed:.1f}s. Final U={final_u:.4f}, "
              f"f_max triggered {n_fmax}/{total_steps} steps")

        results_list.append(result)

    # Select best (skip only inf/nan)
    final_utilities = []
    for r in results_list:
        u_vals = [u for u in r['history']['utility'] if np.isfinite(u)]
        final_utilities.append(u_vals[-1] if u_vals else float('-inf'))
    finite_utils = [u for u in final_utilities if np.isfinite(u)]
    if finite_utils:
        best_idx = int(np.argmax([u if np.isfinite(u) else float('-inf')
                                   for u in final_utilities]))
    else:
        best_idx = 0
    n_diverged = sum(1 for u in final_utilities if not np.isfinite(u))
    if n_diverged > 0:
        print(f"\nWARNING: {n_diverged}/{len(final_utilities)} seeds diverged (inf/nan)")
    print(f"Best seed: {seeds[best_idx]} with U={final_utilities[best_idx]:.4f}")

    # === Pure utility baseline ===
    baseline_result = None
    if not args.no_baseline and not args.no_guidance:
        print(f"\n{'=' * 60}")
        print("Pure utility baseline: LBFGS utility optimization")
        print(f"{'=' * 60}")

        # Create smoothed starting image (same as guided_optimization.py)
        sigma_scaled = SIGMA_SMOOTH * (PNAS_SIZE / n_px_side)
        x_target_2d = x_target.cpu().numpy().reshape(n_px_side, n_px_side)
        x_smoothed_2d = gaussian_filter(x_target_2d, sigma=sigma_scaled)
        x_start_np = x_target_2d.copy().reshape(-1)
        x_start_np[rf_mask_np] = x_smoothed_2d.reshape(-1)[rf_mask_np]
        x_start = torch.tensor(x_start_np, dtype=dtype, device=device)

        print(f"  Start: smoothed target (sigma={sigma_scaled:.2f})")

        t0 = time.time()
        x_final_a, history_a = gradient_ascent_guided(
            model, likelihood, x_start, x_target, rf_mask,
            r_max, f_max, n_px_side,
            N_STEPS, LR, LBFGS_MAX_ITER, LBFGS_MAX_EVAL, LBFGS_HISTORY_SIZE,
            x_background=x_start,
            utility_type=args.utility,
        )
        elapsed = time.time() - t0
        print(f"  Pure utility done in {elapsed:.1f}s. "
              f"Final U={history_a['utility'][-1]:.4f}")

        baseline_result = {
            'x_final': x_final_a,
            'history': history_a,
        }

    # === Visualization ===
    print(f"\n{'=' * 60}")
    print("Visualization")
    print(f"{'=' * 60}")

    # Output filename (see naming convention comment at top of file)
    kernel_name = env['kernel_type']
    util_name = 'standard' if args.utility == 'standard' else 'dist_aware'
    parts = [
        kernel_name,
        util_name,
        f"ntrain={args.n_train}",
        f"{args.sampler}",
        f"w={args.guidance_scale}",
        f"seeds{args.n_samples}",
    ]
    if args.sampler == 'ddim':
        parts.append(f"steps{args.ddim_steps}")
    if args.use_full_gradient:
        parts.append("fullgrad")
    if args.no_guidance:
        parts.append("unconditional")
    if args.beta is not None:
        parts.append(f"beta{args.beta}")
    prefix = 'noise_g_' if args.noisy_gradient else ''
    out_path = _script_dir / f'{prefix}guided_reverse_{"_".join(parts)}.png'

    plot_guided_results(
        results_list, baseline_result, x_target,
        rf_mask, vmin, vmax, n_px_side, config, test_r,
        out_path, target_utility=target_utility,
        utility_type=args.utility, target_index=args.target_index,
        dataset_utility_stats=dataset_utility_stats,
    )

    # === Summary ===
    print(f"\n{'=' * 60}")
    print("Summary")
    print(f"{'=' * 60}")
    print(f"  GP test_r: {test_r:.4f}")
    print(f"  Guided diffusion (best of {args.n_samples}): "
          f"U={final_utilities[best_idx]:.4f} (seed={seeds[best_idx]})")
    if baseline_result is not None:
        print(f"  Pure utility: U={history_a['utility'][-1]:.4f}")
    print(f"  Figure: {out_path}")


if __name__ == '__main__':
    main()
