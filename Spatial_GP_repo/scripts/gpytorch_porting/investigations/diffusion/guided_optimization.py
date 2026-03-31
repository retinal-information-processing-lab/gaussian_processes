"""
Diffusion-guided GP utility optimization on center-cropped PNAS images.

Two-stage script:
  Stage 1 (KEY DELIVERABLE): Train a GP on center-cropped images at any square size.
  Stage 2: Optimize an image using combined GP utility + diffusion score direction.

The GP pipeline (run_single_config) hardcodes 108x108 PNAS data. This script
replicates the pipeline using importable building blocks but with center-cropped
images, enabling GP fitting at any square resolution (32, 48, 64, 80, 108).

Usage:
    # Validate GP at 64x64 (compare to 108x108 baseline):
    python investigations/diffusion/guided_optimization.py --validate-only

    # Validate at a specific crop size:
    python investigations/diffusion/guided_optimization.py --validate-only --crop-size 80

    # Full optimization (GP + diffusion):
    python investigations/diffusion/guided_optimization.py
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
# Path setup (same pattern as gradient.py)
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

# LBFGS optimization constants (same as gradient.py)
N_STEPS = 50
LR = 0.5
LBFGS_MAX_ITER = 20
LBFGS_MAX_EVAL = 25
LBFGS_HISTORY_SIZE = 10

# Image creation (same as gradient.py)
TARGET_INDEX = 9
SIGMA_SMOOTH = 5.0

# Diffusion score constants (Stage 2)
T_SCORE = 50             # timestep for Tweedie denoising
LAMBDA_DIFF = 0.01       # regularization weight for diffusion term

# Original PNAS image size
PNAS_SIZE = 108

# PNAS data path (canonical absolute path -- npz is gitignored, not in worktrees)
PNAS_DATA_PATH = Path.home() / (
    'IDV_code/ClosedLoopProject/gaussian_processes/'
    'Spatial_GP_repo/notebooks/PNAS_paper_sorted_data.npz'
)


# ============================================================================
# Center-crop utility
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


# ============================================================================
# Stage 1: Reusable GP fitting for any square crop size
# ============================================================================

def setup_gp(crop_size=64, kernel_type=None):
    """Train a GP model on center-cropped PNAS images at any square resolution.

    Replicates the run_single_config() pipeline (data loading, STA, kernel
    creation, inducing point selection, training, evaluation) using the same
    importable building blocks, but with center-cropped images.

    The key difference from run_single_config: data loading and flattening
    happen AFTER center-cropping, so n_px_side=crop_size flows through the
    entire pipeline (kernel coordinates, RF mask, STA).

    Neural responses are unchanged -- the neuron was shown the full 108x108
    image when spike counts were recorded. The GP sees a subset of the
    stimulus. If the RF is outside the cropped region, test_r will drop.

    Args:
        crop_size: Square side length to center-crop images to (default: 64).
        kernel_type: Override kernel type. None = use default_params.json.

    Returns:
        dict with keys matching explore_utility.py:setup():
            model: trained GP model (eval mode)
            likelihood: PoissonLikelihood
            X_pool: (N_pool, crop_size^2) pool images (flat, on device)
            X_train: (N_train, crop_size^2) training images
            x_target: (crop_size^2,) first pool image
            config: full config dict
            kernel_type: detected kernel type string
            test_r: Pearson r on test set
            reliability: test reliability
    """
    print(f"\n{'=' * 60}")
    print(f"setup_gp: Training GP on {crop_size}x{crop_size} center-cropped images")
    print(f"{'=' * 60}")

    # ------------------------------------------------------------------
    # 1. Build config from defaults, overriding n_px_side and M/n_train
    # ------------------------------------------------------------------
    overrides = dict(
        mode='default_gpy',
        n_px_side=crop_size,
        M=M,
        n_train=N_TRAIN,
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

    # ------------------------------------------------------------------
    # 2. Load PNAS data and center-crop
    # ------------------------------------------------------------------
    print(f"  Loading data from: {PNAS_DATA_PATH}")
    data = np.load(PNAS_DATA_PATH)

    images_train = data['images_train']  # (N_train, 108, 108, 1)
    images_val = data['images_val']      # (N_val, 108, 108, 1)
    images_test = data['images_test']    # (N_test, 108, 108, 1)

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

    # Combine train + val, convert to torch, flatten to (N, crop_size^2)
    X_all_np = np.concatenate([images_train, images_val], axis=0)
    X_all = torch.tensor(X_all_np, dtype=dtype).reshape(
        X_all_np.shape[0], -1).to(device)
    X_test = torch.tensor(images_test, dtype=dtype).reshape(
        images_test.shape[0], -1).to(device)

    # Responses are unchanged (neuron saw full 108x108 images)
    R_all = torch.cat([
        torch.tensor(data['responses_train'], dtype=dtype),
        torch.tensor(data['responses_val'], dtype=dtype),
    ], dim=0).to(device)
    R_test = torch.tensor(data['responses_test'], dtype=dtype).to(device)

    r = R_all[:, cell]
    r_test = R_test[:, :, cell]  # (30 repeats, 30 images)

    print(f"  Cropped data: X_all={X_all.shape}, X_test={X_test.shape}")

    # ------------------------------------------------------------------
    # 3. Compute RF center from STA on cropped images
    # ------------------------------------------------------------------
    eps_0x, eps_0y = compute_rf_center_from_sta(
        X_all, r, crop_size, zscore=True)
    print(f"  RF center from STA: ({eps_0x:.4f}, {eps_0y:.4f})")

    # ------------------------------------------------------------------
    # 4. Create kernel with RF structure
    # ------------------------------------------------------------------
    kernel = create_kernel(config, crop_size, eps_0x, eps_0y)
    apply_rf_center_bounds(kernel, eps_0x, eps_0y, config)
    kernel = kernel.to(dtype=dtype, device=device)

    # ------------------------------------------------------------------
    # 5. Select inducing points (pivoted Cholesky)
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # 6. Build training set (inducing + extra points)
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # 7. Create model + likelihood
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # 8. Train
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # 9. Evaluate on test set
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # 10. Build pool (non-training images)
    # ------------------------------------------------------------------
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


# ============================================================================
# Stage 2: Diffusion model loading and score computation
# ============================================================================

def load_diffusion_model(checkpoint_path, device):
    """Load trained UNet and cosine schedule from checkpoint.

    Args:
        checkpoint_path: path to .pt checkpoint file
        device: torch device

    Returns:
        dict with keys: unet, schedule, scale_factor, config
    """
    # Import from local diffusion_model.py
    sys.path.insert(0, str(_script_dir))
    from diffusion_model import UNet, cosine_schedule

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    T = checkpoint['config']['T']
    scale_factor = checkpoint['scale_factor']

    unet = UNet(in_channels=1, channels=(32, 64, 128), d_emb=128).to(device)
    unet.load_state_dict(checkpoint['model'])
    unet.eval()

    schedule = cosine_schedule(T)

    print(f"  Loaded diffusion model: epoch={checkpoint['epoch']}, "
          f"T={T}, scale_factor={scale_factor:.4f}")

    return {
        'unet': unet,
        'schedule': schedule,
        'scale_factor': scale_factor,
        'config': checkpoint['config'],
    }


def compute_tweedie_denoised(x_raw_flat, unet, schedule, scale_factor,
                              t, device, seed=None):
    """Compute Tweedie-denoised image estimate from raw pixel flat vector.

    Uses Tweedie's formula: x_0_hat = (x_t - sqrt(1-abar_t) * eps_hat) / sqrt(abar_t)

    The result is DETACHED from the computation graph (no backprop through UNet).

    Args:
        x_raw_flat: (n_pixels,) raw pixel values (GP space)
        unet: trained UNet in eval mode
        schedule: cosine schedule dict (CPU tensors)
        scale_factor: normalization factor (raw / scale_factor -> diffusion space)
        t: timestep for noise level (int, ~50)
        device: torch device
        seed: optional seed for reproducible noise

    Returns:
        x_0_hat_raw_flat: (n_pixels,) denoised estimate in raw pixel space, DETACHED
    """
    n_pixels = x_raw_flat.shape[0]
    side = int(n_pixels ** 0.5)
    assert side * side == n_pixels, f"Non-square pixel count: {n_pixels}"

    with torch.no_grad():
        # Convert to diffusion space
        x_scaled = x_raw_flat / scale_factor  # [-1, 1] range
        x_2d = x_scaled.reshape(1, 1, side, side)  # (1, 1, H, W)

        # Get schedule values (CPU tensors -> index -> move to device)
        sqrt_abar_t = schedule['sqrt_alpha_bar'][t].to(device)
        sqrt_one_minus_abar_t = schedule['sqrt_one_minus_alpha_bar'][t].to(device)

        # Add noise: x_t = sqrt(abar) * x_0 + sqrt(1-abar) * eps
        if seed is not None:
            torch.manual_seed(seed)
        eps = torch.randn_like(x_2d)
        x_t = sqrt_abar_t * x_2d + sqrt_one_minus_abar_t * eps

        # Predict noise
        t_tensor = torch.tensor([t], device=device, dtype=torch.long)
        eps_hat = unet(x_t, t_tensor)

        # Tweedie: x_0_hat = (x_t - sqrt(1-abar) * eps_hat) / sqrt(abar)
        x_0_hat = (x_t - sqrt_one_minus_abar_t * eps_hat) / sqrt_abar_t

        # Convert back to raw pixel space and flatten
        x_0_hat_raw = x_0_hat.squeeze() * scale_factor  # (H, W)
        x_0_hat_raw_flat = x_0_hat_raw.reshape(-1)  # (n_pixels,)

    return x_0_hat_raw_flat


# ============================================================================
# Optimization helpers (adapted from gradient.py)
# ============================================================================

def _reconstruct_image(x_rf, rf_mask, n_pixels, dtype, device, background=None):
    """Place RF pixel values into full image.

    If background is provided, non-RF pixels retain the background values
    (e.g. x_start). This gives the diffusion UNet a natural full image
    instead of zeros outside the RF. The GP kernel ignores non-RF pixels
    regardless (kernels.py:587-589 masks them out before any computation).

    background.detach() ensures no gradients flow through non-RF pixels.
    """
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
                           diffusion_env=None, lambda_diff=LAMBDA_DIFF,
                           t_score=T_SCORE, x_background=None):
    """Gradient ascent maximizing U_DA with optional diffusion regularization.

    Uses LBFGS with strong_wolfe line search. The Tweedie denoised target
    x_0_hat is computed ONCE per outer step (outside the closure) and used
    as a fixed target inside. This ensures function/gradient consistency
    for the line search (proximal/MM pattern).

    If diffusion_env is provided, adds a term pulling x toward the diffusion
    model's denoised estimate: loss += lambda_diff * 0.5 * ||x_rf - x_0_hat_rf||^2

    Args:
        model, likelihood: trained GP model
        x_start: (n_pixels,) starting image
        x_target: (n_pixels,) conditioning target
        rf_mask: boolean mask for RF pixels
        r_max, f_max: utility/firing rate bounds
        n_px_side: image side length
        n_steps, lr, max_iter, max_eval, history_size: LBFGS params
        diffusion_env: dict from load_diffusion_model(), or None for utility-only
        lambda_diff: weight for diffusion regularization
        t_score: timestep for Tweedie denoising
        x_background: (n_pixels,) natural image for non-RF pixels in full image
            reconstruction. If None, non-RF pixels are zeros (legacy behavior).
            Should be x_start so the UNet sees a natural full image.

    Returns:
        x_final: (n_pixels,) optimized image
        history: dict with per-step metrics
    """
    n_pixels = x_start.shape[0]
    device = x_start.device
    dtype = x_start.dtype

    x_rf_init = x_start[rf_mask]
    x_rf = x_rf_init.clone().detach().requires_grad_(True)

    # Always use LBFGS. The previous SGD path was needed when the UNet
    # saw zero-padded images (biased Tweedie estimates). With natural
    # background, the combined loss is smooth enough for strong_wolfe.
    optimizer = torch.optim.LBFGS(
        [x_rf], lr=lr, max_iter=max_iter, max_eval=max_eval,
        history_size=history_size, line_search_fn='strong_wolfe',
    )
    print(f"  Optimizer: LBFGS (lr={lr}, max_iter={max_iter})")

    history = {
        'step': [], 'utility': [], 'grad_norm': [],
        'pearson_r': [], 'proj_coeff': [],
        'loss_utility': [], 'loss_diffusion': [],
    }

    # Record initial state (before any optimization step)
    with torch.no_grad():
        x_full_init = _reconstruct_image(
            x_rf, rf_mask, n_pixels, dtype, device, background=x_background)
        try:
            result_init = distribution_aware_utility(
                model, likelihood,
                x_full_init.unsqueeze(0),
                x_target.unsqueeze(0),
                r_max=r_max,
                adaptive_r_max=False,
                sample_lambda=False,
            )
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
    history['loss_diffusion'].append(0.0)
    print(f"  init   : U={utility_init:.6f}  r={pr_init:.4f}  proj={pc_init:.4f}")

    for step in range(n_steps):
        # Compute Tweedie target ONCE per outer step (fixed for line search).
        # x_0_hat_rf is a constant vector during the LBFGS closure calls,
        # so function and gradient are consistent (proximal/MM pattern).
        if diffusion_env is not None:
            with torch.no_grad():
                x_full_for_tweedie = _reconstruct_image(
                    x_rf, rf_mask, n_pixels, dtype, device,
                    background=x_background)
                x_0_hat_rf = compute_tweedie_denoised(
                    x_full_for_tweedie,
                    diffusion_env['unet'],
                    diffusion_env['schedule'],
                    diffusion_env['scale_factor'],
                    t_score, device,
                    seed=42,
                )[rf_mask]

        # LBFGS closure: combined loss (utility + optional diffusion L2)
        def closure():
            optimizer.zero_grad()
            x_full = _reconstruct_image(
                x_rf, rf_mask, n_pixels, dtype, device,
                background=x_background)

            try:
                result = distribution_aware_utility(
                    model, likelihood,
                    x_full.unsqueeze(0),
                    x_target.unsqueeze(0),
                    r_max=r_max,
                    adaptive_r_max=False,
                    sample_lambda=False,
                )
            except Exception:
                return torch.tensor(float('inf'), device=device)

            mu_g = result['mu_g_marg']
            if torch.exp(mu_g).item() > f_max:
                return torch.tensor(float('inf'), device=device)

            loss = -result['utility'].squeeze()
            if torch.isnan(loss):
                return torch.tensor(float('inf'), device=device)

            # Diffusion L2 penalty toward fixed Tweedie target
            if diffusion_env is not None:
                loss_diff = 0.5 * ((x_rf - x_0_hat_rf) ** 2).sum()
                loss = loss + lambda_diff * loss_diff

            loss.backward()
            return loss

        optimizer.step(closure)

        # Verify gradient on first step
        if step == 0:
            assert x_rf.grad is not None and x_rf.grad.norm() > 0, \
                "No gradient flow through RF reconstruction"
            print(f"  Step 0: |grad|={x_rf.grad.norm().item():.4e}")

        # Track metrics (no grad)
        with torch.no_grad():
            x_full = _reconstruct_image(
                x_rf, rf_mask, n_pixels, dtype, device, background=x_background)
            try:
                result = distribution_aware_utility(
                    model, likelihood,
                    x_full.unsqueeze(0),
                    x_target.unsqueeze(0),
                    r_max=r_max,
                    adaptive_r_max=False,
                    sample_lambda=False,
                )
                utility = result['utility'].item()
            except Exception:
                utility = float('nan')
            pr = rf_pearson_r(x_full, x_target, rf_mask)
            pc = rf_proj_coeff(x_full, x_target, rf_mask)

            if diffusion_env is not None:
                x_0_hat = compute_tweedie_denoised(
                    x_full, diffusion_env['unet'],
                    diffusion_env['schedule'],
                    diffusion_env['scale_factor'],
                    t_score, device, seed=42)
                ld = 0.5 * ((x_rf - x_0_hat[rf_mask]) ** 2).sum().item()
            else:
                ld = 0.0

        grad_norm = x_rf.grad.norm().item() if x_rf.grad is not None else 0.0

        history['step'].append(step)
        history['utility'].append(utility)
        history['grad_norm'].append(grad_norm)
        history['pearson_r'].append(pr)
        history['proj_coeff'].append(pc)
        history['loss_utility'].append(-utility if not np.isnan(utility) else float('nan'))
        history['loss_diffusion'].append(ld)

        if np.isnan(utility) or np.isnan(grad_norm):
            print(f"  step {step}: NaN detected - stopping")
            break

        if step % 5 == 0 or step == n_steps - 1:
            diff_str = f"  L_diff={ld:.4f}" if diffusion_env else ""
            print(f"  step {step:4d}: U={utility:.6f}  "
                  f"r={pr:.4f}  proj={pc:.4f}  |grad|={grad_norm:.4e}{diff_str}")

    with torch.no_grad():
        x_final = _reconstruct_image(
            x_rf, rf_mask, n_pixels, dtype, device, background=x_background)
    return x_final.detach(), history


# ============================================================================
# Visualization
# ============================================================================

def plot_results(x_target, x_start, x_final_util, x_final_guided,
                 history_util, history_guided,
                 rf_mask, kernel, config, diffusion_env,
                 vmin, vmax, test_r, n_px_side, out_path,
                 image_utilities=None):
    """Summary figure: utility-only vs utility+diffusion optimization."""
    mask_2d = rf_mask.cpu().numpy().reshape(n_px_side, n_px_side)

    def to_image(x_flat):
        """Reshape flat vector to 2D image (full size, no cropping)."""
        return x_flat.detach().cpu().numpy().reshape(n_px_side, n_px_side)

    # Check pixel bounds
    def check_bounds(x_flat, label):
        vals = x_flat[rf_mask].cpu().numpy()
        below = vals[vals < vmin]
        above = vals[vals > vmax]
        if len(below) > 0 or len(above) > 0:
            pct = 100 * (len(below) + len(above)) / len(vals)
            print(f"  WARNING: {label} OOB: {pct:.1f}% pixels outside "
                  f"[{vmin:.3f}, {vmax:.3f}]")
            return True
        return False

    from matplotlib.gridspec import GridSpec

    # Top row: images (variable count), bottom row: 3 convergence plots
    images = [
        ('Target', x_target),
        ('Start', x_start),
        ('Final (utility)', x_final_util),
    ]
    if x_final_guided is not None:
        images.append(('Final (util+diff)', x_final_guided))

    n_img_cols = len(images) + (1 if diffusion_env else 0)

    fig = plt.figure(figsize=(4 * n_img_cols, 8))
    gs = GridSpec(2, n_img_cols, figure=fig, hspace=0.3, wspace=0.3)

    # Top row: full images with RF mask contour overlay
    u = image_utilities or {}

    def show_image(ax, x_flat, label, utility_val=None):
        oob = check_bounds(x_flat, label)
        img = to_image(x_flat)
        ax.imshow(img, cmap='gray', vmin=vmin, vmax=vmax, aspect='equal')
        ax.contour(mask_2d, levels=[0.5], colors='cyan', linewidths=0.8)
        title_color = 'red' if oob else 'black'
        title_str = label
        if utility_val is not None and not np.isnan(utility_val):
            title_str += f'\nU={utility_val:.4f}'
        ax.set_title(title_str, fontsize=10, color=title_color)
        ax.axis('off')

    # Utility values ordered to match images list
    image_u_list = [u.get('target'), u.get('start'), u.get('final_util')]
    if x_final_guided is not None:
        image_u_list.append(u.get('final_guided'))

    for i, (label, x) in enumerate(images):
        ax = fig.add_subplot(gs[0, i])
        show_image(ax, x, label,
                   utility_val=image_u_list[i] if i < len(image_u_list) else None)

    # Tweedie denoised image of starting point
    if diffusion_env is not None:
        ax = fig.add_subplot(gs[0, n_img_cols - 1])
        x_0_hat = compute_tweedie_denoised(
            x_start, diffusion_env['unet'], diffusion_env['schedule'],
            diffusion_env['scale_factor'], T_SCORE,
            x_start.device, seed=42)
        show_image(ax, x_0_hat, f'Tweedie denoised\n(t={T_SCORE})')

    # Bottom row: 3 convergence plots spanning the full width
    # Divide bottom row into 3 equal sections using column spans
    plot_width = n_img_cols / 3.0
    plot_starts = [round(i * plot_width) for i in range(3)]
    plot_ends = [round((i + 1) * plot_width) for i in range(3)]

    ax_u = fig.add_subplot(gs[1, plot_starts[0]:plot_ends[0]])
    ax_u.plot(history_util['step'], history_util['utility'], 'b-',
              linewidth=1.5, label='Utility only')
    if history_guided is not None:
        ax_u.plot(history_guided['step'], history_guided['utility'], 'r--',
                  linewidth=1.5, label='Util + diffusion')
    ax_u.set_xlabel('Step')
    ax_u.set_ylabel('U_DA')
    ax_u.set_title('Utility convergence')
    ax_u.legend(fontsize=8)
    ax_u.grid(True, alpha=0.3)

    ax_r = fig.add_subplot(gs[1, plot_starts[1]:plot_ends[1]])
    ax_r.plot(history_util['step'], history_util['pearson_r'], 'b-',
              linewidth=1.5, label='Util only')
    if history_guided is not None:
        ax_r.plot(history_guided['step'], history_guided['pearson_r'], 'r--',
                  linewidth=1.5, label='Util + diff')
    ax_r.set_xlabel('Step')
    ax_r.set_ylabel('Pearson r (RF)')
    ax_r.set_title('Structural similarity')
    ax_r.legend(fontsize=8)
    ax_r.grid(True, alpha=0.3)

    ax_p = fig.add_subplot(gs[1, plot_starts[2]:plot_ends[2]])
    ax_p.plot(history_util['step'], history_util['proj_coeff'], 'b-',
              linewidth=1.5, label='Util only')
    if history_guided is not None:
        ax_p.plot(history_guided['step'], history_guided['proj_coeff'], 'r--',
                  linewidth=1.5, label='Util + diff')
    ax_p.axhline(1.0, color='gray', linestyle=':', alpha=0.3)
    ax_p.set_xlabel('Step')
    ax_p.set_ylabel('Proj coeff (RF)')
    ax_p.set_title('Amplitude convergence')
    ax_p.legend(fontsize=8)
    ax_p.grid(True, alpha=0.3)

    fig.suptitle(
        f'Diffusion-Guided Optimization '
        f'(crop={config["n_px_side"]}, M={config["M"]}, '
        f'test_r={test_r:.3f})',
        fontsize=11)
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {out_path}")
    plt.close(fig)


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Diffusion-guided GP utility optimization')
    parser.add_argument('--crop-size', type=int, default=64,
                        help='Square crop size for GP (default: 64)')
    parser.add_argument('--kernel-type', type=str, default=None,
                        choices=['arc_cosine', 'arc_sine', 'rbf'],
                        help='Kernel type (default: from default_params.json)')
    parser.add_argument('--validate-only', action='store_true',
                        help='Only train GP and report test_r, skip optimization')
    parser.add_argument('--no-diffusion', action='store_true',
                        help='Run utility-only optimization (no diffusion term)')
    parser.add_argument('--lambda-diff', type=float, default=LAMBDA_DIFF,
                        help=f'Diffusion regularization weight (default: {LAMBDA_DIFF})')
    parser.add_argument('--t-score', type=int, default=T_SCORE,
                        help=f'Timestep for Tweedie denoising (default: {T_SCORE})')
    parser.add_argument('--checkpoint', type=str,
                        default=str(_script_dir / 'checkpoints' / 'ddpm_epoch1000.pt'),
                        help='Path to diffusion model checkpoint')
    args = parser.parse_args()

    # === Stage 1: Train GP ===
    env = setup_gp(crop_size=args.crop_size, kernel_type=args.kernel_type)
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
            print("WARNING: test_r < 0.5 -- RF may be outside cropped region. "
                  "Consider a larger crop or different cell.")
        return

    # Warn if test_r is low (RF may be outside cropped region)
    if test_r < 0.5:
        print(f"\nWARNING: test_r={test_r:.4f} < 0.5. RF may be outside "
              f"cropped region. Proceeding anyway.")

    # === Setup images ===
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

    # Target and starting image (same as gradient.py, natural image mode)
    # Scale smoothing sigma by image size: at 108x108, sigma=1.0 is the
    # baseline. At 64x64, pixels are coarser so we need larger sigma to get
    # comparable perturbation. sigma = SIGMA_SMOOTH * (PNAS_SIZE / crop_size).
    sigma_scaled = SIGMA_SMOOTH * (PNAS_SIZE / n_px_side)
    x_target = X_pool[TARGET_INDEX]
    x_target_2d = x_target.cpu().numpy().reshape(n_px_side, n_px_side)
    x_smoothed_2d = gaussian_filter(x_target_2d, sigma=sigma_scaled)
    ## DEBUG: smooth only RF pixels, keep original target outside RF
    x_start_np = x_target_2d.copy().reshape(-1)
    x_start_np[rf_mask_np] = x_smoothed_2d.reshape(-1)[rf_mask_np]
    x_start = torch.tensor(x_start_np, dtype=dtype, device=device)

    print(f"Target: pool image {TARGET_INDEX}")
    print(f"Start: Gaussian smoothing sigma={sigma_scaled:.2f} "
          f"(base={SIGMA_SMOOTH}, scaled by {PNAS_SIZE}/{n_px_side})")
    print(f"  ||target||_RF={x_target[rf_mask].norm().item():.2f}, "
          f"||start||_RF={x_start[rf_mask].norm().item():.2f}")

    # === Utility-only baseline ===
    print(f"\n{'=' * 60}")
    print("Gradient ascent: utility only")
    print(f"{'=' * 60}")
    x_final_util, history_util = gradient_ascent_guided(
        model, likelihood, x_start, x_target, rf_mask,
        r_max, f_max, n_px_side,
        N_STEPS, LR, LBFGS_MAX_ITER, LBFGS_MAX_EVAL, LBFGS_HISTORY_SIZE,
        diffusion_env=None,
        x_background=x_start,
    )

    # === Diffusion-guided optimization ===
    diffusion_env = None
    x_final_guided = None
    history_guided = None

    if not args.no_diffusion:
        print(f"\n{'=' * 60}")
        print("Loading diffusion model")
        print(f"{'=' * 60}")
        diffusion_env = load_diffusion_model(args.checkpoint, device)

        print(f"\n{'=' * 60}")
        print(f"Gradient ascent: utility + diffusion "
              f"(lambda={args.lambda_diff}, t={args.t_score})")
        print(f"{'=' * 60}")
        x_final_guided, history_guided = gradient_ascent_guided(
            model, likelihood, x_start, x_target, rf_mask,
            r_max, f_max, n_px_side,
            N_STEPS, LR, LBFGS_MAX_ITER, LBFGS_MAX_EVAL, LBFGS_HISTORY_SIZE,
            diffusion_env=diffusion_env,
            lambda_diff=args.lambda_diff,
            t_score=args.t_score,
            x_background=x_start,
        )

    # === Compute utilities for all images (for plot annotations) ===
    with torch.no_grad():
        def _compute_utility(x_img):
            try:
                res = distribution_aware_utility(
                    model, likelihood,
                    x_img.unsqueeze(0), x_target.unsqueeze(0),
                    r_max=r_max, adaptive_r_max=False, sample_lambda=False,
                )
                return res['utility'].item()
            except Exception:
                return float('nan')

        u_target = _compute_utility(x_target)
        u_start = _compute_utility(x_start)

    u_final_util = history_util['utility'][-1]
    u_final_guided = history_guided['utility'][-1] if history_guided else None

    image_utilities = {
        'target': u_target,
        'start': u_start,
        'final_util': u_final_util,
        'final_guided': u_final_guided,
    }

    # === Visualization ===
    print(f"\n{'=' * 60}")
    print("Visualization")
    print(f"{'=' * 60}")

    kernel_name = env['kernel_type']
    suffix = f"crop{args.crop_size}_{kernel_name}"
    if diffusion_env:
        suffix += f"_lam{args.lambda_diff}_t{args.t_score}"
    out_path = _script_dir / f'guided_optimization_{suffix}.png'

    plot_results(
        x_target, x_start, x_final_util, x_final_guided,
        history_util, history_guided,
        rf_mask, kernel, config, diffusion_env,
        vmin, vmax, test_r, n_px_side, out_path,
        image_utilities=image_utilities,
    )

    # === Summary ===
    print(f"\n{'=' * 60}")
    print("Summary")
    print(f"{'=' * 60}")
    print(f"  test_r={test_r:.4f}")
    print(f"  Utility-only:  U={history_util['utility'][-1]:.6f}, "
          f"r={history_util['pearson_r'][-1]:.4f}")
    if history_guided is not None:
        print(f"  Util+diffusion: U={history_guided['utility'][-1]:.6f}, "
              f"r={history_guided['pearson_r'][-1]:.4f}")


if __name__ == '__main__':
    main()
