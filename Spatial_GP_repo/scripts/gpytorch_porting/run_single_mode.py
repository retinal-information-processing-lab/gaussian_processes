#!/usr/bin/env python3
"""
run_single_mode.py - Run a single training mode on PNAS neural data.

For experimentation and development. Run ONE training mode with full CLI control.
For structured experiments, use run_experiment.py instead.

Training modes:
  - vargp_old: Original varGP implementation (reference)
  - vargp_direct: Eigenspace-based custom implementation (default)
  - default_gpy: Standard GPyTorch variational inference

Usage:
    python run_single_mode.py --ntilde 50 --mode vargp_old       # Reference
    python run_single_mode.py --ntilde 50 --mode vargp_direct    # Eigenspace (default)
    python run_single_mode.py --ntilde 50 --mode default_gpy     # Standard GPyTorch
    python run_single_mode.py  # Uses defaults: M=100, mode=vargp_direct

Gradient modes (for GPyTorch modes):
    --gradient-mode autograd   # PyTorch autograd (default)
    --gradient-mode vjp        # VJP analytical - same speed as autograd
    --gradient-mode jacobian   # Slow but matches varGP exactly

Default Parameters:
    All parameters loaded from default_params.json for consistency
    across vargp_old, vargp_direct, and default_gpy modes:
    - Kernel: sigma_0=1.0, Amp=1.0, beta=0.1, rho=0.1
    - Link function: A_init=0.01, lambda0_init=1.0
    - Training: n_iterations=50, n_estep=10, n_fstep=10, n_mstep=10, lr=0.1
    - Data: n_train=500, ntilde=100
"""

import sys
import time
import argparse
import json
import numpy as np
from datetime import datetime
from pathlib import Path

# Add paths for imports
sys.path.insert(0, '/home/idv-eqs8-pza/IDV_code/ClosedLoopProject')
sys.path.insert(0, '/home/idv-eqs8-pza/IDV_code/ClosedLoopProject/gaussian_processes/torchlambertw')

import torch
import gpytorch
import matplotlib.pyplot as plt

# NOTE: GP_utils is imported lazily inside vargp_old mode to avoid
# side effects (torch.set_grad_enabled(False) at utils.py line 2)

# Import our GPyTorch components
from kernels import GRADIENT_MODES, KERNEL_TYPES, create_kernel
from likelihoods import PoissonLikelihood
from gpy_model import VariationalGPModel
from gpy_training import train_gpy_default, predict
from metrics import compute_pearson_correlation, compute_explained_variance, compute_adjusted_r_squared
from eigenspace_training import train_eigenspace, predict_eigenspace
from eigenspace_model import DirectVGPModel
from eigenspace_utils import EIGVAL_TOL
from tests.test_utils import set_reproducible_seed


def get_git_commit():
    """Get current git commit hash (short form)."""
    import subprocess
    try:
        result = subprocess.run(
            ['git', 'rev-parse', '--short', 'HEAD'],
            capture_output=True, text=True, check=True,
            cwd=Path(__file__).parent
        )
        return result.stdout.strip()
    except Exception:
        return "unknown"


def load_pnas_data(data_path, dtype=torch.float64):
    """Load PNAS dataset."""
    data = np.load(data_path)
    return {
        'X_train': torch.tensor(data['images_train'], dtype=dtype),
        'X_val': torch.tensor(data['images_val'], dtype=dtype),
        'X_test': torch.tensor(data['images_test'], dtype=dtype),
        'R_train': torch.tensor(data['responses_train'], dtype=dtype),
        'R_val': torch.tensor(data['responses_val'], dtype=dtype),
        'R_test': torch.tensor(data['responses_test'], dtype=dtype),
    }


def _compute_sta_2d(X, r, n_px_side):
    """Compute z-scored STA image as numpy 2D array."""
    X_mean = X.mean(dim=0, keepdim=True)
    X_std = X.std(dim=0, keepdim=True)
    X_norm = (X - X_mean) / (X_std + 1e-8)
    STA = (r[:, None] * X_norm).sum(dim=0) / r.sum()
    return STA.reshape(n_px_side, n_px_side).cpu().numpy()


def plot_fit(r_test_mean, f_pred, cellid, ntilde, test_corr, explained_var, reliability,
             STA_init_2d=None, STA_train_2d=None,
             init_eps_0x=None, init_eps_0y=None, init_beta=None,
             kernel=None, n_px_side=None, output_path=None):
    """Plot actual vs predicted firing rates, with optional STA + RF visualization.

    Args:
        r_test_mean: Mean actual firing rates, shape (n_images,)
        f_pred: Predicted firing rates, shape (n_images,)
        cellid: Cell ID for title
        ntilde: Number of inducing points M
        test_corr: Pearson correlation on test set
        explained_var: Explained variance value
        reliability: Cell reliability
        STA_init_2d: Initial STA image (n_px_side, n_px_side) numpy array, or None
        STA_train_2d: Training-set STA image (n_px_side, n_px_side) numpy array, or None
        init_eps_0x, init_eps_0y: Initial RF center in normalized [-1, 1] coords
        init_beta: Initial beta (natural) value for RF width
        kernel: Trained kernel object (for extracting final RF params), or None
        n_px_side: Image side length for coordinate conversion
        output_path: If provided, save figure to this path instead of showing
    """
    r_actual = r_test_mean.cpu().numpy()
    f_predicted = f_pred.cpu().numpy()

    # Sort by actual firing rate for second subplot
    sort_idx = np.argsort(r_actual)
    r_sorted = r_actual[sort_idx]
    f_sorted = f_predicted[sort_idx]

    has_sta = STA_init_2d is not None and STA_train_2d is not None
    nrows = 2 if has_sta else 1
    fig, axes = plt.subplots(nrows, 2, figsize=(14, 5 * nrows))
    if nrows == 1:
        ax1, ax2 = axes
    else:
        (ax1, ax2), (ax3, ax4) = axes

    n_images = len(r_actual)
    x = np.arange(n_images)

    # Top-left: original order
    for xi in x:
        ax1.axvline(xi, color='gray', linestyle=':', linewidth=0.5, alpha=0.5)
    ax1.plot(x, r_actual, 'k-', linewidth=1.5, label='Actual (mean of 30 reps)')
    ax1.plot(x, f_predicted, 'r-', linewidth=1.5, label='Predicted')
    ax1.set_xlabel('Test image index')
    ax1.set_ylabel('Firing rate (spikes)')
    ax1.set_title('Original order')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)

    # Top-right: sorted by actual firing rate
    for xi in x:
        ax2.axvline(xi, color='gray', linestyle=':', linewidth=0.5, alpha=0.5)
    ax2.plot(x, r_sorted, 'k-', linewidth=1.5, label='Actual (mean of 30 reps)')
    ax2.plot(x, f_sorted, 'r-', linewidth=1.5, label='Predicted')
    ax2.set_xlabel('Test images (sorted by actual firing rate)')
    ax2.set_ylabel('Firing rate (spikes)')
    ax2.set_title('Sorted by actual firing rate')
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)

    # Add metrics text box to right subplot
    metrics_text = (f'M = {ntilde}\n'
                    f'Pearson r = {test_corr:.3f}\n'
                    f'Reliability = {reliability:.3f}\n'
                    f'Expl. var = {explained_var:.3f}')
    ax2.text(0.02, 0.98, metrics_text, transform=ax2.transAxes, fontsize=10,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Bottom row: STA + RF visualizations
    if has_sta:
        def _draw_rf_overlay(ax, eps_0x, eps_0y, beta, label_prefix=''):
            """Draw RF center dot + 1/2-sigma circles on an imshow axis."""
            cx = (eps_0x + 1) / 2 * (n_px_side - 1)
            cy = (eps_0y + 1) / 2 * (n_px_side - 1)
            # sigma from natural beta: locality mask alpha = exp(-[1/(4*beta^2)] * r^2)
            # => sigma = beta * sqrt(2) in normalized coords
            sigma_rf = beta * np.sqrt(2)
            sigma_px = sigma_rf * (n_px_side - 1) / 2

            ax.plot(cx, cy, 'ko', markersize=6)
            circle_1s = plt.Circle((cx, cy), sigma_px, fill=False,
                                   color='black', linewidth=1.5, label=f'1sig ({sigma_px:.0f}px)')
            circle_2s = plt.Circle((cx, cy), 2 * sigma_px, fill=False,
                                   color='black', linewidth=1, linestyle='--', label=f'2sig ({2*sigma_px:.0f}px)')
            ax.add_patch(circle_1s)
            ax.add_patch(circle_2s)
            ax.legend(loc='upper right', fontsize=8)

            param_text = f'{label_prefix}beta={beta:.3f}\neps=({eps_0x:.3f}, {eps_0y:.3f})'
            ax.text(0.02, 0.98, param_text, transform=ax.transAxes, fontsize=8,
                    verticalalignment='top', fontfamily='monospace',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        # Bottom-left: Initial STA + initial RF
        vmax = max(abs(STA_init_2d.min()), abs(STA_init_2d.max()))
        ax3.imshow(STA_init_2d, cmap='RdBu_r', origin='lower', vmin=-vmax, vmax=vmax)
        ax3.set_title('Initial STA + initial RF')
        ax3.set_xlabel('x (pixels)')
        ax3.set_ylabel('y (pixels)')

        if init_eps_0x is not None and init_beta is not None:
            _draw_rf_overlay(ax3, init_eps_0x, init_eps_0y, init_beta, label_prefix='init ')
            ax3.set_xlim(0, n_px_side - 1)
            ax3.set_ylim(0, n_px_side - 1)

        # Bottom-right: Training STA + trained RF overlay
        vmax_tr = max(abs(STA_train_2d.min()), abs(STA_train_2d.max()))
        ax4.imshow(STA_train_2d, cmap='RdBu_r', origin='lower', vmin=-vmax_tr, vmax=vmax_tr)
        ax4.set_title('Training STA + trained RF')
        ax4.set_xlabel('x (pixels)')
        ax4.set_ylabel('y (pixels)')

        if kernel is not None and hasattr(kernel, 'eps_0x'):
            import torch as _torch
            eps_0x_final = kernel.eps_0x.item()
            eps_0y_final = kernel.eps_0y.item()
            beta_final = kernel.beta.item()
            _draw_rf_overlay(ax4, eps_0x_final, eps_0y_final, beta_final, label_prefix='final ')
            ax4.set_xlim(0, n_px_side - 1)
            ax4.set_ylim(0, n_px_side - 1)

    fig.suptitle(f'Cell {cellid} - M={ntilde}', fontsize=12, fontweight='bold')
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150)
        print(f"Figure saved to: {output_path}")
    else:
        plt.show()

    plt.close(fig)
    return fig


# =========================================================================
# Config builders
# =========================================================================

def build_config_from_defaults(**overrides):
    """Build a flat config dict by reading default_params.json.

    This is the mandatory way for standalone scripts (investigations,
    validations) to build a config for run_single_config(). All parameter
    values come from default_params.json — the same file the CLI uses —
    so they stay in sync with the project defaults automatically.

    Everything (mode, seed, cell, M, n_train, ...) comes from
    default_params.json unless explicitly overridden via kwargs.

    Args:
        **overrides: Key-value pairs to override. Any key accepted by
            run_single_config() can be overridden here.

    Returns:
        Flat config dict suitable for run_single_config().

    Example:
        # All defaults from default_params.json:
        config = build_config_from_defaults()

        # Override M for a quick test (everything else from defaults):
        config = build_config_from_defaults(M=50)
    """
    # ------------------------------------------------------------------
    # Read default_params.json — single source of truth for defaults
    # ------------------------------------------------------------------
    defaults_path = Path(__file__).parent / 'default_params.json'
    with open(defaults_path, 'r') as f:
        defaults = json.load(f)

    run = defaults['run']
    ker = defaults['kernel']
    lik = defaults['link_function']
    trn = defaults['training']
    es = defaults['early_stopping']
    mod = defaults['model']
    dat = defaults['data']
    ind = defaults['inducing']
    utl = defaults['utility']

    # ------------------------------------------------------------------
    # Build flat config — every value traces back to default_params.json
    # except where noted
    # ------------------------------------------------------------------
    config = {
        # --- Run point ---
        'mode': run['mode'],
        'M': dat['ntilde'],              # default_params.json -> data.ntilde
        'n_train': dat['n_train'],        # default_params.json -> data.n_train
        'seed': dat['seed'],              # default_params.json -> data.seed
        'cell': dat['cellid'],            # default_params.json -> data.cellid
        'n_iterations': trn['n_iterations'],

        # --- Device/dtype ---
        'device': run['device'],
        'dtype': run['dtype'],

        # --- Kernel (from kernel section) ---
        'kernel_type': ker['type'],
        'sigma_0': ker['sigma_0'],
        'Amp': ker['Amp'],
        'beta': ker['beta'],
        'rho': ker['rho'],
        'eps_0x': None,                   # None = compute from STA
        'eps_0y': None,                   # (default_params.json has 0.0 as placeholder)
        'gradient_mode': ker['gradient_mode'],
        'use_mask': ker['use_mask'],
        'bound_rf_center': ker['bound_rf_center'],
        'n_sigma_rf_bounds': ker['n_sigma_rf_bounds'],
        'lengthscale': ker['lengthscale'],  # only used when kernel_type=rbf

        # --- Likelihood (from link_function section) ---
        'A_init': lik['A_init'],
        'lambda0_init': lik['lambda0_init'],

        # --- Training (from training section) ---
        'n_estep': trn['n_estep'],
        'n_fstep': trn['n_fstep'],
        'n_mstep': trn['n_mstep'],
        'lr': trn['lr'],
        'optimizer': trn['optimizer'],

        # --- Early stopping (from early_stopping section) ---
        'early_stop': es['enabled'],
        'stop_window': es['window'],
        'stop_thresh': es['threshold'],
        'min_iterations': es['min_iterations'],

        # --- Numerical (from model section) ---
        'jitter': mod['jitter'],
        'cholesky_max_tries': mod['cholesky_max_tries'],
        'eigval_tol': mod['eigval_tol'],
        'gpy_lbfgs_max_iter': mod['gpy_lbfgs_max_iter'],
        'lambda_var_clamp': mod['lambda_var_clamp'],
        'stability_threshold': mod['stability_threshold'],

        # --- Data (from data section) ---
        'data_path': dat['path'],
        'n_px_side': dat['n_px_side'],    # null = auto-detect from loaded data
        'use_cache': mod['use_cache'],

        # --- Inducing point selection (from inducing section) ---
        'ip_selection': ind['selection_method'],
        'n_candidates': ind['n_candidates'],
        'n_samples_sta': ind['n_samples_sta'],

        # --- Utility / acquisition (from utility section) ---
        'n_mc_samples': utl['n_mc_samples'],
        'r_max': utl['r_max'],
        'f_max': utl['f_max'],

        # --- Runtime flags (not configurable via default_params.json) ---
        'mstep_analytical': False,
        'unwhitened_variational_dist': False,
        'save_plot': 'none',
        'plot': False,
    }

    # ------------------------------------------------------------------
    # Apply caller overrides — these are the ONLY non-default values.
    # Anything passed here is an intentional, visible deviation.
    # ------------------------------------------------------------------
    config.update(overrides)

    return config


def flatten_yaml_config(yaml_config, mode, M, n_train, seed, cell):
    """Convert nested YAML config + single matrix point to flat config dict.

    This is the bridge between the YAML experiment system and run_single_config().
    The YAML config has nested sections (kernel, training, etc.) and an experiment
    matrix. This function takes one point from the matrix and produces a flat dict.

    Args:
        yaml_config: Parsed YAML config (dict with nested sections)
        mode: Training mode for this run
        M: Number of inducing points for this run
        n_train: Number of training samples for this run
        seed: Random seed for this run
        cell: Cell ID for this run

    Returns:
        Flat config dict suitable for run_single_config()
    """
    exp = yaml_config['experiment']
    ker = yaml_config['kernel']
    lik = yaml_config['likelihood']
    trn = yaml_config['training']
    opt = yaml_config['optimizer']
    es = yaml_config['early_stopping']
    num = yaml_config['numerical']
    ind = yaml_config['inducing']
    utl = yaml_config['utility']
    dat = yaml_config['data']

    return {
        # Run point
        'mode': mode,
        'M': M,
        'n_train': n_train,
        'seed': seed,
        'cell': cell,
        'n_iterations': exp['n_iterations'],

        # Device/dtype
        'device': num['device'],
        'dtype': num['dtype'],

        # Kernel
        'kernel_type': ker['type'],
        'sigma_0': ker['sigma_0'],
        'Amp': ker['Amp'],
        'beta': ker['beta'],
        'rho': ker['rho'],
        'eps_0x': ker['eps_0x'],  # null in YAML = None = compute from STA
        'eps_0y': ker['eps_0y'],
        'gradient_mode': ker['gradient_mode'],
        'use_mask': ker['use_mask'],
        'bound_rf_center': ker['bound_rf_center'],
        'n_sigma_rf_bounds': ker['n_sigma_rf_bounds'],
        'lengthscale': ker['lengthscale'],

        # Likelihood
        'A_init': lik['A_init'],
        'lambda0_init': lik['lambda0_init'],

        # Training
        'n_estep': trn['n_estep'],
        'n_fstep': trn['n_fstep'],
        'n_mstep': trn['n_mstep'],
        'lr': trn['lr'],
        'optimizer': trn['optimizer'],

        # Early stopping
        'early_stop': es['enabled'],
        'stop_window': es['window'],
        'stop_thresh': es['threshold'],
        'min_iterations': es['min_iterations'],

        # Optimizer details
        'gpy_lbfgs_max_iter': opt['gpy_lbfgs_max_iter'],

        # Numerical
        'jitter': num['jitter'],
        'cholesky_max_tries': num['cholesky_max_tries'],
        'eigval_tol': num['eigval_tol'],
        'lambda_var_clamp': num['lambda_var_clamp'],
        'stability_threshold': num['stability_threshold'],

        # Inducing point selection
        'ip_selection': ind['selection_method'],
        'n_candidates': ind['n_candidates'],
        'n_samples_sta': ind['n_samples_sta'],

        # Utility / acquisition
        'n_mc_samples': utl['n_mc_samples'],
        'r_max': utl['r_max'],
        'f_max': utl['f_max'],

        # Data
        'data_path': dat['path'],
        'n_px_side': dat['n_px_side'],    # null = auto-detect from loaded data
        'use_cache': dat['use_cache'],

        # Runtime options (not in YAML, defaults for experiment runs)
        'mstep_analytical': False,
        'unwhitened_variational_dist': False,
        'save_plot': 'none',
        'plot': False,
    }


# =========================================================================
# Parameter consistency check
# =========================================================================

def _validate_model_params(model, likelihood, results):
    """Verify that scalar params in results dict match model/likelihood objects.

    Called before returning from run_single_config to prevent silent divergence
    between the two sources of truth. Only checks parameters available in
    default_gpy and vargp_direct modes.
    """
    kernel = model.covar_module
    checks = {
        'final_A': likelihood.A.item(),
        'final_lambda0': likelihood.lambda0.item(),
        'final_sigma_0': kernel.sigma_0.item(),
        'final_Amp': kernel.Amp.item(),
        'final_beta': kernel.beta.item(),
        'final_rho': kernel.rho.item(),
        'final_eps_0x': kernel.eps_0x.item(),
        'final_eps_0y': kernel.eps_0y.item(),
    }
    for key, model_val in checks.items():
        dict_val = results[key]
        if dict_val is None:
            continue
        if abs(model_val - dict_val) > 1e-6:
            raise ValueError(
                f"Parameter mismatch: results['{key}']={dict_val} "
                f"but model has {model_val}"
            )


# =========================================================================
# Core training function
# =========================================================================

def run_single_config(config):
    """Run one training configuration and return results.

    This is the core function that does data loading, model setup, training,
    and evaluation. Called by both the CLI (main) and run_experiment.py.

    Args:
        config: Flat dict with all parameters for ONE run. Required keys:
            data_path, mode, M, n_train, seed, cell, n_iterations,
            device, dtype,
            sigma_0, Amp, beta, rho, eps_0x, eps_0y, gradient_mode, use_mask,
            A_init, lambda0_init,
            n_estep, n_fstep, n_mstep, lr, optimizer,
            early_stop, stop_window, stop_thresh, min_iterations,
            jitter, eigval_tol,
            n_px_side (null=auto-detect from data), use_cache,
            mstep_analytical, unwhitened_variational_dist,
            ip_selection, n_candidates, n_samples_sta

    Returns:
        dict with metrics, timing, and final parameters. None if training failed.
    """
    mode = config['mode']
    kernel_type = config['kernel_type']
    M = config['M']
    n_train_requested = config['n_train']
    seed = config['seed']
    cell = config['cell']
    n_iterations = config['n_iterations']

    # --- Validation guards for kernel/mode compatibility ---
    if kernel_type != 'arc_cosine' and mode == 'vargp_old':
        raise ValueError(
            f"vargp_old mode only supports arc_cosine kernel (got '{kernel_type}'). "
            f"The old codebase (utils.py:varGP) only implements arc-cosine.")
    if kernel_type != 'arc_cosine' and config['gradient_mode'] in ('vjp', 'jacobian'):
        raise ValueError(
            f"Analytical gradient modes (vjp, jacobian) only support arc_cosine kernel "
            f"(got '{kernel_type}'). Use --gradient-mode autograd for other kernels.")

    device = torch.device(config['device'])
    dtype = torch.float32 if config['dtype'] == 'float32' else torch.float64

    print(f"Device: {device}")
    print(f"Mode: {mode}")
    if kernel_type != 'arc_cosine':
        print(f"Kernel type: {kernel_type}")
    print(f"M={M} inducing points")
    if config['gradient_mode'] != 'autograd':
        print(f"Gradient mode: {config['gradient_mode']}")
    if config['unwhitened_variational_dist']:
        print("Using UnwhitenedVariationalStrategy")

    # Warning for analytical M-step without float32
    if config['mstep_analytical'] and dtype != torch.float32:
        import warnings
        warnings.warn(
            "\n" + "="*70 + "\n"
            "WARNING: --mstep-analytical with float64 is extremely slow (~50s vs ~5s).\n"
            "The analytical gradient implementation has not been optimized for float64.\n"
            "Consider using --dtype float32 for comparable performance to vargp_old.\n"
            + "="*70,
            UserWarning
        )

    # Set seed with explicit CUDA init for reproducibility
    set_reproducible_seed(seed, device=device)
    print(f"Seed: {seed}")

    # Load data — path from config, resolved relative to gpytorch_porting/
    data_path = Path(config['data_path'])
    if not data_path.is_absolute():
        data_path = Path(__file__).parent / data_path
    print(f"Loading data from: {data_path}")
    data = load_pnas_data(data_path, dtype=dtype)
    if dtype == torch.float32:
        print("WARNING: Using float32 - may cause numerical instability")

    # Auto-detect n_px_side from image shape (N, H, W, 1)
    detected_n_px_side = data['X_train'].shape[1]
    assert data['X_train'].shape[1] == data['X_train'].shape[2], \
        f"Expected square images, got {data['X_train'].shape[1]}x{data['X_train'].shape[2]}"
    config_n_px_side = config['n_px_side']  # null/None = auto-detect, int = validate
    if config_n_px_side is not None and config_n_px_side != detected_n_px_side:
        raise ValueError(
            f"Config n_px_side={config_n_px_side} but loaded data is "
            f"{detected_n_px_side}x{detected_n_px_side}. "
            f"Set n_px_side to null for auto-detection or fix the mismatch."
        )
    n_px_side = detected_n_px_side
    config['n_px_side'] = n_px_side  # update config for downstream code
    print(f"Image dimensions: {n_px_side}x{n_px_side} ({n_px_side**2} pixels)")

    # Combine train + val, flatten
    X = torch.cat([data['X_train'], data['X_val']], dim=0)
    R = torch.cat([data['R_train'], data['R_val']], dim=0)
    X = X.reshape(X.shape[0], -1).to(device)  # (N, n_px_side^2)
    R = R.to(device)

    X_test = data['X_test'].reshape(data['X_test'].shape[0], -1).to(device)
    R_test = data['R_test'].to(device)

    # Select cell
    r = R[:, cell]
    r_test = R_test[:, :, cell]  # (30 repeats, 30 images)

    # =========================================================================
    # Step 1: Compute RF center from STA
    # =========================================================================
    # STA is computed BEFORE training set selection because pivoted Cholesky
    # needs the kernel (which needs the RF center). n_samples_sta controls
    # how many random images are used — mimics a real experiment with limited
    # initial data. null = use all available data (idealized).
    from utils import compute_rf_center_from_sta

    ip_selection = config['ip_selection']
    n_samples_sta = config['n_samples_sta']

    if n_samples_sta is not None:
        n_sta = min(n_samples_sta, X.shape[0])
        indices_sta = torch.randperm(X.shape[0], device=device)[:n_sta]
        X_sta = X[indices_sta]
        r_sta = r[indices_sta]
        print(f"\nSTA computed from {n_sta} random images (n_samples_sta={n_samples_sta})")
    else:
        X_sta = X
        r_sta = r
        print(f"\nSTA computed from all {X.shape[0]} available images")

    eps_0x_sta, eps_0y_sta = compute_rf_center_from_sta(
        X_sta, r_sta, n_px_side, zscore=True
    )

    # Compute STA image for visualization
    STA_init_2d = _compute_sta_2d(X_sta, r_sta, n_px_side)

    # Use STA-computed center if config has None (null in YAML)
    eps_0x = config['eps_0x'] if config['eps_0x'] is not None else eps_0x_sta
    eps_0y = config['eps_0y'] if config['eps_0y'] is not None else eps_0y_sta

    print(f"RF center: ({eps_0x:.4f}, {eps_0y:.4f})")
    if eps_0x == eps_0x_sta and eps_0y == eps_0y_sta:
        print(f"  (computed from STA)")
    else:
        print(f"  (from config, STA was: {eps_0x_sta:.4f}, {eps_0y_sta:.4f})")

    # =========================================================================
    # Step 2: Select inducing points
    # =========================================================================
    jitter = config['jitter']
    if dtype == torch.float64 and jitter >= 1e-4:
        import warnings
        warnings.warn(
            f"jitter={jitter} is high for float64 (GPyTorch default for float64 is 1e-6). "
            f"Consider reducing jitter when using --dtype float64."
        )
    n_train = min(n_train_requested, X.shape[0])

    if ip_selection == 'pivoted' and mode != 'vargp_old':
        from utils import select_inducing_points_pivoted

        # Create temporary kernel for pivoted selection (will be re-created
        # by the training code with the same params)
        temp_kernel = create_kernel(config, n_px_side, eps_0x, eps_0y
        ).to(device=device, dtype=dtype)

        ntilde = min(M, X.shape[0])
        n_candidates = config['n_candidates']
        inducing_points, indices_inducing = select_inducing_points_pivoted(
            X, temp_kernel, ntilde,
            n_candidates=n_candidates,
            seed=seed,
            jitter=jitter,
        )
        del temp_kernel  # Free GPU memory

        print(f"\nInducing points: {ntilde} selected via pivoted Cholesky"
              f" (n_candidates={'all' if n_candidates is None else n_candidates})")
    else:
        # Random selection: shuffle and take first M (original behavior)
        all_indices = torch.randperm(X.shape[0], device=device)
        ntilde = min(M, n_train)
        indices_inducing = all_indices[:ntilde]
        inducing_points = X[indices_inducing].clone()
        print(f"\nInducing points: {ntilde} selected randomly")

    # =========================================================================
    # Step 3: Build training set
    # =========================================================================
    # Inducing points are always included in training. If n_train > M,
    # add extra random points from the remaining pool.
    inducing_set = set(indices_inducing.cpu().numpy().tolist())

    if n_train > ntilde:
        # Get indices not already selected as inducing
        remaining = [i for i in range(X.shape[0]) if i not in inducing_set]
        remaining_t = torch.tensor(remaining, device=device)

        # Shuffle remaining and take (n_train - ntilde) extras
        n_extra = n_train - ntilde
        perm = torch.randperm(remaining_t.shape[0], device=device)[:n_extra]
        extra_indices = remaining_t[perm]

        indices_train = torch.cat([indices_inducing, extra_indices])
    else:
        indices_train = indices_inducing
        n_train = ntilde  # Can't have fewer training points than inducing

    X_train = X[indices_train]
    r_train = r[indices_train]

    print(f"\nData shapes:")
    print(f"  X_train: {X_train.shape}")
    print(f"  r_train: {r_train.shape}")
    print(f"  inducing_points: {inducing_points.shape}")
    print(f"  X_test: {X_test.shape}")

    # Training params from config
    lr = config['lr']
    n_estep = config['n_estep']
    n_fstep = config['n_fstep']
    n_mstep = config['n_mstep']
    early_stop = config['early_stop']
    stop_window = config['stop_window']
    stop_thresh = config['stop_thresh']
    min_iterations = config['min_iterations']
    jitter = config['jitter']
    eigval_tol = config['eigval_tol']
    lambda_var_clamp = config['lambda_var_clamp']
    stability_threshold = config['stability_threshold']
    A_init = config['A_init']
    lambda0_init = config['lambda0_init']

    # =========================================================================
    # VARGP_OLD MODE: Use original varGP implementation
    # =========================================================================
    if mode == 'vargp_old':
        # Lazy import to avoid side effects (utils.py disables gradients at module level)
        from gaussian_processes.Spatial_GP_repo import utils as GP_utils

        print(f"\nRunning original varGP (reference implementation)")
        print(f"  maxiter={n_iterations}, nEstep={n_estep}, nMstep={n_mstep}, nFparamstep={n_fstep}")
        if jitter != 1e-4:
            print(f"  NOTE: jitter={jitter} ignored (vargp_old does not use jitter)")

        # Initialize hyperparameters
        beta_t = torch.tensor(config['beta'], device=device)
        rho_t = torch.tensor(config['rho'], device=device)

        theta = {
            'sigma_0': torch.tensor(config['sigma_0'], device=device).requires_grad_(),
            'Amp': torch.tensor(config['Amp'], device=device).requires_grad_(),
            'eps_0x': torch.tensor(eps_0x, device=device).requires_grad_(),
            'eps_0y': torch.tensor(eps_0y, device=device).requires_grad_(),
            '-2log2beta': (-2 * torch.log(2 * beta_t)).requires_grad_(),
            '-log2rho2': (-torch.log(2 * rho_t * rho_t)).requires_grad_(),
        }

        X_train_f32 = X_train.float()
        r_train_f32 = r_train.float()

        hyperparams_tuple = GP_utils.generate_theta(
            x=X_train_f32, r=r_train_f32, n_px_side=n_px_side, display_hyper=False, **theta
        )

        A = torch.tensor(A_init, device=device)
        f_params = {
            'logA': torch.log(A).requires_grad_(),
            'lambda0': torch.tensor(lambda0_init, device=device),
        }

        fit_parameters = {
            'ntilde': ntilde,
            'maxiter': n_iterations,
            'nMstep': n_mstep,
            'nEstep': n_estep,
            'nFparamstep': n_fstep,
            'kernfun': GP_utils.acosker,
            'cellid': cell,
            'n_px_side': n_px_side,
        }

        vargp_old_args = {
            'fit_parameters': fit_parameters,
            'xtilde': inducing_points.float(),
            'hyperparams_tuple': hyperparams_tuple,
            'f_params': f_params,
            'm': torch.zeros(ntilde, device=device),
        }

        print(f"  A_init: {A_init}, lambda0_init: {lambda0_init}")

        start_time = time.time()
        fit_model, err_dict = GP_utils.varGP(X_train_f32, r_train_f32, **vargp_old_args)
        train_time = time.time() - start_time

        if err_dict['is_error']:
            print(f"  ERROR: {err_dict['error']}")
            return None

        _, f_pred, r2, sigma_r2 = GP_utils.test(
            X_test.reshape(-1, n_px_side, n_px_side, 1).float(),
            r_test.float(),
            X_train=X.float(),
            at_iteration=None,
            **fit_model
        )

        if not isinstance(f_pred, torch.Tensor):
            f_pred = torch.tensor(f_pred, device=device)
        else:
            f_pred = f_pred.to(device)

        r_test_mean = r_test.mean(dim=0)
        test_corr = compute_pearson_correlation(r_test_mean.float(), f_pred.float())
        explained_var, reliability = compute_explained_variance(r_test.float(), f_pred.float())
        adjusted_r2 = compute_adjusted_r_squared(r_test.float(), f_pred.float())
        train_corr = float('nan')

        final_loss = fit_model.get('loss', 0.0)
        pred_std = f_pred.std().item()
        pred_mean = f_pred.mean().item()
        pred_min = f_pred.min().item()
        pred_max = f_pred.max().item()

        predictions = {'f_pred': f_pred}
        losses = [final_loss]
        stopped_early = False
        final_iteration = n_iterations

        # Extract final hyperparameters
        theta_final = fit_model['hyperparams_tuple'][0]
        f_p = fit_model['f_params']
        raw_beta = theta_final['-2log2beta'].item() if hasattr(theta_final['-2log2beta'], 'item') else float(theta_final['-2log2beta'])
        raw_rho = theta_final['-log2rho2'].item() if hasattr(theta_final['-log2rho2'], 'item') else float(theta_final['-log2rho2'])
        final_beta = np.exp(-raw_beta / 2) / 2
        final_rho = np.sqrt(np.exp(-raw_rho) / 2)
        logA_val = f_p['logA'].item() if hasattr(f_p['logA'], 'item') else float(f_p['logA'])
        final_A = np.exp(logA_val)
        final_lambda0 = f_p['lambda0'].item() if hasattr(f_p['lambda0'], 'item') else float(f_p['lambda0'])
        final_sigma_0 = theta_final['sigma_0'].item() if hasattr(theta_final['sigma_0'], 'item') else float(theta_final['sigma_0'])
        final_Amp = theta_final['Amp'].item() if hasattr(theta_final['Amp'], 'item') else float(theta_final['Amp'])
        final_eps_0x = theta_final['eps_0x'].item() if hasattr(theta_final['eps_0x'], 'item') else float(theta_final['eps_0x'])
        final_eps_0y = theta_final['eps_0y'].item() if hasattr(theta_final['eps_0y'], 'item') else float(theta_final['eps_0y'])
        time_estep = None
        time_mstep = None

    # =========================================================================
    # VARGP_DIRECT MODE: Eigenspace projection, LBFGS M-step
    # =========================================================================
    elif mode == 'vargp_direct':
        kernel = create_kernel(config, n_px_side, eps_0x, eps_0y)
        from utils import apply_rf_center_bounds
        apply_rf_center_bounds(kernel, eps_0x, eps_0y, config)
        kernel = kernel.to(dtype=dtype, device=device)

        likelihood = PoissonLikelihood(A_init=A_init, lambda0_init=lambda0_init)
        likelihood = likelihood.to(dtype=dtype, device=device)

        model = DirectVGPModel(kernel, likelihood, X_train, inducing_points, eigval_tol,
                               lambda_var_clamp=lambda_var_clamp)

        print(f"\nInitial parameters:")
        print(f"  A_init: {A_init}, lambda0_init: {lambda0_init}")
        print(f"  A: {model.likelihood.A.item():.4f}")
        print(f"  lambda0: {model.likelihood.lambda0.item():.4f}")
        print(f"  Amp: {model.kernel.Amp.item():.6f}")

        mstep_mode = 'analytical' if config['mstep_analytical'] else 'autograd'
        print(f"\nTraining with mode='vargp_direct' (eigenspace projection):")
        print(f"  n_iterations={n_iterations}, n_estep={n_estep}, n_fstep={n_fstep}, n_mstep={n_mstep}")
        print(f"  lr_f={lr}, lr_m={lr}, mstep_mode={mstep_mode}")

        print_every = max(1, n_iterations // 5)
        start_time = time.time()

        with torch.enable_grad():
            result = train_eigenspace(
                model, r_train,
                n_iterations=n_iterations,
                n_estep=n_estep,
                n_fstep=n_fstep,
                n_mstep=n_mstep,
                lr_f=lr,
                lr_m=lr,
                print_every=print_every,
                use_analytical_mstep=config['mstep_analytical'],
                early_stop=early_stop,
                stop_window=stop_window,
                stop_thresh=stop_thresh,
                min_iterations=min_iterations,
                stability_threshold=stability_threshold,
                fix_Amp=config.get('fix_Amp', False),
            )

        train_time = time.time() - start_time
        losses = result['losses']
        time_estep_total = result['time_estep_total']
        time_mstep_total = result['time_mstep_total']
        stopped_early = result.get('stopped_early', False)
        final_iteration = result.get('final_iteration', len(losses))

        print(f"\nTraining time: {train_time:.1f}s")
        print(f"  E-step (+ F-step): {time_estep_total:.1f}s")
        print(f"  M-step:            {time_mstep_total:.1f}s")
        print(f"  Eigenspace dim:    {len(model.state.eigvals_b)}")
        if stopped_early:
            print(f"  Stopped early at iteration {final_iteration}")

        print(f"\nFinal parameters:")
        print(f"  A: {model.likelihood.A.item():.4f}")
        print(f"  lambda0: {model.likelihood.lambda0.item():.4f}")

        print("\nEvaluating on test data...")
        predictions = predict_eigenspace(model, X_test)
        f_pred = predictions['f_pred']

        r_test_mean = r_test.mean(dim=0)
        test_corr = compute_pearson_correlation(r_test_mean, f_pred)
        explained_var, reliability = compute_explained_variance(r_test, f_pred)
        adjusted_r2 = compute_adjusted_r_squared(r_test, f_pred)

        train_preds = predict_eigenspace(model, X_train)
        train_corr = compute_pearson_correlation(r_train, train_preds['f_pred'])

        pred_mean = f_pred.mean().item()
        pred_std = f_pred.std().item()
        pred_min = f_pred.min().item()
        pred_max = f_pred.max().item()

        # Final hyperparameters
        final_A = likelihood.A.item()
        final_lambda0 = likelihood.lambda0.item()
        final_sigma_0 = kernel.sigma_0.item()
        final_Amp = kernel.Amp.item()
        final_beta = kernel.beta.item()
        final_rho = kernel.rho.item()
        final_eps_0x = kernel.eps_0x.item()
        final_eps_0y = kernel.eps_0y.item()
        time_estep = time_estep_total
        time_mstep = time_mstep_total

    # =========================================================================
    # GPYTORCH MODE: default_gpy
    # =========================================================================
    elif mode == 'default_gpy':
        kernel = create_kernel(config, n_px_side, eps_0x, eps_0y)
        from utils import apply_rf_center_bounds
        apply_rf_center_bounds(kernel, eps_0x, eps_0y, config)

        model = VariationalGPModel(
            inducing_points, kernel, jitter=jitter,
            standard_variational_distribution=not config['unwhitened_variational_dist']
        )
        likelihood = PoissonLikelihood(A_init=A_init, lambda0_init=lambda0_init)

        model = model.to(dtype=dtype, device=device)
        likelihood = likelihood.to(dtype=dtype, device=device)

        print(f"\nInitial parameters:")
        print(f"  A_init: {A_init}, lambda0_init: {lambda0_init}")
        print(f"  A: {likelihood.A.item():.4f}")
        print(f"  lambda0: {likelihood.lambda0.item():.4f}")
        print(f"  Amp: {kernel.Amp.item():.6f}")
        print(f"  jitter: {jitter}")

        print(f"\nTraining with mode='default_gpy':")
        print_every = max(1, n_iterations // 5)
        start_time = time.time()

        with torch.enable_grad():
            print(f"  optimizer='{config['optimizer']}', n_iterations={n_iterations}, lr={lr}")
            result = train_gpy_default(
                model, likelihood, X_train, r_train,
                optimizer_name=config['optimizer'],
                lr=lr,
                n_iterations=n_iterations,
                print_every=print_every,
                device=device,
                early_stop=early_stop,
                stop_window=stop_window,
                stop_thresh=stop_thresh,
                min_iterations=min_iterations,
                lbfgs_max_iter=config['gpy_lbfgs_max_iter'],
                jitter=jitter,
                cholesky_max_tries=config['cholesky_max_tries'],
                stability_threshold=stability_threshold,
                lambda_var_clamp=lambda_var_clamp,
            )
            losses = result['losses']
            stopped_early = result.get('stopped_early', False)
            final_iteration = result.get('final_iteration', len(losses))

        train_time = time.time() - start_time
        print(f"\nTraining time: {train_time:.1f}s")
        if stopped_early:
            print(f"  Stopped early at iteration {final_iteration}")

        print(f"\nFinal parameters:")
        print(f"  A: {likelihood.A.item():.4f}")
        print(f"  lambda0: {likelihood.lambda0.item():.4f}")

        print("\nEvaluating on test data...")
        try:
            predictions = predict(model, likelihood, X_test, device=device,
                                  jitter=jitter, cholesky_max_tries=config['cholesky_max_tries'],
                                  lambda_var_clamp=lambda_var_clamp)
            f_pred = predictions['f_pred']

            r_test_mean = r_test.mean(dim=0)
            test_corr = compute_pearson_correlation(r_test_mean, f_pred)
            explained_var, reliability = compute_explained_variance(r_test, f_pred)
            adjusted_r2 = compute_adjusted_r_squared(r_test, f_pred)

            train_preds = predict(model, likelihood, X_train, device=device,
                                  jitter=jitter, cholesky_max_tries=config['cholesky_max_tries'])
            train_corr = compute_pearson_correlation(r_train, train_preds['f_pred'])
        except RuntimeError as e:
            print(f"  Prediction failed: {e}")
            print("  Recording NaN metrics.")
            f_pred = torch.full((X_test.shape[0],), float('nan'), device=device)
            predictions = {'f_pred': f_pred}
            r_test_mean = r_test.mean(dim=0)
            test_corr = float('nan')
            explained_var = float('nan')
            adjusted_r2 = float('nan')
            reliability = compute_explained_variance(r_test, f_pred)[1]
            train_corr = float('nan')

        pred_mean = f_pred.mean().item()
        pred_std = f_pred.std().item()
        pred_min = f_pred.min().item()
        pred_max = f_pred.max().item()

        # Final hyperparameters
        final_A = likelihood.A.item()
        final_lambda0 = likelihood.lambda0.item()
        final_sigma_0 = kernel.sigma_0.item()
        final_Amp = kernel.Amp.item()
        final_beta = kernel.beta.item()
        final_rho = kernel.rho.item()
        final_eps_0x = kernel.eps_0x.item()
        final_eps_0y = kernel.eps_0y.item()
        time_estep = None
        time_mstep = None

    else:
        raise ValueError(f"Unknown mode: {mode}")

    # =========================================================================
    # Results summary
    # =========================================================================
    print(f"\n" + "="*50)
    print(f"RESULTS: mode={mode}, M={M}")
    print(f"="*50)
    print(f"  Training time:   {train_time:.1f}s")
    print(f"  Train Pearson r: {train_corr:.4f}")
    print(f"  Test Pearson r:  {test_corr:.4f}")
    print(f"  Reliability:     {reliability:.4f}")
    print(f"  Explained var:   {explained_var:.4f}")
    print(f"  Final loss:      {losses[-1]:.2f}" if losses else "  Final loss:      N/A (no iterations completed)")
    print(f"  Prediction stats: mean={pred_mean:.3f}, std={pred_std:.3f}, range=[{pred_min:.3f}, {pred_max:.3f}]")

    # Print kernel call stats if analytical gradients were used
    gradient_mode = config['gradient_mode']
    if gradient_mode != 'autograd':
        if gradient_mode == 'vjp':
            from analytical_gradients_vjp import ArcCosineVJPGradients as GradImpl
        else:
            from analytical_gradients import ArcCosineJacobianGradients as GradImpl
        if hasattr(GradImpl, '_call_count'):
            cc = GradImpl._call_count
            total = cc['grad'] + cc['no_grad']
            print(f"  Kernel calls:     {total} total ({cc['grad']} with grads, {cc['no_grad']} without)")
            GradImpl._call_count = {'grad': 0, 'no_grad': 0}

    if pred_std < 0.1:
        print(f"  WARNING: Predictions appear collapsed (std={pred_std:.4f})")

    # Build result dict
    result = {
        'mode': mode,
        'M': M,
        'n_train': n_train,
        'seed': seed,
        'cell': cell,
        'status': 'success',
        'train_time': train_time,
        'train_r': float(train_corr) if not np.isnan(train_corr) else None,
        'test_r': float(test_corr) if not np.isnan(test_corr) else None,
        'explained_var': float(explained_var) if not np.isnan(explained_var) else None,
        'adjusted_r2': float(adjusted_r2) if not np.isnan(adjusted_r2) else None,
        'reliability': float(reliability) if not np.isnan(reliability) else None,
        'final_loss': float(losses[-1]) if losses and not np.isnan(losses[-1]) else None,
        'pred_std': pred_std,
        'time_estep_s': time_estep,
        'time_mstep_s': time_mstep,
        'final_A': final_A,
        'final_lambda0': final_lambda0,
        'final_Amp': final_Amp,
        'final_beta': final_beta,
        'final_rho': final_rho,
        'final_eps_0x': final_eps_0x,
        'final_eps_0y': final_eps_0y,
        'final_sigma_0': final_sigma_0,
        'gradient_mode': gradient_mode if mode != 'vargp_old' else None,
        'n_px_side': n_px_side,
        'n_iterations_run': final_iteration,
        'stopped_early': stopped_early,
        'timestamp': datetime.now().isoformat(timespec='seconds'),
        # Keep references for plotting (not serialized to JSON)
        '_predictions': predictions,
        '_r_test_mean': r_test_mean,
        '_model': model if mode != 'vargp_old' else None,
        '_likelihood': likelihood if mode != 'vargp_old' else None,
        '_indices_train': indices_train,
        '_STA_init_2d': STA_init_2d,
        '_STA_train_2d': _compute_sta_2d(X[indices_train], r[indices_train], n_px_side),
        '_init_eps_0x': eps_0x,
        '_init_eps_0y': eps_0y,
        '_init_beta': config['beta'],
    }

    # Validate that scalar params in dict match model/likelihood objects
    if mode in ('default_gpy'): # Known bug. vargp_direct should be tested too but validate funcitn needs updating
        _validate_model_params(model, likelihood, result)

    return result


# =========================================================================
# CLI entry point
# =========================================================================

def main():
    # --from-config: subprocess mode for run_experiment.py
    # Reads a flat config JSON, runs training, prints result as JSON on stdout.
    # This allows each experiment combo to run in a separate process for GPU
    # memory isolation.
    if len(sys.argv) == 3 and sys.argv[1] == '--from-config':
        config_path = sys.argv[2]
        with open(config_path, 'r') as f:
            config = json.load(f)
        result = run_single_config(config)
        if result is None:
            sys.exit(1)
        # Build serializable result (drop non-serializable GPU objects)
        # Replace NaN/inf with None for valid JSON serialization
        import math
        def _sanitize(v):
            if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
                return None
            return v
        serializable = {k: _sanitize(v) for k, v in result.items()
                        if not k.startswith('_')}
        print(f"RESULT_JSON:{json.dumps(serializable)}")
        sys.exit(0)

    # Load default parameters
    defaults_path = Path(__file__).parent / 'default_params.json'
    with open(defaults_path, 'r') as f:
        defaults = json.load(f)

    parser = argparse.ArgumentParser(description='Test E-step on PNAS data')
    parser.add_argument('--data-path', type=str, default=defaults['data']['path'],
                        help=f'Path to NPZ dataset, relative to gpytorch_porting/ or absolute (default: {defaults["data"]["path"]})')
    parser.add_argument('--cell', type=int, default=defaults['data']['cellid'], help=f'Cell ID (default: {defaults["data"]["cellid"]})')
    parser.add_argument('--ntilde', type=int, default=defaults['data']['ntilde'], help=f'Number of inducing points M (default: {defaults["data"]["ntilde"]})')
    parser.add_argument('--n-train', type=int, default=defaults['data']['n_train'], help=f'Number of training samples (default: {defaults["data"]["n_train"]})')
    parser.add_argument('--n-iterations', type=int, default=defaults['training']['n_iterations'], help=f'Number of EM iterations (default: {defaults["training"]["n_iterations"]})')
    parser.add_argument('--n-estep', type=int, default=defaults['training']['n_estep'], help=f'E-steps per iteration (default: {defaults["training"]["n_estep"]})')
    parser.add_argument('--n-fstep', type=int, default=defaults['training']['n_fstep'], help=f'F-steps per iteration (default: {defaults["training"]["n_fstep"]})')
    parser.add_argument('--n-mstep', type=int, default=defaults['training']['n_mstep'], help=f'M-steps per iteration (default: {defaults["training"]["n_mstep"]})')
    parser.add_argument('--lr', type=float, default=defaults['training']['lr'], help=f'Learning rate (default: {defaults["training"]["lr"]})')
    parser.add_argument('--optimizer', type=str, default=defaults['training']['optimizer'],
                        choices=['adam', 'lbfgs'],
                        help=f'Optimizer for default_gpy mode (default: {defaults["training"]["optimizer"]})')
    parser.add_argument('--device', type=str, default=defaults['run']['device'], help=f'Device (default: {defaults["run"]["device"]})')
    parser.add_argument('--mode', type=str, default=defaults['run']['mode'],
                        choices=['vargp_old', 'default_gpy', 'vargp_direct'],
                        help=f'Training mode (default: {defaults["run"]["mode"]})')

    # Kernel parameters
    parser.add_argument('--kernel-type', type=str, default=defaults['kernel']['type'],
                        choices=list(KERNEL_TYPES),
                        help=f'Kernel type (default: {defaults["kernel"]["type"]})')
    parser.add_argument('--sigma-0', type=float, default=defaults['kernel']['sigma_0'], help=f'Kernel bias variance (default: {defaults["kernel"]["sigma_0"]})')
    parser.add_argument('--Amp', type=float, default=defaults['kernel']['Amp'], help=f'Kernel amplitude (default: {defaults["kernel"]["Amp"]})')
    parser.add_argument('--lengthscale', type=float, default=defaults['kernel']['lengthscale'],
                        help=f'RBF lengthscale, only used with --kernel-type rbf (default: {defaults["kernel"]["lengthscale"]})')
    parser.add_argument('--beta', type=float, default=defaults['kernel']['beta'], help=f'RF size (default: {defaults["kernel"]["beta"]})')
    parser.add_argument('--rho', type=float, default=defaults['kernel']['rho'], help=f'Smoothness (default: {defaults["kernel"]["rho"]})')
    parser.add_argument('--eps-0x', type=float, default=None, help='RF center x (default: compute from STA)')
    parser.add_argument('--eps-0y', type=float, default=None, help='RF center y (default: compute from STA)')

    # Link function parameters
    parser.add_argument('--A-init', type=float, default=defaults['link_function']['A_init'], help=f'Initial gain A (default: {defaults["link_function"]["A_init"]})')
    parser.add_argument('--lambda0-init', type=float, default=defaults['link_function']['lambda0_init'], help=f'Initial bias lambda0 (default: {defaults["link_function"]["lambda0_init"]})')
    parser.add_argument('--use-mask', action='store_true', default=defaults['kernel']['use_mask'],
                        help=f'Use pixel masking (default: {defaults["kernel"]["use_mask"]})')
    parser.add_argument('--no-mask', action='store_false', dest='use_mask',
                        help='Disable pixel masking (WARNING: uses full 11664x11664 C matrix)')
    parser.add_argument('--gradient-mode', type=str, default=defaults['kernel']['gradient_mode'],
                        choices=list(GRADIENT_MODES),
                        help=f'Gradient computation mode (default: {defaults["kernel"]["gradient_mode"]})')
    parser.add_argument('--mstep-analytical', action='store_true',
                        help='Use analytical gradients for M-step in vargp_direct mode (faster, matches varGP)')

    # RF center bounds
    ker_defaults = defaults['kernel']
    parser.add_argument('--bound-rf-center', action='store_true',
                        default=ker_defaults['bound_rf_center'], dest='bound_rf_center',
                        help=f'Constrain RF center to STA neighborhood (default: {ker_defaults["bound_rf_center"]})')
    parser.add_argument('--no-bound-rf-center', action='store_false', dest='bound_rf_center',
                        help='Disable RF center bounds (allow full [-1, 1] range)')
    parser.add_argument('--n-sigma-rf-bounds', type=float, default=ker_defaults['n_sigma_rf_bounds'],
                        help=f'Allowed RF center deviation in sigma_rf units (default: {ker_defaults["n_sigma_rf_bounds"]})')

    # Performance options
    parser.add_argument('--use-cache', action='store_true', default=defaults['model']['use_cache'],
                        help=f'Use kernel caching in E-step (default: {defaults["model"]["use_cache"]}, 11.7x fewer kernel calls)')
    parser.add_argument('--no-cache', action='store_false', dest='use_cache',
                        help='Disable kernel caching (for testing fallback path)')
    parser.add_argument('--jitter', type=float, default=defaults['model']['jitter'],
                        help=f'Jitter for numerical stability (default: {defaults["model"]["jitter"]})')
    parser.add_argument('--cholesky-max-tries', type=int, default=defaults['model']['cholesky_max_tries'],
                        help=f'Max Cholesky retry attempts (default: {defaults["model"]["cholesky_max_tries"]})')
    parser.add_argument('--lambda-var-clamp', type=float, default=defaults['model']['lambda_var_clamp'],
                        help=f'Min posterior variance clamp (default: {defaults["model"]["lambda_var_clamp"]})')
    parser.add_argument('--stability-threshold', type=float, default=defaults['model']['stability_threshold'],
                        help=f'Max mean firing rate before step rejection (default: {defaults["model"]["stability_threshold"]})')

    parser.add_argument('--unwhitened-variational-dist', action='store_true',
                        help='Use UnwhitenedVariationalStrategy (stores natural params directly, no L_K dependency)')

    # Plotting options
    parser.add_argument('--plot', action='store_true',
                        help='Show plot of actual vs predicted firing rates')
    parser.add_argument('--save-plot', type=str, default='auto',
                        help='Save plot path. "auto" saves to imgs/{mode}_M{ntilde}.png, "none" to disable')
    parser.add_argument('--seed', type=int, default=defaults['data']['seed'],
                        help=f'Random seed for reproducibility (default: {defaults["data"]["seed"]})')
    parser.add_argument('--dtype', type=str, default=defaults['run']['dtype'],
                        choices=['float32', 'float64'],
                        help=f'Data type (default: {defaults["run"]["dtype"]})')

    # JSON output for benchmark tracking
    parser.add_argument('--json-append', type=str, default=None,
                        help='Append results as JSON line to specified file (for benchmark tracking)')

    # Early stopping options (defaults from default_params.json)
    es_defaults = defaults['early_stopping']
    parser.add_argument('--no-early-stop', action='store_true',
                        help='Disable early stopping (early stopping is ON by default)')
    parser.add_argument('--stop-window', type=int, default=es_defaults['window'],
                        help=f'Number of iterations to look back for improvement (default: {es_defaults["window"]})')
    parser.add_argument('--stop-thresh', type=float, default=es_defaults['threshold'],
                        help=f'Minimum relative improvement over window to continue (default: {es_defaults["threshold"]})')
    parser.add_argument('--min-iterations', type=int, default=es_defaults['min_iterations'],
                        help=f'Minimum iterations before early stopping can trigger (default: {es_defaults["min_iterations"]})')

    # Inducing point selection
    ind_defaults = defaults['inducing']
    parser.add_argument('--ip-selection', type=str, default=ind_defaults['selection_method'],
                        choices=['pivoted', 'random'],
                        help=f'Inducing point selection method (default: {ind_defaults["selection_method"]})')
    parser.add_argument('--n-candidates', type=int, default=ind_defaults['n_candidates'],
                        help='Number of candidate points for pivoted selection (default: null = all available)')
    parser.add_argument('--n-samples-sta', type=int, default=ind_defaults['n_samples_sta'],
                        help='Number of random images for initial STA estimate (default: null = all available)')

    args = parser.parse_args()

    # eps_0x/eps_0y: None (not provided) → compute from STA.
    # Any float value (including 0.0) → use that value.
    eps_0x = args.eps_0x  # None if not passed, float if passed
    eps_0y = args.eps_0y

    # Build config from default_params.json, then overlay CLI args.
    # All argparse defaults already come from the same JSON, so only
    # user-provided CLI flags actually change anything.
    config = build_config_from_defaults(
        data_path=args.data_path,
        mode=args.mode,
        kernel_type=args.kernel_type,
        M=args.ntilde,
        n_train=args.n_train,
        seed=args.seed,
        cell=args.cell,
        n_iterations=args.n_iterations,
        device=args.device,
        dtype=args.dtype,
        sigma_0=args.sigma_0,
        Amp=args.Amp,
        beta=args.beta,
        rho=args.rho,
        lengthscale=args.lengthscale,
        eps_0x=eps_0x,
        eps_0y=eps_0y,
        gradient_mode=args.gradient_mode,
        use_mask=args.use_mask,
        bound_rf_center=args.bound_rf_center,
        n_sigma_rf_bounds=args.n_sigma_rf_bounds,
        A_init=args.A_init,
        lambda0_init=args.lambda0_init,
        n_estep=args.n_estep,
        n_fstep=args.n_fstep,
        n_mstep=args.n_mstep,
        lr=args.lr,
        optimizer=args.optimizer,
        early_stop=not args.no_early_stop,
        stop_window=args.stop_window,
        stop_thresh=args.stop_thresh,
        min_iterations=args.min_iterations,
        jitter=args.jitter,
        cholesky_max_tries=args.cholesky_max_tries,
        lambda_var_clamp=args.lambda_var_clamp,
        stability_threshold=args.stability_threshold,
        use_cache=args.use_cache,
        mstep_analytical=args.mstep_analytical,
        unwhitened_variational_dist=args.unwhitened_variational_dist,
        save_plot=args.save_plot,
        plot=args.plot,
        ip_selection=args.ip_selection,
        n_candidates=args.n_candidates,
        n_samples_sta=args.n_samples_sta,
    )

    # Run
    result = run_single_config(config)
    if result is None:
        return None

    # Plot handling
    save_path = None
    if args.save_plot == 'auto':
        imgs_dir = Path(__file__).parent / 'imgs'
        imgs_dir.mkdir(exist_ok=True)
        save_path = imgs_dir / f"{args.mode}_M{args.ntilde}.png"
    elif args.save_plot and args.save_plot.lower() != 'none':
        save_path = Path(args.save_plot)
        save_path.parent.mkdir(parents=True, exist_ok=True)

    if args.plot or save_path:
        model_obj = result.get('_model')
        # DirectVGPModel uses .kernel, VariationalGPModel uses .covar_module
        kernel_obj = None
        if model_obj is not None:
            kernel_obj = getattr(model_obj, 'covar_module', None) or getattr(model_obj, 'kernel', None)
        plot_fit(result['_r_test_mean'], result['_predictions']['f_pred'],
                 args.cell, args.ntilde,
                 result['test_r'], result['explained_var'], result['reliability'],
                 STA_init_2d=result.get('_STA_init_2d'),
                 STA_train_2d=result.get('_STA_train_2d'),
                 init_eps_0x=result.get('_init_eps_0x'),
                 init_eps_0y=result.get('_init_eps_0y'),
                 init_beta=result.get('_init_beta'),
                 kernel=kernel_obj,
                 n_px_side=result['n_px_side'],
                 output_path=save_path)

    # Write JSON if requested
    if args.json_append:
        json_record = {
            'timestamp': result['timestamp'],
            'commit': get_git_commit(),
            'seed': args.seed,
            'mode': args.mode,
            'M': args.ntilde,
            'ntrain': result['n_train'],
            'niter': args.n_iterations,
            'nestep': args.n_estep,
            'nmstep': args.n_mstep,
            'nfstep': args.n_fstep,
            'cell_id': args.cell,
            'gradient_mode': result['gradient_mode'],
            'test_r': round(result['test_r'], 4) if result['test_r'] is not None else None,
            'explained_var': round(result['explained_var'], 4) if result['explained_var'] is not None else None,
            'final_loss': round(result['final_loss'], 4) if result['final_loss'] is not None else None,
            'time_total_s': round(result['train_time'], 2),
            'time_estep_s': round(result['time_estep_s'], 2) if result['time_estep_s'] is not None else None,
            'time_mstep_s': round(result['time_mstep_s'], 2) if result['time_mstep_s'] is not None else None,
            'final_A': round(result['final_A'], 6),
            'final_lambda0': round(result['final_lambda0'], 6),
            'final_Amp': round(result['final_Amp'], 6),
            'final_beta': round(result['final_beta'], 6),
            'final_rho': round(result['final_rho'], 6),
            'final_eps_0x': round(result['final_eps_0x'], 6),
            'final_eps_0y': round(result['final_eps_0y'], 6),
            'final_sigma_0': round(result['final_sigma_0'], 6),
        }

        json_path = Path(args.json_append)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        with open(json_path, 'a') as f:
            f.write(json.dumps(json_record) + '\n')
        print(f"\nJSON result appended to: {json_path}")

    return result


if __name__ == '__main__':
    main()
