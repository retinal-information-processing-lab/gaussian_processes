"""
Checkpoint save/load for vargp_direct (DirectVGPModel).

Saves model state (kernel params, likelihood params, variational params m_b/V_b)
plus pool indices + integrity metadata into a single .pt file.

Design notes:
- X_train is NOT saved. Reconstructed from pool_indices on load. All runs share
  the same PNAS pool, so this keeps checkpoints small (~1 KB vs ~2 MB).
- Derived eigenspace state (B, eigvals_b, K_tilde_b, K_b, KKtilde_inv_b, Kvec)
  is NOT saved. The DirectVGPModel constructor recomputes it from
  (kernel, X_train, eigval_tol). Only m_b and V_b need re-injection.
- Integrity safety net: pool_shape and pool_sum are saved and verified on
  load to catch "wrong dataset" bugs without crypto hashing.

Mirrors checkpoint.py (which handles default_gpy / VariationalGPModel) but is
a separate file because the two model classes have incompatible state
representations: DirectVGPModel is not an nn.Module and has no
variational_strategy.
"""

from datetime import datetime
from pathlib import Path
import warnings

import torch

from kernels import create_kernel
from likelihoods import PoissonLikelihood
from eigenspace_model import DirectVGPModel


def save_eigenspace_checkpoint(
    model,
    config,
    metrics,
    pool_indices,
    pool_shape,
    pool_sum,
    checkpoint_path,
):
    """Save DirectVGPModel state to a .pt file.

    Args:
        model: Trained DirectVGPModel.
        config: Flat config dict used for training (needed for reconstruction).
        metrics: Dict of final metrics to embed in the checkpoint.
        pool_indices: LongTensor (n_training,), indices into the PNAS pool.
        pool_shape: list of ints, e.g. [3160, 11664]. Integrity tag.
        pool_sum: float, X_pool.sum().item(). Integrity tag.
        checkpoint_path: Path to the output .pt file. Parent dirs are created.
    """
    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    kernel = model.kernel
    likelihood = model.likelihood

    # Extract final hyperparameters for quick inspection without loading the
    # full checkpoint. Mirrors checkpoint.py convention.
    hyperparams = {
        'A': likelihood.A.item(),
        'lambda0': likelihood.lambda0.item(),
        'sigma_0': kernel.sigma_0.item(),
        'Amp': kernel.Amp.item(),
        'beta': kernel.beta.item(),
        'rho': kernel.rho.item(),
        'eps_0x': kernel.eps_0x.item(),
        'eps_0y': kernel.eps_0y.item(),
    }

    # RF center bounds (if any) are plain instance attributes on the kernel,
    # NOT in state_dict. They are set once at training time by
    # apply_rf_center_bounds(kernel, initial_eps_0x, initial_eps_0y, config)
    # using the initial STA estimate as the anchor, and they stay fixed for
    # the entire run. We must save them explicitly here so a reloaded model
    # can be retrained with the same trainable region — otherwise the original
    # STA anchor is lost (only the drifted trained eps_0x survives in state_dict).
    if hasattr(kernel, '_eps_x_min'):
        rf_center_bounds = {
            'eps_x_min': kernel._eps_x_min,
            'eps_x_max': kernel._eps_x_max,
            'eps_y_min': kernel._eps_y_min,
            'eps_y_max': kernel._eps_y_max,
        }
    else:
        rf_center_bounds = None

    checkpoint = {
        'kernel_state_dict': kernel.state_dict(),
        'likelihood_state_dict': likelihood.state_dict(),
        'm_b': model.state.m_b.cpu(),
        'V_b': model.state.V_b.cpu(),
        'pool_indices': pool_indices.cpu(),
        'pool_shape': list(pool_shape),
        'pool_sum': float(pool_sum),
        'rf_center_bounds': rf_center_bounds,
        'config': config,
        'metrics': metrics,
        'hyperparams': hyperparams,
        'metadata': {
            'cell_id': config['cell'],
            'n_px_side': config['n_px_side'],
            'M': config['M'],
            'kernel_type': config['kernel_type'],
            'eigval_tol': config['eigval_tol'],
            'lambda_var_clamp': config['lambda_var_clamp'],
            'data_path': str(config['data_path']),
            'saved_at': datetime.now().isoformat(timespec='seconds'),
        },
    }

    torch.save(checkpoint, checkpoint_path)


def load_eigenspace_checkpoint(checkpoint_path, X_pool, pool_sum_tolerance,
                               device=None):
    """Load a DirectVGPModel checkpoint and reconstruct the model.

    Verifies pool_shape and pool_sum against the provided X_pool before
    reconstructing. Raises AssertionError on mismatch (likely wrong dataset).
    Also prints a warning if the saved data_path differs from the current
    config (non-fatal: files can legitimately move).

    Args:
        checkpoint_path: Path to the .pt file.
        X_pool: (N_pool, n_pixels) tensor. Same pool ordering as at save time
                (usually cat([X_train, X_val]) flattened from the same .npz).
        pool_sum_tolerance: float. Maximum absolute difference allowed between
                the pool_sum saved in the checkpoint and X_pool.sum() at load
                time. Must be passed explicitly — read it from
                default_params.json -> active_learning.checkpoint_pool_sum_tolerance.
                No hidden default here; single source of truth is the config.
        device: Optional device override. Default 'cpu'.

    Returns:
        dict with keys:
            'model': reconstructed DirectVGPModel
            'config': flat config dict from training
            'metrics': training metrics dict
            'hyperparams': final hyperparameter values
            'metadata': cell_id, data_path, saved_at, etc.
            'pool_indices': LongTensor indices into X_pool

    Raises:
        AssertionError: if pool_shape or pool_sum does not match X_pool.
    """
    if device is None:
        device = 'cpu'

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint['config']
    metadata = checkpoint['metadata']

    # ----- Integrity checks (before any reconstruction) -----
    saved_shape = checkpoint['pool_shape']
    saved_sum = checkpoint['pool_sum']
    current_shape = list(X_pool.shape)
    current_sum = X_pool.sum().item()

    if current_shape != saved_shape:
        raise AssertionError(
            f"Pool shape mismatch: checkpoint has {saved_shape}, "
            f"X_pool has {current_shape}. Likely wrong dataset."
        )
    if abs(current_sum - saved_sum) > pool_sum_tolerance:
        raise AssertionError(
            f"Pool sum mismatch: checkpoint has {saved_sum:.6f}, "
            f"X_pool has {current_sum:.6f} "
            f"(tolerance {pool_sum_tolerance}). "
            f"Likely content differs from save time."
        )

    saved_data_path = metadata.get('data_path')
    current_data_path = config.get('data_path')
    if (saved_data_path and current_data_path
            and str(saved_data_path) != str(current_data_path)):
        warnings.warn(
            f"Checkpoint was saved with data_path='{saved_data_path}' but "
            f"current config has data_path='{current_data_path}'. Integrity "
            f"checks passed so probably fine, but double-check the pool source."
        )

    # ----- Reconstruct training data from indices -----
    pool_indices = checkpoint['pool_indices']
    X_train = X_pool[pool_indices].to(device=device)

    # ----- Reconstruct kernel -----
    # eps_0x and eps_0y are nn.Parameters registered directly in the kernel,
    # so they live in the state_dict. Extract them first for create_kernel()
    # construction; load_state_dict will re-set them immediately afterward.
    kernel_state = checkpoint['kernel_state_dict']
    eps_0x = kernel_state['eps_0x'].item()
    eps_0y = kernel_state['eps_0y'].item()
    n_px_side = metadata['n_px_side']

    kernel = create_kernel(config, n_px_side, eps_0x, eps_0y)
    kernel.load_state_dict(kernel_state)

    # Restore RF center bounds directly from the saved values (if any).
    # We do NOT call apply_rf_center_bounds() here because it would anchor
    # the bounds to the TRAINED eps_0x (already drifted from the initial STA),
    # which silently changes the trainable region of a reloaded model. The
    # bounds were fixed at training time from the initial STA estimate and
    # must be restored byte-identical — not re-derived from the current
    # parameter values.
    rf_center_bounds = checkpoint.get('rf_center_bounds')
    if rf_center_bounds is not None:
        kernel._eps_x_min = rf_center_bounds['eps_x_min']
        kernel._eps_x_max = rf_center_bounds['eps_x_max']
        kernel._eps_y_min = rf_center_bounds['eps_y_min']
        kernel._eps_y_max = rf_center_bounds['eps_y_max']

    # Determine dtype from saved variational parameters.
    dtype = checkpoint['m_b'].dtype
    kernel = kernel.to(device=device, dtype=dtype)

    # ----- Reconstruct likelihood -----
    likelihood = PoissonLikelihood(
        A_init=config['A_init'],
        lambda0_init=config['lambda0_init'],
    )
    likelihood.load_state_dict(checkpoint['likelihood_state_dict'])
    likelihood = likelihood.to(device=device, dtype=dtype)

    X_train = X_train.to(dtype=dtype)

    # ----- Construct the model -----
    # DirectVGPModel's __init__ computes the fresh eigenspace (B, eigvals_b,
    # K_tilde_b, K_b, KKtilde_inv_b, Kvec) from (kernel, X_train, eigval_tol).
    # Since the kernel has just been loaded with the saved state, the
    # eigendecomposition is deterministic on the same hardware and should
    # match what was at save time.
    eigval_tol = metadata['eigval_tol']
    lambda_var_clamp = metadata['lambda_var_clamp']
    model = DirectVGPModel(
        kernel=kernel,
        likelihood=likelihood,
        X_train=X_train,
        X_tilde=X_train,  # M == n_train invariant in the active loop
        eigval_tol=eigval_tol,
        lambda_var_clamp=lambda_var_clamp,
    )

    # Inject saved variational parameters, overriding the constructor's init.
    m_b = checkpoint['m_b'].to(device=device, dtype=dtype)
    V_b = checkpoint['V_b'].to(device=device, dtype=dtype)
    model.update_variational_params(m_b, V_b)

    return {
        'model': model,
        'config': config,
        'metrics': checkpoint['metrics'],
        'hyperparams': checkpoint['hyperparams'],
        'metadata': metadata,
        'pool_indices': pool_indices,
    }
