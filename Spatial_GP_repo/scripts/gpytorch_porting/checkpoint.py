"""
Checkpoint save/load for default_gpy models.

Saves model + likelihood state, config, metrics, and metadata into a single .pt file.
Reconstructs the full model from a checkpoint without needing the original training code.
"""

import math
from datetime import datetime
from pathlib import Path

import torch

from kernels import create_kernel
from likelihoods import PoissonLikelihood
from gpy_model import VariationalGPModel


def save_checkpoint(model, likelihood, config, metrics, checkpoint_path):
    """Save a trained default_gpy model to a .pt checkpoint file.

    Args:
        model: Trained VariationalGPModel
        likelihood: Trained PoissonLikelihood
        config: Flat config dict (from build_config_from_defaults or run_single_config)
        metrics: Dict with at least 'test_r'. May also have 'train_r', 'explained_var',
                 'reliability', 'train_time', 'final_loss', etc.
        checkpoint_path: Path to save the .pt file
    """
    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    # Extract final hyperparameters from the trained model/likelihood
    kernel = model.covar_module
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

    checkpoint = {
        'model_state_dict': model.state_dict(),
        'likelihood_state_dict': likelihood.state_dict(),
        'config': config,
        'metrics': metrics,
        'hyperparams': hyperparams,
        'metadata': {
            'cell_id': config['cell'],
            'n_px_side': config['n_px_side'],
            'M': config['M'],
            'kernel_type': config['kernel_type'],
            'saved_at': datetime.now().isoformat(timespec='seconds'),
        },
    }

    torch.save(checkpoint, checkpoint_path)


def load_checkpoint(checkpoint_path, device=None):
    """Load a trained default_gpy model from a .pt checkpoint file.

    Reconstructs the kernel, model, and likelihood from saved state.

    Args:
        checkpoint_path: Path to the .pt checkpoint file
        device: Device to load onto (default: 'cpu'). Use 'cuda' for GPU inference.

    Returns:
        dict with keys:
            'model': Reconstructed VariationalGPModel (eval mode)
            'likelihood': Reconstructed PoissonLikelihood (eval mode)
            'config': The config dict used for training
            'metrics': Training metrics (test_r, etc.)
            'hyperparams': Final kernel + likelihood parameter values
            'metadata': Cell ID, image size, M, etc.
    """
    if device is None:
        device = 'cpu'

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    config = checkpoint['config']
    model_state = checkpoint['model_state_dict']

    # Get n_px_side and RF center from config/hyperparams
    n_px_side = checkpoint['metadata']['n_px_side']
    hyperparams = checkpoint['hyperparams']
    eps_0x = hyperparams['eps_0x']
    eps_0y = hyperparams['eps_0y']

    # Reconstruct kernel from config
    kernel = create_kernel(config, n_px_side, eps_0x, eps_0y)

    # Get inducing points shape from state dict to create model
    inducing_points = model_state['variational_strategy.inducing_points']

    # Determine dtype from saved inducing points
    dtype = inducing_points.dtype

    # Reconstruct model
    model = VariationalGPModel(
        inducing_points,
        kernel,
        jitter=config['jitter'],
        standard_variational_distribution=not config.get('unwhitened_variational_dist', False),
    )

    # Load saved state
    model.load_state_dict(model_state)
    model = model.to(dtype=dtype, device=device)
    model.eval()

    # Reconstruct likelihood
    # Use dummy init values — state_dict will overwrite them
    likelihood = PoissonLikelihood(A_init=0.01, lambda0_init=1.0)
    likelihood.load_state_dict(checkpoint['likelihood_state_dict'])
    likelihood = likelihood.to(dtype=dtype, device=device)
    likelihood.eval()

    return {
        'model': model,
        'likelihood': likelihood,
        'config': config,
        'metrics': checkpoint['metrics'],
        'hyperparams': hyperparams,
        'metadata': checkpoint['metadata'],
    }
