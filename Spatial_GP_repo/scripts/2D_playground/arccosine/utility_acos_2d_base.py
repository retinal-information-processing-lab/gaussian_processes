"""
Arc-Cosine GP Utility Playground 2D
===================================
Arc-Cosine version of utility_2d_rbf_base.py - same analysis, different kernel.

NON-STATIONARITY NOTE:
Arc-Cosine kernel has k(x,x) = ||x||² + σ₀² which varies with input magnitude.
This causes extreme utility values at domain corners. Values are clipped to
99th percentile for visualization.

Related files:
- utility_2d_rbf_base.py (RBF version in parent folder)
- train_acos_2d.py (creates checkpoint)
"""

import sys
import torch
import numpy as np
from pathlib import Path
import warnings

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "1D_playground"))

from utility_2d_rbf_base import (
    plot_results_2d,
    evaluate_distribution_aware_utility_2d,
    lambda_true_2d,
    DEVICE, DTYPE,
    X_MIN, X_MAX, Y_MIN, Y_MAX,
    DEFAULT_P_X_MEAN_2D, DEFAULT_P_X_STD_2D,
)

from gp_utility_playground import VariationalGP, evaluate_nd_utility_new

# Arc-Cosine kernel imports
GPYTORCH_PORTING_PATH = Path(__file__).parent.parent.parent / 'gpytorch_porting'
sys.path.insert(0, str(GPYTORCH_PORTING_PATH))
from kernels import ArcCosineKernel
from likelihoods import PoissonLikelihood

warnings.filterwarnings("ignore", message=".*torch.cuda.*")
warnings.filterwarnings("ignore", message=".*torch.sparse.*")

CHECKPOINT_PATH = Path(__file__).parent / 'trained_arccosine_2d_checkpoint.pt'


def load_checkpoint(filepath):
    """Load Arc-Cosine checkpoint (kernel swap before state_dict load)."""
    checkpoint = torch.load(filepath, map_location=DEVICE, weights_only=False)

    inducing_points = checkpoint['inducing_points']
    model = VariationalGP(inducing_points, jitter=1e-4).to(DEVICE)

    # Swap to Arc-Cosine kernel BEFORE loading state dict
    kp = checkpoint['kernel_params']
    model.covar_module = ArcCosineKernel(sigma_0=kp['sigma_0'], Amp=kp['Amp'], C=None).to(DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    lp = checkpoint['likelihood_params']
    likelihood = PoissonLikelihood(A_init=lp['A'], lambda0_init=lp['lambda_0']).to(DEVICE)
    likelihood.load_state_dict(checkpoint['likelihood_state_dict'])
    likelihood.eval()

    print(f"Loaded: {filepath}")
    print(f"  Kernel: sigma_0={kp['sigma_0']:.4f}, Amp={kp['Amp']:.4f}")
    print(f"  Likelihood: A={lp['A']:.4f}, lambda_0={lp['lambda_0']:.3f}")

    return model, likelihood, checkpoint


def main():
    torch.manual_seed(42)
    np.random.seed(42)

    print("=" * 60)
    print("Arc-Cosine GP Utility Playground 2D")
    print("=" * 60)

    model, likelihood, checkpoint = load_checkpoint(CHECKPOINT_PATH)
    train_x = checkpoint['train_x']
    train_y = checkpoint['train_y']

    # Evaluation grid
    n_eval = 40
    x_eval = torch.linspace(X_MIN, X_MAX, n_eval, dtype=DTYPE, device=DEVICE)
    y_eval = torch.linspace(Y_MIN, Y_MAX, n_eval, dtype=DTYPE, device=DEVICE)
    eval_grid_x, eval_grid_y = torch.meshgrid(x_eval, y_eval, indexing='xy')
    eval_points = torch.stack([eval_grid_x.flatten(), eval_grid_y.flatten()], dim=-1)

    print(f"\nEvaluation grid: {n_eval}x{n_eval}")

    # Compute utilities
    print("\nComputing standard utility...")
    utility_std = evaluate_nd_utility_new(model, eval_points)

    print("Computing distribution-aware utility...")
    utility_da = evaluate_distribution_aware_utility_2d(
        model, eval_points, n_mc_samples=500,
        p_x_mean=DEFAULT_P_X_MEAN_2D, p_x_std=DEFAULT_P_X_STD_2D
    )

    # Report non-finite values
    n_inf = np.sum(~np.isfinite(utility_std.cpu().numpy()))
    if n_inf > 0:
        print(f"\nNote: {n_inf} non-finite values in standard utility (clipped for visualization)")

    # Max locations
    utility_std_clean = torch.where(torch.isfinite(utility_std), utility_std, torch.tensor(float('-inf')))
    max_std_loc = eval_points[torch.argmax(utility_std_clean)].cpu().numpy()
    max_da_loc = eval_points[torch.argmax(utility_da)].cpu().numpy()
    p_x_center = DEFAULT_P_X_MEAN_2D.cpu().numpy()

    print(f"\nMax locations:")
    print(f"  Standard: ({max_std_loc[0]:.1f}, {max_std_loc[1]:.1f})")
    print(f"  Distr-aware: ({max_da_loc[0]:.1f}, {max_da_loc[1]:.1f})")
    print(f"  p(x) center: ({p_x_center[0]:.1f}, {p_x_center[1]:.1f})")

    # Plot
    print("\nGenerating plot...")
    save_path = Path(__file__).parent / 'utility_acos_2d_base.png'
    plot_results_2d(
        model, train_x, train_y, lambda_true_2d,
        eval_grid_x, eval_grid_y,
        utility_std, utility_da,
        p_x_mean=DEFAULT_P_X_MEAN_2D, p_x_std=DEFAULT_P_X_STD_2D,
        save_path=save_path
    )

    print("\nDone!")


if __name__ == "__main__":
    main()
