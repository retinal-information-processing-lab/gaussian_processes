"""
Normalized Arc-Cosine GP Utility Playground 2D
===============================================
Normalized arc-cosine version of utility_acos_2d_base.py.

The normalized kernel has K_bar(x,x) = 1 for all x, so GP posterior variance
is bounded and does not grow with ||x||. This should eliminate the divergence
to domain corners seen with the unnormalized arc-cosine kernel.

Related files:
- train_acos_normalized_2d.py (creates checkpoint)
- ../arccosine/utility_acos_2d_base.py (unnormalized version for comparison)
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
    lambda_true_2d,
    DEVICE, DTYPE,
    X_MIN, X_MAX, Y_MIN, Y_MAX,
    DEFAULT_P_X_MEAN_2D, DEFAULT_P_X_STD_2D,
    ADAPTIVE_SAFETY_K, ADAPTIVE_MAX_RMAX, ADAPTIVE_MIN_RMAX,
    # From gpytorch_porting (re-exported by utility_2d_rbf_base)
    SimpleArcCosineNormalizedKernel,
    PoissonLikelihood,
    standard_utility,
    distribution_aware_utility,
)

from gp_utility_playground import VariationalGP

warnings.filterwarnings("ignore", message=".*torch.cuda.*")
warnings.filterwarnings("ignore", message=".*torch.sparse.*")

CHECKPOINT_PATH = Path(__file__).parent / 'trained_arccosine_normalized_2d_checkpoint.pt'


def load_normalized_arccosine_2d_checkpoint(filepath):
    """Load normalized arc-cosine checkpoint (kernel swap before state_dict load)."""
    checkpoint = torch.load(filepath, map_location=DEVICE, weights_only=False)

    inducing_points = checkpoint['inducing_points']
    model = VariationalGP(inducing_points, jitter=1e-4).to(DEVICE)

    # Swap to normalized arc-cosine kernel
    kp = checkpoint['kernel_params']
    model.covar_module = SimpleArcCosineNormalizedKernel(sigma_0=kp['sigma_0']).to(DEVICE)

    # Load state dict with strict=False (kernel params may differ from default RBF)
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    model.eval()

    # Reconstruct likelihood from gpytorch_porting
    lp = checkpoint['likelihood_params']
    likelihood = PoissonLikelihood(A_init=lp['A'], lambda0_init=lp['lambda_0']).to(DEVICE)
    likelihood.eval()

    print(f"Loaded: {filepath}")
    print(f"  Kernel: SimpleArcCosineNormalizedKernel (K_bar(x,x) = 1)")
    print(f"  sigma_0={kp['sigma_0']:.4f}")
    print(f"  Likelihood: A={likelihood.A.item():.4f}, lambda_0={likelihood.lambda0.item():.3f}")

    return model, likelihood, checkpoint


def main():
    torch.manual_seed(42)
    np.random.seed(42)

    print("=" * 60)
    print("Normalized Arc-Cosine GP Utility Playground 2D")
    print("=" * 60)

    model, likelihood, checkpoint = load_normalized_arccosine_2d_checkpoint(CHECKPOINT_PATH)
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
    print("\nComputing standard utility (adaptive r_max)...")
    with torch.no_grad():
        result_std = standard_utility(
            model, likelihood, eval_points, r_max=None, adaptive_r_max=True,
            adaptive_safety_k=ADAPTIVE_SAFETY_K, adaptive_max_rmax=ADAPTIVE_MAX_RMAX,
            adaptive_min_rmax=ADAPTIVE_MIN_RMAX
        )
    utility_std = result_std['utility']

    print("Computing distribution-aware utility...")
    # Draw MC samples from p(x)
    n_mc = 500
    x_samples = DEFAULT_P_X_MEAN_2D + DEFAULT_P_X_STD_2D * torch.randn(
        n_mc, 2, dtype=DTYPE, device=DEVICE
    )
    with torch.no_grad():
        # sample_lambda=False: use posterior mean instead of sampling (reduces MC noise)
        result_da = distribution_aware_utility(
            model, likelihood, eval_points, x_samples,
            r_max=None, adaptive_r_max=True,
            sample_lambda=False,
            adaptive_safety_k=ADAPTIVE_SAFETY_K, adaptive_max_rmax=ADAPTIVE_MAX_RMAX,
            adaptive_min_rmax=ADAPTIVE_MIN_RMAX
        )
    utility_da = result_da['utility']

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
    save_path = Path(__file__).parent / 'utility_acos_normalized_2d_base.png'
    plot_results_2d(
        model, train_x, train_y, lambda_true_2d,
        eval_grid_x, eval_grid_y,
        utility_std, utility_da,
        p_x_mean=DEFAULT_P_X_MEAN_2D, p_x_std=DEFAULT_P_X_STD_2D,
        save_path=save_path,
        clip_utility=False
    )

    print("\nDone!")


if __name__ == "__main__":
    main()
