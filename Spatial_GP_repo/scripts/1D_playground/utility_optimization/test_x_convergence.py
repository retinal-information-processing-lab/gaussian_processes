"""
Test: Utility Optimization Converges to x_sample
================================================
Created by Claude

PURPOSE: Verify that gradient ascent on distribution-aware utility converges
to x_sample when X_SET = [x_sample]. This validates the interpretation that
U(x*) measures "how informative is x* for learning about f(x_sample)".

CONTEXT: The distribution-aware utility is defined as:
    U(x*) = I(f(x); R | x*, D) where x ~ p(x)
This is the mutual information between f(x) and the response R at x*.
When p(x) = delta(x - x_sample), we ask: "how informative is observing at x*
for learning about f(x_sample)?" The answer should peak at x* = x_sample.

CONFIGURATION:
- x_sample = 0.25 (the single point in X_SET)
- x_init = -1.5 (starting point, far from x_sample)
- Training data: standard GP training across domain

EXPECTED OUTCOME:
- x* should converge from x_init toward x_sample
- Final x* should be within tolerance of x_sample
- Trajectory should show clear "crawling" toward target

USAGE:
    python test_x_convergence.py          # E[λ] approximation (default)
    python test_x_convergence.py --mc     # MC sampling of λ
"""

import sys
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Add parent directories to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))  # 1D_playground
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))  # Spatial_GP_repo

# Import from gp_utility_playground.py
from gp_utility_playground import (
    VariationalGP,
    PoissonLikelihood,
    train_gp,
    generate_poisson_data,
    lambda_true,
    get_marginal_moments,  # For landscape evaluation (non-differentiable is fine)
    get_conditional_moments,  # For landscape evaluation
    compute_H,  # For landscape evaluation
    DEVICE,
    DTYPE,
    MAX_R,
    X_MIN,
    X_MAX,
)

# Import SetPX from set_based_utility
sys.path.insert(0, str(Path(__file__).parent.parent / "set_based_utility"))
from test_finite_set import SetPX

import warnings
warnings.filterwarnings("ignore", message=".*torch.cuda.*DtypeTensor.*")
warnings.filterwarnings("ignore", message=".*torch.sparse.SparseTensor.*")

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
# Target point (what we want to learn about)
X_SAMPLE = 0.25

# Starting point for optimization
# NOTE: Starting too far (e.g., -1.5) lands in flat utility region with unreliable gradients.
# Start from a position where gradient is meaningful (within ~1 lengthscale of x_sample).
X_INIT = -0.25

# Optimization parameters
N_OPT_STEPS = 150
LR = 0.1

# Training configuration
N_TRAIN = 15
N_INDUCING = 20

# MC samples for utility - higher for less noisy gradients
N_MC_SAMPLES = 200

# Convergence tolerance
TOLERANCE = 0.1

# -----------------------------------------------------------------------------
# Command-Line Arguments
# -----------------------------------------------------------------------------
parser = argparse.ArgumentParser(description='Test utility optimization convergence to x_sample')
parser.add_argument('--mc', action='store_true',
                    help='Use MC sampling of lambda instead of E[lambda] approximation')
args = parser.parse_args()

# Utility computation mode (set by command-line argument)
USE_MC_SAMPLING = args.mc  # False = E[λ] approximation (default), True = full MC sampling


# -----------------------------------------------------------------------------
# Differentiable Moment Functions (maintain gradient flow)
# -----------------------------------------------------------------------------
def get_marginal_moments_differentiable(model, x_star):
    """Get marginal GP posterior moments - differentiable version.

    Unlike the parent function, this does NOT use torch.no_grad().
    """
    model.eval()
    posterior = model(x_star)
    return posterior.mean, posterior.variance


def get_conditional_moments_differentiable(model, x_star, x_sample, lambda_sample):
    """Compute conditional GP moments - differentiable version.

    Uses Gaussian conditioning but maintains gradient flow for x_star.

    Args:
        model: Trained GP model
        x_star: Query points (K,) - can require grad
        x_sample: Single observation point (scalar, detached)
        lambda_sample: Observed lambda value at x_sample (scalar, detached)

    Returns:
        mu_cond: (K,) conditional posterior means
        sigma2_cond: (K,) conditional posterior variances
    """
    model.eval()
    dtype = x_star.dtype
    device = x_star.device

    x_sample_tensor = torch.tensor([x_sample], dtype=dtype, device=device)

    # Concatenate sample point with query points
    all_x = torch.cat([x_sample_tensor, x_star])
    posterior = model(all_x)
    full_covar = posterior.covariance_matrix

    mu_sample = posterior.mean[0]
    var_sample = full_covar[0, 0]
    mu_star = posterior.mean[1:]
    var_star = full_covar.diag()[1:]
    cross_cov = full_covar[0, 1:]

    # Standard Gaussian conditioning formula
    innovation = lambda_sample - mu_sample
    mu_cond = mu_star + cross_cov * (innovation / var_sample)
    sigma2_cond = var_star - (cross_cov ** 2) / var_sample
    sigma2_cond = torch.clamp(sigma2_cond, min=1e-8)

    return mu_cond, sigma2_cond


def compute_H_differentiable(mu, sigma2, r_max=MAX_R, a=1.0, lambda0=0.0):
    """Compute entropy H(R | mu, sigma2) - differentiable version.

    Uses Laplace approximation but maintains gradient flow.
    """
    from utility import laplace_approximations_new

    device = mu.device
    dtype = mu.dtype
    r_values = torch.arange(0, r_max, dtype=dtype, device=device)

    logf_mean = a * mu + lambda0
    logf_var = a**2 * sigma2

    p_r, log_p_r = laplace_approximations_new(mu=logf_mean, sigma2=logf_var, r=r_values)
    H = -torch.sum(p_r * log_p_r, dim=1)

    return H


# -----------------------------------------------------------------------------
# Differentiable Utility Computation
# -----------------------------------------------------------------------------
def compute_utility_at_x_star_deterministic(model, x_star, px_set, r_max=MAX_R):
    """Compute distribution-aware utility at x* using E[λ] approximation - DETERMINISTIC.

    WARNING: This is an APPROXIMATION!
    We approximate E_lambda[H_cond(x* | lambda)] ≈ H_cond(x* | E[lambda])
    i.e., we use the posterior mean of lambda(x_sample) instead of sampling.
    This eliminates MC noise for cleaner gradients during optimization.

    Works for both single-point and multi-point X_SET by averaging H_cond over the set.

    Args:
        model: Trained GP
        x_star: Single query point (tensor with requires_grad=True)
        px_set: SetPX with target point(s)
        r_max: Max spike count for entropy computation

    Returns:
        utility: Scalar utility value (differentiable)
    """
    device = x_star.device
    dtype = x_star.dtype

    # Ensure x_star is 2D for model input
    if x_star.dim() == 0:
        x_star_input = x_star.unsqueeze(0)
    else:
        x_star_input = x_star

    # H_marg at x_star (differentiable)
    mu_marg, sigma2_marg = get_marginal_moments_differentiable(model, x_star_input)
    H_marg = compute_H_differentiable(mu_marg, sigma2_marg, r_max=r_max)

    # Average H_cond over all points in X_SET
    x_set = px_set.get_set()
    H_cond_sum = torch.zeros_like(H_marg)

    for x_sample_tensor in x_set:
        x_sample = x_sample_tensor.item()

        # Get GP posterior mean at x_sample (use mean instead of sampling)
        x_sample_input = torch.tensor([x_sample], dtype=dtype, device=device)
        with torch.no_grad():
            post_sample = model(x_sample_input)
            lambda_mean = post_sample.mean[0].item()

        # H_cond at x_star given (x_sample, lambda_mean) - differentiable w.r.t. x_star
        mu_cond, sigma2_cond = get_conditional_moments_differentiable(model, x_star_input, x_sample, lambda_mean)
        H_cond_i = compute_H_differentiable(mu_cond, sigma2_cond, r_max=r_max)
        H_cond_sum = H_cond_sum + H_cond_i

    H_cond = H_cond_sum / px_set.n_points
    utility = H_marg - H_cond

    return utility.squeeze()


def compute_utility_at_x_star(model, x_star, px_set, n_mc_samples=N_MC_SAMPLES, r_max=MAX_R,
                               _warned=[False]):
    """Compute distribution-aware utility at a single point x*.

    Uses global USE_MC_SAMPLING flag to choose between:
    - E[λ] approximation (default): Uses posterior mean, cleaner gradients
    - MC sampling: Samples λ from posterior, exact in the MC limit

    Args:
        model: Trained GP
        x_star: Single query point (tensor with requires_grad=True)
        px_set: SetPX with target point(s)
        n_mc_samples: MC samples for conditional entropy (when USE_MC_SAMPLING=True)
        r_max: Max spike count for entropy computation

    Returns:
        utility: Scalar utility value (differentiable)
    """
    # Print mode once
    if not _warned[0]:
        if USE_MC_SAMPLING:
            print("  [MODE: MC SAMPLING]")
            print(f"    Sampling lambda from posterior ({n_mc_samples} samples per step)")
        else:
            print("  [MODE: E[λ] APPROXIMATION]")
            print("    Using E[lambda] instead of sampling - cleaner gradients but approximate")
        _warned[0] = True

    # Use deterministic E[λ] approximation
    if not USE_MC_SAMPLING:
        return compute_utility_at_x_star_deterministic(model, x_star, px_set, r_max=r_max)

    # Multi-point case or MC mode: use MC sampling
    device = x_star.device
    dtype = x_star.dtype

    if x_star.dim() == 0:
        x_star_input = x_star.unsqueeze(0)
    else:
        x_star_input = x_star

    mu_marg, sigma2_marg = get_marginal_moments_differentiable(model, x_star_input)
    H_marg = compute_H_differentiable(mu_marg, sigma2_marg, r_max=r_max)

    H_cond_sum = torch.zeros(H_marg.shape, dtype=dtype, device=device)

    for _ in range(n_mc_samples):
        x_i = px_set.sample(n=1, dtype=dtype, device=device).item()

        with torch.no_grad():
            post_i = model(torch.tensor([x_i], dtype=dtype, device=device))
            mu_i = post_i.mean[0]
            std_i = post_i.variance[0].sqrt()
            lambda_i = (mu_i + std_i * torch.randn(1, dtype=dtype, device=device)).item()

        mu_cond, sigma2_cond = get_conditional_moments_differentiable(model, x_star_input, x_i, lambda_i)
        H_cond_i = compute_H_differentiable(mu_cond, sigma2_cond, r_max=r_max)
        H_cond_sum = H_cond_sum + H_cond_i

    H_cond = H_cond_sum / n_mc_samples
    utility = H_marg - H_cond

    return utility.squeeze()


# -----------------------------------------------------------------------------
# Optimization Loop
# -----------------------------------------------------------------------------
def optimize_x_star(model, x_init, px_set, n_steps=N_OPT_STEPS, lr=LR, n_mc_samples=N_MC_SAMPLES):
    """Run gradient ascent to find x* that maximizes utility.

    NOTE: utils.py disables gradients globally with torch.set_grad_enabled(False).
    We use torch.enable_grad() to override this locally.

    Args:
        model: Trained GP
        x_init: Starting point (scalar)
        px_set: SetPX with target point(s)
        n_steps: Number of optimization steps
        lr: Learning rate
        n_mc_samples: MC samples per utility evaluation

    Returns:
        x_final: Optimized x*
        trajectory: List of x* values at each step
        utilities: List of utility values at each step
    """
    trajectory = [x_init]
    utilities = []

    print(f"\nStarting optimization from x* = {x_init}")
    print(f"Target: x_sample = {px_set.get_set()[0].item()}")
    print("-" * 50)

    # CRITICAL: Enable gradients (utils.py disables them globally)
    with torch.enable_grad():
        # Initialize x_star as optimizable parameter
        x_star = torch.tensor([x_init], dtype=DTYPE, device=DEVICE, requires_grad=True)
        optimizer = torch.optim.Adam([x_star], lr=lr)

        for step in range(n_steps):
            optimizer.zero_grad()

            # Compute utility (gradient enabled)
            utility = compute_utility_at_x_star(model, x_star, px_set, n_mc_samples=n_mc_samples)

            # Maximize utility = minimize negative utility
            loss = -utility
            loss.backward()

            optimizer.step()

            # Clamp to domain bounds
            with torch.no_grad():
                x_star.data.clamp_(X_MIN, X_MAX)

            # Record trajectory (detach to avoid graph issues)
            current_x = x_star.detach().item()
            current_u = utility.detach().item()
            trajectory.append(current_x)
            utilities.append(current_u)

            # Clear GPyTorch caches to avoid memory buildup
            model.train()
            model.eval()

            # Print progress
            if step % 20 == 0 or step == n_steps - 1:
                dist_to_target = abs(current_x - px_set.get_set()[0].item())
                print(f"Step {step:3d}: x* = {current_x:7.4f}, U = {current_u:7.4f}, dist = {dist_to_target:.4f}")

    return x_star.item(), trajectory, utilities


# -----------------------------------------------------------------------------
# Utility Landscape Evaluation (for visualization)
# -----------------------------------------------------------------------------
def evaluate_utility_landscape(model, px_set, n_points=100, n_mc_samples=50):
    """Evaluate utility across the domain for visualization.

    Uses fewer MC samples for speed since this is just for plotting.
    """
    x_grid = torch.linspace(X_MIN, X_MAX, n_points, dtype=DTYPE, device=DEVICE)
    utilities = []

    print("\nEvaluating utility landscape for visualization...")
    for i, x in enumerate(x_grid):
        x_tensor = x.unsqueeze(0)
        with torch.no_grad():
            u = compute_utility_at_x_star(model, x_tensor, px_set, n_mc_samples=n_mc_samples)
        utilities.append(u.item())

        if (i + 1) % 25 == 0:
            print(f"  {i+1}/{n_points} points evaluated")

    return x_grid.cpu().numpy(), np.array(utilities)


# -----------------------------------------------------------------------------
# Main Test
# -----------------------------------------------------------------------------
def main():
    print("=" * 70)
    print("Test: Utility Optimization Converges to x_sample")
    print("=" * 70)

    # Set seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    # Create SetPX with single target point
    px_set = SetPX([X_SAMPLE])

    print(f"\nConfiguration:")
    print(f"  Domain: x in [{X_MIN}, {X_MAX}]")
    print(f"  x_sample (target): {X_SAMPLE}")
    print(f"  x_init (start): {X_INIT}")
    print(f"  Optimization steps: {N_OPT_STEPS}")
    print(f"  Learning rate: {LR}")
    print(f"  MC samples per step: {N_MC_SAMPLES}")

    # Generate training data across the domain
    train_x = torch.linspace(X_MIN, X_MAX, N_TRAIN, dtype=DTYPE, device=DEVICE)
    train_y = generate_poisson_data(train_x, lambda_true)

    print(f"\nTraining data: {N_TRAIN} points across domain")

    # Create and train GP
    inducing_points = torch.linspace(X_MIN, X_MAX, N_INDUCING, dtype=DTYPE, device=DEVICE)
    model = VariationalGP(inducing_points, jitter=1e-4).to(DEVICE)
    likelihood = PoissonLikelihood().to(DEVICE)

    print("\nTraining GP...")
    model, likelihood = train_gp(model, likelihood, train_x, train_y, n_iterations=400, lr=0.1)

    # Run optimization
    model.eval()
    x_final, trajectory, utilities = optimize_x_star(model, X_INIT, px_set)

    # Analyze results
    distance_to_target = abs(x_final - X_SAMPLE)
    converged = distance_to_target < TOLERANCE

    print("\n" + "=" * 70)
    print("RESULTS:")
    print("=" * 70)
    print(f"Starting point: x* = {X_INIT}")
    print(f"Final point: x* = {x_final:.4f}")
    print(f"Target: x_sample = {X_SAMPLE}")
    print(f"Distance to target: {distance_to_target:.4f}")
    print(f"Tolerance: {TOLERANCE}")

    # Evaluate utility landscape for visualization
    x_landscape, u_landscape = evaluate_utility_landscape(model, px_set, n_points=100, n_mc_samples=50)

    # Find max of landscape
    max_idx = np.argmax(u_landscape)
    x_max_landscape = x_landscape[max_idx]

    print(f"\nUtility landscape max at x = {x_max_landscape:.4f}")

    # Visualization
    print("\nGenerating plot...")
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Get GP posterior for panel 1
    x_plot = torch.linspace(X_MIN, X_MAX, 200, dtype=DTYPE, device=DEVICE)
    with torch.no_grad():
        posterior = model(x_plot)
        gp_mean = posterior.mean.cpu().numpy()
        gp_std = posterior.variance.sqrt().cpu().numpy()
        true_lambda_vals = lambda_true(x_plot).cpu().numpy()

    x_plot_np = x_plot.cpu().numpy()
    trajectory_np = np.array(trajectory)

    # Panel 1: GP Posterior + Trajectory
    ax1 = axes[0]
    ax1.plot(x_plot_np, true_lambda_vals, 'k--', linewidth=2, label='True lambda(x)')
    ax1.plot(x_plot_np, gp_mean, 'b-', linewidth=2, label='GP mean')
    ax1.fill_between(x_plot_np, gp_mean - 2*gp_std, gp_mean + 2*gp_std,
                     alpha=0.3, color='blue', label='GP +/- 2 std')

    # Mark training data
    train_x_np = train_x.cpu().numpy()
    ax1.scatter(train_x_np, lambda_true(train_x).cpu().numpy(),
                c='red', s=50, zorder=5, marker='o', label='Training data')

    # Mark x_sample (target)
    ax1.axvline(x=X_SAMPLE, color='orange', linestyle='-', linewidth=3, alpha=0.8, label=f'x_sample = {X_SAMPLE}')

    # Mark trajectory
    for i, x_traj in enumerate(trajectory_np[::10]):  # Every 10th point
        alpha = 0.3 + 0.7 * (i / (len(trajectory_np[::10]) - 1)) if len(trajectory_np[::10]) > 1 else 1.0
        ax1.axvline(x=x_traj, color='green', linestyle=':', linewidth=1, alpha=alpha)

    ax1.scatter([X_INIT], [gp_mean[np.argmin(np.abs(x_plot_np - X_INIT))]],
                c='green', s=100, marker='o', zorder=6, label=f'x_init = {X_INIT}')
    ax1.scatter([x_final], [gp_mean[np.argmin(np.abs(x_plot_np - x_final))]],
                c='darkgreen', s=150, marker='*', zorder=6, label=f'x_final = {x_final:.2f}')

    ax1.set_xlabel('x')
    ax1.set_ylabel('lambda(x)')
    ax1.set_title('GP Posterior + Optimization Trajectory')
    ax1.legend(loc='upper right', fontsize=8)
    ax1.grid(True, alpha=0.3)

    # Panel 2: Utility Landscape + Trajectory
    ax2 = axes[1]
    ax2.plot(x_landscape, u_landscape, 'b-', linewidth=2, label='U(x*)')
    ax2.fill_between(x_landscape, 0, u_landscape, alpha=0.3, color='blue')

    # Mark x_sample
    ax2.axvline(x=X_SAMPLE, color='orange', linestyle='-', linewidth=3, alpha=0.8, label=f'x_sample = {X_SAMPLE}')

    # Mark landscape max
    ax2.scatter([x_max_landscape], [u_landscape[max_idx]], c='darkblue', s=150, marker='*',
                zorder=5, label=f'U max at {x_max_landscape:.2f}')

    # Plot trajectory points on utility curve
    # Interpolate utility values at trajectory points
    traj_utilities = np.interp(trajectory_np, x_landscape, u_landscape)
    colors = plt.cm.Greens(np.linspace(0.3, 1.0, len(trajectory_np)))
    ax2.scatter(trajectory_np, traj_utilities, c=colors, s=30, zorder=4, alpha=0.7)

    # Mark start and end
    ax2.scatter([X_INIT], [traj_utilities[0]], c='green', s=100, marker='o', zorder=6, label='Start')
    ax2.scatter([x_final], [traj_utilities[-1]], c='darkgreen', s=150, marker='*', zorder=6, label='Final')

    ax2.set_xlabel('x*')
    ax2.set_ylabel('Utility U(x*)')
    ax2.set_title('Utility Landscape + Optimization Path')
    ax2.legend(loc='upper right', fontsize=8)
    ax2.grid(True, alpha=0.3)

    # Panel 3: Convergence Plot
    ax3 = axes[2]
    iterations = np.arange(len(trajectory_np))
    ax3.plot(iterations, trajectory_np, 'g-', linewidth=2, marker='o', markersize=3, label='x* trajectory')

    # Horizontal line at x_sample
    ax3.axhline(y=X_SAMPLE, color='orange', linestyle='-', linewidth=3, alpha=0.8, label=f'x_sample = {X_SAMPLE}')

    # Tolerance band
    ax3.axhspan(X_SAMPLE - TOLERANCE, X_SAMPLE + TOLERANCE, alpha=0.2, color='orange', label=f'Tolerance = {TOLERANCE}')

    ax3.set_xlabel('Optimization Step')
    ax3.set_ylabel('x*')
    ax3.set_title('Convergence: x* vs Iteration')
    ax3.legend(loc='upper right' if X_INIT > X_SAMPLE else 'lower right', fontsize=8)
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim(-5, len(trajectory_np) + 5)

    # Add verdict and mode to figure title
    mode_str = "MC Sampling" if USE_MC_SAMPLING else "E[λ] Approximation"
    if converged:
        fig.suptitle(f'PASS: x* converged to x_sample (dist = {distance_to_target:.4f}) | Mode: {mode_str}',
                     fontsize=13, fontweight='bold', color='green')
    else:
        fig.suptitle(f'CHECK: x* did not converge (dist = {distance_to_target:.4f}) | Mode: {mode_str}',
                     fontsize=13, fontweight='bold', color='red')

    plt.tight_layout()

    # Save figure with mode-specific filename
    mode_suffix = "mc" if USE_MC_SAMPLING else "E_lambda"
    save_path = Path(__file__).parent / f'test_x_convergence_result_{mode_suffix}.png'
    plt.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {save_path}")

    # Final verdict
    print("\n" + "=" * 70)
    print("VERDICT:")
    print("=" * 70)
    if converged:
        print(f"PASS: x* converged to x_sample!")
        print(f"      Started at x* = {X_INIT}")
        print(f"      Ended at x* = {x_final:.4f}")
        print(f"      Target was x_sample = {X_SAMPLE}")
        print(f"      Final distance: {distance_to_target:.4f} < tolerance {TOLERANCE}")
    else:
        print(f"CHECK: x* did not fully converge to x_sample.")
        print(f"      Started at x* = {X_INIT}")
        print(f"      Ended at x* = {x_final:.4f}")
        print(f"      Target was x_sample = {X_SAMPLE}")
        print(f"      Final distance: {distance_to_target:.4f} >= tolerance {TOLERANCE}")
        print(f"      Utility landscape max at: {x_max_landscape:.4f}")


if __name__ == "__main__":
    main()
