"""
Test: Distribution-Aware Utility Under Poor GP Fit (No Training Data)
=====================================================================
Created by Claude

PURPOSE: Verify that distribution-aware utility peaks inside p(x) region even when
GP has zero training data (posterior = prior, maximum uncertainty everywhere).

SUPPORTS DIFFERENT p(x) DISTRIBUTIONS:
- gaussian: N(mean, std^2) - concentrated at center
- uniform: U[a, b] - flat over interval
- bimodal: mixture of two Gaussians - two peaks

EXPECTED OUTCOME:
- Standard utility: Flat (stationary kernel gives constant variance)
- Distribution-aware utility: Peaks where p(x) has support

USAGE:
    python test_no_training_data.py                    # Default: gaussian
    python test_no_training_data.py --px_type gaussian
    python test_no_training_data.py --px_type uniform
    python test_no_training_data.py --px_type bimodal
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
    evaluate_nd_utility_new,
    get_marginal_moments,
    get_conditional_moments,
    compute_H,
    DEVICE,
    DTYPE,
    MAX_R,
)

import warnings
warnings.filterwarnings("ignore", message=".*torch.cuda.*DtypeTensor.*")
warnings.filterwarnings("ignore", message=".*torch.sparse.SparseTensor.*")

# -----------------------------------------------------------------------------
# Configuration - Extended domain to make effect visible
# -----------------------------------------------------------------------------
X_MIN, X_MAX = -5.0, 5.0   # Extended domain (wider than standard playground)
N_INDUCING = 30            # Inducing points for variational GP

# Kernel hyperparameters (set manually since no training)
LENGTHSCALE = 1.0          # Reasonable for domain scale
OUTPUTSCALE = 1.0          # Standard output variance
MEAN_CONSTANT = 2.0        # Baseline (firing rate ~ exp(2) ~ 7)

# -----------------------------------------------------------------------------
# p(x) Distribution Classes
# -----------------------------------------------------------------------------
class GaussianPX:
    """Gaussian distribution p(x) = N(mean, std^2)"""
    def __init__(self, mean=0.0, std=1.0):
        self.mean = mean
        self.std = std
        self.name = f"Gaussian N({mean}, {std}^2)"

    def sample(self, n=1, dtype=DTYPE, device=DEVICE):
        """Sample n points from p(x)"""
        return self.mean + self.std * torch.randn(n, dtype=dtype, device=device)

    def pdf(self, x):
        """Evaluate PDF at x (numpy array)"""
        return np.exp(-0.5 * ((x - self.mean) / self.std) ** 2) / (self.std * np.sqrt(2 * np.pi))

    def support_region(self):
        """Return (center, half_width) for checking if utility peaks are inside"""
        return self.mean, 2 * self.std


class UniformPX:
    """Uniform distribution p(x) = U[a, b]"""
    def __init__(self, a=-1.5, b=1.5):
        self.a = a
        self.b = b
        self.name = f"Uniform U[{a}, {b}]"

    def sample(self, n=1, dtype=DTYPE, device=DEVICE):
        """Sample n points from p(x)"""
        return self.a + (self.b - self.a) * torch.rand(n, dtype=dtype, device=device)

    def pdf(self, x):
        """Evaluate PDF at x (numpy array)"""
        pdf = np.zeros_like(x)
        mask = (x >= self.a) & (x <= self.b)
        pdf[mask] = 1.0 / (self.b - self.a)
        return pdf

    def support_region(self):
        """Return (center, half_width) for checking if utility peaks are inside"""
        center = (self.a + self.b) / 2
        half_width = (self.b - self.a) / 2
        return center, half_width


class BimodalPX:
    """Bimodal distribution: mixture of two Gaussians"""
    def __init__(self, mean1=-2.0, mean2=2.0, std=0.5, weight1=0.5):
        self.mean1 = mean1
        self.mean2 = mean2
        self.std = std
        self.weight1 = weight1
        self.weight2 = 1 - weight1
        self.name = f"Bimodal: {weight1:.0%} N({mean1},{std}^2) + {1-weight1:.0%} N({mean2},{std}^2)"

    def sample(self, n=1, dtype=DTYPE, device=DEVICE):
        """Sample n points from p(x)"""
        # Decide which component each sample comes from
        component = torch.rand(n, device=device) < self.weight1
        samples = torch.zeros(n, dtype=dtype, device=device)
        n1 = component.sum().item()
        n2 = n - n1
        if n1 > 0:
            samples[component] = self.mean1 + self.std * torch.randn(n1, dtype=dtype, device=device)
        if n2 > 0:
            samples[~component] = self.mean2 + self.std * torch.randn(n2, dtype=dtype, device=device)
        return samples

    def pdf(self, x):
        """Evaluate PDF at x (numpy array)"""
        pdf1 = np.exp(-0.5 * ((x - self.mean1) / self.std) ** 2) / (self.std * np.sqrt(2 * np.pi))
        pdf2 = np.exp(-0.5 * ((x - self.mean2) / self.std) ** 2) / (self.std * np.sqrt(2 * np.pi))
        return self.weight1 * pdf1 + self.weight2 * pdf2

    def support_region(self):
        """Return list of (center, half_width) for each mode"""
        return [(self.mean1, 2 * self.std), (self.mean2, 2 * self.std)]


def get_px_distribution(px_type):
    """Factory function to create p(x) distribution"""
    if px_type == "gaussian":
        return GaussianPX(mean=0.0, std=1.0)
    elif px_type == "uniform":
        return UniformPX(a=-1.5, b=1.5)
    elif px_type == "bimodal":
        return BimodalPX(mean1=-2.0, mean2=2.0, std=0.5)
    else:
        raise ValueError(f"Unknown px_type: {px_type}. Choose from: gaussian, uniform, bimodal")


# -----------------------------------------------------------------------------
# Custom Distribution-Aware Utility (supports arbitrary p(x))
# -----------------------------------------------------------------------------
def evaluate_distribution_aware_utility_custom(model, x_candidates, px_dist,
                                                n_mc_samples=500, r_max=MAX_R):
    """Evaluate distribution-aware utility with custom p(x) distribution.

    U(x*) = H_marg(x*) - E[H_cond(x* | x, λ)]

    where expectation is over x ~ p(x), λ ~ q(λ|x).

    Args:
        model: GP model
        x_candidates: (n_candidates,) query points
        px_dist: Distribution object with .sample() method
        n_mc_samples: Number of MC samples
        r_max: Max spike count for entropy

    Returns:
        utility: (n_candidates,) utility at each candidate
    """
    model.eval()
    device = x_candidates.device
    dtype = x_candidates.dtype

    # Marginal entropy
    mu_marg, sigma2_marg = get_marginal_moments(model, x_candidates)
    H_marg = compute_H(mu_marg, sigma2_marg, r_max=r_max)

    # Average conditional entropy over MC samples
    H_cond_sum = torch.zeros_like(H_marg)

    with torch.no_grad():
        for _ in range(n_mc_samples):
            # Sample x from p(x) using custom distribution
            x_i = px_dist.sample(n=1, dtype=dtype, device=device).item()
            x_i_tensor = torch.tensor([x_i], dtype=dtype, device=device)

            # Get GP posterior at x_i
            post_i = model(x_i_tensor)
            mu_i = post_i.mean[0]
            std_i = post_i.variance[0].sqrt()

            # Sample lambda from GP posterior at x_i
            lambda_i = (mu_i + std_i * torch.randn(1, dtype=dtype, device=device)).item()

            # Conditional entropy
            mu_cond_i, sigma2_cond_i = get_conditional_moments(model, x_candidates, x_i, lambda_i)
            H_cond_i = compute_H(mu_cond_i, sigma2_cond_i, r_max=r_max)
            H_cond_sum += H_cond_i

    H_cond = H_cond_sum / n_mc_samples
    utility = H_marg - H_cond

    return utility


# -----------------------------------------------------------------------------
# True Function (for visualization only - NOT used for training)
# -----------------------------------------------------------------------------
def lambda_sin(x):
    """True latent function - sinusoidal, gives firing rates in reasonable range."""
    return (torch.sin(2 * np.pi * x / 3) + 1.0) * 2  # Period ~6, range [0, 4]


# -----------------------------------------------------------------------------
# Helper: Check if x is inside p(x) support region
# -----------------------------------------------------------------------------
def is_inside_support(x, px_dist):
    """Check if x is inside the support region of p(x)"""
    support = px_dist.support_region()
    if isinstance(support, list):
        # Bimodal: list of (center, half_width) tuples
        for center, half_width in support:
            if abs(x - center) < half_width:
                return True
        return False
    else:
        # Single mode: (center, half_width)
        center, half_width = support
        return abs(x - center) < half_width


# -----------------------------------------------------------------------------
# Main Test
# -----------------------------------------------------------------------------
def main(px_type="gaussian"):
    print("=" * 70)
    print("Test: Distribution-Aware Utility Under Poor GP Fit (No Training Data)")
    print("=" * 70)

    # Create p(x) distribution
    px_dist = get_px_distribution(px_type)

    print(f"\nConfiguration:")
    print(f"  Domain: x in [{X_MIN}, {X_MAX}]")
    print(f"  p(x) = {px_dist.name}")
    print(f"  Kernel lengthscale: {LENGTHSCALE}")
    print(f"  NO TRAINING DATA (pure GP prior)")

    # Set seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    # Create inducing points covering the domain
    inducing_points = torch.linspace(X_MIN, X_MAX, N_INDUCING, dtype=DTYPE, device=DEVICE)

    # Initialize GP model (no training)
    model = VariationalGP(inducing_points, jitter=1e-4).to(DEVICE)
    likelihood = PoissonLikelihood().to(DEVICE)

    # Set kernel hyperparameters manually
    model.covar_module.base_kernel.lengthscale = LENGTHSCALE
    model.covar_module.outputscale = OUTPUTSCALE
    model.mean_module.constant = MEAN_CONSTANT

    print(f"\nGP Hyperparameters (manually set):")
    print(f"  lengthscale = {model.covar_module.base_kernel.lengthscale.item():.4f}")
    print(f"  outputscale = {model.covar_module.outputscale.item():.4f}")
    print(f"  mean = {model.mean_module.constant.item():.4f}")

    # Evaluation grid
    x_candidates = torch.linspace(X_MIN, X_MAX, 200, dtype=DTYPE, device=DEVICE)

    # Evaluate utilities
    print("\nEvaluating standard utility...")
    model.eval()
    utility_standard = evaluate_nd_utility_new(model, x_candidates)

    print("Evaluating distribution-aware utility (may take a moment)...")
    utility_distr_aware = evaluate_distribution_aware_utility_custom(
        model, x_candidates,
        px_dist=px_dist,
        n_mc_samples=500,
        r_max=MAX_R
    )

    # Convert to numpy
    x_np = x_candidates.cpu().numpy()
    utility_std_np = utility_standard.cpu().numpy()
    utility_da_np = utility_distr_aware.cpu().numpy()

    # Find max utility locations
    max_idx_std = np.argmax(utility_std_np)
    max_idx_da = np.argmax(utility_da_np)
    x_max_std = x_np[max_idx_std]
    x_max_da = x_np[max_idx_da]

    print("\n" + "=" * 70)
    print("RESULTS:")
    print("=" * 70)
    print(f"Standard utility max at x = {x_max_std:.3f} (value = {utility_std_np[max_idx_std]:.4f})")
    print(f"Distribution-aware utility max at x = {x_max_da:.3f} (value = {utility_da_np[max_idx_da]:.4f})")

    # Check if distribution-aware peaks inside p(x) region
    in_px_region = is_inside_support(x_max_da, px_dist)
    print(f"\nDistribution-aware max inside p(x) support region? {in_px_region}")

    # Check utility variation
    std_range = utility_std_np.max() - utility_std_np.min()
    da_range = utility_da_np.max() - utility_da_np.min()
    print(f"\nUtility range (max - min):")
    print(f"  Standard: {std_range:.4f}")
    print(f"  Distribution-aware: {da_range:.4f}")

    # Visualization
    print("\nGenerating plot...")
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

    # Get GP posterior for visualization
    with torch.no_grad():
        posterior = model(x_candidates)
        mean = posterior.mean.cpu().numpy()
        std = posterior.variance.sqrt().cpu().numpy()
        true_lambda = lambda_sin(x_candidates).cpu().numpy()

    # Panel 1: True function + GP prior
    ax1 = axes[0]
    ax1.plot(x_np, true_lambda, 'k--', linewidth=2, label='True lambda(x) [sin]')
    ax1.plot(x_np, mean, 'b-', linewidth=2, label='GP prior mean')
    ax1.fill_between(x_np, mean - 2*std, mean + 2*std, alpha=0.3, color='blue', label='GP prior +/- 2 std')
    ax1.axhline(y=MEAN_CONSTANT, color='gray', linestyle=':', alpha=0.7, label=f'Prior mean = {MEAN_CONSTANT}')
    ax1.set_ylabel('lambda(x)')
    ax1.set_title('GP Prior (No Training Data) vs True Function')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)

    # Panel 2: Standard utility
    ax2 = axes[1]
    ax2.plot(x_np, utility_std_np, 'g-', linewidth=2)
    ax2.fill_between(x_np, 0, utility_std_np, alpha=0.3, color='green')
    ax2.scatter([x_max_std], [utility_std_np[max_idx_std]], c='darkgreen', s=150,
                zorder=5, marker='*', label=f'Max at x={x_max_std:.2f}')
    ax2.set_ylabel('Standard Utility')
    ax2.set_title(f'Standard Utility (Expected: flat or peaks at boundaries +/-{X_MAX})')
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)

    # Panel 3: Distribution-aware utility with p(x)
    ax3 = axes[2]

    # Plot p(x) as background (using the distribution's PDF)
    p_x = px_dist.pdf(x_np)
    p_x_scaled = p_x / p_x.max() * utility_da_np.max() * 0.5 if p_x.max() > 0 else p_x
    ax3.fill_between(x_np, 0, p_x_scaled, alpha=0.2, color='orange', label='p(x) (scaled)')

    ax3.plot(x_np, utility_da_np, 'b-', linewidth=2)
    ax3.fill_between(x_np, 0, utility_da_np, alpha=0.3, color='blue')
    ax3.scatter([x_max_da], [utility_da_np[max_idx_da]], c='darkblue', s=150,
                zorder=5, marker='*', label=f'Max at x={x_max_da:.2f}')

    # Draw support region boundaries
    support = px_dist.support_region()
    if isinstance(support, list):
        # Bimodal: multiple regions
        for i, (center, half_width) in enumerate(support):
            label = 'p(x) support' if i == 0 else None
            ax3.axvline(x=center - half_width, color='orange', linestyle='--', alpha=0.5)
            ax3.axvline(x=center + half_width, color='orange', linestyle='--', alpha=0.5, label=label)
    else:
        # Single region
        center, half_width = support
        ax3.axvline(x=center - half_width, color='orange', linestyle='--', alpha=0.5)
        ax3.axvline(x=center + half_width, color='orange', linestyle='--', alpha=0.5, label='p(x) support')

    ax3.set_xlabel('x')
    ax3.set_ylabel('Distribution-Aware Utility')
    ax3.set_title(f'Distribution-Aware Utility with p(x) = {px_dist.name}')
    ax3.legend(loc='upper right')
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save figure with px_type in filename
    save_path = Path(__file__).parent / f'test_no_training_data_{px_type}_result.png'
    plt.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {save_path}")

    # Final verdict
    print("\n" + "=" * 70)
    print("VERDICT:")
    print("=" * 70)
    if in_px_region:
        print("PASS: Distribution-aware utility correctly peaks inside p(x) support region")
        print("      even with no training data (uninformative GP).")
    else:
        print("CHECK: Results may not match expectations. Review the plot.")
        print(f"  - Distribution-aware max (x={x_max_da:.2f}) is OUTSIDE p(x) support region")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test distribution-aware utility under poor GP fit")
    parser.add_argument("--px_type", type=str, default="gaussian",
                        choices=["gaussian", "uniform", "bimodal"],
                        help="Type of p(x) distribution (default: gaussian)")
    args = parser.parse_args()

    main(px_type=args.px_type)
