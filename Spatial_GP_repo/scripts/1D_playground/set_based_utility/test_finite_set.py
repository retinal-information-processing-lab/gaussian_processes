"""
Test: Distribution-Aware Utility with Finite X Set
===================================================
Created by Claude

PURPOSE: Verify that distribution-aware utility peaks at/near members of a
finite set when p(x) samples uniformly from that set (instead of continuous distribution).

CONTEXT: This test is part of investigating distribution-aware utility behavior.
The goal is to check that when we replace a continuous p(x) with just a SET of points,
the utility will peak in that area, validating that distribution-aware utility
correctly prioritizes regions where p(x) has mass.

CONFIGURATION:
- X_SET: Hardcoded points clustered in one region (default: [0.2, 0.25, 0.3])
- Training data: GP trained on data OUTSIDE the set region

EXPECTED OUTCOME:
- Standard utility: Peaks based on GP uncertainty (typically outside training)
- Distribution-aware utility: Peaks near X_SET members

USAGE:
    python test_finite_set.py
"""

import sys
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
    evaluate_nd_utility_new,
    get_marginal_moments,
    get_conditional_moments,
    compute_H,
    DEVICE,
    DTYPE,
    MAX_R,
    X_MIN,
    X_MAX,
)

import warnings
warnings.filterwarnings("ignore", message=".*torch.cuda.*DtypeTensor.*")
warnings.filterwarnings("ignore", message=".*torch.sparse.SparseTensor.*")

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
# Domain (from parent, but we redefine for clarity)
X_DOMAIN_MIN, X_DOMAIN_MAX = X_MIN, X_MAX  # -2.0, 2.0

# X_SET configuration - hardcoded finite set clustered in one region
# X_SET = [0.2, 0.25, 0.3]
X_SET = [0.2]

# Training configuration
N_INDUCING = 20
N_TRAIN = 10  # Total training points (split between left and right of set)

# MC sampling for utility
N_MC_SAMPLES = 500

# Tolerance for verdict (how close to set member counts as "near")
TOLERANCE = 0.15


# -----------------------------------------------------------------------------
# SetPX Class - Core Test Component
# -----------------------------------------------------------------------------
class SetPX:
    """Distribution that samples uniformly from a finite set of x values.

    p(x) = 1/N for x in {x1, x2, ..., xN}, 0 otherwise.

    This is the key class for this test - it replaces continuous distributions
    (Gaussian, Uniform) with a discrete finite set.
    """

    def __init__(self, x_values):
        """Initialize with a list of x values.

        Args:
            x_values: List of x values defining the set
        """
        if isinstance(x_values, list):
            x_values = torch.tensor(x_values, dtype=DTYPE, device=DEVICE)
        self.x_values = x_values.to(DEVICE)
        self.n_points = len(x_values)
        self.name = f"FiniteSet({self.n_points} points: {', '.join(f'{x:.2f}' for x in self.x_values.tolist())})"

    def sample(self, n=1, dtype=DTYPE, device=DEVICE):
        """Sample n points uniformly from the set (with replacement).

        Returns:
            Tensor of n samples
        """
        indices = torch.randint(0, self.n_points, (n,), device=device)
        return self.x_values[indices].to(dtype=dtype)

    def get_set(self):
        """Return the full set of x values."""
        return self.x_values

    def support_region(self):
        """Return (min, max) of the set members."""
        return self.x_values.min().item(), self.x_values.max().item()

    def distance_to_nearest(self, x):
        """Compute distance from x to nearest set member.

        Args:
            x: scalar or tensor

        Returns:
            Minimum distance to any set member
        """
        if isinstance(x, (int, float)):
            x = torch.tensor([x], dtype=DTYPE, device=DEVICE)
        distances = torch.abs(x.unsqueeze(-1) - self.x_values)
        return distances.min(dim=-1).values


# -----------------------------------------------------------------------------
# Custom Distribution-Aware Utility (for finite set)
# -----------------------------------------------------------------------------
def evaluate_distribution_aware_utility_set(model, x_candidates, px_set,
                                             n_mc_samples=N_MC_SAMPLES, r_max=MAX_R):
    """Evaluate distribution-aware utility using a finite set for p(x).

    U(x*) = H_marg(x*) - E[H_cond(x* | x, lambda)]

    where expectation is over x ~ uniform(X_SET), lambda ~ q(lambda|x).

    Args:
        model: Trained GP model
        x_candidates: (n_candidates,) query points to evaluate utility at
        px_set: SetPX object defining the finite set
        n_mc_samples: Number of MC samples for expectation
        r_max: Max spike count for entropy computation

    Returns:
        utility: (n_candidates,) utility at each candidate
    """
    model.eval()
    device = x_candidates.device
    dtype = x_candidates.dtype

    # Marginal entropy at all candidates
    mu_marg, sigma2_marg = get_marginal_moments(model, x_candidates)
    H_marg = compute_H(mu_marg, sigma2_marg, r_max=r_max)

    # Average conditional entropy over MC samples
    H_cond_sum = torch.zeros_like(H_marg)

    with torch.no_grad():
        for _ in range(n_mc_samples):
            # Sample x uniformly from the finite set
            x_i = px_set.sample(n=1, dtype=dtype, device=device).item()
            x_i_tensor = torch.tensor([x_i], dtype=dtype, device=device)

            # Get GP posterior at x_i
            post_i = model(x_i_tensor)
            mu_i = post_i.mean[0]
            std_i = post_i.variance[0].sqrt()

            # Sample lambda from GP posterior at x_i
            lambda_i = (mu_i + std_i * torch.randn(1, dtype=dtype, device=device)).item()

            # Conditional entropy at all candidates given (x_i, lambda_i)
            mu_cond_i, sigma2_cond_i = get_conditional_moments(model, x_candidates, x_i, lambda_i)
            H_cond_i = compute_H(mu_cond_i, sigma2_cond_i, r_max=r_max)
            H_cond_sum += H_cond_i

    H_cond = H_cond_sum / n_mc_samples
    utility = H_marg - H_cond

    return utility


# -----------------------------------------------------------------------------
# Training Data Generation
# -----------------------------------------------------------------------------
def generate_training_data_outside_set(px_set, n_train=N_TRAIN, margin=0.2):
    """Generate training data OUTSIDE the set region.

    Creates an interesting test case where:
    - GP is well-fitted in training region (left + right edges)
    - GP has uncertainty in the set region (center)
    - Standard utility wants to explore everywhere uncertain
    - Distribution-aware should specifically target set members

    Args:
        px_set: SetPX object defining where NOT to train
        n_train: Total number of training points
        margin: Extra margin around set to avoid

    Returns:
        train_x, train_y: Training data
    """
    # Get set bounds
    set_min, set_max = px_set.support_region()
    avoid_min = set_min - margin
    avoid_max = set_max + margin

    # Split training between left and right of the set
    n_left = n_train // 2
    n_right = n_train - n_left

    # Generate training points outside the avoidance region
    left_x = torch.linspace(X_DOMAIN_MIN, avoid_min, n_left, dtype=DTYPE, device=DEVICE)
    right_x = torch.linspace(avoid_max, X_DOMAIN_MAX, n_right, dtype=DTYPE, device=DEVICE)

    train_x = torch.cat([left_x, right_x])
    train_y = generate_poisson_data(train_x, lambda_true)

    return train_x, train_y


# -----------------------------------------------------------------------------
# Verdict Logic
# -----------------------------------------------------------------------------
def analyze_utility_peaks(x_candidates, utility, px_set, tolerance=TOLERANCE):
    """Analyze whether utility peaks are near set members.

    Args:
        x_candidates: (N,) grid of x values
        utility: (N,) utility values
        px_set: SetPX object
        tolerance: Maximum distance to consider "near" the set

    Returns:
        dict with analysis results
    """
    x_np = x_candidates.cpu().numpy()
    u_np = utility.cpu().numpy()
    set_tensor = px_set.get_set()

    # Find global max
    global_max_idx = np.argmax(u_np)
    global_max_x = x_np[global_max_idx]
    global_max_utility = u_np[global_max_idx]

    # Distance from global max to nearest set member
    dist_to_set = px_set.distance_to_nearest(global_max_x).item()

    # Check if max is near set
    max_near_set = dist_to_set < tolerance

    # Get set bounds for "in region" check
    set_min, set_max = px_set.support_region()
    max_in_region = (global_max_x >= set_min - tolerance) and (global_max_x <= set_max + tolerance)

    return {
        'global_max_x': global_max_x,
        'global_max_utility': global_max_utility,
        'dist_to_nearest_set_member': dist_to_set,
        'max_near_set': max_near_set,
        'max_in_region': max_in_region,
        'passes': max_near_set or max_in_region,
    }


def compare_utilities(std_analysis, da_analysis, px_set):
    """Compare standard utility vs distribution-aware utility results."""
    std_dist = std_analysis['dist_to_nearest_set_member']
    da_dist = da_analysis['dist_to_nearest_set_member']

    improvement = std_dist - da_dist  # Positive = DA is closer to set

    return {
        'standard_max_x': std_analysis['global_max_x'],
        'standard_dist_to_set': std_dist,
        'da_max_x': da_analysis['global_max_x'],
        'da_dist_to_set': da_dist,
        'improvement': improvement,
        'da_closer_to_set': da_dist < std_dist,
        'da_peaks_at_set': da_analysis['max_near_set'],
    }


# -----------------------------------------------------------------------------
# Main Test
# -----------------------------------------------------------------------------
def main():
    print("=" * 70)
    print("Test: Distribution-Aware Utility with Finite X Set")
    print("=" * 70)

    # Set seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    # Create finite set
    px_set = SetPX(X_SET)

    print(f"\nConfiguration:")
    print(f"  Domain: x in [{X_DOMAIN_MIN}, {X_DOMAIN_MAX}]")
    print(f"  p(x) = {px_set.name}")
    print(f"  Training: {N_TRAIN} points OUTSIDE set region")
    print(f"  MC samples for utility: {N_MC_SAMPLES}")

    # Generate training data outside set region
    train_x, train_y = generate_training_data_outside_set(px_set, N_TRAIN)
    print(f"\nTraining data locations: {train_x.cpu().numpy()}")
    print(f"  Avoiding set region: [{px_set.support_region()[0]:.2f}, {px_set.support_region()[1]:.2f}]")

    # Create inducing points covering the domain
    inducing_points = torch.linspace(X_DOMAIN_MIN, X_DOMAIN_MAX, N_INDUCING, dtype=DTYPE, device=DEVICE)

    # Initialize GP model
    model = VariationalGP(inducing_points, jitter=1e-4).to(DEVICE)
    likelihood = PoissonLikelihood().to(DEVICE)

    # Train GP
    print("\nTraining GP...")
    model, likelihood = train_gp(model, likelihood, train_x, train_y, n_iterations=400, lr=0.1)

    # Evaluation grid
    x_candidates = torch.linspace(X_DOMAIN_MIN, X_DOMAIN_MAX, 200, dtype=DTYPE, device=DEVICE)

    # Evaluate utilities
    print("\nEvaluating standard utility...")
    model.eval()
    utility_standard = evaluate_nd_utility_new(model, x_candidates)

    print("Evaluating distribution-aware utility (may take a moment)...")
    utility_da = evaluate_distribution_aware_utility_set(
        model, x_candidates,
        px_set=px_set,
        n_mc_samples=N_MC_SAMPLES,
        r_max=MAX_R
    )

    # Convert to numpy for analysis and plotting
    x_np = x_candidates.cpu().numpy()
    utility_std_np = utility_standard.cpu().numpy()
    utility_da_np = utility_da.cpu().numpy()

    # Analyze results
    std_analysis = analyze_utility_peaks(x_candidates, utility_standard, px_set)
    da_analysis = analyze_utility_peaks(x_candidates, utility_da, px_set)
    comparison = compare_utilities(std_analysis, da_analysis, px_set)

    print("\n" + "=" * 70)
    print("RESULTS:")
    print("=" * 70)
    print(f"Standard utility max at x = {std_analysis['global_max_x']:.3f}")
    print(f"  Distance to nearest set member: {std_analysis['dist_to_nearest_set_member']:.3f}")
    print(f"\nDistribution-aware utility max at x = {da_analysis['global_max_x']:.3f}")
    print(f"  Distance to nearest set member: {da_analysis['dist_to_nearest_set_member']:.3f}")
    print(f"\nComparison:")
    print(f"  DA closer to set than standard? {comparison['da_closer_to_set']}")
    print(f"  Improvement (distance reduction): {comparison['improvement']:.3f}")

    # Visualization
    print("\nGenerating plot...")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Get GP posterior for visualization
    with torch.no_grad():
        posterior = model(x_candidates)
        gp_mean = posterior.mean.cpu().numpy()
        gp_std = posterior.variance.sqrt().cpu().numpy()
        true_lambda = lambda_true(x_candidates).cpu().numpy()

    # Panel 1: GP Posterior + Training Data + Set Members
    ax1 = axes[0, 0]
    ax1.plot(x_np, true_lambda, 'k--', linewidth=2, label='True lambda(x)')
    ax1.plot(x_np, gp_mean, 'b-', linewidth=2, label='GP posterior mean')
    ax1.fill_between(x_np, gp_mean - 2*gp_std, gp_mean + 2*gp_std, alpha=0.3, color='blue', label='GP +/- 2 std')

    # Mark training data
    train_x_np = train_x.cpu().numpy()
    train_y_np = train_y.cpu().numpy()
    ax1.scatter(train_x_np, np.log(train_y_np + 1), c='red', s=100, zorder=5, marker='o', label='Training data (log scale)')

    # Mark set members with vertical lines
    set_np = px_set.get_set().cpu().numpy()
    for i, x_set in enumerate(set_np):
        label = 'X_SET members' if i == 0 else None
        ax1.axvline(x=x_set, color='orange', linestyle='-', linewidth=2, alpha=0.7, label=label)

    ax1.set_xlabel('x')
    ax1.set_ylabel('lambda(x)')
    ax1.set_title('GP Posterior (trained OUTSIDE set region)')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)

    # Panel 2: Standard Utility
    ax2 = axes[0, 1]
    ax2.plot(x_np, utility_std_np, 'g-', linewidth=2)
    ax2.fill_between(x_np, 0, utility_std_np, alpha=0.3, color='green')
    ax2.scatter([std_analysis['global_max_x']], [std_analysis['global_max_utility']],
                c='darkgreen', s=150, zorder=5, marker='*',
                label=f'Max at x={std_analysis["global_max_x"]:.2f}')

    # Mark set members
    for i, x_set in enumerate(set_np):
        label = 'X_SET members' if i == 0 else None
        ax2.axvline(x=x_set, color='orange', linestyle='-', linewidth=2, alpha=0.7, label=label)

    ax2.set_xlabel('x')
    ax2.set_ylabel('Standard Utility')
    ax2.set_title('Standard Utility (ignores p(x))')
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)

    # Panel 3: Distribution-Aware Utility
    ax3 = axes[1, 0]
    ax3.plot(x_np, utility_da_np, 'b-', linewidth=2)
    ax3.fill_between(x_np, 0, utility_da_np, alpha=0.3, color='blue')
    ax3.scatter([da_analysis['global_max_x']], [da_analysis['global_max_utility']],
                c='darkblue', s=150, zorder=5, marker='*',
                label=f'Max at x={da_analysis["global_max_x"]:.2f}')

    # Mark set members
    for i, x_set in enumerate(set_np):
        label = 'X_SET members' if i == 0 else None
        ax3.axvline(x=x_set, color='orange', linestyle='-', linewidth=2, alpha=0.7, label=label)

    ax3.set_xlabel('x')
    ax3.set_ylabel('Distribution-Aware Utility')
    ax3.set_title(f'Distribution-Aware Utility (p(x) = finite set)')
    ax3.legend(loc='upper right')
    ax3.grid(True, alpha=0.3)

    # Panel 4: Comparison (normalized overlay)
    ax4 = axes[1, 1]

    # Normalize both utilities to [0, 1] for comparison
    utility_std_norm = (utility_std_np - utility_std_np.min()) / (utility_std_np.max() - utility_std_np.min() + 1e-8)
    utility_da_norm = (utility_da_np - utility_da_np.min()) / (utility_da_np.max() - utility_da_np.min() + 1e-8)

    ax4.plot(x_np, utility_std_norm, 'g-', linewidth=2, alpha=0.7, label='Standard (normalized)')
    ax4.plot(x_np, utility_da_norm, 'b-', linewidth=2, alpha=0.7, label='Distribution-Aware (normalized)')

    # Mark maxima
    ax4.scatter([std_analysis['global_max_x']], [utility_std_norm[np.argmax(utility_std_np)]],
                c='darkgreen', s=150, zorder=5, marker='*')
    ax4.scatter([da_analysis['global_max_x']], [utility_da_norm[np.argmax(utility_da_np)]],
                c='darkblue', s=150, zorder=5, marker='*')

    # Mark set members
    for i, x_set in enumerate(set_np):
        label = 'X_SET members' if i == 0 else None
        ax4.axvline(x=x_set, color='orange', linestyle='-', linewidth=2, alpha=0.7, label=label)

    # Verdict in title
    if da_analysis['passes']:
        verdict_str = "PASS: DA peaks near set"
    elif comparison['da_closer_to_set']:
        verdict_str = "PARTIAL: DA closer to set"
    else:
        verdict_str = "CHECK: DA not closer to set"

    ax4.set_xlabel('x')
    ax4.set_ylabel('Normalized Utility')
    ax4.set_title(f'Comparison - {verdict_str}')
    ax4.legend(loc='upper right')
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save figure
    save_path = Path(__file__).parent / 'test_finite_set_result.png'
    plt.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {save_path}")

    # Final verdict
    print("\n" + "=" * 70)
    print("VERDICT:")
    print("=" * 70)
    if da_analysis['passes']:
        print("PASS: Distribution-aware utility correctly peaks near/at set members.")
        print(f"      Max at x={da_analysis['global_max_x']:.3f}, distance to set: {da_analysis['dist_to_nearest_set_member']:.3f}")
    elif comparison['da_closer_to_set']:
        print("PARTIAL PASS: Distribution-aware utility is closer to set than standard utility,")
        print("              but does not peak directly at a set member.")
        print(f"      Standard max at x={std_analysis['global_max_x']:.3f} (dist: {std_analysis['dist_to_nearest_set_member']:.3f})")
        print(f"      DA max at x={da_analysis['global_max_x']:.3f} (dist: {da_analysis['dist_to_nearest_set_member']:.3f})")
    else:
        print("CHECK: Results may not match expectations. Review the plot.")
        print(f"      Standard max at x={std_analysis['global_max_x']:.3f} (dist: {std_analysis['dist_to_nearest_set_member']:.3f})")
        print(f"      DA max at x={da_analysis['global_max_x']:.3f} (dist: {da_analysis['dist_to_nearest_set_member']:.3f})")


if __name__ == "__main__":
    main()
