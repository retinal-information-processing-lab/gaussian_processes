"""
GP Utility Playground - Created by Claude
==========================================
A simple 1D Gaussian Process playground with Poisson likelihood.

USE: Visualize the utility funciton shape as a snapshot after training the gp on some samples.

Uses GPyTorch with a custom Poisson likelihood and the nd_utility acquisition function.
"""

import sys
import time
import torch
import gpytorch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Add Spatial_GP_repo to path for imports (two levels up from 1D_playground)
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from utility import nd_utility_new, nd_utility_NUMERICAL

import warnings
warnings.filterwarnings("ignore", message=".*torch.cuda.*DtypeTensor.*")
warnings.filterwarnings("ignore", message=".*torch.sparse.SparseTensor.*")

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float64
torch.set_default_dtype(DTYPE)

# X_MIN, X_MAX = -2.0, 2.0  # Input domain bounds

X_MIN, X_MAX = -15.0, 15.0  # Input domain bounds


N_INDUCING = 20          # Number of inducing points for variational GP
MAX_R = 500              # Max spike count for utility computation

# Default p(x) distribution parameters for distribution-aware utility
DEFAULT_P_X_MEAN = 3.0
DEFAULT_P_X_STD = 1.5

# -----------------------------------------------------------------------------
# Default parameters for diagnostic scripts (diagnose_1d_utility_w_fixed_ntrain.py, etc.)
# These are NOT used in this script - they are exported for import by other scripts
# that analyze conditioning behavior (before/after observing lambda at a point).
# -----------------------------------------------------------------------------
# X_SAMPLE: The x-location where we imagine observing a new response.
#           Diagnostic scripts show how conditioning on lambda(X_SAMPLE) updates the GP.
#           Not used here
X_SAMPLE = DEFAULT_P_X_MEAN

# LAMBDA_SEED: Random seed for sampling lambda_obs from GP posterior at X_SAMPLE.
#              Ensures reproducibility across runs.
LAMBDA_SEED = 12

# N_LAMBDA_SAMPLES: Number of MC samples for averaging conditional entropy.
#                   Higher = more accurate utility estimate, but slower.
N_LAMBDA_SAMPLES = 1000

# SAMPLE_X: Controls how x is chosen in the MC loop for utility computation.
#           True  = sample x ~ p(x) (Gaussian) - "distribution-aware" utility
#           False = use fixed x = X_SAMPLE - utility at a single point
# SAMPLE_X = True
SAMPLE_X = False

# -----------------------------------------------------------------------------
# Ground Truth Function
# -----------------------------------------------------------------------------
def lambda_sin(x):
    """True latent function. GP learns this from Poisson observations.

    f(x) = exp(lambda(x)) is the firing rate.
    Keep lambda in reasonable range to avoid numerical issues with exp().
    """
    # Simple sinusoidal - gives firing rates roughly in [0.5, 7]
    return (torch.sin(2 * np.pi * x) + 1.0)*2

def lambda_true_asymmetric(x):
    """
    Asymmetric bump: peak at x=0.3, steep left side, gradual right decay.
    Looks like a skewed distribution (not periodic).
    """
    peak = 0.3
    sigma_left = 0.1    # Steep rise
    sigma_right = 0.35  # Gradual decay
    amplitude = 3.     # Peak firing rate ~ exp(amplitude) ≈ 400 <- 6
    baseline = 0.5      # Minimum firing rate ~ exp(baseline) ≈ 1.6

    # Asymmetric Gaussian shape
    left_mask = x < peak
    z = torch.zeros_like(x)
    z[left_mask] = ((x[left_mask] - peak) / sigma_left) ** 2
    z[~left_mask] = ((x[~left_mask] - peak) / sigma_right) ** 2

    # λ(x) so that f(x) = exp(λ) has the bump shape
    lam = baseline + (amplitude - baseline) * torch.exp(-0.5 * z)
    return lam

lambda_true = lambda_true_asymmetric
# lambda_true = lambda_sin

def generate_poisson_data(x, lambda_fn):
    """Generate Poisson spike counts from the true latent function."""
    lam = lambda_fn(x)
    rates = torch.exp(lam)
    # Sample from Poisson distribution
    counts = torch.poisson(rates)
    return counts


# -----------------------------------------------------------------------------
# Custom Poisson Likelihood for GPyTorch
# -----------------------------------------------------------------------------
class PoissonLikelihood(gpytorch.likelihoods.Likelihood):
    """Poisson likelihood for count data with log-link.

    Model: y ~ Poisson(exp(f)) where f is the GP latent function.

    For variational inference, we need E_q(f)[log p(y|f)] where q(f) = N(mu, sigma^2).
    This has closed form:
        E[log p(y|f)] = E[y*f - exp(f) - log(y!)]
                      = y*mu - exp(mu + sigma^2/2) - log(y!)
    """

    def expected_log_prob(self, target, input):
        """Expected log probability under the variational distribution.

        Args:
            target: Observed counts (batch_shape)
            input: GP output distribution (MultivariateNormal)

        Returns:
            Expected log probability (scalar after sum)
        """
        mean = input.mean
        var = input.variance

        # E[y*f - exp(f)] = y*mu - exp(mu + sigma^2/2)
        # Note: we omit -log(y!) as it's constant w.r.t. parameters
        log_prob = target * mean - torch.exp(mean + var / 2)
        return log_prob.sum(-1)

    def forward(self, function_samples):
        """Return Poisson distribution given function samples."""
        return torch.distributions.Poisson(rate=torch.exp(function_samples))


# -----------------------------------------------------------------------------
# Variational GP Model
# -----------------------------------------------------------------------------
class VariationalGP(gpytorch.models.ApproximateGP):
    """Simple variational GP with RBF kernel."""

    def __init__(self, inducing_points, jitter=1e-4):
        # Variational distribution q(u) over inducing point values
        variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(
            inducing_points.size(0)
        )

        # Variational strategy: how to compute q(f) from q(u)
        variational_strategy = gpytorch.variational.VariationalStrategy(
            self,
            inducing_points,
            variational_distribution,
            learn_inducing_locations=False
        )

        super().__init__(variational_strategy)

        # Mean and kernel
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel()
        )
        self.jitter = jitter  # Store jitter for numerical stability

    def forward(self, x):
        mean = self.mean_module(x)
        covar = self.covar_module(x)
        # Add jitter to diagonal for numerical stability with many inducing points
        covar = covar.add_jitter(self.jitter)
        return gpytorch.distributions.MultivariateNormal(mean, covar)


# -----------------------------------------------------------------------------
# Training
# -----------------------------------------------------------------------------
def compute_elbo(model, likelihood, train_x, train_y):
    """Compute variational ELBO manually.

    ELBO = E_q[log p(y|f)] - KL(q(u) || p(u))

    This is more explicit than using gpytorch.mlls.VariationalELBO.
    """
    # Get variational distribution q(f) at training points
    output = model(train_x)

    # Expected log likelihood: E_q[log p(y|f)]
    # For Poisson: E[y*f - exp(f)] = y*mu - exp(mu + sigma^2/2)
    mean = output.mean
    var = output.variance
    expected_log_lik = (train_y * mean - torch.exp(mean + var / 2)).sum()

    # KL divergence: KL(q(u) || p(u))
    kl_div = model.variational_strategy.kl_divergence().sum()

    # ELBO = expected log likelihood - KL divergence
    elbo = expected_log_lik - kl_div

    return elbo, expected_log_lik, kl_div


def train_gp(model, likelihood, train_x, train_y, n_iterations=500, lr=0.1):
    """Train the variational GP by maximizing ELBO."""
    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Note: utility.py sets torch.set_grad_enabled(False) at module level,
    # so we need to explicitly enable gradients here
    with torch.enable_grad():
        for i in range(n_iterations):
            optimizer.zero_grad()

            elbo, ell, kl = compute_elbo(model, likelihood, train_x, train_y)
            loss = -elbo  # Minimize negative ELBO

            loss.backward()
            optimizer.step()

            if (i + 1) % 100 == 0:
                print(f"Iter {i+1}/{n_iterations}, ELBO: {elbo.item():.2f} "
                      f"(ELL: {ell.item():.2f}, KL: {kl.item():.2f})")

    # Print learned hyperparameters
    lengthscale = model.covar_module.base_kernel.lengthscale.item()
    outputscale = model.covar_module.outputscale.item()
    mean_const = model.mean_module.constant.item()
    print(f"\nLearned hyperparameters:")
    print(f"  lengthscale = {lengthscale:.4f}")
    print(f"  outputscale = {outputscale:.4f}")
    print(f"  mean = {mean_const:.4f}")

    return model, likelihood


def train_gp_fixed_lengthscale(model, likelihood, train_x, train_y,
                                n_iterations=500, lr=None,
                                fixed_lengthscale=0.2, fixed_outputscale=1.0):
    """Train GP but keep lengthscale and outputscale fixed.

    Args:
        model: VariationalGP model
        likelihood: PoissonLikelihood
        train_x: Training inputs
        train_y: Training outputs (observed counts)
        n_iterations: Number of optimization iterations
        lr: Learning rate (if None, adaptive based on number of inducing points)
        fixed_lengthscale: Fixed lengthscale value
        fixed_outputscale: Fixed outputscale value

    Returns:
        model, likelihood: Trained model and likelihood
    """
    # Set and freeze lengthscale
    model.covar_module.base_kernel.lengthscale = fixed_lengthscale
    model.covar_module.base_kernel.raw_lengthscale.requires_grad = False

    # Set and freeze outputscale
    model.covar_module.outputscale = fixed_outputscale
    model.covar_module.raw_outputscale.requires_grad = False

    model.train()

    # Adaptive learning rate based on number of inducing points
    n_inducing = len(model.variational_strategy.inducing_points)
    if lr is None:
        lr = 0.1 if n_inducing <= 50 else 0.05 if n_inducing <= 80 else 0.01
    print(f"  Using lr={lr} for {n_inducing} inducing points")

    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr
    )

    with torch.enable_grad():
        for i in range(n_iterations):
            optimizer.zero_grad()

            output = model(train_x)
            mean = output.mean
            var = output.variance

            # ELBO components
            expected_log_lik = (train_y * mean - torch.exp(mean + var / 2)).sum()
            kl_div = model.variational_strategy.kl_divergence().sum()
            elbo = expected_log_lik - kl_div

            loss = -elbo
            loss.backward()
            optimizer.step()

            if (i + 1) % 200 == 0:
                print(f"Iter {i+1}/{n_iterations}, ELBO: {elbo.item():.2f}")

    return model, likelihood


# -----------------------------------------------------------------------------
# GP Moment Functions
# -----------------------------------------------------------------------------
def get_marginal_moments(model, x_star):
    """Get marginal GP posterior moments at query points.

    Args:
        model: Trained GP model
        x_star: Query points (n_points,)

    Returns:
        mu: (n_points,) posterior means
        sigma2: (n_points,) posterior variances
    """
    model.eval()
    with torch.no_grad():
        posterior = model(x_star)
        return posterior.mean, posterior.variance


def get_conditional_moments(model, x_star, x_sample, lambda_sample):
    """Compute conditional GP moments after observing lambda(x_sample) = lambda_sample.

    Uses GPyTorch's covariance_matrix for Gaussian conditioning.

    Args:
        model: Trained GP model
        x_star: Query points (K,)
        x_sample: Single observation point (scalar)
        lambda_sample: Observed lambda value at x_sample (scalar)

    Returns:
        mu_cond: (K,) conditional posterior means
        sigma2_cond: (K,) conditional posterior variances
    """
    model.eval()
    x_sample_tensor = torch.tensor([x_sample], dtype=x_star.dtype, device=x_star.device)

    with torch.no_grad():
        all_x = torch.cat([x_sample_tensor, x_star])
        posterior = model(all_x)
        full_covar = posterior.covariance_matrix

        mu_sample = posterior.mean[0]
        var_sample = full_covar[0, 0]
        mu_star = posterior.mean[1:]
        var_star = full_covar.diag()[1:]
        cross_cov = full_covar[0, 1:]

        innovation = lambda_sample - mu_sample
        mu_cond = mu_star + cross_cov * (innovation / var_sample)
        sigma2_cond = var_star - (cross_cov ** 2) / var_sample
        sigma2_cond = torch.clamp(sigma2_cond, min=1e-8)

    return mu_cond, sigma2_cond


# -----------------------------------------------------------------------------
# Entropy Computation
# -----------------------------------------------------------------------------
def compute_H(mu, sigma2, r_max=MAX_R, a=1.0, lambda0=0.0):
    """Compute entropy H(R | mu, sigma2) using Laplace approximation.

    Args:
        mu: (n_points,) latent GP means
        sigma2: (n_points,) latent GP variances (must be non-negative)
        r_max: Maximum spike count for Laplace approximation
        a: Firing rate scaling (f = exp(a*lambda + lambda0))
        lambda0: Firing rate offset

    Returns:
        H: (n_points,) tensor of entropies

    Raises:
        ValueError: If sigma2 contains any negative values
    """
    # Validate variance is non-negative
    if torch.any(sigma2 < 0):
        raise ValueError(f"sigma2 must be non-negative. Found min value: {sigma2.min().item()}")

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
# Helper Functions
# -----------------------------------------------------------------------------
def remove_near_duplicates(x, y, min_dist=1e-4):
    """Remove points closer than min_dist to keep kernel well-conditioned.

    Args:
        x: (N,) tensor of input locations (assumed sorted)
        y: (N,) tensor of corresponding observations
        min_dist: minimum allowed distance between points

    Returns:
        x_filtered, y_filtered: filtered tensors with near-duplicates removed
    """
    if len(x) <= 1:
        return x, y

    keep_mask = torch.ones(len(x), dtype=torch.bool, device=x.device)
    last_kept_idx = 0
    for i in range(1, len(x)):
        if x[i] - x[last_kept_idx] >= min_dist:
            last_kept_idx = i
        else:
            keep_mask[i] = False

    return x[keep_mask], y[keep_mask]


# -----------------------------------------------------------------------------
# Utility Evaluation ( distribution-aware utility )
# -----------------------------------------------------------------------------
def evaluate_distribution_aware_utility(model, x_candidates, n_mc_samples=1000,
                                        p_x_mean=0.0, p_x_std=0.2, r_max=MAX_R):
    """Evaluate distribution-aware utility at candidate points.

    U(x*) = H_marg(x*) - E[H_cond(x* | x, λ)]

    where expectation is over x ~ N(p_x_mean, p_x_std²), λ ~ q(λ|x).
    Always uses Monte Carlo sampling for the expectation.

    Args:
        model: Trained GP model
        x_candidates: (n_candidates,) query points
        n_mc_samples: Number of MC samples for expectation
        p_x_mean: Mean of Gaussian p(x) distribution
        p_x_std: Std of Gaussian p(x) distribution
        r_max: Max spike count for entropy computation

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
            # Sample x from Gaussian p(x)
            x_i = p_x_mean + p_x_std * torch.randn(1, dtype=dtype, device=device).item()
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
# Utility Evaluation ( standard nd_utility )
# -----------------------------------------------------------------------------

def evaluate_nd_utility_new(model, x_candidates, max_r=MAX_R):
    """
    Evaluate nd_utility_new (Laplace approximation) at candidate points.

    This uses the log-space Laplace approximation which is numerically stable
    and correctly normalized (Σp_Lap(r) ≈ 1.0) even for high uncertainty.
    """
    model.eval()

    with torch.no_grad():
        posterior = model(x_candidates)
        mu = posterior.mean
        sigma2 = posterior.variance

        utility = nd_utility_new(mu, sigma2, r_max=max_r)

    return utility

def evaluate_nd_utility_NUMERICAL(model, x_candidates, max_r=MAX_R, n_quadrature=100):
    """Evaluate nd_utility_NUMERICAL (Gauss-Hermite quadrature) at candidate points.
    
    Use this as ground truth to compare against Laplace approximation.
    """
    model.eval()

    with torch.no_grad():
        posterior = model(x_candidates)
        mu = posterior.mean
        sigma2 = posterior.variance

        utility = nd_utility_NUMERICAL(mu, sigma2, r_max=max_r, n_quadrature=n_quadrature)

    return utility

# -----------------------------------------------------------------------------
# Visualization
# -----------------------------------------------------------------------------
def plot_results(model, likelihood, train_x, train_y, lambda_fn, save_path=None,
                 p_x_mean=None, p_x_std=None):
    """Create visualization of GP fit and utility landscape.

    Args:
        model: Trained GP model
        likelihood: Likelihood object
        train_x: Training inputs
        train_y: Training outputs (observed counts)
        lambda_fn: Ground truth function
        save_path: Optional path to save figure. If None, uses default name.
        p_x_mean: Optional mean of p(x) Gaussian distribution (for distribution-aware utility)
        p_x_std: Optional std of p(x) Gaussian distribution (for distribution-aware utility)
    """
    model.eval()
    likelihood.eval()

    # Dense grid for plotting
    x_plot = torch.linspace(3*X_MIN, 3*X_MAX, 200, dtype=DTYPE, device=DEVICE)

    with torch.no_grad():
        # GP posterior
        posterior = model(x_plot)
        mean = posterior.mean
        std = posterior.variance.sqrt()

        # True function
        true_lambda = lambda_fn(x_plot)

        # Compute both utilities
        utility_standard = evaluate_nd_utility_new(model, x_plot)
        if p_x_mean is not None and p_x_std is not None:
            utility_distr_aware = evaluate_distribution_aware_utility(model, x_plot, p_x_mean=p_x_mean, p_x_std=p_x_std)
        else:
            utility_distr_aware = None

    # Convert to numpy for plotting
    x_np = x_plot.cpu().numpy()
    mean_np = mean.cpu().numpy()
    std_np = std.cpu().numpy()
    true_np = true_lambda.cpu().numpy()
    utility_std_np = utility_standard.cpu().numpy()
    utility_da_np = utility_distr_aware.cpu().numpy() if utility_distr_aware is not None else None
    train_x_np = train_x.cpu().numpy()
    train_y_np = train_y.cpu().numpy()

    # Compute firing rates: f(x) = exp(λ(x))
    true_rate_np = np.exp(true_np)
    pred_rate_np = np.exp(mean_np)
    # Approximate uncertainty in rate space (delta method: var(exp(x)) ≈ exp(2μ) * var(x))
    pred_rate_std_np = pred_rate_np * std_np

    fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)

    # Plot 1: GP posterior vs true function (log-rate / λ space)
    ax1 = axes[0]
    ax1.plot(x_np, true_np, 'k--', label='True λ(x)', linewidth=2)
    ax1.plot(x_np, mean_np, 'b-', label='GP mean', linewidth=2)
    ax1.fill_between(x_np, mean_np - 2*std_np, mean_np + 2*std_np,
                     alpha=0.3, color='blue', label='±2σ')

    for i, tx in enumerate(train_x_np):
        label = 'Training x' if i == 0 else None
        ax1.axvline(x=tx, color='red', alpha=0.5, linewidth=1.5, label=label)

    ax1.set_ylabel('λ(x) = log(f)')
    ax1.set_title('Latent Function (log firing rate)')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)

    # Plot 2: Firing rate space with observations
    ax2 = axes[1]
    ax2.plot(x_np, true_rate_np, 'k--', label='True f(x) = exp(λ)', linewidth=2)
    ax2.plot(x_np, pred_rate_np, 'b-', label='Predicted f(x)', linewidth=2)
    ax2.fill_between(x_np,
                     np.maximum(0, pred_rate_np - 2*pred_rate_std_np),
                     pred_rate_np + 2*pred_rate_std_np,
                     alpha=0.3, color='blue', label='±2σ (approx)')

    # Plot observed spike counts
    ax2.scatter(train_x_np, train_y_np, c='red', s=80, zorder=5,
                edgecolors='darkred', linewidths=1.5,
                label=f'Observed counts (n={len(train_y_np)})')

    ax2.set_ylabel('Firing rate / Spike count')
    ax2.set_title('Firing Rate Space (observations shown as red dots)')
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)

    # Plot 3: Utility landscape (both utilities on separate y-axes)
    ax3 = axes[2]

    # Add p(x) distribution as background (scaled to fit, no axis)
    if p_x_mean is not None and p_x_std is not None:
        p_x = np.exp(-0.5 * ((x_np - p_x_mean) / p_x_std) ** 2)
        p_x_scaled = p_x / p_x.max() * utility_std_np.max() * 0.5  # Scale to ~50% of standard utility max
        ax3.fill_between(x_np, 0, p_x_scaled, alpha=0.15, color='orange', zorder=1)

    # Standard utility on left y-axis (green)
    ax3.plot(x_np, utility_std_np, 'g-', linewidth=2, label='Standard utility')
    ax3.fill_between(x_np, 0, utility_std_np, alpha=0.2, color='green', zorder=2)
    ax3.set_ylabel('Standard Utility', color='green')
    ax3.tick_params(axis='y', labelcolor='green')

    # Distribution-aware utility on right y-axis (blue) if available
    if utility_da_np is not None:
        ax3_twin = ax3.twinx()
        ax3_twin.plot(x_np, utility_da_np, 'b-', linewidth=2, label='Distribution-aware utility')
        ax3_twin.fill_between(x_np, 0, utility_da_np, alpha=0.2, color='blue', zorder=2)
        ax3_twin.set_ylabel('Distribution-Aware Utility', color='blue')
        ax3_twin.tick_params(axis='y', labelcolor='blue')

    # Training point markers
    for tx in train_x_np:
        ax3.axvline(x=tx, color='red', alpha=0.3, linewidth=1, zorder=3)

    # Mark max utility points
    max_idx_std = np.argmax(utility_std_np)
    ax3.scatter([x_np[max_idx_std]], [utility_std_np[max_idx_std]], c='darkgreen', s=100,
                zorder=5, marker='*', label=f'Max standard at x={x_np[max_idx_std]:.2f}')

    if utility_da_np is not None:
        max_idx_da = np.argmax(utility_da_np)
        ax3_twin.scatter([x_np[max_idx_da]], [utility_da_np[max_idx_da]], c='darkblue', s=100,
                         zorder=5, marker='*', label=f'Max distr-aware at x={x_np[max_idx_da]:.2f}')

    ax3.set_xlabel('x')
    ax3.set_title('Utility Landscape: Standard (green, left) vs Distribution-Aware (blue, right)')
    ax3.grid(True, alpha=0.3)

    # Combined legend
    lines1, labels1 = ax3.get_legend_handles_labels()
    if utility_da_np is not None:
        lines2, labels2 = ax3_twin.get_legend_handles_labels()
        # Add p(x) placeholder to legend
        ax3.fill_between([], [], [], alpha=0.15, color='orange', label='p(x)')
        lines1, labels1 = ax3.get_legend_handles_labels()
        ax3.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=8)
    else:
        ax3.legend(loc='upper right', fontsize=8)

    plt.tight_layout()
    if save_path is None:
        save_path = Path(__file__).parent / 'gp_utility_playground_result.png'
    plt.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {save_path}")

    return fig


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    # Set seed for reproducibility (change this to get different random samples)
    SEED = 42
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    # print("=" * 60)
    # print(f"GP Utility Playground (seed={SEED})")
    # print("=" * 60)

    # Generate initial training data 
    n_train = 20
    train_x = torch.linspace(X_MIN, X_MAX, n_train, dtype=DTYPE, device=DEVICE)

    train_y = generate_poisson_data(train_x, lambda_true)

    inducing_points = train_x.clone()

    model = VariationalGP(inducing_points, jitter=0).to(DEVICE)
    likelihood = PoissonLikelihood().to(DEVICE)

    # Train
    print("\nTraining GP...")

    model, likelihood = train_gp(model, likelihood, train_x, train_y, n_iterations=500)

    # Evaluate both utilities
    print("\nEvaluating utilities across domain...")
    x_candidates = torch.linspace(X_MIN, X_MAX, 100, dtype=DTYPE, device=DEVICE)

    utility_standard = evaluate_nd_utility_new(model, x_candidates)
    utility_distr_aware = evaluate_distribution_aware_utility(
        model, x_candidates, p_x_mean=DEFAULT_P_X_MEAN, p_x_std=DEFAULT_P_X_STD
    )

    # Find max utility points
    max_idx_std = torch.argmax(utility_standard)
    max_idx_da = torch.argmax(utility_distr_aware)
    print(f"\nStandard utility max at x = {x_candidates[max_idx_std].item():.4f}")
    print(f"Distribution-aware utility max at x = {x_candidates[max_idx_da].item():.4f}")

    # Visualize
    print("\nPlotting results...")
    plot_results(model, likelihood, train_x, train_y, lambda_true,
                 p_x_mean=DEFAULT_P_X_MEAN, p_x_std=DEFAULT_P_X_STD)

    print("\nDone!")


if __name__ == "__main__":
    main()
