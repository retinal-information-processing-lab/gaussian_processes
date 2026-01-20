"""
Test to empirically verify how GPyTorch interprets variational parameters
in VariationalStrategy vs UnwhitenedVariationalStrategy.
"""

import torch
import gpytorch
from gpytorch.variational import (
    CholeskyVariationalDistribution,
    VariationalStrategy,
    UnwhitenedVariationalStrategy
)

# Set seed for reproducibility
torch.manual_seed(42)

# Create simple test data
n_samples = 50
n_inducing = 10
n_features = 5

X = torch.randn(n_samples, n_features, dtype=torch.float64)
inducing_points = torch.randn(n_inducing, n_features, dtype=torch.float64)
X_test = torch.randn(5, n_features, dtype=torch.float64)

print("=" * 80)
print("Test: VariationalStrategy vs UnwhitenedVariationalStrategy Parameter Interpretation")
print("=" * 80)

# Define a simple kernel
class SimpleModel(gpytorch.models.ApproximateGP):
    def __init__(self, inducing_points, variational_strategy_class):
        variational_distribution = CholeskyVariationalDistribution(inducing_points.size(0))
        variational_strategy = variational_strategy_class(
            self,
            inducing_points,
            variational_distribution,
            learn_inducing_locations=False
        )
        super().__init__(variational_strategy)
        self.mean_module = gpytorch.means.ZeroMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    def forward(self, x):
        mean = self.mean_module(x)
        covar = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean, covar)

# Create two models
print("\n1. Creating models...")
model_whitened = SimpleModel(inducing_points.clone(), VariationalStrategy).double()
model_unwhitened = SimpleModel(inducing_points.clone(), UnwhitenedVariationalStrategy).double()

print(f"   Whitened model strategy: {type(model_whitened.variational_strategy).__name__}")
print(f"   Unwhitened model strategy: {type(model_unwhitened.variational_strategy).__name__}")

# Get the variational distributions
var_dist_whitened = model_whitened.variational_strategy.variational_distribution
var_dist_unwhitened = model_unwhitened.variational_strategy.variational_distribution

print("\n2. Examining variational distribution parameters...")
print(f"   Whitened - variational_mean shape: {var_dist_whitened.variational_mean.shape}")
print(f"   Whitened - chol_variational_covar shape: {var_dist_whitened.chol_variational_covar.shape}")
print(f"   Unwhitened - variational_mean shape: {var_dist_unwhitened.variational_mean.shape}")
print(f"   Unwhitened - chol_variational_covar shape: {var_dist_unwhitened.chol_variational_covar.shape}")

# Get initial mean and covar from both
m_whitened_init = var_dist_whitened.variational_mean.clone()
L_whitened_init = var_dist_whitened.chol_variational_covar.clone()

m_unwhitened_init = var_dist_unwhitened.variational_mean.clone()
L_unwhitened_init = var_dist_unwhitened.chol_variational_covar.clone()

# Compute V from L
V_whitened_init = L_whitened_init @ L_whitened_init.T
V_unwhitened_init = L_unwhitened_init @ L_unwhitened_init.T

print("\n3. Initial variational parameters (before modification)...")
print(f"   Whitened mean norm: {m_whitened_init.norm().item():.6f}")
print(f"   Whitened L norm: {L_whitened_init.norm().item():.6f}")
print(f"   Unwhitened mean norm: {m_unwhitened_init.norm().item():.6f}")
print(f"   Unwhitened L norm: {L_unwhitened_init.norm().item():.6f}")

# Get the K_tilde matrix (inducing kernel)
# We need to call the model to get the covariance structure
with torch.no_grad():
    posterior_whitened = model_whitened(inducing_points)
    K_tilde = posterior_whitened.covariance_matrix
    K_tilde_diag = K_tilde.diag()

print("\n4. Inducing point kernel K_tilde...")
print(f"   K_tilde shape: {K_tilde.shape}")
print(f"   K_tilde diag (first 5): {K_tilde_diag[:5]}")

# Set the SAME variational parameters in both models
print("\n5. Setting SAME variational parameters in both models...")

# Create a new mean and covariance
m_new = torch.ones(n_inducing, dtype=torch.float64) * 0.5
V_new = torch.eye(n_inducing, dtype=torch.float64) * 2.0
L_new = torch.linalg.cholesky(V_new)

print(f"   New mean: ones * 0.5, norm = {m_new.norm().item():.6f}")
print(f"   New V: 2*I, shape = {V_new.shape}")
print(f"   New L (Cholesky of V): shape = {L_new.shape}")

# Set parameters in both models
with torch.no_grad():
    var_dist_whitened.variational_mean.copy_(m_new)
    var_dist_whitened.chol_variational_covar.copy_(L_new)
    
    var_dist_unwhitened.variational_mean.copy_(m_new)
    var_dist_unwhitened.chol_variational_covar.copy_(L_new)

print("   ✓ Parameters set in both models")

# Now compare model outputs
print("\n6. Comparing model outputs on test data...")
print(f"   Test input shape: {X_test.shape}")

model_whitened.eval()
model_unwhitened.eval()

with torch.no_grad():
    output_whitened = model_whitened(X_test)
    output_unwhitened = model_unwhitened(X_test)

mean_whitened = output_whitened.mean
mean_unwhitened = output_unwhitened.mean
var_whitened = output_whitened.variance
var_unwhitened = output_unwhitened.variance

print(f"\n   Whitened model output:")
print(f"      Mean: {mean_whitened}")
print(f"      Variance: {var_whitened}")

print(f"\n   Unwhitened model output:")
print(f"      Mean: {mean_unwhitened}")
print(f"      Variance: {var_unwhitened}")

# Compute differences
mean_diff = (mean_whitened - mean_unwhitened).abs()
var_diff = (var_whitened - var_unwhitened).abs()

print(f"\n   Absolute differences:")
print(f"      Mean difference (max): {mean_diff.max().item():.2e}")
print(f"      Mean difference (mean): {mean_diff.mean().item():.2e}")
print(f"      Variance difference (max): {var_diff.max().item():.2e}")
print(f"      Variance difference (mean): {var_diff.mean().item():.2e}")

# Check if they're essentially the same
tolerance = 1e-6
means_same = (mean_diff.max() < tolerance).item()
vars_same = (var_diff.max() < tolerance).item()

print("\n" + "=" * 80)
print("CONCLUSION:")
print("=" * 80)
if means_same and vars_same:
    print("✓ SAME: Both strategies produce IDENTICAL outputs when given identical parameters")
    print("  → They interpret (m, L) in the SAME way")
    print("  → The whitened/unwhitened distinction might only affect the OPTIMIZATION dynamics")
else:
    print("✗ DIFFERENT: Strategies produce DIFFERENT outputs despite identical parameters")
    print(f"  → Whitening interprets parameters differently")
    print(f"  → Mean difference: {mean_diff.max().item():.2e}")
    print(f"  → Variance difference: {var_diff.max().item():.2e}")

print("=" * 80)

# Additional investigation: Check the actual forward computation
print("\n7. Investigating internal representations...")

# For whitened: u ~ N(m, V), f|u = f_u, so variational params are in u-space
# For unwhitened: f ~ N(m_f, V_f), so variational params are in f-space

# Get the actual variational distributions
print(f"\n   Whitened strategy - internal variable name: {type(model_whitened.variational_strategy).__name__}")
print(f"   Unwhitened strategy - internal variable name: {type(model_unwhitened.variational_strategy).__name__}")

# Check if there's a whitening matrix
if hasattr(model_whitened.variational_strategy, '_whiten'):
    print(f"   Whitened has _whiten: {model_whitened.variational_strategy._whiten}")
if hasattr(model_unwhitened.variational_strategy, '_whiten'):
    print(f"   Unwhitened has _whiten: {model_unwhitened.variational_strategy._whiten}")

