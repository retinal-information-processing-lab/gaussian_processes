"""
Variational GP Model for GPyTorch

This module implements the sparse variational GP model using GPyTorch's
ApproximateGP framework with the arc-cosine kernel.
"""

import torch
import gpytorch
from gpytorch.models import ApproximateGP
from gpytorch.variational import CholeskyVariationalDistribution, VariationalStrategy


class VariationalGPModel(ApproximateGP):
    """Sparse Variational GP model.

    Uses inducing points for scalability and CholeskyVariationalDistribution
    for the variational posterior q(λ̃) = N(m, V).

    Parameters
    ----------
    inducing_points : Tensor, shape (n_inducing, n_features)
        Locations of inducing points
    kernel : gpytorch.kernels.Kernel
        Kernel function to use (e.g., ArcCosineKernel)
    learn_inducing_locations : bool
        Whether to optimize inducing point locations (default: False)
    jitter : float
        Jitter to add for numerical stability (default: 1e-4)

    Attributes
    ----------
    variational_strategy : VariationalStrategy
        GPyTorch's strategy for computing q(f) from q(u)
    mean_module : ZeroMean
        Mean function (zero for our model)
    covar_module : Kernel
        Covariance function
    """

    def __init__(self, inducing_points, kernel, learn_inducing_locations=False, jitter=1e-4):
        # Variational distribution q(u) = N(m, LLᵀ)
        # Uses Cholesky parameterization for numerical stability
        variational_distribution = CholeskyVariationalDistribution(
            inducing_points.size(0)
        )

        # Variational strategy: how to compute q(f) from q(u)
        # IMPORTANT: Pass jitter_val to ensure GPyTorch uses the same jitter as our code
        variational_strategy = VariationalStrategy(
            self,
            inducing_points,
            variational_distribution,
            learn_inducing_locations=learn_inducing_locations,
            jitter_val=jitter
        )

        super().__init__(variational_strategy)

        # Mean function: zero mean (as in custom implementation)
        self.mean_module = gpytorch.means.ZeroMean()

        # Covariance function
        self.covar_module = kernel

        # Store jitter for adding stability
        self.jitter = jitter

    def forward(self, x):
        """Compute the GP prior distribution at input points x.

        Args:
            x: Input tensor, shape (n_points, n_features)

        Returns:
            MultivariateNormal distribution representing the GP prior at x
        """
        mean = self.mean_module(x)
        covar = self.covar_module(x)

        # Add jitter for numerical stability
        if self.jitter > 0:
            covar = covar.add_jitter(self.jitter)

        return gpytorch.distributions.MultivariateNormal(mean, covar)

    def get_variational_parameters(self):
        """Get the variational parameters (m, V).

        Returns a dictionary with:
        - 'mean': variational mean m, shape (n_inducing,)
        - 'covar': variational covariance V, shape (n_inducing, n_inducing)
        """
        var_dist = self.variational_strategy.variational_distribution
        return {
            'mean': var_dist.mean,
            'covar': var_dist.covariance_matrix
        }

    def get_inducing_points(self):
        """Get the inducing point locations."""
        return self.variational_strategy.inducing_points


def test_model():
    """Basic test for VariationalGPModel."""
    from kernels import ArcCosineKernel

    torch.manual_seed(42)

    # Create small test data
    n_train = 50
    n_inducing = 10
    n_features = 100

    X = torch.randn(n_train, n_features)

    # Select inducing points as subset of training data
    indices = torch.randperm(n_train)[:n_inducing]
    inducing_points = X[indices]

    # Create kernel and model
    kernel = ArcCosineKernel(sigma_0=1.0)
    model = VariationalGPModel(inducing_points, kernel)

    # Test forward pass
    model.eval()
    with torch.no_grad():
        output = model(X[:5])
        print(f"Output type: {type(output)}")
        print(f"Mean shape: {output.mean.shape}")
        print(f"Variance shape: {output.variance.shape}")
        print(f"Mean values: {output.mean}")
        print(f"Variance values: {output.variance}")

    # Test variational parameters
    var_params = model.get_variational_parameters()
    print(f"\nVariational mean shape: {var_params['mean'].shape}")
    print(f"Variational covar shape: {var_params['covar'].shape}")

    # Check inducing points
    inducing = model.get_inducing_points()
    print(f"\nInducing points shape: {inducing.shape}")

    # Test train mode
    model.train()
    output = model(X[:5])
    print(f"\nTrain mode output type: {type(output)}")

    print("\nPASS: VariationalGPModel test passed!")
    return True


if __name__ == '__main__':
    test_model()
