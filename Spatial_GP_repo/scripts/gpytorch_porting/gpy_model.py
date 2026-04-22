"""
Variational GP Model for GPyTorch

This module implements the sparse variational GP model using GPyTorch's
ApproximateGP framework with the arc-cosine kernel.
"""

import numpy as np
import torch
from _constants import JITTER
import gpytorch
from gpytorch.models import ApproximateGP
from gpytorch.variational import (
    CholeskyVariationalDistribution,
    TrilNaturalVariationalDistribution,
    VariationalStrategy,
    UnwhitenedVariationalStrategy,
)


class VariationalGPModel(ApproximateGP):
    """Sparse Variational GP model.

    Uses inducing points for scalability. Three variational-distribution
    parameterizations are supported:

    - `CholeskyVariationalDistribution` (default) — standard whitened
      parameterization, optimized jointly with hyperparams via LBFGS/Adam.
    - `TrilNaturalVariationalDistribution` — Tril-natural parameters, must
      be optimized by `gpytorch.optim.NGD`. Used by `mode='ngd'`.
    - `None` (unwhitened strategy) — see `standard_variational_distribution=False`.

    Parameters
    ----------
    inducing_points : Tensor, shape (n_inducing, n_features)
    kernel : gpytorch.kernels.Kernel
    jitter : float
        Jitter to add for numerical stability.
    standard_variational_distribution : bool
        If True (default), use `VariationalStrategy` (whitened).
        If False, use `UnwhitenedVariationalStrategy`.
    variational_distribution_cls : {'cholesky', 'tril_natural'}, default 'cholesky'
        Which variational-distribution class to instantiate. Only used when
        `standard_variational_distribution=True`. `'tril_natural'` requires
        updates via `gpytorch.optim.NGD` — plain Adam/LBFGS would break PSD
        constraints (see natural_variational_distribution.py:103-107).
    learn_inducing_locations : bool

    Attributes
    ----------
    variational_strategy : VariationalStrategy or UnwhitenedVariationalStrategy
    mean_module : ZeroMean
    covar_module : Kernel
    variational_distribution_kind : {'cholesky', 'tril_natural'}
        Flag for downstream code (e.g. training loop) to pick the right
        optimizer. Replaces the old `standard_variational_distribution`
        bool-only flag, which conflated strategy and distribution choice.
    """

    def __init__(self, inducing_points, kernel, jitter, standard_variational_distribution,
                 learn_inducing_locations=False,
                 variational_distribution_cls='cholesky'):
        if variational_distribution_cls == 'cholesky':
            variational_distribution = CholeskyVariationalDistribution(
                inducing_points.size(0)
            )
        elif variational_distribution_cls == 'tril_natural':
            if not standard_variational_distribution:
                raise ValueError(
                    "variational_distribution_cls='tril_natural' requires "
                    "standard_variational_distribution=True (UnwhitenedVariationalStrategy "
                    "is not supported for natural params)."
                )
            variational_distribution = TrilNaturalVariationalDistribution(
                num_inducing_points=inducing_points.size(0)
            )
        else:
            raise ValueError(
                f"Unknown variational_distribution_cls: {variational_distribution_cls!r}. "
                "Use 'cholesky' or 'tril_natural'."
            )

        # Variational strategy: how to compute q(f) from q(u)
        # IMPORTANT: Pass jitter_val to ensure GPyTorch uses the same jitter as our code
        if standard_variational_distribution:
            variational_strategy = VariationalStrategy(
                self,
                inducing_points,
                variational_distribution,
                learn_inducing_locations=learn_inducing_locations,
                jitter_val=jitter
            )
        else:
            variational_strategy = UnwhitenedVariationalStrategy(
                self,
                inducing_points,
                variational_distribution,
                learn_inducing_locations=learn_inducing_locations,
                jitter_val=jitter
            )

        super().__init__(variational_strategy)

        # Store strategy type for downstream code (e.g., estep.py needs to know
        # whether to do explicit L_K whitening conversions)
        self.standard_variational_distribution = standard_variational_distribution
        # Track which variational-distribution class was used; training loops
        # (e.g. train_ngd) inspect this to pick the right optimizer.
        self.variational_distribution_kind = variational_distribution_cls

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

        # NOTE: No jitter added here. GPyTorch's VariationalStrategy handles
        # jitter internally via jitter_val (passed in __init__), adding it to
        # K_uu before Cholesky and to K_XX for predictive covariance.

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
    # Assume square image
    n_px_side = int(np.sqrt(n_features))  # 10x10 image
    kernel = ArcCosineKernel(n_px_side=n_px_side, sigma_0=1.0,
                             beta=0.1, rho=0.1, eps_0x=0.0, eps_0y=0.0)
    model = VariationalGPModel(inducing_points, kernel,
                               jitter=JITTER, standard_variational_distribution=True)

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
