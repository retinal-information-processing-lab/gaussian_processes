"""
Poisson Likelihood for GPyTorch

This module implements the Poisson likelihood with exponential link function
for modeling neural spike count data.

Model: r ~ Poisson(f(x)) where f(x) = exp(A·λ(x) + λ₀)
- λ(x) is the latent GP function
- A is the gain parameter (learnable)
- λ₀ is the bias parameter (baseline log-firing rate, learnable)
"""

import torch
import gpytorch
from gpytorch.likelihoods import Likelihood
from gpytorch.constraints import Positive


class PoissonLikelihood(Likelihood):
    """Poisson likelihood with exponential link function.

    Model: r ~ Poisson(exp(A·λ + λ₀))

    For variational inference with q(λ) = N(μ, σ²), the expected log-likelihood is:
        E[log p(r|λ)] = E[r·(A·λ + λ₀) - exp(A·λ + λ₀) - log(r!)]
                      = r·(A·μ + λ₀) - exp(A·μ + A²σ²/2 + λ₀) - log(r!)

    We omit log(r!) since it's constant w.r.t. parameters.

    Parameters
    ----------
    A_init : float
        Initial value for gain parameter A (default: 1.0)
    lambda0_init : float
        Initial value for bias parameter λ₀ (default: 0.0)

    Attributes
    ----------
    raw_A : Parameter
        Unconstrained parameter (logA), where A = exp(raw_A)
    A : Property
        Constrained (positive) A value
    lambda0 : Parameter
        Bias parameter (unconstrained)
    """

    def __init__(self, A_init=1.0, lambda0_init=0.0):
        super().__init__()

        # Register A parameter with positivity constraint
        self.register_parameter(
            name='raw_A',
            parameter=torch.nn.Parameter(torch.zeros(1))
        )

        # Use exp/log transform (A = exp(raw_A), matching varGP's logA)
        self.register_constraint('raw_A', Positive(transform=torch.exp, inv_transform=torch.log))

        self.A = A_init  # Set via property to apply inverse transform

        # Register lambda0 parameter (unconstrained)
        self.register_parameter(
            name='lambda0',
            parameter=torch.nn.Parameter(torch.tensor([lambda0_init]))
        )

    @property
    def A(self):
        """Get the constrained A value (positive)."""
        return self.raw_A_constraint.transform(self.raw_A)

    @A.setter
    def A(self, value):
        """Set A via inverse transform."""
        if not torch.is_tensor(value):
            value = torch.as_tensor(value, dtype=torch.float)
        self.initialize(raw_A=self.raw_A_constraint.inverse_transform(value))

    def expected_log_prob(self, target, input):
        """Expected log probability under the variational distribution.

        Args:
            target: Observed spike counts, shape (n_samples,)
            input: GP output MultivariateNormal with mean and variance

        Returns:
            Expected log probability, summed over samples (scalar)
        """
        mu = input.mean        # (n_samples,)
        var = input.variance   # (n_samples,)

        A = self.A.squeeze()
        lambda0 = self.lambda0.squeeze()

        # E[r·(A·λ + λ₀) - exp(A·λ + λ₀)]
        # = r·(A·μ + λ₀) - exp(A·μ + A²σ²/2 + λ₀)
        log_prob = target * (A * mu + lambda0) - torch.exp(A * mu + 0.5 * A**2 * var + lambda0)

        return log_prob.sum(-1)

    def forward(self, function_samples):
        """Return Poisson distribution given function samples.

        Used for sampling from the likelihood.

        Args:
            function_samples: Samples of λ from the GP, shape (..., n_samples)

        Returns:
            Poisson distribution with rate = exp(A·λ + λ₀)
        """
        A = self.A.squeeze()
        lambda0 = self.lambda0.squeeze()
        rate = torch.exp(A * function_samples + lambda0)
        return torch.distributions.Poisson(rate=rate)


def test_likelihood():
    """Basic test for PoissonLikelihood."""
    import numpy as np

    # Create likelihood
    likelihood = PoissonLikelihood(A_init=0.5, lambda0_init=1.0)

    print(f"A: {likelihood.A.item():.4f}")
    print(f"lambda0: {likelihood.lambda0.item():.4f}")

    # Create fake GP output
    mean = torch.tensor([0.0, 1.0, 2.0])
    var = torch.tensor([0.5, 0.5, 0.5])

    # Create MultivariateNormal-like object
    class FakeGPOutput:
        def __init__(self, mean, variance):
            self.mean = mean
            self.variance = variance

    gp_output = FakeGPOutput(mean, var)
    target = torch.tensor([1.0, 5.0, 10.0])

    # Compute expected log prob
    log_prob = likelihood.expected_log_prob(target, gp_output)
    print(f"Expected log prob: {log_prob.item():.4f}")

    # Verify manually
    A = likelihood.A.squeeze()
    lambda0 = likelihood.lambda0.squeeze()
    manual = (target * (A * mean + lambda0) - torch.exp(A * mean + 0.5 * A**2 * var + lambda0)).sum()
    print(f"Manual computation: {manual.item():.4f}")

    assert torch.allclose(log_prob, manual), "Mismatch!"
    print("PASS: PoissonLikelihood test passed!")

    return True


if __name__ == '__main__':
    test_likelihood()
