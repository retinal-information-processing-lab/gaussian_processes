# GPyTorch Patterns Reference

**Authoritative source for**: Working GPyTorch code patterns and examples

This document contains code examples from `scripts/1D_playground/gp_utility_playground.py` that demonstrate correct GPyTorch usage for this project.

---

## 1. Poisson Likelihood

```python
class PoissonLikelihood(gpytorch.likelihoods.Likelihood):
    """Poisson likelihood with log-link: r ~ Poisson(exp(f))

    For our model: f = A*lambda + lambda_0, so rate = exp(A*lambda + lambda_0)

    Expected log-likelihood under q(lambda) = N(mu, var):
        E[r*f - exp(f)] = r*(A*mu + lambda_0) - exp(A*mu + A^2*var/2 + lambda_0)
    """

    def __init__(self, A_init=1.0, lambda0_init=0.0):
        super().__init__()
        # Learnable parameters
        self.register_parameter('raw_A', torch.nn.Parameter(torch.tensor(A_init)))
        self.register_parameter('lambda0', torch.nn.Parameter(torch.tensor(lambda0_init)))

    @property
    def A(self):
        # A = exp(raw_A), where raw_A = logA (matching varGP)
        return self.raw_A_constraint.transform(self.raw_A)

    def expected_log_prob(self, target, input):
        """
        Args:
            target: Observed spike counts (n_samples,)
            input: GP output MultivariateNormal with mean, variance
        Returns:
            Expected log probability (summed over samples)
        """
        mu = input.mean      # (n_samples,)
        var = input.variance # (n_samples,)
        A = self.A
        lambda0 = self.lambda0

        # E[r*(A*lambda + lambda_0) - exp(A*lambda + lambda_0)]
        # = r*(A*mu + lambda_0) - exp(A*mu + A^2*var/2 + lambda_0)
        log_prob = target * (A * mu + lambda0) - torch.exp(A * mu + 0.5 * A**2 * var + lambda0)
        return log_prob.sum(-1)

    def forward(self, function_samples):
        """For sampling: return Poisson distribution given function samples."""
        A = self.A
        lambda0 = self.lambda0
        rate = torch.exp(A * function_samples + lambda0)
        return torch.distributions.Poisson(rate=rate)
```

---

## 2. Variational GP Model

```python
class VariationalGPModel(gpytorch.models.ApproximateGP):
    def __init__(self, inducing_points, kernel):
        # Variational distribution q(u) = N(m, LL^T)
        variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(
            inducing_points.size(0)
        )
        # Strategy for computing q(f) from q(u)
        variational_strategy = gpytorch.variational.VariationalStrategy(
            self,
            inducing_points,
            variational_distribution,
            learn_inducing_locations=False  # Keep inducing points fixed
        )
        super().__init__(variational_strategy)

        self.mean_module = gpytorch.means.ZeroMean()
        self.covar_module = kernel

    def forward(self, x):
        mean = self.mean_module(x)
        covar = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean, covar)
```

---

## 3. Training Loop

```python
def train_gpy_default(model, likelihood, train_x, train_y, n_iterations=500, lr=0.1):
    model.train()

    # Optimize both model and likelihood parameters
    optimizer = torch.optim.Adam([
        {'params': model.parameters()},
        {'params': likelihood.parameters()}
    ], lr=lr)

    with torch.enable_grad():  # Important: utility.py disables grad globally
        for i in range(n_iterations):
            optimizer.zero_grad()

            # Forward pass
            output = model(train_x)

            # ELBO = E_q[log p(y|f)] - KL(q(u) || p(u))
            expected_log_lik = likelihood.expected_log_prob(train_y, output)
            kl_div = model.variational_strategy.kl_divergence()

            # Minimize negative ELBO
            loss = -expected_log_lik + kl_div

            loss.backward()
            optimizer.step()

            if (i + 1) % 100 == 0:
                print(f"Iter {i+1}/{n_iterations}, Loss: {loss.item():.2f}")

    return model, likelihood
```

---

## 4. Prediction

```python
def predict(model, likelihood, test_x):
    model.eval()
    with torch.no_grad():
        # Get posterior q(lambda*) at test points
        posterior = model(test_x)
        mu = posterior.mean
        var = posterior.variance

        # Predicted firing rate: E[exp(A*lambda + lambda_0)] = exp(A*mu + A^2*var/2 + lambda_0)
        A = likelihood.A
        lambda0 = likelihood.lambda0
        f_pred = torch.exp(A * mu + 0.5 * A**2 * var + lambda0)

    return f_pred, mu, var
```

---

## 5. Key Implementation Notes

1. **Order matters**: Call `.double()` before `.to(device)` to avoid dtype issues
2. **Gradient scope**: Use `torch.enable_grad()` explicitly since utility.py may disable it globally
3. **Inducing points**: Keep fixed (`learn_inducing_locations=False`) - we select from training data
4. **Avoid `.data`**: Use `torch.no_grad() + copy()` instead of direct `.data` access

---

*Extracted from CLAUDE.md Section 11, January 2025*
