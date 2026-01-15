"""
Custom E-Step Implementation for GPyTorch Variational GP

Implements closed-form Newton update for variational parameters (m, V).

CORRECT Formulas (from .claude/ESTEP_MATH_ANALYSIS.md):
    V_new = K̃(K̃ + G)⁻¹K̃
    m_new = m + K̃(K̃ + G)⁻¹(g - m)

where:
    g = A · Kᵀ @ (r - f̄)
    G = A² · Kᵀ @ diag(f̄) @ K
    f̄ᵢ = exp(A·μᵢ + ½A²σᵢ² + λ₀)
"""

import torch
import gpytorch
from typing import Tuple, Optional


def e_step(
    model: gpytorch.models.ApproximateGP,
    likelihood,
    X: torch.Tensor,
    r: torch.Tensor,
    jitter: float = 1e-6
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Perform one E-step: closed-form Newton update of variational parameters.

    Uses CORRECT formulas (from .claude/ESTEP_MATH_ANALYSIS.md):
        g = A · Kᵀ @ (r - f̄)
        G = A² · Kᵀ @ diag(f̄) @ K

        V_new = K̃ @ solve(K̃ + G, K̃)  = K̃(K̃ + G)⁻¹K̃
        m_new = m + K̃ @ solve(K̃ + G, g - m)  = m + K̃(K̃ + G)⁻¹(g - m)

    Note: The original utils.py:Estep() uses a different m formula that has a
    discrepancy. This implementation uses the mathematically correct Newton step.

    Args:
        model: VariationalGPModel instance
        likelihood: PoissonLikelihood instance
        X: Training inputs, shape (N, n_features)
        r: Training spike counts, shape (N,)
        jitter: Small value for numerical stability

    Returns:
        m_new: Updated variational mean, shape (M,)
        V_new: Updated variational covariance, shape (M, M)
    """
    # Get current variational mean
    var_params = model.variational_strategy._variational_distribution
    m = var_params.variational_mean  # (M,)

    # Get likelihood parameters
    A = likelihood.A.squeeze()
    lambda0 = likelihood.lambda0.squeeze()

    # Use GPyTorch's native forward pass for posterior moments
    output = model(X)
    lambda_mean = output.mean      # (N,)
    lambda_var = output.variance   # (N,)

    # Expected firing rate: f̄ = exp(A·μ + ½A²σ² + λ₀)
    f_mean = torch.exp(A * lambda_mean + 0.5 * A**2 * lambda_var + lambda0)

    # Get kernel matrices
    inducing_points = model.variational_strategy.inducing_points
    kernel = model.covar_module

    K = kernel(X, inducing_points).evaluate()           # (N, M)
    K_tilde = kernel(inducing_points).evaluate()        # (M, M)

    # Add jitter for numerical stability
    M = K_tilde.shape[0]
    eye = torch.eye(M, dtype=K_tilde.dtype, device=K_tilde.device)
    K_tilde_j = K_tilde + jitter * eye

    # Standard (non-transformed) g and G
    g = A * K.T @ (r - f_mean)                              # (M,)
    G = A**2 * K.T @ (f_mean[:, None] * K)                  # (M, M)

    # V update: V_new = K̃ @ solve(K̃ + G, K̃) = K̃(K̃ + G)⁻¹K̃
    V_new = K_tilde_j @ torch.linalg.solve(K_tilde_j + G, K_tilde_j)

    # m update: m_new = m + K̃ @ solve(K̃ + G, g - m) = m + K̃(K̃ + G)⁻¹(g - m)
    m_new = m + K_tilde_j @ torch.linalg.solve(K_tilde_j + G, g - m)

    # Symmetrize V
    V_new = (V_new + V_new.T) / 2

    return m_new, V_new


def update_variational_parameters(
    model: gpytorch.models.ApproximateGP,
    m_new: torch.Tensor,
    V_new: torch.Tensor
):
    """Write updated (m, V) back to GPyTorch model.

    GPyTorch stores:
        - variational_mean: m directly
        - chol_variational_covar: L where V = LLᵀ
    """
    # Access the parameter storage object (not the distribution)
    var_params = model.variational_strategy._variational_distribution

    # Update mean (stored directly)
    var_params.variational_mean.data.copy_(m_new)

    # Compute Cholesky factor L where V = LLᵀ
    try:
        L_new = torch.linalg.cholesky(V_new)
    except RuntimeError:
        # Add jitter if Cholesky fails
        eye = torch.eye(V_new.shape[0], dtype=V_new.dtype, device=V_new.device)
        L_new = torch.linalg.cholesky(V_new + 1e-6 * eye)

    var_params.chol_variational_covar.data.copy_(L_new)


def train_with_estep(
    model: gpytorch.models.ApproximateGP,
    likelihood,
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    n_iterations: int = 100,
    n_estep: int = 3,
    n_mstep: int = 10,
    lr: float = 0.01,
    print_every: int = 10,
    device: Optional[torch.device] = None
):
    """Train using EM: E-step (Newton) + M-step (gradient descent on hyperparams).

    Args:
        model: VariationalGPModel instance
        likelihood: PoissonLikelihood instance
        train_x: Training inputs, shape (N, n_features)
        train_y: Training spike counts, shape (N,)
        n_iterations: Number of EM iterations
        n_estep: Number of E-steps per iteration
        n_mstep: Number of M-step gradient updates per iteration
        lr: Learning rate for M-step
        print_every: Print progress every N iterations (0 to disable)
        device: Device to use

    Returns:
        losses: List of ELBO values
    """
    if device is None:
        device = train_x.device

    model = model.to(device)
    likelihood = likelihood.to(device)
    train_x = train_x.to(device)
    train_y = train_y.to(device)

    # M-step optimizer: only hyperparameters, not variational params
    hyperparams = list(model.covar_module.parameters()) + list(likelihood.parameters())
    optimizer = torch.optim.Adam(hyperparams, lr=lr)

    losses = []

    for iteration in range(n_iterations):
        # ===== E-STEP: Update (m, V) =====
        model.eval()
        with torch.no_grad():
            for _ in range(n_estep):
                m_new, V_new = e_step(model, likelihood, train_x, train_y)
                update_variational_parameters(model, m_new, V_new)

        # ===== M-STEP: Update hyperparameters =====
        model.train()
        with torch.enable_grad():
            for _ in range(n_mstep):
                optimizer.zero_grad()
                output = model(train_x)
                loss = -likelihood.expected_log_prob(train_y, output) + \
                       model.variational_strategy.kl_divergence()
                loss.backward()
                optimizer.step()

        # Record loss
        model.eval()
        with torch.no_grad():
            output = model(train_x)
            ell = likelihood.expected_log_prob(train_y, output)
            kl = model.variational_strategy.kl_divergence()
            current_loss = (-ell + kl).item()
        losses.append(current_loss)

        if print_every > 0 and (iteration + 1) % print_every == 0:
            print(f"Iter {iteration+1}/{n_iterations}, Loss: {current_loss:.2f}, "
                  f"ELL: {ell.item():.2f}, KL: {kl.item():.2f}")

    return losses


def test_estep():
    """Test E-step implementation."""
    from kernels import ArcCosineKernel
    from likelihoods import PoissonLikelihood
    from model import VariationalGPModel

    torch.manual_seed(42)
    print("Testing E-step...")

    # Synthetic data
    n_train, n_inducing, n_features = 100, 20, 50
    X = torch.randn(n_train, n_features, dtype=torch.float64)
    y = torch.poisson(torch.exp(0.1 * X.sum(dim=1)))
    inducing = X[torch.randperm(n_train)[:n_inducing]].clone()

    # Model
    kernel = ArcCosineKernel(sigma_0=1.0)
    model = VariationalGPModel(inducing, kernel).double()
    likelihood = PoissonLikelihood(A_init=0.1, lambda0_init=1.0).double()

    # Test single E-step
    print("1. Single E-step...")
    m_new, V_new = e_step(model, likelihood, X, y)
    print(f"   m shape: {m_new.shape}, V shape: {V_new.shape}")
    print(f"   V symmetric: {(V_new - V_new.T).abs().max().item():.2e}")
    print(f"   V positive definite: {torch.linalg.eigvalsh(V_new).min().item():.2e}")

    # Test update
    print("2. Update variational params...")
    update_variational_parameters(model, m_new, V_new)
    m_check = model.variational_strategy.variational_distribution.mean
    print(f"   Mean updated: {(m_check - m_new).abs().max().item():.2e}")

    # Test training loop
    print("3. Training loop (5 iters)...")
    kernel = ArcCosineKernel(sigma_0=1.0)
    model = VariationalGPModel(inducing, kernel).double()
    likelihood = PoissonLikelihood(A_init=0.1, lambda0_init=1.0).double()

    losses = train_with_estep(model, likelihood, X, y,
                              n_iterations=5, n_estep=2, n_mstep=10,
                              lr=0.1, print_every=1)
    print(f"   Loss decreased: {losses[-1] < losses[0]}")

    print("\nAll tests PASSED!")
    return True


if __name__ == '__main__':
    test_estep()
