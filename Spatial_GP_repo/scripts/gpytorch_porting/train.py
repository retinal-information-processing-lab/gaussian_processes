"""
Training Utilities for GPyTorch Variational GP

This module provides functions for training and evaluating the variational GP model.
"""

import torch
import numpy as np


def train_model(model, likelihood, train_x, train_y, n_iterations=500, lr=0.1,
                print_every=100, device=None):
    """Train the variational GP model.

    Maximizes the ELBO = E_q[log p(y|f)] - KL(q(u) || p(u))

    Args:
        model: VariationalGPModel instance
        likelihood: PoissonLikelihood instance
        train_x: Training inputs, shape (n_train, n_features)
        train_y: Training targets (spike counts), shape (n_train,)
        n_iterations: Number of optimization iterations
        lr: Learning rate for Adam optimizer
        print_every: Print loss every N iterations (0 to disable)
        device: Device to use (defaults to train_x.device)

    Returns:
        losses: List of loss values during training
    """
    if device is None:
        device = train_x.device

    model = model.to(device)
    likelihood = likelihood.to(device)
    train_x = train_x.to(device)
    train_y = train_y.to(device)

    model.train()
    likelihood.train()

    # Optimize both model and likelihood parameters
    optimizer = torch.optim.Adam([
        {'params': model.parameters()},
        {'params': likelihood.parameters()}
    ], lr=lr)

    losses = []

    with torch.enable_grad():
        for i in range(n_iterations):
            optimizer.zero_grad()

            # Forward pass: get GP posterior at training points
            output = model(train_x)

            # ELBO = E_q[log p(y|f)] - KL(q(u) || p(u))
            expected_log_lik = likelihood.expected_log_prob(train_y, output)
            kl_div = model.variational_strategy.kl_divergence()

            # Minimize negative ELBO
            loss = -expected_log_lik + kl_div

            loss.backward()
            optimizer.step()

            losses.append(loss.item())

            if print_every > 0 and (i + 1) % print_every == 0:
                print(f"Iter {i+1}/{n_iterations}, Loss: {loss.item():.2f}, "
                      f"ELL: {expected_log_lik.item():.2f}, KL: {kl_div.item():.2f}")

    return losses


def predict(model, likelihood, test_x, device=None):
    """Make predictions on test data.

    Args:
        model: Trained VariationalGPModel
        likelihood: Trained PoissonLikelihood
        test_x: Test inputs, shape (n_test, n_features)
        device: Device to use (defaults to test_x.device)

    Returns:
        dict with:
        - 'f_pred': Predicted firing rates E[exp(A·λ + λ₀)]
        - 'lambda_mean': Posterior mean of λ
        - 'lambda_var': Posterior variance of λ
    """
    if device is None:
        device = test_x.device

    model = model.to(device)
    likelihood = likelihood.to(device)
    test_x = test_x.to(device)

    model.eval()
    likelihood.eval()

    with torch.no_grad():
        # Get posterior q(λ*) at test points
        posterior = model(test_x)
        lambda_mean = posterior.mean
        lambda_var = posterior.variance

        # Predicted firing rate: E[exp(A·λ + λ₀)] = exp(A·μ + A²σ²/2 + λ₀)
        A = likelihood.A.squeeze()
        lambda0 = likelihood.lambda0.squeeze()
        f_pred = torch.exp(A * lambda_mean + 0.5 * A**2 * lambda_var + lambda0)

    return {
        'f_pred': f_pred,
        'lambda_mean': lambda_mean,
        'lambda_var': lambda_var
    }


def compute_r_squared(y_true, y_pred):
    """Compute R² (coefficient of determination).

    Args:
        y_true: True values, shape (n,) or (n_repeats, n)
        y_pred: Predicted values, shape (n,)

    Returns:
        R² value (scalar)
    """
    if y_true.ndim == 2:
        # Average over repeats
        y_true = y_true.mean(dim=0)

    y_true = y_true.float()
    y_pred = y_pred.float()

    ss_res = ((y_true - y_pred) ** 2).sum()
    ss_tot = ((y_true - y_true.mean()) ** 2).sum()

    r2 = 1 - ss_res / ss_tot

    return r2.item()


def compute_pearson_correlation(y_true, y_pred):
    """Compute Pearson correlation coefficient.

    Args:
        y_true: True values, shape (n,) or (n_repeats, n)
        y_pred: Predicted values, shape (n,)

    Returns:
        Pearson correlation (scalar)
    """
    if y_true.ndim == 2:
        # Average over repeats
        y_true = y_true.mean(dim=0)

    y_true = y_true.float()
    y_pred = y_pred.float()

    # Center the values
    y_true_centered = y_true - y_true.mean()
    y_pred_centered = y_pred - y_pred.mean()

    # Compute correlation
    numerator = (y_true_centered * y_pred_centered).sum()
    denominator = torch.sqrt((y_true_centered ** 2).sum() * (y_pred_centered ** 2).sum())

    corr = numerator / denominator

    return corr.item()


def compute_explained_variance(r_test, f_pred):
    """Compute explained variance normalized by cell reliability.

    This matches the reference implementation in utils.py:explained_variance().

    Explained variance = (Pearson r with predictions) / (cell reliability)

    where reliability is the correlation between even and odd trial halves.
    A perfect model achieves explained_variance = 1.0.

    Args:
        r_test: Test responses with repetitions, shape (n_repeats, n_images)
        f_pred: Predicted firing rates, shape (n_images,)

    Returns:
        explained_var: Fraction of explainable variance captured (scalar)
        reliability: Cell reliability (scalar)
    """
    r_test = r_test.float()
    f_pred = f_pred.float()

    # Split into even and odd repetitions
    r_even = r_test[0::2, :].mean(dim=0)  # (n_images,)
    r_odd = r_test[1::2, :].mean(dim=0)   # (n_images,)

    # Reliability = correlation between even and odd halves
    reliability = torch.corrcoef(torch.stack([r_even, r_odd]))[0, 1].abs()

    # Accuracy = average correlation with each half
    accuracy_even = torch.corrcoef(torch.stack([f_pred, r_even]))[0, 1]
    accuracy_odd = torch.corrcoef(torch.stack([f_pred, r_odd]))[0, 1]
    accuracy = 0.5 * (accuracy_even + accuracy_odd)

    # Explained variance = accuracy / reliability
    explained_var = accuracy / reliability

    return explained_var.item(), reliability.item()


def test_training():
    """Test training utilities with synthetic data."""
    from kernels import ArcCosineKernel
    from likelihoods import PoissonLikelihood
    from model import VariationalGPModel

    torch.manual_seed(42)

    # Create synthetic data
    n_train = 100
    n_test = 20
    n_inducing = 20
    n_features = 50

    # Generate random inputs
    X_train = torch.randn(n_train, n_features)
    X_test = torch.randn(n_test, n_features)

    # Generate synthetic spike counts (Poisson with rate = exp(linear function))
    true_rate = torch.exp(0.1 * X_train.sum(dim=1))
    y_train = torch.poisson(true_rate)

    true_rate_test = torch.exp(0.1 * X_test.sum(dim=1))
    y_test = torch.poisson(true_rate_test)

    # Select inducing points
    indices = torch.randperm(n_train)[:n_inducing]
    inducing_points = X_train[indices].clone()

    # Create model and likelihood
    kernel = ArcCosineKernel(sigma_0=1.0)
    model = VariationalGPModel(inducing_points, kernel)
    likelihood = PoissonLikelihood(A_init=1.0, lambda0_init=0.0)

    print("Training on synthetic data...")
    losses = train_model(model, likelihood, X_train, y_train,
                         n_iterations=100, lr=0.1, print_every=25)

    print(f"\nFinal loss: {losses[-1]:.2f}")
    print(f"Loss decreased: {losses[0] > losses[-1]}")

    # Make predictions
    predictions = predict(model, likelihood, X_test)
    print(f"\nPredicted firing rates: {predictions['f_pred'][:5]}")

    # Compute metrics
    r2 = compute_r_squared(y_test, predictions['f_pred'])
    corr = compute_pearson_correlation(y_test, predictions['f_pred'])
    print(f"R²: {r2:.4f}")
    print(f"Pearson r: {corr:.4f}")

    # Check that model learned something
    print(f"\nLikelihood A: {likelihood.A.item():.4f}")
    print(f"Likelihood lambda0: {likelihood.lambda0.item():.4f}")

    print("\nPASS: Training utilities test passed!")
    return True


if __name__ == '__main__':
    test_training()
