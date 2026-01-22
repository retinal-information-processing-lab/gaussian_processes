"""
M-Step Function for GPyTorch Variational GP

Handles optimization of kernel hyperparameters while holding variational
parameters (m, V) and firing rate parameters (A, λ₀) fixed.

Key function:
- m_step(): Adam-based optimization of kernel hyperparameters

DEPRECATED FUNCTIONS (deleted):
- m_step_lbfgs(): Did not support hyperparameter clamping
- m_step_lbfgs_grouped(): Did not support hyperparameter clamping
"""

import torch
import gpytorch


def m_step(
    model: gpytorch.models.ApproximateGP,
    likelihood,
    X: torch.Tensor,
    r: torch.Tensor,
    n_mstep: int,
    lr: float,  # Required - no default to prevent silent bugs
    verbose: bool = False
):
    """M-step: Optimize kernel hyperparameters with Adam.

    Structural change from baseline: Only kernel hyperparameters are optimized here,
    not A or lambda0 (those are handled in F-step).

    Args:
        model: VariationalGPModel instance
        likelihood: PoissonLikelihood instance
        X: Training inputs, shape (N, n_features)
        r: Training spike counts, shape (N,)
        n_mstep: Number of Adam iterations
        lr: Learning rate for Adam
        verbose: Print debug info
    """
    if n_mstep == 0:
        return

    optimizer = torch.optim.Adam(model.covar_module.parameters(), lr=lr)

    for _ in range(n_mstep):
        optimizer.zero_grad()
        output = model(X)
        loss = -likelihood.expected_log_prob(r, output) + \
               model.variational_strategy.kl_divergence()
        loss.backward()
        optimizer.step()

        # Clamp hyperparameters to valid bounds (projected gradient descent)
        # Since kernel is now ArcCosineKernel directly (not ScaleKernel wrapper),
        # clamp_hyperparameters is called on model.covar_module directly
        kernel = model.covar_module
        if hasattr(kernel, 'clamp_hyperparameters'):
            kernel.clamp_hyperparameters()
