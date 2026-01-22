"""
Whitening Conversion Functions for GPyTorch Variational GP

Handles conversion between whitened (GPyTorch storage) and natural (E-step computation)
parameterizations of variational parameters (m, V).

GPyTorch stores whitened variational params, but our E-step produces natural params.
These functions convert between the two representations at the storage boundary.

GPyTorch formula: λ_m = K_XZ @ L_K⁻ᵀ @ m_stored  (expects whitened m)
Standard SVGP:    λ_m = K_XZ @ K̃⁻¹ @ m          (expects natural m)

When we store natural m directly, GPyTorch computes wrong λ_m (~8x smaller).
Fix: Convert m at read/write boundaries.
"""

import warnings
from typing import Optional

import torch
import gpytorch


def _validate_jitter(jitter: Optional[float], model: gpytorch.models.ApproximateGP) -> float:
    """Validate and return jitter value, warning if mismatch detected.

    CRITICAL: All jitter values MUST match model.jitter to ensure consistency
    between whitening conversions and GPyTorch's internal computations.
    Mismatched jitter causes whitening conversion failures in E-step paths.

    Args:
        jitter: Explicit jitter value or None (use model.jitter)
        model: VariationalGPModel instance (must have .jitter attribute)

    Returns:
        jitter: The validated jitter value (always model.jitter)
    """
    if jitter is None:
        return model.jitter

    if jitter != model.jitter:
        warnings.warn(
            f"Jitter mismatch: explicit jitter={jitter} but model.jitter={model.jitter}. "
            f"This can cause incorrect results in whitened E-step paths. "
            f"Using model.jitter={model.jitter} instead.",
            UserWarning,
            stacklevel=3  # Point to the caller of the function that calls _validate_jitter
        )
    return model.jitter


def set_kernel_requires_grad(model: gpytorch.models.ApproximateGP, requires_grad: bool):
    """Toggle requires_grad for kernel parameters.

    This is used to skip gradient computation during E-step and F-step where
    kernel gradients are not needed. Disabling requires_grad allows the
    analytical gradients optimization to skip dK computation.

    Args:
        model: VariationalGPModel instance
        requires_grad: Whether to enable gradient computation for kernel params
    """
    for name, param in model.covar_module.named_parameters():
        param.requires_grad = requires_grad


def get_variational_mean(model: gpytorch.models.ApproximateGP) -> torch.Tensor:
    """Get the variational mean m from the model."""
    return model.variational_strategy._variational_distribution.variational_mean


def get_variational_covar(model: gpytorch.models.ApproximateGP) -> torch.Tensor:
    """Get the variational covariance V from the model (V = LLᵀ)."""
    L = model.variational_strategy._variational_distribution.chol_variational_covar
    return L @ L.T


def get_variational_mean_with_L_K(
    model: gpytorch.models.ApproximateGP,
    L_K: torch.Tensor
) -> torch.Tensor:
    """Read variational mean and convert whitened → natural.

    GPyTorch ALWAYS interprets m_stored as WHITENED mean, meaning:
        m_actual = L_K @ m_stored

    At initialization:
        m_stored = 0 (zeros from __init__)
        m_actual = L_K @ 0 = 0 (prior mean)
        The conversion is a no-op, but we apply it for consistency with V.

    After we store whitened m:
        m_stored = m_whitened = L_K⁻¹ @ m_natural
        m_actual = L_K @ m_whitened = m_natural

    Args:
        model: VariationalGPModel instance
        L_K: Cholesky factor of K̃ + jitter (from kernel cache)

    Returns:
        m_natural: Natural variational mean, shape (M,)

    Complexity: O(M²) - matrix-vector multiply
    """
    m_stored = model.variational_strategy._variational_distribution.variational_mean

    # ALWAYS apply conversion - GPyTorch ALWAYS interprets m_stored as whitened
    # At init: m_stored=0 means m_actual=0 (prior mean)
    # After update: m_stored=m_whitened means m_actual=m_natural
    return L_K @ m_stored


def update_variational_mean_with_L_K(
    model: gpytorch.models.ApproximateGP,
    m_natural: torch.Tensor,
    L_K: torch.Tensor
):
    """Convert natural → whitened and store in GPyTorch model.

    Our E-step produces natural m. This function converts it to whitened form
    before storing, so GPyTorch computes correct λ_m.

    Formula: m_whitened = L_K⁻¹ @ m_natural (solved via triangular solve)

    IMPORTANT: Also sets variational_params_initialized = True to prevent
    GPyTorch's automatic whitening on first model(X) call.

    Args:
        model: VariationalGPModel instance
        m_natural: Natural variational mean from E-step, shape (M,)
        L_K: Cholesky factor of K̃ + jitter (from kernel cache)

    Complexity: O(M²) - triangular solve
    """
    # Solve L_K @ m_whitened = m_natural for m_whitened
    m_whitened = torch.linalg.solve_triangular(
        L_K, m_natural.unsqueeze(-1), upper=False
    ).squeeze(-1)

    # Update using torch.no_grad() with .copy_() (best practice)
    with torch.no_grad():
        model.variational_strategy._variational_distribution.variational_mean.copy_(m_whitened)

    # CRITICAL: Set flag to prevent GPyTorch's automatic whitening
    # Without this, GPyTorch would double-whiten our already-whitened m
    model.variational_strategy.variational_params_initialized.fill_(True)


def update_variational_covar(
    model: gpytorch.models.ApproximateGP,
    V_new: torch.Tensor
):
    """Update variational covariance V in GPyTorch model.

    GPyTorch stores L where V = LLᵀ (Cholesky). This function computes L from V.

    Note: V whitening is NOT addressed here (separate task). This function
    stores V directly as Cholesky, same as the original update_variational_parameters().

    Args:
        model: VariationalGPModel instance
        V_new: New variational covariance, shape (M, M)
    """
    try:
        L_new = torch.linalg.cholesky(V_new)
    except RuntimeError:
        # Add jitter if Cholesky fails
        eye = torch.eye(V_new.shape[0], dtype=V_new.dtype, device=V_new.device)
        L_new = torch.linalg.cholesky(V_new + 1e-6 * eye)

    # Update using torch.no_grad() with .copy_() (best practice)
    with torch.no_grad():
        model.variational_strategy._variational_distribution.chol_variational_covar.copy_(L_new)


def get_variational_covar_with_L_K(
    model: gpytorch.models.ApproximateGP,
    L_K: torch.Tensor
) -> torch.Tensor:
    """Read variational covariance and convert whitened → natural.

    GPyTorch stores V as Cholesky factor L where V_stored = L @ L.T.
    GPyTorch ALWAYS interprets V_stored as WHITENED covariance, meaning:
        V_actual = L_K @ V_stored @ L_K.T

    At initialization:
        L_stored = I, so V_stored = I
        GPyTorch interprets this as V_actual = L_K @ I @ L_K.T = K̃ (the prior!)
        We MUST apply this conversion, otherwise cached path uses V=I instead of V=K̃

    After we store whitened V:
        V_stored = V_whitened = L_K⁻¹ @ V_natural @ L_K⁻ᵀ
        V_actual = L_K @ V_whitened @ L_K.T = V_natural

    Args:
        model: VariationalGPModel instance
        L_K: Cholesky factor of K̃ + jitter (from kernel cache)

    Returns:
        V_natural: Natural variational covariance, shape (M, M)

    Complexity: O(M³) - two matrix multiplications
    """
    L_stored = model.variational_strategy._variational_distribution.chol_variational_covar
    V_stored = L_stored @ L_stored.T

    # ALWAYS apply conversion - GPyTorch ALWAYS interprets V_stored as whitened
    # At init: V_stored=I means V_actual=K̃ (prior)
    # After update: V_stored=V_whitened means V_actual=V_natural
    return L_K @ V_stored @ L_K.T


def update_variational_covar_with_L_K(
    model: gpytorch.models.ApproximateGP,
    V_natural: torch.Tensor,
    L_K: torch.Tensor
):
    """Convert natural → whitened and store in GPyTorch model.

    Our E-step produces natural V. This function converts it to whitened form
    before storing, so GPyTorch computes correct variance and KL divergence.

    Formula: V_whitened = L_K⁻¹ @ V_natural @ L_K⁻ᵀ (via triangular solves)

    IMPORTANT:
    - Requires cache clearing after call (variance is cached unlike mean)
    - Sets variational_params_initialized = True (same as m whitening)

    Args:
        model: VariationalGPModel instance
        V_natural: Natural variational covariance from E-step, shape (M, M)
        L_K: Cholesky factor of K̃ + jitter (from kernel cache)

    Complexity: O(M³) - two triangular solves + one Cholesky
    """
    # Step 1: Convert to whitened
    # V_whitened = L_K⁻¹ @ V_natural @ L_K⁻ᵀ
    temp = torch.linalg.solve_triangular(L_K, V_natural, upper=False)
    V_whitened = torch.linalg.solve_triangular(L_K, temp.T, upper=False).T

    # Ensure symmetry (numerical stability)
    V_whitened = (V_whitened + V_whitened.T) / 2

    # Step 2: Compute Cholesky of whitened V
    try:
        L_whitened = torch.linalg.cholesky(V_whitened)
    except RuntimeError:
        # Add jitter if Cholesky fails
        M = V_whitened.shape[0]
        eye = torch.eye(M, dtype=V_whitened.dtype, device=V_whitened.device)
        L_whitened = torch.linalg.cholesky(V_whitened + 1e-6 * eye)

    # Step 3: Store whitened Cholesky using torch.no_grad() with .copy_() (best practice)
    with torch.no_grad():
        model.variational_strategy._variational_distribution.chol_variational_covar.copy_(L_whitened)

    # Set flag to indicate whitening has been applied (same as m whitening)
    model.variational_strategy.variational_params_initialized.fill_(True)


def update_variational_parameters(
    model: gpytorch.models.ApproximateGP,
    m_new: torch.Tensor,
    V_new: torch.Tensor,
    jitter: float
):
    """Write updated (m, V) back to GPyTorch model.

    GPyTorch stores:
        - variational_mean: m directly
        - chol_variational_covar: L where V = LLᵀ
    """
    # Access the parameter storage object (not the distribution)
    var_params = model.variational_strategy._variational_distribution

    # Compute Cholesky factor L where V = LLᵀ
    try:
        L_new = torch.linalg.cholesky(V_new)
    except RuntimeError:
        # Add jitter if Cholesky fails
        eye = torch.eye(V_new.shape[0], dtype=V_new.dtype, device=V_new.device)
        L_new = torch.linalg.cholesky(V_new + jitter * eye)

    # Update parameters using torch.no_grad() with .copy_() (best practice)
    # This ensures no computation graph is attached to the parameter updates
    with torch.no_grad():
        var_params.variational_mean.copy_(m_new)
        var_params.chol_variational_covar.copy_(L_new)


def clear_variational_cache(model: gpytorch.models.ApproximateGP):
    """Clear GPyTorch's memoized cache for variational covariance.

    MUST be called after updating chol_variational_covar.
    NOT needed after updating variational_mean (mean updates are immediate).

    Without clearing, GPyTorch returns stale variance values.
    """
    from gpytorch.utils.memoize import clear_cache_hook
    clear_cache_hook(model.variational_strategy)
    if hasattr(model.variational_strategy, '_memoize_cache'):
        model.variational_strategy._memoize_cache.clear()
