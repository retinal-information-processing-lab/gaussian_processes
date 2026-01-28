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

PERFORMANCE OPTIMIZATION (2026-01-18):
    Kernel matrices K and K̃ are cached via compute_kernel_cache() and reused
    within e_step_loop() via compute_moments_from_kernel_cache() and
    e_step_with_kernel_cache(). This bypasses GPyTorch's model(X) calls and
    reduces kernel computations from 35 to 3 per loop (11.7x reduction).
    Use `use_cache=False` in train_varGP_style() to disable for testing.
"""

import warnings

import torch
import gpytorch
from typing import Tuple, Optional, Dict, Literal

# Type alias for moment source selection
MomentSource = Literal['cache', 'gpytorch']

# Import whitening functions from dedicated module
from whitening import (
    _validate_jitter,
    set_kernel_requires_grad,
    get_variational_mean,
    get_variational_covar,
    get_variational_mean_with_L_K,
    update_variational_mean_with_L_K,
    get_variational_covar_with_L_K,
    update_variational_covar_with_L_K,
    update_variational_parameters,
    update_variational_covar,
    clear_variational_cache,
)


# =============================================================================
# Stability Constants (C1 fix - unified thresholds)
# =============================================================================

# Maximum allowed mean firing rate before instability is detected.
# When f_mean.mean() exceeds this, we revert to previous state (E-step) or
# return inf to halt optimization (F-step). Value chosen empirically for
# neural spike data where firing rates rarely exceed ~50 Hz.
STABILITY_THRESHOLD = 1000


# =============================================================================
# Kernel Caching Functions (2026-01-18 optimization)
# =============================================================================

def compute_kernel_cache(
    model: gpytorch.models.ApproximateGP,
    X: torch.Tensor,
    jitter: Optional[float] = None
) -> Dict[str, torch.Tensor]:
    """Compute and cache kernel matrices for E-step reuse.

    This function computes K and K̃ once, which can then be reused across
    multiple Newton steps in e_step_loop(). This reduces kernel calls from
    35 to 3 per loop (11.7x improvement).

    Args:
        model: VariationalGPModel instance
        X: Training inputs, shape (N, n_features)
        jitter: Jitter value. If None (default), uses model.jitter.
                If provided but mismatches model.jitter, warns and overrides.

    Returns:
        Dict with:
            'K': Cross-kernel matrix (N, M)
            'K_tilde': Inducing point kernel matrix (M, M)
            'K_tilde_j': K_tilde + jitter * I for stability
            'k0': Diagonal of kernel at X (for variance), shape (N,)
    """
    # Validate jitter - must match model.jitter for consistency
    jitter = _validate_jitter(jitter, model)

    inducing_points = model.variational_strategy.inducing_points
    kernel = model.covar_module

    # Compute kernel matrices ONCE
    K = kernel(X, inducing_points).evaluate()           # (N, M)
    K_tilde = kernel(inducing_points).evaluate()        # (M, M)

    # Diagonal of kernel at X - needed for variance computation
    # k0[i] = k(x_i, x_i)
    k0 = kernel(X, diag=True)  # (N,)
    if hasattr(k0, 'evaluate'):
        k0 = k0.evaluate()

    # Add jitter for stability
    M = K_tilde.shape[0]
    eye = torch.eye(M, dtype=K_tilde.dtype, device=K_tilde.device)
    K_tilde_j = K_tilde + jitter * eye

    # Cholesky factor for whitening conversions (O(M³) - done once per E-step)
    # L_K @ L_K.T = K_tilde_j
    # Only compute if model uses standard (whitened) variational distribution
    # (needed for whitened <-> natural conversions)
    if not hasattr(model, 'standard_variational_distribution'):
        raise AttributeError(
            "Model does not have 'standard_variational_distribution' attribute. "
            "Use VariationalGPModel which defines this attribute."
        )
    if model.standard_variational_distribution:
        # J2 fix: Add fallback with warning if Cholesky fails
        try:
            L_K = torch.linalg.cholesky(K_tilde_j)
        except RuntimeError:
            min_eig = torch.linalg.eigvalsh(K_tilde_j).min().item()
            warnings.warn(
                f"Cholesky failed on K_tilde in compute_kernel_cache() "
                f"(shape={tuple(K_tilde_j.shape)}, min_eigenvalue={min_eig:.2e}). "
                f"Adding extra jitter={jitter:.1e} and retrying.",
                RuntimeWarning
            )
            K_tilde_j = K_tilde_j + jitter * eye
            L_K = torch.linalg.cholesky(K_tilde_j)
    else:
        L_K = None  # Not needed for UnwhitenedVariationalStrategy

    return {
        'K': K,
        'K_tilde': K_tilde,
        'K_tilde_j': K_tilde_j,
        'k0': k0,
        'L_K': L_K,  # For whitened <-> natural m conversions (None for unwhitened)
    }


def compute_L_K(
    model: gpytorch.models.ApproximateGP,
    jitter: Optional[float] = None
) -> torch.Tensor:
    """Compute Cholesky factor L_K of inducing point kernel.

    L_K @ L_K.T = K̃ + jitter * I

    Used for whitening conversions when kernel cache is not available.
    This is a lightweight alternative to compute_kernel_cache() when only
    L_K is needed (e.g., for non-cached path whitening).

    Args:
        model: VariationalGPModel instance
        jitter: Jitter value. If None (default), uses model.jitter.
                If provided but mismatches model.jitter, warns and overrides.

    Returns:
        L_K: Lower triangular Cholesky factor, shape (M, M)
    """
    # Validate jitter - must match model.jitter for consistency
    jitter = _validate_jitter(jitter, model)

    inducing_points = model.variational_strategy.inducing_points
    K_tilde = model.covar_module(inducing_points).evaluate()
    M = K_tilde.shape[0]
    eye = torch.eye(M, dtype=K_tilde.dtype, device=K_tilde.device)
    K_tilde_j = K_tilde + jitter * eye

    # J2 fix: Add fallback with warning if Cholesky fails
    try:
        return torch.linalg.cholesky(K_tilde_j)
    except RuntimeError:
        min_eig = torch.linalg.eigvalsh(K_tilde_j).min().item()
        warnings.warn(
            f"Cholesky failed on K_tilde in compute_L_K() "
            f"(shape={tuple(K_tilde_j.shape)}, min_eigenvalue={min_eig:.2e}). "
            f"Adding extra jitter={jitter:.1e} and retrying.",
            RuntimeWarning
        )
        K_tilde_j = K_tilde_j + jitter * eye
        return torch.linalg.cholesky(K_tilde_j)


def compute_moments_from_kernel_cache(
    kernel_cache: Dict[str, torch.Tensor],
    m: torch.Tensor,
    V: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute lambda moments using cached kernel matrices (bypasses GPyTorch model(X)).

    This computes posterior moments directly from the cached K and K̃ matrices,
    avoiding GPyTorch's model(X) which would recompute kernels.

    Formulas:
        u = K̃⁻¹ @ Kᵀ  (projection vectors, shape M x N)
        λ_mean = u.T @ m = K @ K̃⁻¹ @ m  (shape N,)
        λ_var = k0 - diag(K @ K̃⁻¹ @ Kᵀ) + diag(K @ K̃⁻¹ @ V @ K̃⁻¹ @ Kᵀ)
              = k0 + diag(K @ K̃⁻¹ @ (V - K̃) @ K̃⁻¹ @ Kᵀ)

    Args:
        kernel_cache: Dict from compute_kernel_cache() containing K, K_tilde, k0
        m: Variational mean, shape (M,)
        V: Variational covariance, shape (M, M)

    Returns:
        lambda_m: Posterior mean at X, shape (N,)
        lambda_var: Posterior variance at X, shape (N,)
    """
    K = kernel_cache['K']              # (N, M)
    K_tilde_j = kernel_cache['K_tilde_j']  # (M, M)
    k0 = kernel_cache['k0']            # (N,)

    # u = K̃⁻¹ @ Kᵀ, shape (M, N) - solve K̃ @ u = Kᵀ
    u = torch.linalg.solve(K_tilde_j, K.T)  # (M, N)

    # λ_mean = uᵀ @ m = K @ K̃⁻¹ @ m
    lambda_m = u.T @ m  # (N,)

    # λ_var = k0 + diag(K @ K̃⁻¹ @ (V - K̃) @ K̃⁻¹ @ Kᵀ)
    #       = k0 + diag(uᵀ @ (V - K̃) @ u)
    # For efficiency, compute (V - K̃) @ u first, then dot with u
    V_minus_K = V - kernel_cache['K_tilde']
    Vu = V_minus_K @ u  # (M, N)
    # diag(uᵀ @ Vu) = sum(u * Vu, dim=0)
    lambda_var = k0 + (u * Vu).sum(dim=0)  # (N,)

    # J3 fix: Warn if negative variance detected (before clamping)
    if (lambda_var < 0).any():
        n_negative = (lambda_var < 0).sum().item()
        n_total = lambda_var.numel()
        min_val = lambda_var.min().item()
        mean_val = lambda_var.mean().item()
        warnings.warn(
            f"Negative variance detected: {n_negative}/{n_total} values. "
            f"min={min_val:.2e}, mean={mean_val:.2e}. Clamping to 1e-6.",
            RuntimeWarning
        )

    # Ensure variance is positive
    lambda_var = torch.clamp(lambda_var, min=1e-6)

    return lambda_m, lambda_var


def _newton_update(
    m: torch.Tensor,
    K: torch.Tensor,
    K_tilde_j: torch.Tensor,
    A: torch.Tensor,
    f_mean: torch.Tensor,
    r: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Core Newton update for E-step (shared by all E-step variants).

    Computes one Newton step for the variational parameters (m, V).

    Formulas:
        g = A · Kᵀ @ (r - f̄)
        G = A² · Kᵀ @ diag(f̄) @ K

        V_new = K̃ @ solve(K̃ + G, K̃)  = K̃(K̃ + G)⁻¹K̃
        m_new = m + K̃ @ solve(K̃ + G, g - m)  = m + K̃(K̃ + G)⁻¹(g - m)

    Args:
        m: Current variational mean, shape (M,)
        K: Cross-kernel K(X, Z̃), shape (N, M)
        K_tilde_j: Inducing kernel K̃ + jitter*I, shape (M, M)
        A: Gain parameter (scalar)
        f_mean: Expected firing rate exp(A·μ + ½A²σ² + λ₀), shape (N,)
        r: Training spike counts, shape (N,)

    Returns:
        m_new: Updated variational mean, shape (M,)
        V_new: Updated variational covariance, shape (M, M)
    """
    # Gradient and Hessian approximation
    g = A * K.T @ (r - f_mean)                  # (M,)
    G = A**2 * K.T @ (f_mean[:, None] * K)      # (M, M)

    # V update: V_new = K̃(K̃ + G)⁻¹K̃
    V_new = K_tilde_j @ torch.linalg.solve(K_tilde_j + G, K_tilde_j)

    # m update: m_new = m + K̃(K̃ + G)⁻¹(g - m)
    m_new = m + K_tilde_j @ torch.linalg.solve(K_tilde_j + G, g - m)

    # Symmetrize V for numerical stability
    V_new = (V_new + V_new.T) / 2

    return m_new, V_new


def _estep_single(
    m: torch.Tensor,
    V: torch.Tensor,
    r: torch.Tensor,
    A: torch.Tensor,
    lambda0: torch.Tensor,
    moment_source: MomentSource,
    *,
    kernel_cache: Optional[Dict[str, torch.Tensor]] = None,
    model: Optional[gpytorch.models.ApproximateGP] = None,
    likelihood=None,
    X: Optional[torch.Tensor] = None,
    jitter: Optional[float] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Unified E-step: single Newton update for variational parameters (m, V).

    This function replaces the three old E-step variants:
    - e_step() → _estep_single(moment_source='gpytorch')
    - e_step_explicit() → _estep_single(moment_source='gpytorch')
    - e_step_with_kernel_cache() → _estep_single(moment_source='cache')

    Key difference from old e_step(): m is ALWAYS passed explicitly (never read
    from model internally), making the function stateless and easier to reason about.

    Args:
        m: Current variational mean in NATURAL parameterization, shape (M,)
        V: Current variational covariance, shape (M, M)
        r: Training spike counts, shape (N,)
        A: Gain parameter (scalar tensor)
        lambda0: Bias parameter (scalar tensor)
        moment_source: How to compute posterior moments:
            'cache': Use kernel_cache (fast, requires kernel_cache)
            'gpytorch': Use model(X) (slow, requires model, likelihood, X)

        kernel_cache: Required when moment_source='cache'.
                      Dict from compute_kernel_cache() with K, K_tilde_j, k0.
        model: Required when moment_source='gpytorch'. VariationalGPModel instance.
        likelihood: Required when moment_source='gpytorch'. PoissonLikelihood instance.
        X: Required when moment_source='gpytorch'. Training inputs, shape (N, n_features).
        jitter: Required when moment_source='gpytorch'. Jitter value.

    Returns:
        m_new: Updated variational mean in NATURAL parameterization, shape (M,)
        V_new: Updated variational covariance, shape (M, M)

    Raises:
        ValueError: If required arguments for the selected moment_source are missing.
    """
    if moment_source == 'cache':
        # =====================================================================
        # CACHED PATH: Compute moments from pre-cached kernel matrices
        # =====================================================================
        if kernel_cache is None:
            raise ValueError("kernel_cache required when moment_source='cache'")

        K = kernel_cache['K']
        K_tilde_j = kernel_cache['K_tilde_j']

        # Compute moments using cached kernel matrices (not GPyTorch model(X))
        lambda_m, lambda_var = compute_moments_from_kernel_cache(kernel_cache, m, V)

        # Expected firing rate: f̄ = exp(A·μ + ½A²σ² + λ₀)
        f_mean = torch.exp(A * lambda_m + 0.5 * A**2 * lambda_var + lambda0)

        return _newton_update(m, K, K_tilde_j, A, f_mean, r)

    elif moment_source == 'gpytorch':
        # =====================================================================
        # GPYTORCH PATH: Compute moments via model(X) forward pass
        # =====================================================================
        if model is None or X is None:
            raise ValueError("model and X required when moment_source='gpytorch'")
        if jitter is None:
            jitter = _validate_jitter(None, model)

        # Moments via GPyTorch (uses stored whitened params)
        output = model(X)
        lambda_mean = output.mean
        lambda_var = output.variance
        f_mean = torch.exp(A * lambda_mean + 0.5 * A**2 * lambda_var + lambda0)

        # Compute kernel matrices (NOT cached)
        inducing_points = model.variational_strategy.inducing_points
        kernel = model.covar_module
        K = kernel(X, inducing_points).evaluate()
        K_tilde = kernel(inducing_points).evaluate()
        M = K_tilde.shape[0]
        K_tilde_j = K_tilde + jitter * torch.eye(M, dtype=K_tilde.dtype, device=K_tilde.device)

        return _newton_update(m, K, K_tilde_j, A, f_mean, r)

    else:
        raise ValueError(f"Invalid moment_source: {moment_source}. Must be 'cache' or 'gpytorch'.")


def e_step_with_kernel_cache(
    m: torch.Tensor,
    V: torch.Tensor,
    kernel_cache: Dict[str, torch.Tensor],
    A: torch.Tensor,
    lambda0: torch.Tensor,
    r: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Perform one E-step using cached kernel matrices (bypasses GPyTorch).

    This is a backward-compatible wrapper around _estep_single(moment_source='cache').

    Args:
        m: Current variational mean, shape (M,)
        V: Current variational covariance, shape (M, M)
        kernel_cache: Dict from compute_kernel_cache() containing K, K_tilde, k0
        A: Gain parameter (scalar)
        lambda0: Bias parameter (scalar)
        r: Training spike counts, shape (N,)

    Returns:
        m_new: Updated variational mean, shape (M,)
        V_new: Updated variational covariance, shape (M, M)
    """
    return _estep_single(m, V, r, A, lambda0, 'cache', kernel_cache=kernel_cache)


# =============================================================================
# Original E-step (non-cached)
# =============================================================================

def e_step(
    model: gpytorch.models.ApproximateGP,
    likelihood,
    X: torch.Tensor,
    r: torch.Tensor,
    jitter: Optional[float] = None
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
        jitter: Jitter value. If None (default), uses model.jitter.
                If provided but mismatches model.jitter, warns and overrides.

    Returns:
        m_new: Updated variational mean, shape (M,)
        V_new: Updated variational covariance, shape (M, M)
    """
    # Validate jitter - must match model.jitter for consistency
    jitter = _validate_jitter(jitter, model)

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

    return _newton_update(m, K, K_tilde_j, A, f_mean, r)


def e_step_explicit(
    m: torch.Tensor,
    model: gpytorch.models.ApproximateGP,
    likelihood,
    X: torch.Tensor,
    r: torch.Tensor,
    jitter: Optional[float] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """E-step with explicit m parameter for whitened workflows.

    Unlike e_step(), this takes m as an argument rather than reading from model.
    Uses model(X) for moment computation (expects stored params to be whitened).

    This function is used in the non-cached path when whitening is enabled:
    - Stored params are whitened (for correct GPyTorch moment computation)
    - But E-step formula needs natural m, so we pass it explicitly

    Args:
        m: Current variational mean in NATURAL parameterization, shape (M,)
        model: VariationalGPModel (stored params should be whitened)
        likelihood: PoissonLikelihood instance
        X: Training inputs, shape (N, n_features)
        r: Training spike counts, shape (N,)
        jitter: Jitter value. If None (default), uses model.jitter.
                If provided but mismatches model.jitter, warns and overrides.

    Returns:
        m_new: Updated variational mean in NATURAL parameterization, shape (M,)
        V_new: Updated variational covariance, shape (M, M)
    """
    # Validate jitter - must match model.jitter for consistency
    jitter = _validate_jitter(jitter, model)

    A = likelihood.A.squeeze()
    lambda0 = likelihood.lambda0.squeeze()

    # Moments via GPyTorch (uses stored whitened params - correct!)
    output = model(X)
    lambda_mean = output.mean
    lambda_var = output.variance
    f_mean = torch.exp(A * lambda_mean + 0.5 * A**2 * lambda_var + lambda0)

    # Compute kernel matrices (NOT cached - this is the non-cached path)
    inducing_points = model.variational_strategy.inducing_points
    kernel = model.covar_module
    K = kernel(X, inducing_points).evaluate()
    K_tilde = kernel(inducing_points).evaluate()
    M = K_tilde.shape[0]
    K_tilde_j = K_tilde + jitter * torch.eye(M, dtype=K_tilde.dtype, device=K_tilde.device)

    return _newton_update(m, K, K_tilde_j, A, f_mean, r)


def compute_moments(
    model: gpytorch.models.ApproximateGP,
    likelihood,
    X: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute lambda moments and f_mean from current variational params.

    This is the GPyTorch equivalent of the old code's:
        lambda_m, lambda_var = lambda_moments(...)
        f_mean = mean_f_given_lambda_moments(...)

    Args:
        model: VariationalGPModel instance
        likelihood: PoissonLikelihood instance
        X: Input points, shape (N, n_features)

    Returns:
        lambda_m: Posterior mean of λ at X, shape (N,)
        lambda_var: Posterior variance of λ at X, shape (N,)
        f_mean: Expected firing rate exp(A·μ + ½A²σ² + λ₀), shape (N,)
    """
    output = model(X)
    lambda_m = output.mean
    lambda_var = output.variance

    A = likelihood.A.squeeze()
    lambda0 = likelihood.lambda0.squeeze()
    f_mean = torch.exp(A * lambda_m + 0.5 * A**2 * lambda_var + lambda0)

    return lambda_m, lambda_var, f_mean


def e_step_loop(
    model: gpytorch.models.ApproximateGP,
    likelihood,
    X: torch.Tensor,
    r: torch.Tensor,
    n_estep: int,
    jitter: Optional[float] = None,
    verbose: bool = False,
    kernel_cache: Optional[Dict[str, torch.Tensor]] = None,
    *,  # Force keyword-only arguments below
    explicit_unwhitening: bool,  # REQUIRED: whether to do L_K conversions
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Run Newton loop with moment recomputation and stability checks.

    This matches the structure of the old varGP E-step loop (utils.py:5664-5712):
    - Save previous state before each Newton step
    - Recompute moments after each Newton step (CRITICAL)
    - Stability check: revert if f_mean.mean() > STABILITY_THRESHOLD or NaN detected
    - Early stopping: break if rel_change < 1e-5

    PERFORMANCE OPTIMIZATION (2026-01-18):
        When kernel_cache is provided, K and K̃ matrices are reused instead of
        recomputed via GPyTorch. This reduces kernel calls from 35 to 3 per loop.

    EXPLICIT UNWHITENING (2026-01-23):
        When explicit_unwhitening=True, variational parameters are converted between
        whitened (GPyTorch storage) and natural (E-step computation) forms via L_K.
        This is required when using VariationalStrategy (standard whitened distribution).
        Set explicit_unwhitening=False when using UnwhitenedVariationalStrategy.

    Args:
        model: VariationalGPModel instance
        likelihood: PoissonLikelihood instance
        X: Training inputs, shape (N, n_features)
        r: Training spike counts, shape (N,)
        n_estep: Number of Newton iterations
        jitter: Jitter value. If None (default), uses model.jitter.
                If provided but mismatches model.jitter, warns and overrides.
        verbose: Print debug info
        kernel_cache: Optional pre-computed kernel cache from compute_kernel_cache().
                      When provided, uses direct math formulas instead of GPyTorch model(X).
        explicit_unwhitening: Whether to do explicit L_K whitening conversions.
                              Must be explicitly specified (no auto-detection).
                              Set True for standard variational distribution.
                              Set False for unwhitened variational strategy.

    Returns:
        lambda_m: Final posterior mean of λ at X, shape (N,)
        lambda_var: Final posterior variance of λ at X, shape (N,)
    """
    # Validate jitter - must match model.jitter for consistency
    jitter = _validate_jitter(jitter, model)

    # Validate explicit_unwhitening matches model's variational distribution
    if not hasattr(model, 'standard_variational_distribution'):
        raise AttributeError(
            "Model does not have 'standard_variational_distribution' attribute. "
            "Use VariationalGPModel which defines this attribute."
        )
    if explicit_unwhitening and not model.standard_variational_distribution:
        raise ValueError(
            "explicit_unwhitening=True but model uses UnwhitenedVariationalStrategy "
            "(standard_variational_distribution=False). "
            "Set explicit_unwhitening=False."
        )
    if not explicit_unwhitening and model.standard_variational_distribution:
        raise ValueError(
            "explicit_unwhitening=False but model uses standard VariationalStrategy "
            "(standard_variational_distribution=True). "
            "Set explicit_unwhitening=True."
        )

    # Get likelihood parameters
    A = likelihood.A.squeeze()
    lambda0 = likelihood.lambda0.squeeze()

    # Use cached kernels if provided, otherwise use original (non-cached) path
    if kernel_cache is not None:
        # =====================================================================
        # CACHED PATH: Use pre-computed kernel matrices (bypasses GPyTorch)
        # =====================================================================
        if explicit_unwhitening:
            L_K = kernel_cache['L_K']
            # Read with whitening conversion (whitened → natural)
            m = get_variational_mean_with_L_K(model, L_K).clone()
            V = get_variational_covar_with_L_K(model, L_K).clone()
        else:
            # NO WHITENING - read directly (old behavior before whitening was added)
            m = get_variational_mean(model).clone()
            V = get_variational_covar(model).clone()

        # Initial moment computation using cached kernels (not GPyTorch model(X))
        lambda_m, lambda_var = compute_moments_from_kernel_cache(kernel_cache, m, V)
        f_mean = torch.exp(A * lambda_m + 0.5 * A**2 * lambda_var + lambda0)

        for i in range(n_estep):
            # Save previous state
            m_prev = m.clone()
            V_prev = V.clone()
            f_mean_prev = f_mean.clone()

            # Newton update using cached kernels (not GPyTorch)
            m, V = _estep_single(m, V, r, A, lambda0, moment_source='cache', kernel_cache=kernel_cache)

            # Recompute moments using cached kernels
            lambda_m, lambda_var = compute_moments_from_kernel_cache(kernel_cache, m, V)
            f_mean = torch.exp(A * lambda_m + 0.5 * A**2 * lambda_var + lambda0)

            # Stability check: revert if f_mean is too large or contains NaN (C1 fix)
            if f_mean.mean() > STABILITY_THRESHOLD or torch.any(torch.isnan(f_mean)):
                if verbose:
                    print(f"f_mean instability: mean={f_mean.mean():.1f}, has_nan={torch.any(torch.isnan(f_mean)).item()}, reverting")
                m, V = m_prev, V_prev
                lambda_m, lambda_var = compute_moments_from_kernel_cache(kernel_cache, m, V)
                f_mean = torch.exp(A * lambda_m + 0.5 * A**2 * lambda_var + lambda0)
                break

            # Convergence check (early stopping)
            if i > 0:
                rel_change = (f_mean - f_mean_prev).norm() / (f_mean_prev.norm() + 1e-6)
                if rel_change < 1e-5:
                    if verbose:
                        print(f"E-step converged after {i+1} iterations (rel_change={rel_change:.2e})")
                    break

        # Write final m, V back to model
        if explicit_unwhitening:
            update_variational_mean_with_L_K(model, m, L_K)
            update_variational_covar_with_L_K(model, V, L_K)
            # Clear GPyTorch cache (required for covariance updates)
            clear_variational_cache(model)
        else:
            # NO WHITENING - write directly
            update_variational_parameters(model, m, V, jitter)

    else:
        # =====================================================================
        # NON-CACHED PATH: Uses GPyTorch model(X) for moment computation
        # =====================================================================
        if explicit_unwhitening:
            # Compute L_K for whitening conversions
            L_K = compute_L_K(model, jitter)

            # Read with whitening (whitened → natural)
            m = get_variational_mean_with_L_K(model, L_K).clone()
            V = get_variational_covar_with_L_K(model, L_K).clone()

            # Initial moments (model(X) uses whitened storage - correct!)
            lambda_m, lambda_var, f_mean = compute_moments(model, likelihood, X)

            for i in range(n_estep):
                m_prev, V_prev, f_mean_prev = m.clone(), V.clone(), f_mean.clone()

                # Newton update using explicit m (natural space)
                m, V = _estep_single(m, V, r, A, lambda0, 'gpytorch',
                                     model=model, X=X, jitter=jitter)

                # Write back whitened (needed for next model(X) call)
                update_variational_mean_with_L_K(model, m, L_K)
                update_variational_covar_with_L_K(model, V, L_K)
                clear_variational_cache(model)

                # Recompute moments via model(X)
                lambda_m, lambda_var, f_mean = compute_moments(model, likelihood, X)

                # Stability check (C1 fix: unified threshold + NaN check)
                if f_mean.mean() > STABILITY_THRESHOLD or torch.any(torch.isnan(f_mean)):
                    if verbose:
                        print(f"f_mean instability: mean={f_mean.mean():.1f}, has_nan={torch.any(torch.isnan(f_mean)).item()}, reverting")
                    m, V = m_prev, V_prev
                    update_variational_mean_with_L_K(model, m, L_K)
                    update_variational_covar_with_L_K(model, V, L_K)
                    clear_variational_cache(model)
                    lambda_m, lambda_var, f_mean = compute_moments(model, likelihood, X)
                    break

                # Convergence check
                if i > 0:
                    rel_change = (f_mean - f_mean_prev).norm() / (f_mean_prev.norm() + 1e-6)
                    if rel_change < 1e-5:
                        if verbose:
                            print(f"E-step converged after {i+1} iterations (rel_change={rel_change:.2e})")
                        break

        else:
            # NO WHITENING - original behavior (--no-whitening flag)
            # Read m, V from model for non-whitened path
            m = get_variational_mean(model).clone()
            V = get_variational_covar(model).clone()
            lambda_m, lambda_var, f_mean = compute_moments(model, likelihood, X)

            for i in range(n_estep):
                # Save previous state
                m_prev = m.clone()
                V_prev = V.clone()
                f_mean_prev = f_mean.clone()

                # Newton update
                m, V = _estep_single(m, V, r, A, lambda0, 'gpytorch',
                                     model=model, X=X, jitter=jitter)
                update_variational_parameters(model, m, V, jitter)

                # Recompute moments (CRITICAL - old code does this after each Newton step)
                lambda_m, lambda_var, f_mean = compute_moments(model, likelihood, X)

                # Stability check: revert if f_mean is too large or contains NaN (C1 fix)
                if f_mean.mean() > STABILITY_THRESHOLD or torch.any(torch.isnan(f_mean)):
                    if verbose:
                        print(f"f_mean instability: mean={f_mean.mean():.1f}, has_nan={torch.any(torch.isnan(f_mean)).item()}, reverting")
                    m, V = m_prev, V_prev
                    update_variational_parameters(model, m, V, jitter)
                    lambda_m, lambda_var, f_mean = compute_moments(model, likelihood, X)
                    break

                # Convergence check (early stopping)
                if i > 0:
                    rel_change = (f_mean - f_mean_prev).norm() / (f_mean_prev.norm() + 1e-6)
                    if rel_change < 1e-5:
                        if verbose:
                            print(f"E-step converged after {i+1} iterations (rel_change={rel_change:.2e})")
                        break

    return lambda_m, lambda_var


# =============================================================================
# Re-export whitening functions for backward compatibility
# =============================================================================
# Note: These imports at the module level make functions available via:
#   from estep import get_variational_mean, update_variational_parameters, etc.
# This maintains backward compatibility with existing code that imports from estep.
__all__ = [
    # E-step core
    'e_step_loop',
    '_estep_single',
    'e_step',
    'e_step_explicit',
    'e_step_with_kernel_cache',
    '_newton_update',
    # Kernel caching
    'compute_kernel_cache',
    'compute_moments_from_kernel_cache',
    'compute_L_K',
    'compute_moments',
    # Whitening (re-exported from whitening.py)
    '_validate_jitter',
    'set_kernel_requires_grad',
    'get_variational_mean',
    'get_variational_covar',
    'get_variational_mean_with_L_K',
    'update_variational_mean_with_L_K',
    'get_variational_covar_with_L_K',
    'update_variational_covar_with_L_K',
    'update_variational_parameters',
    'update_variational_covar',
    'clear_variational_cache',
]
