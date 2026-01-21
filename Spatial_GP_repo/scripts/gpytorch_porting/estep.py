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
from typing import Tuple, Optional, Dict


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
    # Only compute if model uses whitening (needed for whitened <-> natural conversions)
    if getattr(model, 'whitening', True):
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
    K_tilde_j = K_tilde + jitter * torch.eye(M, dtype=K_tilde.dtype, device=K_tilde.device)
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


def e_step_with_kernel_cache(
    m: torch.Tensor,
    V: torch.Tensor,
    kernel_cache: Dict[str, torch.Tensor],
    A: torch.Tensor,
    lambda0: torch.Tensor,
    r: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Perform one E-step using cached kernel matrices (bypasses GPyTorch).

    This is the optimized version of e_step() that reuses pre-computed K and K̃
    matrices instead of calling GPyTorch's model(X).

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
    K = kernel_cache['K']
    K_tilde_j = kernel_cache['K_tilde_j']

    # Compute moments using cached kernel matrices (not GPyTorch model(X))
    lambda_m, lambda_var = compute_moments_from_kernel_cache(kernel_cache, m, V)

    # Expected firing rate: f̄ = exp(A·μ + ½A²σ² + λ₀)
    f_mean = torch.exp(A * lambda_m + 0.5 * A**2 * lambda_var + lambda0)

    return _newton_update(m, K, K_tilde_j, A, f_mean, r)


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


# =============================================================================
# varGP-style training functions (matching utils.py:varGP structure)
# =============================================================================

def get_variational_mean(model: gpytorch.models.ApproximateGP) -> torch.Tensor:
    """Get the variational mean m from the model."""
    return model.variational_strategy._variational_distribution.variational_mean


def get_variational_covar(model: gpytorch.models.ApproximateGP) -> torch.Tensor:
    """Get the variational covariance V from the model (V = LLᵀ)."""
    L = model.variational_strategy._variational_distribution.chol_variational_covar
    return L @ L.T


# =============================================================================
# Whitening Conversion Functions (2026-01-19)
# =============================================================================
# GPyTorch stores whitened variational params, but our E-step produces natural params.
# These functions convert between the two representations at the storage boundary.
#
# GPyTorch formula: λ_m = K_XZ @ L_K⁻ᵀ @ m_stored  (expects whitened m)
# Standard SVGP:    λ_m = K_XZ @ K̃⁻¹ @ m          (expects natural m)
#
# When we store natural m directly, GPyTorch computes wrong λ_m (~8x smaller).
# Fix: Convert m at read/write boundaries.
#
# STATUS: IMPLEMENTED but DISABLED in e_step_loop() (2026-01-19)
# These functions work correctly (verified by tests/test_m_whitening.py Test 1 & 2).
# However, using whitened m with non-whitened V causes incorrect KL divergence
# computation in M-step, leading to worse training results.
# The whitening will be re-enabled after implementing V whitening (separate task).

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


def lambda0_given_A(
    A: torch.Tensor,
    r: torch.Tensor,
    lambda_m: torch.Tensor,
    lambda_var: torch.Tensor
) -> torch.Tensor:
    """Closed-form optimal lambda0 given A.

    Derived from setting dL/d(lambda0) = 0 where L is the expected log-likelihood.
    This matches utils.py:lambda0_given_logA() but takes A directly (not logA).

    The expected log-likelihood contains:
        E[r*lambda0 - exp(A*lambda + lambda0)]
      = r*lambda0 - exp(lambda0)*E[exp(A*lambda)]
      = r*lambda0 - exp(lambda0)*exp(A*lambda_m + 0.5*A^2*lambda_var)

    Setting d/d(lambda0) = 0:
        sum(r) = exp(lambda0) * sum(exp(A*lambda_m + 0.5*A^2*lambda_var))

    Solution:
        lambda0 = log(sum(r)) - log(sum(exp(A*lambda_m + 0.5*A^2*lambda_var)))

    Args:
        A: Gain parameter (scalar tensor)
        r: Spike counts, shape (N,)
        lambda_m: GP posterior mean, shape (N,)
        lambda_var: GP posterior variance, shape (N,)

    Returns:
        Optimal lambda0 (scalar tensor)
    """
    sumr = r.sum()
    expexpr = torch.exp(A * lambda_m + 0.5 * A * A * lambda_var)
    sumexpr = expexpr.sum()
    return torch.log(sumr) - torch.log(sumexpr)


def e_step_loop(
    model: gpytorch.models.ApproximateGP,
    likelihood,
    X: torch.Tensor,
    r: torch.Tensor,
    n_estep: int,
    jitter: Optional[float] = None,
    verbose: bool = False,
    kernel_cache: Optional[Dict[str, torch.Tensor]] = None,
    use_whitening: Optional[bool] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Run Newton loop with moment recomputation and stability checks.

    This matches the structure of the old varGP E-step loop (utils.py:5664-5712):
    - Save previous state before each Newton step
    - Recompute moments after each Newton step (CRITICAL)
    - Stability check: revert if f_mean.mean() > 1000
    - Early stopping: break if rel_change < 1e-5

    PERFORMANCE OPTIMIZATION (2026-01-18):
        When kernel_cache is provided, K and K̃ matrices are reused instead of
        recomputed via GPyTorch. This reduces kernel calls from 35 to 3 per loop.

    WHITENING (2026-01-19):
        When use_whitening=True, variational parameters are converted between
        whitened (GPyTorch storage) and natural (E-step computation) forms.
        This ensures mathematically correct behavior. Set use_whitening=False to
        use the original "wrong but self-consistent" behavior for comparison.

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
        use_whitening: If None (default), auto-detect from model.whitening attribute.
                       If True, convert between whitened and natural params.
                       If False, use original behavior (no whitening conversions).

    Returns:
        lambda_m: Final posterior mean of λ at X, shape (N,)
        lambda_var: Final posterior variance of λ at X, shape (N,)
    """
    # Validate jitter - must match model.jitter for consistency
    jitter = _validate_jitter(jitter, model)

    # Auto-detect whitening from model if not specified
    if use_whitening is None:
        use_whitening = getattr(model, 'whitening', True) # BUG, SHOULD raise error if model has no whitening attr

    # Get likelihood parameters
    A = likelihood.A.squeeze()
    lambda0 = likelihood.lambda0.squeeze()

    # Use cached kernels if provided, otherwise use original (non-cached) path
    if kernel_cache is not None:
        # =====================================================================
        # CACHED PATH: Use pre-computed kernel matrices (bypasses GPyTorch)
        # =====================================================================
        if use_whitening:
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
            m, V = e_step_with_kernel_cache(m, V, kernel_cache, A, lambda0, r)

            # Recompute moments using cached kernels
            lambda_m, lambda_var = compute_moments_from_kernel_cache(kernel_cache, m, V)
            f_mean = torch.exp(A * lambda_m + 0.5 * A**2 * lambda_var + lambda0)

            # Stability check: revert if f_mean is too large
            if f_mean.mean() > 1000:
                if verbose:
                    print(f"f_mean.mean() = {f_mean.mean():.1f} > 1000, reverting to previous state")
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
        if use_whitening:
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
        if use_whitening:
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
                m, V = e_step_explicit(m, model, likelihood, X, r, jitter)

                # Write back whitened (needed for next model(X) call)
                update_variational_mean_with_L_K(model, m, L_K)
                update_variational_covar_with_L_K(model, V, L_K)
                clear_variational_cache(model)

                # Recompute moments via model(X)
                lambda_m, lambda_var, f_mean = compute_moments(model, likelihood, X)

                # Stability check
                if f_mean.mean() > 1000:
                    if verbose:
                        print(f"f_mean.mean() = {f_mean.mean():.1f} > 1000, reverting")
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
            lambda_m, lambda_var, f_mean = compute_moments(model, likelihood, X)

            for i in range(n_estep):
                # Save previous state
                m_prev = get_variational_mean(model).clone()
                V_prev = get_variational_covar(model).clone()
                f_mean_prev = f_mean.clone()

                # Newton update
                m_new, V_new = e_step(model, likelihood, X, r, jitter)
                update_variational_parameters(model, m_new, V_new, jitter)

                # Recompute moments (CRITICAL - old code does this after each Newton step)
                lambda_m, lambda_var, f_mean = compute_moments(model, likelihood, X)

                # Stability check: revert if f_mean is too large
                if f_mean.mean() > 1000:
                    if verbose:
                        print(f"f_mean.mean() = {f_mean.mean():.1f} > 1000, reverting to previous state")
                    update_variational_parameters(model, m_prev, V_prev, jitter)
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


def f_step(
    model: gpytorch.models.ApproximateGP,
    likelihood,
    X: torch.Tensor,
    r: torch.Tensor,
    lambda_m: torch.Tensor,
    lambda_var: torch.Tensor,
    n_fstep: int,
    lr: float = 0.01,
    verbose: bool = False
):
    """F-step: Optimize A with Adam, lambda0 computed analytically.

    Structural change from baseline: lambda0 is set analytically (not optimized).
    Only A is optimized via gradient descent.

    Args:
        model: VariationalGPModel instance
        likelihood: PoissonLikelihood instance
        X: Training inputs, shape (N, n_features)
        r: Training spike counts, shape (N,)
        lambda_m: GP posterior mean (held fixed), shape (N,)
        lambda_var: GP posterior variance (held fixed), shape (N,)
        n_fstep: Number of Adam iterations
        lr: Learning rate for Adam
        verbose: Print debug info
    """
    # First set analytical lambda0
    A = likelihood.A.squeeze()
    new_lambda0 = lambda0_given_A(A, r, lambda_m, lambda_var)
    with torch.no_grad():
        likelihood.lambda0.copy_(new_lambda0.unsqueeze(0))

    if n_fstep == 0:
        return

    optimizer = torch.optim.Adam([likelihood.raw_A], lr=lr)

    for _ in range(n_fstep):
        optimizer.zero_grad()

        # Update lambda0 analytically for current A
        A = likelihood.A.squeeze()
        with torch.no_grad():
            new_lambda0 = lambda0_given_A(A, r, lambda_m, lambda_var)
            likelihood.lambda0.copy_(new_lambda0.unsqueeze(0))

        # Compute loss
        output = model(X)
        loss = -likelihood.expected_log_prob(r, output) + \
               model.variational_strategy.kl_divergence()

        loss.backward()
        optimizer.step()

    # Final lambda0 update
    A = likelihood.A.squeeze()
    with torch.no_grad():
        new_lambda0 = lambda0_given_A(A, r, lambda_m, lambda_var)
        likelihood.lambda0.copy_(new_lambda0.unsqueeze(0))


def f_step_lbfgs(
    model: gpytorch.models.ApproximateGP,
    likelihood,
    X: torch.Tensor,
    r: torch.Tensor,
    lambda_m: torch.Tensor,
    lambda_var: torch.Tensor,
    n_fstep: int,
    lr: float = 0.1,
    verbose: bool = False
):
    """F-step using LBFGS optimizer - matches original varGP exactly.

    This replicates utils.py:varGP() F-step structure:
    - Uses LBFGS with strong_wolfe line search
    - Optimizes logA (raw_A = logA since A = exp(raw_A))
    - Computes gradients manually via analytical formula
    - lambda0 set analytically inside closure
    - Stability check: returns inf if f_mean.mean() > 100

    Args:
        model: VariationalGPModel instance
        likelihood: PoissonLikelihood instance
        X: Training inputs, shape (N, n_features)
        r: Training spike counts, shape (N,)
        lambda_m: GP posterior mean (held fixed), shape (N,)
        lambda_var: GP posterior variance (held fixed), shape (N,)
        n_fstep: Number of LBFGS iterations (max_iter)
        lr: Learning rate for LBFGS (default 0.1 matches varGP)
        verbose: Print debug info
    """
    if n_fstep == 0:
        return

    # Get current A and convert to logA (varGP uses logA parameterization)
    A_current = likelihood.A.squeeze().detach()
    logA = torch.log(A_current).clone().requires_grad_(True)

    # Initial lambda0 update
    A = torch.exp(logA)
    new_lambda0 = lambda0_given_A(A, r, lambda_m, lambda_var)
    with torch.no_grad():
        likelihood.lambda0.copy_(new_lambda0.reshape(likelihood.lambda0.shape))

    # Track f_mean across closure calls (nonlocal update like original)
    f_mean_container = [None]

    # LBFGS optimizer matching original varGP settings
    optimizer = torch.optim.LBFGS(
        [logA],
        lr=lr,
        max_iter=n_fstep,
        tolerance_change=1e-9,
        tolerance_grad=1e-7,
        history_size=n_fstep,
        line_search_fn='strong_wolfe'
    )

    closure_counter = [0]

    def closure():
        closure_counter[0] += 1
        optimizer.zero_grad()

        # Get current A from logA
        A = torch.exp(logA)

        # Update lambda0 analytically (inside closure, like original)
        with torch.no_grad():
            lambda0 = lambda0_given_A(A, r, lambda_m, lambda_var)
            likelihood.lambda0.copy_(lambda0.reshape(likelihood.lambda0.shape))

        # Compute f_mean = exp(A*lambda_m + 0.5*A^2*lambda_var + lambda0)
        f_mean = torch.exp(A * lambda_m + 0.5 * A * A * lambda_var + lambda0)
        f_mean_container[0] = f_mean

        # Stability check: return inf if f_mean is too large (like original)
        if f_mean.mean() > 100 or torch.any(torch.isnan(f_mean)):
            if verbose:
                print(f"f_mean.mean() = {f_mean.mean():.1f} at closure call {closure_counter[0]}, returning inf")
            return torch.tensor(float('inf'), device=logA.device, dtype=logA.dtype)

        # Compute loglikelihood: L = A*r@lambda_m + lambda0*sum(r) - sum(f_mean)
        rlambda_m = r @ lambda_m
        sum_r = r.sum()
        loglikelihood = A * rlambda_m + lambda0 * sum_r - f_mean.sum()

        # Compute gradient of loglikelihood w.r.t. logA (analytical, like original)
        # dL/dlogA = A * (r@lambda_m - (lambda_m + A*lambda_var) @ f_mean)
        dloglikelihood_dlogA = A * (rlambda_m - torch.dot(lambda_m + A * lambda_var, f_mean))

        # Set gradient manually (negative because LBFGS minimizes)
        logA.grad = -dloglikelihood_dlogA

        # Return negative loglikelihood (minimize)
        return -loglikelihood

    # Run LBFGS
    optimizer.step(closure)

    # Final updates after optimization
    with torch.no_grad():
        # Update likelihood's raw_A from optimized logA
        # raw_A = logA since A = exp(raw_A)
        likelihood.raw_A.copy_(logA.reshape(likelihood.raw_A.shape))

        # Final lambda0 update (like original: "the optimal logA value found by
        # the optimizer might not be the one used in the last closure call")
        A_check = likelihood.A.squeeze()
        new_lambda0 = lambda0_given_A(A_check, r, lambda_m, lambda_var)
        likelihood.lambda0.copy_(new_lambda0.reshape(likelihood.lambda0.shape))

    if verbose:
        print(f"F-step LBFGS: {closure_counter[0]} closure calls, A: {likelihood.A.item():.4f}")


def m_step(
    model: gpytorch.models.ApproximateGP,
    likelihood,
    X: torch.Tensor,
    r: torch.Tensor,
    n_mstep: int,
    lr: float = 0.01,
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


def m_step_lbfgs(
    model: gpytorch.models.ApproximateGP,
    likelihood,
    X: torch.Tensor,
    r: torch.Tensor,
    n_mstep: int,
    lr: float = 0.1,  # varGP default
    verbose: bool = False,
    debug: bool = False
):
    """M-step: Optimize kernel hyperparameters using LBFGS.

    Uses autograd for gradients (not analytical).
    Matches varGP LBFGS settings: lr=0.1, strong_wolfe line search.

    Key feature: Returns infinite loss when parameters exceed bounds (matching
    original varGP behavior). This forces LBFGS to try smaller steps.

    Args:
        model: VariationalGPModel instance
        likelihood: PoissonLikelihood instance
        X: Training inputs, shape (N, n_features)
        r: Training spike counts, shape (N,)
        n_mstep: Number of LBFGS iterations (max_iter)
        lr: Learning rate for LBFGS (default 0.1 matches varGP)
        verbose: Print debug info
        debug: Print detailed debugging info

    DEPRECATED: This function does not support hyperparameter clamping.
    Use m_step() instead.
    """
    raise NotImplementedError(
        "m_step_lbfgs() is deprecated and does not support hyperparameter clamping. "
        "Use m_step() (Adam-based) instead."
    )

    if n_mstep == 0:
        return

    # Get kernel parameters (kernel is ArcCosineKernel directly, not wrapped)
    kernel_params = list(model.covar_module.parameters())

    # Debug: capture initial state
    if debug:
        with torch.no_grad():
            output_init = model(X)
            ell_init = likelihood.expected_log_prob(r, output_init)
            kl_init = model.variational_strategy.kl_divergence()
            loss_init = (-ell_init + kl_init).item()
        param_init = {name: p.clone().detach() for name, p in model.covar_module.named_parameters()}
        print(f"  M-step LBFGS DEBUG: initial loss = {loss_init:.2f}")
        for name, p in param_init.items():
            print(f"    {name}: {p.item():.6f}" if p.numel() == 1 else f"    {name}: shape {p.shape}")

    # LBFGS optimizer matching varGP settings
    optimizer = torch.optim.LBFGS(
        kernel_params,
        lr=lr,
        max_iter=n_mstep,
        tolerance_change=1e-9,
        tolerance_grad=1e-7,
        history_size=100,
        line_search_fn='strong_wolfe'
    )

    closure_counter = [0]
    bounds_violations = [0]
    grad_norms = []

    def closure():
        closure_counter[0] += 1
        optimizer.zero_grad()

        # Check parameter bounds (matching varGP behavior)
        # If bounds violated, set gradient to inf and return inf loss
        # This tells LBFGS to try a smaller step
        kernel = model.covar_module  # ArcCosineKernel directly (not wrapped)
        return_infinite_loss = False
        if hasattr(kernel, 'eps_0x') and hasattr(kernel, 'eps_0y'):
            eps_0x = kernel.eps_0x.item()
            eps_0y = kernel.eps_0y.item()
            # Bounds: eps_0 should be in [-0.99, 0.99] (slightly inside image boundary)
            # Using 0.99 instead of 1.0 to keep RF center well inside image
            if not (-0.99 <= eps_0x <= 0.99):
                return_infinite_loss = True
                if kernel.eps_0x.requires_grad:
                    kernel.eps_0x.grad = torch.full_like(kernel.eps_0x, float('inf'))
            if not (-0.99 <= eps_0y <= 0.99):
                return_infinite_loss = True
                if kernel.eps_0y.requires_grad:
                    kernel.eps_0y.grad = torch.full_like(kernel.eps_0y, float('inf'))

        if return_infinite_loss:
            bounds_violations[0] += 1
            if debug:
                print(f"  Bounds violation at closure {closure_counter[0]}: "
                      f"eps_0=({eps_0x:.3f}, {eps_0y:.3f})")
            return torch.tensor(float('inf'), device=X.device, dtype=X.dtype)

        # Forward pass
        output = model(X)

        # Compute ELBO loss
        ell = likelihood.expected_log_prob(r, output)
        kl = model.variational_strategy.kl_divergence()
        loss = -ell + kl

        # Check for NaN/inf
        if torch.isnan(loss) or torch.isinf(loss):
            bounds_violations[0] += 1
            # Set all gradients to inf
            for p in kernel_params:
                if p.requires_grad:
                    p.grad = torch.full_like(p, float('inf'))
            return torch.tensor(float('inf'), device=X.device, dtype=X.dtype)

        # Backward pass (autograd)
        loss.backward()

        # Debug: track gradient norms
        if debug:
            total_grad_norm = 0.0
            for p in kernel_params:
                if p.grad is not None:
                    total_grad_norm += p.grad.norm().item() ** 2
            grad_norms.append(total_grad_norm ** 0.5)

        return loss

    # Run LBFGS (single step, but closure called multiple times)
    optimizer.step(closure)

    # Debug: capture final state
    if debug:
        with torch.no_grad():
            output_final = model(X)
            ell_final = likelihood.expected_log_prob(r, output_final)
            kl_final = model.variational_strategy.kl_divergence()
            loss_final = (-ell_final + kl_final).item()
        print(f"  M-step LBFGS DEBUG: final loss = {loss_final:.2f} (delta = {loss_final - loss_init:.2f})")
        print(f"  M-step LBFGS DEBUG: {closure_counter[0]} closure calls, {bounds_violations[0]} bounds violations")
        print(f"  M-step LBFGS DEBUG: grad norms: {grad_norms[:5]}...")
        for name, p in model.covar_module.named_parameters():
            p_init = param_init[name]
            delta = (p - p_init).abs().max().item()
            print(f"    {name}: {p.item():.6f} (delta={delta:.6f})" if p.numel() == 1 else f"    {name}: max_delta={delta:.6f}")

    if verbose:
        print(f"M-step LBFGS: {closure_counter[0]} closure calls, {bounds_violations[0]} bounds violations")


def m_step_lbfgs_grouped(
    model: gpytorch.models.ApproximateGP,
    likelihood,
    X: torch.Tensor,
    r: torch.Tensor,
    n_mstep: int,
    lr_center: float = 0.1,
    lr_sigma0: float = 1.0,  # 10x larger for sigma_0
    lr_other: float = 0.1,
    verbose: bool = False
):
    """M-step with grouped LBFGS: separate optimizers for different parameter groups.

    Splits kernel parameters into 3 groups with different learning rates:
    1. RF center (eps_0x, eps_0y): lr_center, with bounds checking
    2. sigma_0: lr_sigma0 (larger, since gradient is small)
    3. Other (Amp, beta, rho): lr_other

    This is block coordinate descent - each group optimized while others held fixed.

    Args:
        model: VariationalGPModel instance
        likelihood: PoissonLikelihood instance
        X: Training inputs, shape (N, n_features)
        r: Training spike counts, shape (N,)
        n_mstep: Number of LBFGS iterations per group
        lr_center: Learning rate for RF center
        lr_sigma0: Learning rate for sigma_0 (default 10x larger)
        lr_other: Learning rate for other parameters
        verbose: Print debug info

    DEPRECATED: This function does not support hyperparameter clamping.
    Use m_step() instead.
    """
    raise NotImplementedError(
        "m_step_lbfgs_grouped() is deprecated and does not support hyperparameter clamping. "
        "Use m_step() (Adam-based) instead."
    )

    if n_mstep == 0:
        return

    kernel = model.covar_module  # ArcCosineKernel directly (not wrapped)

    # Identify parameter groups
    center_params = []
    sigma0_params = []
    other_params = []

    for name, p in model.covar_module.named_parameters():
        if 'eps_0x' in name or 'eps_0y' in name:
            center_params.append(p)
        elif 'sigma_0' in name:
            sigma0_params.append(p)
        else:
            other_params.append(p)

    def make_closure(params_to_optimize):
        """Create closure that only computes gradients for specified params."""
        def closure():
            # Zero all gradients
            for p in model.covar_module.parameters():
                if p.grad is not None:
                    p.grad.zero_()

            # Forward pass
            output = model(X)
            ell = likelihood.expected_log_prob(r, output)
            kl = model.variational_strategy.kl_divergence()
            loss = -ell + kl

            if torch.isnan(loss) or torch.isinf(loss):
                return torch.tensor(float('inf'), device=X.device, dtype=X.dtype)

            loss.backward()
            return loss
        return closure

    def make_center_closure():
        """Closure for RF center with bounds checking."""
        def closure():
            for p in model.covar_module.parameters():
                if p.grad is not None:
                    p.grad.zero_()

            # Bounds check for eps_0
            if hasattr(kernel, 'eps_0x') and hasattr(kernel, 'eps_0y'):
                eps_0x = kernel.eps_0x.item()
                eps_0y = kernel.eps_0y.item()
                if not (-0.99 <= eps_0x <= 0.99):
                    if kernel.eps_0x.requires_grad:
                        kernel.eps_0x.grad = torch.full_like(kernel.eps_0x, float('inf'))
                    return torch.tensor(float('inf'), device=X.device, dtype=X.dtype)
                if not (-0.99 <= eps_0y <= 0.99):
                    if kernel.eps_0y.requires_grad:
                        kernel.eps_0y.grad = torch.full_like(kernel.eps_0y, float('inf'))
                    return torch.tensor(float('inf'), device=X.device, dtype=X.dtype)

            output = model(X)
            ell = likelihood.expected_log_prob(r, output)
            kl = model.variational_strategy.kl_divergence()
            loss = -ell + kl

            if torch.isnan(loss) or torch.isinf(loss):
                return torch.tensor(float('inf'), device=X.device, dtype=X.dtype)

            loss.backward()
            return loss
        return closure

    total_closures = [0]

    # Group 1: sigma_0 - Use ADAM (handles small gradients well)
    # LBFGS overshoots for this parameter due to tiny gradient magnitude
    if sigma0_params:
        optimizer = torch.optim.Adam(sigma0_params, lr=lr_sigma0)
        for _ in range(n_mstep):
            optimizer.zero_grad()
            output = model(X)
            ell = likelihood.expected_log_prob(r, output)
            kl = model.variational_strategy.kl_divergence()
            loss = -ell + kl
            if not (torch.isnan(loss) or torch.isinf(loss)):
                loss.backward()
                optimizer.step()
        total_closures[0] += n_mstep

    # Group 2: RF center (eps_0x, eps_0y) with bounds
    if center_params:
        optimizer = torch.optim.LBFGS(
            center_params, lr=lr_center, max_iter=n_mstep,
            tolerance_change=1e-9, tolerance_grad=1e-7,
            history_size=100, line_search_fn='strong_wolfe'
        )
        closure = make_center_closure()
        closure_count = [0]
        def counted_closure():
            closure_count[0] += 1
            return closure()
        optimizer.step(counted_closure)
        total_closures[0] += closure_count[0]

    # Group 3: Other params (Amp, beta, rho)
    if other_params:
        optimizer = torch.optim.LBFGS(
            other_params, lr=lr_other, max_iter=n_mstep,
            tolerance_change=1e-9, tolerance_grad=1e-7,
            history_size=100, line_search_fn='strong_wolfe'
        )
        closure = make_closure(other_params)
        closure_count = [0]
        def counted_closure():
            closure_count[0] += 1
            return closure()
        optimizer.step(counted_closure)
        total_closures[0] += closure_count[0]

    if verbose:
        print(f"M-step LBFGS grouped: {total_closures[0]} total closure calls")


# =============================================================================
# Training loops moved to train.py
# =============================================================================
# train_varGP_style and train_efm are now in train.py to consolidate all
# training loops in one module. Import from there:
#   from train import train_varGP_style, train_efm
# =============================================================================
