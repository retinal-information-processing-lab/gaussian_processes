"""
Utility functions for GPyTorch porting.

Standalone utilities that avoid importing from utils.py (which has side effects
like torch.pi reinitialization).

Functions:
- compute_rf_center_from_sta: Initialize RF center from spike-triggered average
- lambda0_given_A: Closed-form optimal lambda0 given A (added 2025-02)
- compute_f_mean: Expected firing rate computation (added 2025-02)
- select_inducing_points_pivoted: Pivoted Cholesky inducing point selection (added 2025-02)
- get_gp_marginal_moments: Differentiable GP posterior moments (added 2025-02)
- get_gp_conditional_moments: Differentiable Gaussian conditioning (added 2025-02)
- Differentiable Laplace pipeline: compute_H, nd_utility_new (added 2026-02)
"""

import torch


def compute_rf_center_from_sta(X, r, n_px_side, zscore):
    """
    Compute receptive field center from spike-triggered average.

    The spike-triggered average (STA) is the weighted average of stimuli
    preceding spikes. For images, this reveals which spatial locations
    drive neural responses. We find the center-of-mass of |STA| to get
    a robust estimate of the RF center.

    DETERMINISTIC: Same inputs always produce same outputs.

    Args:
        X: Images tensor, shape (N, n_pixels) where n_pixels = n_px_side^2
        r: Spike counts for one cell, shape (N,)
        n_px_side: Image side length (e.g., 108 for PNAS data)
        zscore: bool, REQUIRED (no default).
            If True: Z-score normalize each pixel across images before
                     computing weighted average. Recommended for natural
                     images where pixel variance varies spatially.
            If False: Use raw pixel values. Simpler but may be biased
                      toward high-variance image regions.

    Returns:
        eps_0x: RF center x-coordinate in normalized [-1, 1] range
        eps_0y: RF center y-coordinate in normalized [-1, 1] range

    Example:
        >>> eps_0x, eps_0y = compute_rf_center_from_sta(X_train, r_train, 108, zscore=True)
        >>> print(f"RF center: ({eps_0x:.3f}, {eps_0y:.3f})")
    """
    # Validate inputs
    if X.shape[1] != n_px_side * n_px_side:
        raise ValueError(f"X has {X.shape[1]} features, expected {n_px_side**2}")
    if X.shape[0] != r.shape[0]:
        raise ValueError(f"X has {X.shape[0]} samples, r has {r.shape[0]}")
    if r.sum() == 0:
        raise ValueError("No spikes in r, cannot compute STA")

    # Step 1: Optionally Z-score normalize images (per-pixel across samples)
    if zscore:
        X_mean = X.mean(dim=0, keepdim=True)
        X_std = X.std(dim=0, keepdim=True)
        X_norm = (X - X_mean) / (X_std + 1e-8)  # Avoid division by zero
    else:
        X_norm = X

    # Step 2: Compute spike-triggered average
    # STA = E[stimulus | spike] ≈ sum(r_i * x_i) / sum(r_i)
    STA = (r[:, None] * X_norm).sum(dim=0) / r.sum()
    STA_2d = STA.reshape(n_px_side, n_px_side)

    # Step 3: Find center-of-mass of |STA|
    # More robust than argmax for noisy estimates
    STA_abs = STA_2d.abs()
    total_mass = STA_abs.sum()

    # Create coordinate grids
    y_coords = torch.arange(n_px_side, device=X.device, dtype=X.dtype)
    x_coords = torch.arange(n_px_side, device=X.device, dtype=X.dtype)

    # Weighted average of coordinates
    # Sum over columns to get row weights, then weight by y-coords
    eps_pix_y = (STA_abs.sum(dim=1) * y_coords).sum() / total_mass
    # Sum over rows to get column weights, then weight by x-coords
    eps_pix_x = (STA_abs.sum(dim=0) * x_coords).sum() / total_mass

    # Step 4: Convert pixel coordinates [0, n_px_side-1] to normalized [-1, 1]
    # pixel 0 → -1, pixel (n_px_side-1) → +1
    eps_0x = (eps_pix_x / (n_px_side - 1)) * 2 - 1
    eps_0y = (eps_pix_y / (n_px_side - 1)) * 2 - 1

    return eps_0x.item(), eps_0y.item()


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

    Raises:
        ValueError: If sum(r) <= 0 (no spikes in training data)
    """
    sumr = r.sum()

    # Guard against zero spike count which would cause log(0) = -inf
    if sumr <= 0:
        raise ValueError("All training spikes are zero. Data problem.")

    expexpr = torch.exp(A * lambda_m + 0.5 * A * A * lambda_var)
    sumexpr = expexpr.sum()
    return torch.log(sumr) - torch.log(sumexpr)


def compute_f_mean(
    lambda_m: torch.Tensor,
    lambda_var: torch.Tensor,
    A: torch.Tensor,
    lambda0: torch.Tensor
) -> torch.Tensor:
    """Compute expected firing rate.

    f_mean = exp(A * lambda_m + 0.5 * A^2 * lambda_var + lambda0)

    Args:
        lambda_m: Posterior mean, shape (N,)
        lambda_var: Posterior variance, shape (N,)
        A: Gain parameter (scalar)
        lambda0: Bias parameter (scalar)

    Returns:
        f_mean: Expected firing rate, shape (N,)
    """
    return torch.exp(A * lambda_m + 0.5 * A * A * lambda_var + lambda0)


def select_inducing_points_pivoted(X, kernel, n_inducing, n_candidates=None, seed=None, jitter=1e-4):
    """Select inducing points via pivoted Cholesky decomposition.

    Greedily selects points that maximize kernel diversity, avoiding
    near-duplicate points that cause ill-conditioned K_uu matrices.
    Each iteration picks the point adding the most new variance beyond
    already-selected points (maximizes Schur complement diagonal).

    Ported from spatiotemporal_gpy/utils.py:154-252.

    Args:
        X: All available data points, shape (n_samples, n_features).
        kernel: Instantiated GPyTorch kernel (e.g., ArcCosineKernel).
            Must be on the same device as X.
        n_inducing: Number of inducing points to select.
        n_candidates: Number of candidate points to consider (subsampled
            from X). None = use all samples. Larger = better selection
            but O(n_candidates^2) memory for the kernel matrix.
        seed: Random seed for candidate subsampling. The pivoted
            selection itself is deterministic given the candidates.
        jitter: Diagonal jitter for numerical stability. Should match
            model jitter (default 1e-4).

    Returns:
        inducing_points: (n_inducing, n_features) selected points,
            ordered by informativeness (first = most informative).
        indices: (n_inducing,) indices into the original X tensor.
    """
    from gpytorch.functions import pivoted_cholesky

    if seed is not None:
        torch.manual_seed(seed)

    n_samples = X.shape[0]
    device = X.device
    dtype = X.dtype

    # Determine number of candidates
    if n_candidates is None:
        n_candidates = n_samples
    n_candidates = min(n_candidates, n_samples)

    if n_inducing > n_candidates:
        import warnings
        warnings.warn(
            f"Requested {n_inducing} inducing points but only {n_candidates} "
            f"candidates. Increasing n_candidates to {n_inducing}."
        )
        n_candidates = n_inducing

    # Subsample candidates
    if n_candidates < n_samples:
        candidate_idx = torch.randperm(n_samples, device=device)[:n_candidates]
    else:
        candidate_idx = torch.arange(n_samples, device=device)

    X_candidates = X[candidate_idx]

    # Compute kernel matrix on candidates
    with torch.no_grad():
        K = kernel(X_candidates).evaluate()
        K = K + jitter * torch.eye(K.shape[0], device=K.device, dtype=K.dtype)

        # Pivoted Cholesky — returns pivots in order of informativeness
        _, pivots = pivoted_cholesky(K, rank=n_inducing, return_pivots=True)

    # Map pivots back to original indices
    selected_candidate_idx = pivots[:n_inducing]
    original_indices = candidate_idx[selected_candidate_idx.to(candidate_idx.device)]

    inducing_points = X[original_indices].clone()

    return inducing_points, original_indices


def get_gp_marginal_moments(model, x_star):
    """Get marginal GP posterior moments at query points.

    Differentiable version — does NOT wrap in torch.no_grad(). Gradient
    flows from x_star through kernel(x_star, inducing) into mu and sigma2.

    Local replacement for gp_utility_playground.get_marginal_moments(),
    which wraps in torch.no_grad() and blocks gradient flow.

    Args:
        model: Trained GP model in eval mode. Must support model(X)
            returning an object with .mean and .variance.
        x_star: (N, d) or (N,) query points.

    Returns:
        mu: (N,) posterior means at x_star.
        sigma2: (N,) posterior variances at x_star.
    """
    model.eval()
    posterior = model(x_star)
    return posterior.mean, posterior.variance


def get_gp_conditional_moments(model, x_star, x_sample, lambda_sample):
    """Compute conditional GP moments after observing lambda(x_sample).

    Differentiable version — does NOT wrap in torch.no_grad(). Supports
    gradient flow from x_star and lambda_sample through the Gaussian
    conditioning formulas.

    Local replacement for utility_2d_rbf_base.get_conditional_moments_nd(),
    which wraps in torch.no_grad() and blocks gradient flow.

    Gaussian conditioning on joint [x_sample; x_star]:
        mu_cond    = mu_star + cross_cov * (lambda_sample - mu_sample) / var_sample
        sigma2_cond = var_star - cross_cov^2 / var_sample

    Args:
        model: Trained GP model in eval mode. Must support
            model(X).covariance_matrix (default_gpy mode).
        x_star: (K, d) query points.
        x_sample: (d,) single observation point tensor.
        lambda_sample: Observed lambda value at x_sample. Must be a
            tensor (scalar or 0-d), NOT a Python float — this preserves
            gradient flow through the reparameterization trick.

    Returns:
        mu_cond: (K,) conditional posterior means.
        sigma2_cond: (K,) conditional posterior variances (clamped >= 1e-8).
    """
    model.eval()

    # Build joint input: [x_sample (1,d); x_star (K,d)]
    x_sample_2d = x_sample.unsqueeze(0)  # (1, d)
    all_x = torch.cat([x_sample_2d, x_star])  # (K+1, d)

    # Joint posterior — full covariance needed for conditioning
    posterior = model(all_x)
    full_covar = posterior.covariance_matrix  # (K+1, K+1)

    # Extract blocks
    mu_sample = posterior.mean[0]
    var_sample = full_covar[0, 0]
    mu_star = posterior.mean[1:]
    var_star = full_covar.diagonal()[1:]
    cross_cov = full_covar[0, 1:]

    # Gaussian conditioning
    innovation = lambda_sample - mu_sample
    mu_cond = mu_star + cross_cov * (innovation / var_sample)
    sigma2_cond = var_star - (cross_cov ** 2) / var_sample
    sigma2_cond = torch.clamp(sigma2_cond, min=1e-8)

    return mu_cond, sigma2_cond


# ============================================================================
# Differentiable Laplace Approximation Pipeline
# ============================================================================
# Local, gradient-compatible versions of the Laplace approximation for
# Poisson-GP entropy computation. Replaces utility.py:laplace_approximations_new
# and utility.py:nd_utility_new, which use indexed assignment (torch.empty +
# __setitem__) that breaks autograd.
#
# Also replaces gp_utility_playground.py:compute_H.
#
# The Lambert W function is copied from utility.py:LambertWLogFunction for
# self-containment — it has a proper custom backward (dW/dy = W/(1+W)).
# ============================================================================

_TINY = 1e-30


class _LambertWLogFunction(torch.autograd.Function):
    """Compute W_0(exp(y)) without computing exp(y).

    Solves w + log(w) = y via Newton iteration.
    Copied from utility.py:LambertWLogFunction for local self-containment.
    """

    @staticmethod
    def forward(ctx, y):
        safe_log_input = y.clamp(min=1.0)
        w = torch.where(y >= 2.0, y - torch.log(safe_log_input),
            torch.where(y >= 0.0, 0.567 + 0.5 * y,
                        torch.exp(y)))
        for _ in range(10):
            w = w.clamp(min=_TINY)
            log_w = torch.log(w)
            w = w - w * (w + log_w - y) / (w + 1.0)
        w = w.clamp(min=_TINY)
        ctx.save_for_backward(w)
        return w

    @staticmethod
    def backward(ctx, grad_output):
        w, = ctx.saved_tensors
        # dW/dy = W / (1 + W)  — the e^y terms cancel
        return grad_output * w / (1.0 + w)


def _lambertw0_log(y):
    """Compute W_0(exp(y)) via custom autograd function."""
    return _LambertWLogFunction.apply(y)


def _diff_argmax_g(r, sigma2, mu):
    """Mode of the Laplace approximation.

    g_bar = r*sigma2 + mu - W_0(sigma2 * exp(r*sigma2 + mu))

    Differentiable version of utility.py:argmax_g.

    Args:
        r: (R,) spike count values.
        sigma2: (N,) variance of log-firing rate.
        mu: (N,) mean of log-firing rate.

    Returns:
        g_bar: (N, R) mode values.
    """
    rsigma2 = sigma2[:, None] * r[None, :]  # (N, R)
    y = torch.log(sigma2.clamp(min=_TINY))[:, None] + rsigma2 + mu[:, None]  # (N, R)
    lambert_W_result = _lambertw0_log(y)
    return rsigma2 + mu[:, None] - lambert_W_result


def _diff_laplace_log_probs(mu, sigma2, r):
    """Laplace approximation of log p(r|x,D), fully differentiable.

    Replaces utility.py:laplace_approximations_new. The key difference:
    uses torch.where instead of indexed assignment into a pre-allocated
    tensor, which preserves the autograd computational graph.

    For samples with sigma2 < 1e-6, falls back to exact Poisson via
    torch.where (differentiable branching).

    Args:
        mu: (N,) mean of log-firing rate g = A*lambda + lambda0.
        sigma2: (N,) variance of log-firing rate.
        r: (R,) spike count values.

    Returns:
        p_r: (N, R) probabilities.
        log_p_r: (N, R) log probabilities.
    """
    log_r_fact = torch.lgamma(r + 1)  # (R,)

    # Laplace path (clamp sigma2 for numerical safety)
    sigma2_safe = sigma2.clamp(min=1e-10)
    g_bar = _diff_argmax_g(r, sigma2_safe, mu)  # (N, R)
    exp_g_bar = torch.exp(g_bar)
    log_p_laplace = (g_bar * r[None, :]
                     - exp_g_bar
                     - ((g_bar - mu[:, None]) ** 2) / (2 * sigma2_safe[:, None])
                     - 0.5 * torch.log1p(sigma2_safe[:, None] * exp_g_bar)
                     - log_r_fact[None, :])

    # Exact Poisson path (for small sigma2 where Laplace is unstable)
    log_p_poisson = (mu[:, None] * r[None, :]
                     - torch.exp(mu[:, None])
                     - log_r_fact[None, :])

    # Select via torch.where (differentiable branching)
    small_var = (sigma2 < 1e-6).unsqueeze(1)  # (N, 1)
    log_p = torch.where(small_var, log_p_poisson, log_p_laplace)

    return torch.exp(log_p), log_p


def compute_H(mu, sigma2, r_max=100, a=1.0, lambda0=0.0):
    """Compute entropy H(R | mu, sigma2) using Laplace approximation.

    Differentiable — supports gradient flow through mu and sigma2.
    Local copy of gp_utility_playground.compute_H, rewritten to use
    torch.where instead of indexed assignment (which breaks autograd).
    Transforms raw GP moments to log-firing rate, then computes entropy.

    Args:
        mu: (N,) raw GP posterior means (lambda, NOT log-firing rate).
        sigma2: (N,) raw GP posterior variances.
        r_max: Max spike count for truncation.
        a: Firing rate scaling (g = a*lambda + lambda0).
        lambda0: Firing rate offset.

    Returns:
        H: (N,) entropy values.
    """
    if torch.any(sigma2 < 0):
        raise ValueError(f"sigma2 must be non-negative. Min: {sigma2.min().item()}")

    r = torch.arange(0, r_max, dtype=mu.dtype, device=mu.device)
    logf_mean = a * mu + lambda0
    logf_var = a ** 2 * sigma2
    p_r, log_p_r = _diff_laplace_log_probs(logf_mean, logf_var, r)
    H = -torch.sum(p_r * log_p_r, dim=1)
    return H


def nd_utility_new(mu_g, sigma2_g, r_max=100):
    """Compute standard utility U = H_marg - E[H_noise].

    Differentiable — supports gradient flow through mu_g and sigma2_g.
    Local copy of utility.py:nd_utility_new, rewritten to use torch.where
    instead of indexed assignment (which breaks autograd).
    Takes pre-transformed log-firing rate moments (g = A*lambda + lambda0).

    Args:
        mu_g: (N,) mean of log-firing rate.
        sigma2_g: (N,) variance of log-firing rate.
        r_max: Max spike count.

    Returns:
        utility: (N,) utility values.
    """
    if sigma2_g.ndim == 0:
        sigma2_g = sigma2_g[None]
        mu_g = mu_g[None]

    r = torch.arange(0, r_max + 1, dtype=mu_g.dtype, device=mu_g.device)

    # Laplace approximation
    p_r, log_p_r = _diff_laplace_log_probs(mu_g, sigma2_g, r)

    # Marginal entropy H(R|x,D)
    H_marg = -torch.sum(p_r * log_p_r, dim=1)

    # Conditional entropy E[H(R|f,x)] (Eq. 33 PNAS)
    log_r_fact = torch.lgamma(r + 1)
    p_times_logr_sum = torch.sum(p_r * log_r_fact[None, :], dim=1)
    E_H_noise = -torch.exp(mu_g + 0.5 * sigma2_g) * (mu_g + sigma2_g - 1) + p_times_logr_sum

    return H_marg - E_H_noise
