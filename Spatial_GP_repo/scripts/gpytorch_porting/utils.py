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


def compute_rf_center_from_sta(X, r, n_px_side, zscore, blur_sigma=3.0):
    """
    Compute receptive field center from spike-triggered average.

    The spike-triggered average (STA) is the weighted average of stimuli
    preceding spikes. For images, this reveals which spatial locations
    drive neural responses. We find the peak of the Gaussian-smoothed
    |STA| to estimate the RF center.

    Uses smoothed argmax rather than center-of-mass because CoM is pulled
    by diffuse background noise across the full image (e.g., 12.9px off
    from the STA peak for PNAS cell 8). Gaussian smoothing before argmax
    reduces noise sensitivity without the CoM bias.

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
        blur_sigma: float, Gaussian blur sigma in pixels before argmax
            (default: 3.0). Algorithmic constant, not an experiment parameter.

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

    # Step 3: Find RF center via smoothed argmax of |STA|
    STA_abs = STA_2d.abs()

    # Build 2D Gaussian blur kernel
    kernel_size = int(4 * blur_sigma + 1) | 1  # ensure odd, covers ~4 sigma
    x_1d = torch.arange(kernel_size, device=X.device, dtype=X.dtype) - kernel_size // 2
    g_1d = torch.exp(-x_1d**2 / (2 * blur_sigma**2))
    g_2d = g_1d[:, None] * g_1d[None, :]
    g_2d = g_2d / g_2d.sum()

    # Smooth |STA| with Gaussian blur
    # conv2d expects (batch, channel, H, W) input and (out_ch, in_ch, kH, kW) kernel
    STA_smooth = torch.nn.functional.conv2d(
        STA_abs.unsqueeze(0).unsqueeze(0),
        g_2d.unsqueeze(0).unsqueeze(0),
        padding=kernel_size // 2
    ).squeeze()

    # Argmax of smoothed |STA|
    peak_flat = STA_smooth.argmax()
    eps_pix_y = (peak_flat // n_px_side).to(X.dtype)
    eps_pix_x = (peak_flat % n_px_side).to(X.dtype)

    # Step 4: Convert pixel coordinates [0, n_px_side-1] to normalized [-1, 1]
    # pixel 0 → -1, pixel (n_px_side-1) → +1
    eps_0x = (eps_pix_x / (n_px_side - 1)) * 2 - 1
    eps_0y = (eps_pix_y / (n_px_side - 1)) * 2 - 1

    return eps_0x.item(), eps_0y.item()


def apply_rf_center_bounds(kernel, eps_0x, eps_0y, config):
    """Apply RF center bounds if enabled in config.

    Constrains the RF center (eps_0x, eps_0y) to stay within
    n_sigma_rf_bounds * sigma_rf of the initial STA estimate,
    where sigma_rf = beta * sqrt(2).

    Warns if n_samples_sta is low (< 100), since the STA-based
    center may be imprecise and tight bounds could lock in a bad estimate.

    Args:
        kernel: ArcCosineKernel (or subclass) with set_center_bounds method
        eps_0x: Initial RF center x (normalized coords)
        eps_0y: Initial RF center y (normalized coords)
        config: Flat config dict with keys: bound_rf_center, n_sigma_rf_bounds,
                beta, n_samples_sta
    """
    import warnings
    import math

    if not config['bound_rf_center']:
        return

    n_sigma = config['n_sigma_rf_bounds']
    sigma_rf = config['beta'] * math.sqrt(2)
    radius = n_sigma * sigma_rf

    n_samples_sta = config['n_samples_sta']
    if n_samples_sta is not None and n_samples_sta < 100:
        warnings.warn(
            f"bound_rf_center=True but n_samples_sta={n_samples_sta} (< 100). "
            f"STA-based RF center may be imprecise — bounds may constrain to wrong location."
        )

    kernel.set_center_bounds(eps_0x, eps_0y, radius)


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
        K = kernel(X_candidates).to_dense()
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

    TODO: model.eval() assumes an nn.Module. DirectVGPModel uses no-op
        eval()/train() shims to satisfy this. Refactor to either use a
        proper protocol/ABC or guard the call with hasattr.
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


def compute_adaptive_rmax(mu_g, sigma2_g, safety_k, max_rmax, min_rmax):
    """Compute adaptive r_max for Laplace truncation.

    Computes an r_max value that ensures the Laplace sum captures the
    probability mass regardless of (mu_g, sigma2_g). Prevents entropy
    collapse when mu_g is high (common with non-stationary kernels).

    Raises ValueError if the needed r_max exceeds max_rmax, because the
    entropy would be wrong (truncation too aggressive).

    Args:
        mu_g: (N,) mean of log-firing rate g = A*lambda + lambda0.
        sigma2_g: (N,) variance of log-firing rate.
        safety_k: Number of standard deviations for upper tail (default 3.0).
        max_rmax: Upper clamp (default 10000).
        min_rmax: Lower clamp / floor (default 200).

    Returns:
        rmax: Integer, adaptive truncation value.

    Raises:
        ValueError: If the needed r_max exceeds max_rmax.
    """
    _EXP_CLAMP = 80.0  # exp(80) ≈ 5.5e34, safely below float32 overflow (~exp(88))

    if max_rmax > int(torch.exp(torch.tensor(_EXP_CLAMP, dtype=mu_g.dtype)).item()):
        raise ValueError(
            f"max_rmax={max_rmax} exceeds exp({_EXP_CLAMP})={torch.exp(torch.tensor(_EXP_CLAMP)).item():.0e}. "
            f"Overflow clamp would hide failures. Use a smaller max_rmax for Laplace approximation."
        )

    upper_logf = mu_g + safety_k * torch.sqrt(sigma2_g)
    max_upper_logf = upper_logf.max().item()

    # Clamp to prevent exp() overflow. The check above guarantees
    # max_rmax < exp(_EXP_CLAMP), so any clamped value still produces
    # needed >> max_rmax and the check below catches it.
    safe_logf = min(max_upper_logf, _EXP_CLAMP)
    upper_rate = float(torch.exp(torch.tensor(safe_logf, dtype=mu_g.dtype)))
    needed = int(upper_rate + 5 * (max(upper_rate, 1.0) ** 0.5) + 10)

    if needed > max_rmax:
        raise ValueError(
            f"Adaptive r_max: needed={needed} exceeds max_rmax={max_rmax} "
            f"(upper_logf={max_upper_logf:.2f}). "
            f"Entropy would be wrong. Check model convergence or A/lambda0 values."
        )

    return max(needed, min_rmax)


def compute_H(mu, sigma2, r_max, a, lambda0):
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


def compute_H_MC(mu, sigma2, n_samples=1000, a=1.0, lambda0=0.0, max_rate=1e10,
                 max_log_contrib=50.0, return_clip_fraction=False):
    """Monte Carlo entropy estimation H(R | mu, sigma2) with variance reduction.

    # BUG RELEVANT WHEN USING THIS FUNCITON: IMPORTANT THIS FUNCITON IS USING HARD CODED VALUES. RAISE TO USER IMEDIATELY
    
    INVESTIGATION CONTEXT:
    Created for investigations/understanding_utility/ (Feb 2026) to trace
    entropy behavior under norm scaling beyond the r_max=100 valid region.
    The fixed-r_max approach (compute_H) breaks when sigma2_g grows large
    (entropy_landscape.png shows the valid region is roughly triangular).
    This MC approach works at any scale by sampling from p(r) instead of
    summing over all r values.

    VARIANCE REDUCTION (clipping):
    For heavy-tailed distributions (large sigma2_g), the naive MC estimator
    has catastrophic variance: a few tail samples with -log p(r) ~ 10^9
    dominate the average. We clip -log p(r) to max_log_contrib, introducing
    bias but making the estimator practical. This is a pragmatic compromise
    for pedagogical exploration.

    AFFIDABILITY METRIC (clip_fraction):
    The fraction of samples that hit the clip threshold is an indicator of
    reliability, analogous to z_safe for Laplace:
      clip_fraction < 0.05: Reliable (bias < 10%)
      clip_fraction > 0.10: Unreliable (bias > 20%, H is overestimated)
    Use return_clip_fraction=True to get this diagnostic.

    USE CASES:
    - Analysis/pedagogy: understanding how H_marg and H_cond behave as images
      are scaled up (c >> 1), where the GP variance grows beyond the
      r_max=100 boundary.
    - NOT for production acquisition functions (not differentiable, biased).
    - NOT needed for natural images (they stay well within the r_max=100
      safe zone with z_safe >> 10).

    Algorithm:
        1. Sample g ~ N(mu_g, sigma2_g)    [log-firing rate from posterior]
        2. Sample r ~ Poisson(exp(g))       [spike count given rate]
        3. Compute log p(r) via Laplace     [single evaluation per sample]
        4. H ≈ -mean(clamp(log p(r), min=-max_log_contrib))  [robust average]

    NOT differentiable (uses torch.no_grad() and discrete Poisson sampling).

    Args:
        mu: (N,) raw GP posterior means (lambda, NOT log-firing rate).
        sigma2: (N,) raw GP posterior variances.
        n_samples: Number of MC samples per query point. 1000 gives ~3% error,
            10000 gives ~1% error.
        a: Firing rate scaling (g = a*lambda + lambda0).
        lambda0: Firing rate offset.
        max_rate: Clamp firing rates above this to avoid Poisson sampler issues.
            Default 1e10 is conservative (torch.poisson works up to ~1e15 but
            float32 loses integer precision above 16.7M).
        max_log_contrib: Clip -log p(r) contributions above this value to
            reduce variance from tail samples. Default 50.0 allows H up to ~50
            nats but prevents extreme outliers from dominating. Introduces bias
            for very large sigma2_g but extends practical range to c~20.
        return_clip_fraction: If True, return (H, clip_fraction) tuple where
            clip_fraction[i] is the fraction of samples that were clipped for
            query point i. This is a reliability metric: < 0.05 is reliable,
            > 0.10 indicates significant bias.

    Returns:
        H: (N,) entropy estimates (biased for large sigma2_g due to clipping).
        clip_fraction: (N,) fraction of clipped samples per query point.
            Only returned if return_clip_fraction=True.

    References:
        investigations/understanding_utility/entropy_landscape.md
        investigations/understanding_utility/H_scaling_comparison.png
    """
    # Not differentiable — sampling from Poisson is discrete
    with torch.no_grad():
        N = mu.shape[0]
        device = mu.device
        dtype = mu.dtype

        # Transform to log-firing rate space
        mu_g = a * mu + lambda0
        sigma2_g = a ** 2 * sigma2

        H = torch.zeros(N, dtype=dtype, device=device)
        clip_fractions = torch.zeros(N, dtype=dtype, device=device) if return_clip_fraction else None

        # Process each query point independently
        # (Could be batched more efficiently but this is clearer pedagogically)
        for i in range(N):
            # Step 1: Sample log-firing rates from the GP posterior
            # g ~ N(mu_g[i], sigma2_g[i])
            g_samples = (torch.randn(n_samples, dtype=dtype, device=device)
                        * torch.sqrt(sigma2_g[i]) + mu_g[i])  # (n_samples,)

            # Step 2: For each g, sample a spike count from Poisson(exp(g))
            # r ~ Poisson(f) where f = exp(g) is the firing rate
            rates = torch.exp(g_samples).clamp(max=max_rate)  # (n_samples,)
            r_samples = torch.poisson(rates)  # (n_samples,) discrete counts

            # Step 3: Evaluate log p(r) at each sampled r using Laplace
            # _diff_laplace_log_probs expects:
            #   mu, sigma2: (M,) — here M=1 (single query point)
            #   r: (R,) — here R=n_samples (the sampled spike counts)
            # Returns: p_r (1, n_samples), log_p_r (1, n_samples)
            mu_i = mu_g[i:i+1]  # (1,) — unsqueeze for batch dimension
            sigma2_i = sigma2_g[i:i+1]  # (1,)
            _, log_p_r = _diff_laplace_log_probs(mu_i, sigma2_i, r_samples)
            # log_p_r shape: (1, n_samples)

            # Step 4: Entropy is the expectation E[-log p(R)]
            # Clip extreme contributions to reduce variance (introduces bias)
            neg_log_p_r = -log_p_r[0]  # (n_samples,)
            neg_log_p_r_clipped = torch.clamp(neg_log_p_r, max=max_log_contrib)

            # Track clipping rate (affidability metric)
            if return_clip_fraction:
                clipped = (neg_log_p_r > max_log_contrib).float()
                clip_fractions[i] = clipped.mean()

            # Monte Carlo estimate: average over clipped samples
            H[i] = neg_log_p_r_clipped.mean()

        if return_clip_fraction:
            return H, clip_fractions
        return H


def nd_utility_new(mu_g, sigma2_g, r_max):
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
