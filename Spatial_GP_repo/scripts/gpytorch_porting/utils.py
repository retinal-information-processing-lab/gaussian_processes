"""
Utility functions for GPyTorch porting.

Standalone utilities that avoid importing from utils.py (which has side effects
like torch.pi reinitialization).

Functions:
- compute_rf_center_from_sta: Initialize RF center from spike-triggered average
- lambda0_given_A: Closed-form optimal lambda0 given A (added 2025-02)
- compute_f_mean: Expected firing rate computation (added 2025-02)
- select_inducing_points_pivoted: Pivoted Cholesky inducing point selection (added 2025-02)
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
