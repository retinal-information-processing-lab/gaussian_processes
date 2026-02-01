"""
Utility functions for GPyTorch porting.

Standalone utilities that avoid importing from utils.py (which has side effects
like torch.pi reinitialization).
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
