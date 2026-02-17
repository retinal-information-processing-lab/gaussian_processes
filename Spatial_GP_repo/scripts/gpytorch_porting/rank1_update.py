"""
Rank-1 Model Extension for Active Learning

Creates a NEW DirectVGPModel with M+1 training/inducing points, warm-started
from an existing model with M points. Used during Phase 2 of the closed-loop
experiment when a new image is added to the training set.

Approach (first version): Let the DirectVGPModel constructor recompute K_tilde
from scratch. At M=300 on GPU, the cost is negligible vs eigendecomposition.
TODO (future optimization): Replace full K_tilde recompute with efficient
rank-1 column append to avoid redundant kernel evaluations.

Warm-start follows legacy set_new_model_variational_params() (utils.py:384):
1. Expand m_b, V_b from old eigenspace to full M-dimensional space
2. Pad to M+1 dimensions (identity variance for new point, mean(m) for new mean)
3. Project into new model's eigenspace

Reference: utils.py:set_new_model_variational_params (line 384)
"""

import torch
from eigenspace_model import DirectVGPModel


def extend_model_with_new_point(
    current_model: DirectVGPModel,
    x_new: torch.Tensor,
    kernel,
    likelihood,
    eigval_tol: float
) -> DirectVGPModel:
    """Create a new DirectVGPModel with one additional training/inducing point.

    The new model has M+1 points and warm-started variational parameters
    projected from the old model's eigenspace into the new eigenspace.

    Args:
        current_model: Existing trained model with M training/inducing points.
        x_new: New image to add, shape (n_pixels,) or (1, n_pixels).
        kernel: Deep copy of current_model.kernel (caller must copy, since
                train_eigenspace modifies kernel in-place).
        likelihood: Deep copy of current_model.likelihood (same reason).
        eigval_tol: Eigenvalue tolerance for eigenspace projection.

    Returns:
        New DirectVGPModel with M+1 points and warm-started variational params.
    """
    # Ensure x_new is 2D: (1, n_pixels)
    if x_new.dim() == 1:
        x_new = x_new.unsqueeze(0)

    # --- Step 1: Save old eigenspace info ---
    B_old = current_model.state.B          # (M, n_b_old)
    m_b_old = current_model.state.m_b      # (n_b_old,)
    V_b_old = current_model.state.V_b      # (n_b_old, n_b_old)
    M = current_model.X_tilde.shape[0]

    # --- Step 2: Extend data ---
    # Inducing points = training points (always, per plan)
    X_tilde_new = torch.cat([current_model.X_tilde, x_new], dim=0)  # (M+1, n_pixels)

    # --- Step 3: Create new model (constructor computes fresh eigenspace) ---
    new_model = DirectVGPModel(kernel, likelihood, X_tilde_new, X_tilde_new, eigval_tol)
    B_new = new_model.state.B  # (M+1, n_b_new)

    # --- Step 4: Warm-start variational parameters ---
    # Matching legacy set_new_model_variational_params (utils.py:384-415)

    # 4a. Expand to full M-dimensional space
    m_full = B_old @ m_b_old                     # (M,)
    V_full = B_old @ V_b_old @ B_old.T           # (M, M)
    V_full = 0.5 * (V_full + V_full.T)           # Symmetrize (legacy does this too)

    # 4b. Pad to M+1 dimensions
    # New point gets mean(m_full) for mean, identity (1.0) variance for covariance
    m_full_ext = torch.cat([m_full, m_full.mean().unsqueeze(0)])  # (M+1,)

    V_full_ext = torch.eye(M + 1, dtype=V_full.dtype, device=V_full.device)
    V_full_ext[:M, :M] = V_full                  # (M+1, M+1)

    # 4c. Project into new eigenspace
    m_b_new = B_new.T @ m_full_ext               # (n_b_new,)
    V_b_new = B_new.T @ V_full_ext @ B_new       # (n_b_new, n_b_new)

    # --- Step 5: Set warm-started params ---
    new_model.update_variational_params(m_b_new, V_b_new)

    return new_model
