"""
Rank-1 Model Extension for Active Learning

Creates a NEW DirectVGPModel with M+1 training/inducing points, warm-started
from an existing model with M points. Used during Phase 2 of the closed-loop
experiment when a new image is added to the training set.

Approach: Efficient column append to K_tilde.
Instead of recomputing the full (M+1)x(M+1) K_tilde from scratch (O(M^2)
kernel evaluations), compute only the new column (O(M) kernel evaluations)
and append it to the existing K_tilde. The eigendecomposition is O(M^3) in
both cases and is unavoidable. The savings come purely from avoiding
redundant kernel evaluations.

Reference: utils.py:add_one_img_to_kernel (line 443),
           utils.py:get_new_model_kernels (line 471),
           utils.py:generate_new_active_model (line 502).

Warm-start follows legacy set_new_model_variational_params() (utils.py:384):
1. Expand m_b, V_b from old eigenspace to full M-dimensional space
2. Pad to M+1 dimensions (identity variance for new point, mean(m) for new mean)
3. Project into new model's eigenspace
"""

import torch
from eigenspace_model import DirectVGPModel, DirectVariationalState
from eigenspace_utils import eigendecompose_K_tilde


def extend_model_with_new_point(
    current_model: DirectVGPModel,
    x_new: torch.Tensor,
    kernel,
    likelihood,
    eigval_tol: float
) -> DirectVGPModel:
    """Create a new DirectVGPModel with one additional training/inducing point.

    Uses efficient column append: computes only the new column of K_tilde
    (O(M) kernel evaluations) instead of the full matrix (O(M^2)).

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
    K_tilde_prev = current_model.state.K_tilde  # (M, M)
    M = current_model.X_tilde.shape[0]

    # --- Step 2: Extend data ---
    X_tilde_new = torch.cat([current_model.X_tilde, x_new], dim=0)  # (M+1, n_pixels)

    # --- Step 3: Efficient K_tilde column append ---
    # Compute only the new column: kernel(all M+1 inducing points, x_new)
    # This is O(M+1) kernel evaluations instead of O((M+1)^2).
    with torch.no_grad():
        K_tilde_column = kernel(X_tilde_new, x_new).to_dense()  # (M+1, 1)

    # Append column and row to build (M+1) x (M+1) K_tilde
    # K_tilde_column[:-1] = K(old M points, x_new) — the new column
    # K_tilde_column.T    = K(x_new, all M+1 points) — the new row
    K_tilde_new = torch.cat(
        [K_tilde_prev, K_tilde_column[:M]], dim=1  # (M, M+1)
    )
    K_tilde_new = torch.cat(
        [K_tilde_new, K_tilde_column.T], dim=0     # (M+1, M+1)
    )

    # --- Step 4: Eigendecompose new K_tilde ---
    B_new, eigvals_b, _ = eigendecompose_K_tilde(K_tilde_new, eigval_tol)
    n_b_new = len(eigvals_b)

    # --- Step 5: Compute derived eigenspace quantities ---
    # Since X_train == X_tilde (active loop invariant), K = K_tilde
    K_b = K_tilde_new @ B_new  # (M+1, n_b_new)
    K_tilde_b = torch.diag(eigvals_b)  # (n_b_new, n_b_new) diagonal
    K_times_Ktilde_inv_b = K_b / eigvals_b.unsqueeze(0)  # (M+1, n_b_new)

    # Kvec: append only the new diagonal entry k(x_new, x_new)
    with torch.no_grad():
        Kvec_new_entry = kernel(x_new, diag=True)  # (1,)
    Kvec = torch.cat([current_model.state.Kvec, Kvec_new_entry])  # (M+1,)

    # Mask from kernel (same as before — kernel params haven't changed)
    mask = kernel._cached_mask if hasattr(kernel, '_cached_mask') else None

    # --- Step 6: Warm-start variational parameters ---
    # Matching legacy set_new_model_variational_params (utils.py:384-415)

    # 6a. Expand to full M-dimensional space
    m_full = B_old @ m_b_old                     # (M,)
    V_full = B_old @ V_b_old @ B_old.T           # (M, M)
    V_full = 0.5 * (V_full + V_full.T)           # Symmetrize (legacy does this too)

    # 6b. Pad to M+1 dimensions
    # New point gets mean(m_full) for mean, identity (1.0) variance for covariance
    m_full_ext = torch.cat([m_full, m_full.mean().unsqueeze(0)])  # (M+1,)

    V_full_ext = torch.eye(M + 1, dtype=V_full.dtype, device=V_full.device)
    V_full_ext[:M, :M] = V_full                  # (M+1, M+1)

    # 6c. Project into new eigenspace
    m_b_new = B_new.T @ m_full_ext               # (n_b_new,)
    V_b_new = B_new.T @ V_full_ext @ B_new       # (n_b_new, n_b_new)

    # --- Step 7: Build state and model ---
    state = DirectVariationalState(
        m_b=m_b_new,
        V_b=V_b_new,
        B=B_new,
        eigvals_b=eigvals_b,
        K_tilde_b=K_tilde_b,
        K_b=K_b,
        KKtilde_inv_b=K_times_Ktilde_inv_b,
        Kvec=Kvec,
        K_tilde=K_tilde_new,
        mask=mask,
    )

    new_model = DirectVGPModel(
        kernel, likelihood, X_tilde_new, X_tilde_new, eigval_tol,
        _precomputed_state=state,
    )

    return new_model
