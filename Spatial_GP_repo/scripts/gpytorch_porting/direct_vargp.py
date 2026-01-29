"""
Direct Variational GP Training (vargp_direct mode)

This module implements a "true copy" of the original varGP implementation,
using GPyTorch only as a kernel calculator while managing variational
parameters directly with eigenspace projection.

Key differences from vargp_style:
1. Eigenspace projection: stores m_b, V_b in reduced space (~10-11 dims)
2. LBFGS M-step with autograd (not Adam)
3. CORRECT E-step m_new formula (differs from old buggy code)

CRITICAL: E-step Math Discrepancy
---------------------------------
This implementation uses the mathematically CORRECT m_new formula:
    m_new = m + K_tilde @ solve(K_tilde + G, g - m)

The old varGP code (utils.py line 4247) uses an INCORRECT formula:
    m_new = V_new @ (G @ m + g)  # WRONG - has extra K_tilde factor

See .claude/ESTEP_MATH_ANALYSIS.md for full derivation.

Reference: utils.py:varGP() lines 5293-5975
"""

import time
import warnings
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch

from eigenspace import (
    EIGVAL_TOL,
    compute_eigenspace,
    project_to_eigenspace,
    reproject_variational_params,
    compute_KKtilde_inv_b,
    compute_K_tilde_b_diagonal,
)
from fstep import lambda0_given_A


# Stability threshold for f_mean (same as estep.py)
STABILITY_THRESHOLD = 1e6


# ==============================================================================
# Analytical Gradient Functions for M-step Optimization
# ==============================================================================

def compute_C_and_gradients(kernel):
    """Compute C matrix and all dC/d(hyperparameters) matrices.

    Ports utils.py:localker(grad=True) lines 3577-3631.

    The C matrix encodes receptive field (RF) structure:
        C = Amp * alpha[:, None] * C_smooth * alpha[None, :]

    where:
        alpha[i] = exp(-beta_factor * dist_center[i])
        C_smooth[i,j] = exp(-rho2_factor * dist_pairwise[i,j])
        beta_factor = exp(raw_m2log2beta)
        rho2_factor = exp(raw_mlog2rho2)

    Args:
        kernel: ArcCosineKernel instance with RF parameters

    Returns:
        C: Structured covariance matrix, shape (n_masked, n_masked)
        mask: Boolean mask for active pixels, shape (n_px,)
        dC: Dict of gradient matrices, each shape (n_masked, n_masked)
            Keys: 'Amp', 'eps_0x', 'eps_0y', 'raw_m2log2beta', 'raw_mlog2rho2'

    Reference:
        utils.py:localker() lines 3617-3627
    """
    # Get parameters
    Amp = kernel.Amp.squeeze()
    eps_0x = kernel.eps_0x.squeeze()
    eps_0y = kernel.eps_0y.squeeze()

    # Transform log-space parameters
    # beta_factor = exp(raw_m2log2beta) = 1/(4*beta^2)
    # rho2_factor = exp(raw_mlog2rho2) = 1/(2*rho^2)
    beta_factor = torch.exp(kernel.raw_m2log2beta.squeeze())
    rho2_factor = torch.exp(kernel.raw_mlog2rho2.squeeze())

    # Get pixel coordinates
    xcord = kernel.xcord.clone()
    ycord = kernel.ycord.clone()

    # Compute mask (with detached params for structural stability)
    mask = kernel.compute_mask()
    xcord = xcord[mask]
    ycord = ycord[mask]

    # Locality weights: distance from RF center
    dist_sq_center = (xcord - eps_0x)**2 + (ycord - eps_0y)**2
    logalpha = -beta_factor * dist_sq_center
    alpha = torch.exp(logalpha)

    # Smoothness kernel: pairwise pixel distances
    dx = xcord[:, None] - xcord[None, :]
    dy = ycord[:, None] - ycord[None, :]
    dist_sq_pairwise = dx**2 + dy**2
    logCsmooth = -rho2_factor * dist_sq_pairwise
    C_smooth = torch.exp(logCsmooth)

    # Full C matrix
    C = Amp * alpha[:, None] * C_smooth * alpha[None, :]

    # Symmetrize for numerical stability
    C = (C + C.T) / 2

    # Compute gradients (exact formulas from utils.py lines 3619-3626)
    dC = {}

    # dC/d(Amp) = C / Amp
    dC['Amp'] = C / Amp

    # dC/d(eps_0x) = 2 * beta_factor * C * (xi + xj - 2*eps_0x)
    dC['eps_0x'] = 2 * beta_factor * C * (xcord[:, None] + xcord[None, :] - 2 * eps_0x)

    # dC/d(eps_0y) = 2 * beta_factor * C * (yi + yj - 2*eps_0y)
    dC['eps_0y'] = 2 * beta_factor * C * (ycord[:, None] + ycord[None, :] - 2 * eps_0y)

    # dC/d(raw_m2log2beta) = C * (logalpha_i + logalpha_j)
    dC['raw_m2log2beta'] = C * (logalpha[:, None] + logalpha[None, :])

    # dC/d(raw_mlog2rho2) = C * logCsmooth
    dC['raw_mlog2rho2'] = C * logCsmooth

    return C, mask, dC


def compute_kernel_and_gradients(x1, x2, C, dC, sigma_0, diag=False):
    """Compute kernel matrix K and all dK/d(hyperparameters) matrices.

    Ports utils.py:acosker(dC=...) lines 3663-3813.

    For full matrix case (diag=False):
        K[i,j] = M[i,j] * J(θ[i,j])
        where M = sqrt(V1 * V2), V1 = x1ᵀCx1 + σ₀², etc.

    For diagonal case (diag=True):
        K[i] = x1[i]ᵀ @ C @ x1[i] + σ₀²

    Args:
        x1: First input, shape (n1, n_features)
        x2: Second input, shape (n2, n_features) or None for diagonal
        C: Structured covariance matrix, shape (n_features, n_features)
        dC: Dict of dC matrices from compute_C_and_gradients()
        sigma_0: Kernel bias variance (scalar tensor)
        diag: If True, return only diagonal elements

    Returns:
        K: Kernel matrix, shape (n1, n2) or (n1,) if diag=True
        dK: Dict of gradient matrices, same shape as K
            Keys: 'sigma_0', 'Amp', 'eps_0x', 'eps_0y', 'raw_m2log2beta', 'raw_mlog2rho2'

    Reference:
        utils.py:acosker() lines 3699-3813
    """
    sigma_0_sq = sigma_0 ** 2
    n1 = x1.shape[0]

    # Compute quadratic forms x1ᵀCx1
    # CX1[i,:] = C @ x1[i,:], so CX1 has shape (n1, n_features)
    CX1 = x1 @ C  # (n1, n_features)
    V1 = (CX1 * x1).sum(dim=-1) + sigma_0_sq  # (n1,)

    if diag:
        # Diagonal case: K[i] = V1[i] = x1[i]ᵀ @ C @ x1[i] + σ₀²
        K = V1

        # Gradients
        dK = {}

        # dK/d(sigma_0) = 2 * sigma_0
        dK['sigma_0'] = torch.full_like(K, 2 * sigma_0.item())

        # dK/d(C-params): dK[key][i] = x1[i]ᵀ @ dC[key] @ x1[i]
        for key, dC_val in dC.items():
            if key == 'sigma_0':
                continue
            # dCX1[i,:] = dC[key] @ x1[i,:], so (dCX1 * x1).sum(-1) = diag(x1 @ dC @ x1.T)
            dCX1 = x1 @ dC_val  # (n1, n_features)
            dK[key] = (dCX1 * x1).sum(dim=-1)  # (n1,)

        return K, dK

    # Full matrix case
    n2 = x2.shape[0]

    CX2 = x2 @ C  # (n2, n_features)
    V2 = (CX2 * x2).sum(dim=-1) + sigma_0_sq  # (n2,)

    X1 = torch.sqrt(V1)  # (n1,)
    X2 = torch.sqrt(V2)  # (n2,)

    # M = sqrt(V1 * V2) = X1 * X2
    X1X2 = X1[:, None] * X2[None, :]  # (n1, n2)

    # Cross-term: x1ᵀ C x2 + σ₀²
    x1x2 = (CX1 @ x2.T) + sigma_0_sq  # (n1, n2)

    # Normalized inner product
    eps = 1e-7
    cosdelta = torch.clamp(x1x2 / (X1X2 + eps), -1.0 + eps, 1.0 - eps)

    # Angle
    delta = torch.arccos(cosdelta)

    # Angular term: J(θ) = (sin(θ) + (π - θ)cos(θ)) / π
    sin_delta = torch.sqrt(torch.clamp(1.0 - cosdelta ** 2, min=eps))
    J = (sin_delta + (torch.pi - delta) * cosdelta) / torch.pi

    # Kernel: K = M * J
    K = X1X2 * J

    # ====== Gradients ======
    dK = {}

    # --- dK/d(sigma_0) ---
    # From utils.py lines 3722-3730
    # dX1X2 = sigma_0² * (X2/X1 + X1/X2) in broadcasted form
    dX1X2_sigma0 = sigma_0_sq * (X2[None, :] / X1[:, None] + X1[:, None] / X2[None, :])
    dcosdelta_sigma0 = (2 * sigma_0_sq - cosdelta * dX1X2_sigma0) / X1X2
    dJ_sigma0 = -(delta - torch.pi) * dcosdelta_sigma0 / torch.pi
    dK['sigma_0'] = (X1X2 * dJ_sigma0 + dX1X2_sigma0 * J) / sigma_0

    # --- dK/d(C-params) ---
    # From utils.py lines 3734-3747
    for key, dC_val in dC.items():
        if key == 'sigma_0':
            continue

        # dX1[i] = 0.5 * x1[i]ᵀ @ dC @ x1[i] / X1[i]
        dCX1 = x1 @ dC_val  # (n1, n_features)
        dX1 = 0.5 * (dCX1 * x1).sum(dim=-1) / X1  # (n1,)

        # dX2[j] = 0.5 * x2[j]ᵀ @ dC @ x2[j] / X2[j]
        dCX2 = x2 @ dC_val  # (n2, n_features)
        dX2 = 0.5 * (dCX2 * x2).sum(dim=-1) / X2  # (n2,)

        # dX1X2[i,j] = dX1[i] * X2[j] + X1[i] * dX2[j]
        dX1X2 = dX1[:, None] * X2[None, :] + X1[:, None] * dX2[None, :]

        # dcosdelta = (x1.T @ dC @ x2 - cosdelta * dX1X2) / X1X2
        dx1x2 = dCX1 @ x2.T  # (n1, n2)
        dcosdelta = (dx1x2 - cosdelta * dX1X2) / X1X2

        # dJ = -(delta - π) * dcosdelta / π
        dJ = -(delta - torch.pi) * dcosdelta / torch.pi

        # dK[key] = X1X2 * dJ + dX1X2 * J
        dK[key] = X1X2 * dJ + dX1X2 * J

    return K, dK


def compute_lambda_moments_and_gradients(
    K_b, K_tilde_b, Kvec, m_b, V_b,
    dK_b, dK_tilde_b, dKvec,
    eigvals_b
):
    """Compute posterior moments and their gradients w.r.t. hyperparameters.

    Ports utils.py:lambda_moments() lines 3937-3952.

    Moments:
        a = K_b @ K_tilde_inv_b  (projection vector, exploits diagonal K_tilde_b)
        lambda_m = a @ m_b
        lambda_var = Kvec + diag(a @ (V_b - K_tilde_b) @ a.T)

    Gradients:
        da[key] = (dK_b[key] - a @ dK_tilde_b[key]) @ K_tilde_inv_b
        dlambda_m[key] = da[key] @ m_b
        dlambda_var[key] = dKvec[key]
                         + 2 * diag(da[key] @ V_b @ a.T)
                         - diag(dK_b[key] @ a.T)
                         - diag(K_b @ da[key].T)

    Args:
        K_b: Cross-kernel in eigenspace, shape (N, n_b)
        K_tilde_b: Inducing kernel in eigenspace (DIAGONAL), shape (n_b, n_b)
        Kvec: Diagonal self-kernel, shape (N,)
        m_b: Variational mean in eigenspace, shape (n_b,)
        V_b: Variational covariance in eigenspace, shape (n_b, n_b)
        dK_b: Dict of dK_b matrices, each shape (N, n_b)
        dK_tilde_b: Dict of dK_tilde_b matrices, each shape (n_b, n_b)
        dKvec: Dict of dKvec vectors, each shape (N,)
        eigvals_b: Eigenvalues for K_tilde_b diagonal, shape (n_b,)

    Returns:
        lambda_m: Posterior mean, shape (N,)
        lambda_var: Posterior variance, shape (N,)
        dlambda_m: Dict of mean gradients, each shape (N,)
        dlambda_var: Dict of variance gradients, each shape (N,)

    Reference:
        utils.py:lambda_moments() lines 3942-3952
    """
    # K_tilde_b is diagonal with eigenvalues on diagonal
    # K_tilde_inv_b[i,i] = 1/eigvals_b[i]
    K_tilde_inv_b_diag = 1.0 / eigvals_b  # (n_b,)

    # Projection vector: a = K_b @ K_tilde_inv_b
    # Since K_tilde_inv_b is diagonal, this is element-wise:
    a = K_b * K_tilde_inv_b_diag[None, :]  # (N, n_b)

    # Posterior mean
    lambda_m = a @ m_b  # (N,)

    # Posterior variance: lambda_var = Kvec + diag(a @ (V - K) @ a.T)
    V_minus_K = V_b - K_tilde_b  # (n_b, n_b)
    aV = a @ V_minus_K  # (N, n_b)
    lambda_var = Kvec + (a * aV).sum(dim=-1)  # (N,)

    # Clamp for numerical stability
    lambda_var = torch.clamp(lambda_var, min=1e-6)

    # ====== Gradients ======
    dlambda_m = {}
    dlambda_var = {}

    for key in dK_b.keys():
        # da/dθ = (dK/dθ - a @ dK_tilde/dθ) @ K_tilde_inv
        # In eigenspace with diagonal K_tilde_inv:
        # da[key] = (dK_b[key] - a @ dK_tilde_b[key]) * K_tilde_inv_b_diag
        da_key = (dK_b[key] - a @ dK_tilde_b[key]) * K_tilde_inv_b_diag[None, :]  # (N, n_b)

        # dlambda_m[key] = da[key] @ m
        dlambda_m[key] = da_key @ m_b  # (N,)

        # dlambda_var[key] = dKvec[key]
        #                  + 2*diag(da @ V @ a.T)   = einsum('ij,ji->i', 2*da, V@a.T)
        #                  - diag(dK @ a.T)        = einsum('ij,ij->i', dK, a)
        #                  - diag(K @ da.T)        = einsum('ij,ij->i', K, da)

        # Term 1: dKvec[key]
        term1 = dKvec[key]

        # Term 2: 2 * diag(da @ V @ a.T)
        # = 2 * sum(da * (V @ a.T).T, dim=1) = 2 * sum(da * (a @ V.T).T, dim=1)
        # Since V is symmetric: V.T = V
        Va_T = V_b @ a.T  # (n_b, N)
        term2 = 2 * torch.einsum('ij,ji->i', da_key, Va_T)  # (N,)

        # Term 3: -diag(dK @ a.T) = -sum(dK * a, dim=1)
        term3 = -torch.einsum('ij,ij->i', dK_b[key], a)  # (N,)

        # Term 4: -diag(K @ da.T) = -sum(K * da, dim=1)
        term4 = -torch.einsum('ij,ij->i', K_b, da_key)  # (N,)

        dlambda_var[key] = term1 + term2 + term3 + term4  # (N,)

    return lambda_m, lambda_var, dlambda_m, dlambda_var


def compute_loss_gradients(
    r, f_mean, A, m_b, V_b, K_tilde_b, eigvals_b,
    dlambda_m, dlambda_var, dK_tilde_b
):
    """Compute gradients of negative ELBO w.r.t. kernel hyperparameters.

    Loss = -loglikelihood + KL
    dL/dθ = -dloglikelihood/dθ + dKL/dθ

    Ports utils.py:compute_loglikelihood() lines 4111-4117 and
    utils.py:compute_KL_div() lines 4143-4150.

    Args:
        r: Spike counts, shape (N,)
        f_mean: Expected firing rate, shape (N,)
        A: Gain parameter (scalar)
        m_b: Variational mean in eigenspace, shape (n_b,)
        V_b: Variational covariance in eigenspace, shape (n_b, n_b)
        K_tilde_b: Inducing kernel in eigenspace (DIAGONAL), shape (n_b, n_b)
        eigvals_b: Eigenvalues for K_tilde_b diagonal, shape (n_b,)
        dlambda_m: Dict of mean gradients from compute_lambda_moments_and_gradients()
        dlambda_var: Dict of variance gradients from compute_lambda_moments_and_gradients()
        dK_tilde_b: Dict of dK_tilde_b matrices, each shape (n_b, n_b)

    Returns:
        dL: Dict of loss gradients w.r.t. each hyperparameter (scalars)

    Reference:
        utils.py:compute_loglikelihood() lines 4111-4117
        utils.py:compute_KL_div() lines 4143-4150
    """
    # K_tilde_inv_b is diagonal: K_tilde_inv_b[i,i] = 1/eigvals_b[i]
    K_tilde_inv_b_diag = 1.0 / eigvals_b  # (n_b,)

    # ====== dloglikelihood/dθ ======
    # From utils.py line 4116:
    # dloglikelihood[key] = A*r@dlambda_m[key] - A*f_mean@dlambda_m[key] - 0.5*A²*f_mean@dlambda_var[key]

    dloglikelihood = {}
    for key in dlambda_m.keys():
        dloglikelihood[key] = (
            A * (r @ dlambda_m[key])
            - A * (f_mean @ dlambda_m[key])
            - 0.5 * A * A * (f_mean @ dlambda_var[key])
        )

    # ====== dKL/dθ ======
    # From utils.py lines 4133-4148:
    # c = V @ K_tilde_inv
    # b = K_tilde_inv @ m
    # B = dK_tilde[key] @ K_tilde_inv
    # dKL[key] = 0.5*trace(B) - 0.5*trace(c@B) - 0.5*b.T@(B@m)

    # In eigenspace with diagonal K_tilde_inv:
    # c = V_b @ K_tilde_inv_b (diagonal on right)
    c = V_b * K_tilde_inv_b_diag[None, :]  # (n_b, n_b) - scales columns

    # b = K_tilde_inv_b @ m_b (diagonal on left)
    b = K_tilde_inv_b_diag * m_b  # (n_b,)

    dKL = {}
    for key in dK_tilde_b.keys():
        # B = dK_tilde_b[key] @ K_tilde_inv_b (diagonal on right)
        B = dK_tilde_b[key] * K_tilde_inv_b_diag[None, :]  # (n_b, n_b)

        # Term 1: 0.5 * trace(B)
        term1 = 0.5 * torch.trace(B)

        # Term 2: -0.5 * trace(c @ B)
        term2 = -0.5 * torch.trace(c @ B)

        # Term 3: -0.5 * b.T @ (B @ m)
        term3 = -0.5 * (b @ (B @ m_b))

        dKL[key] = term1 + term2 + term3

    # ====== Final gradient: dL = -dloglikelihood + dKL ======
    dL = {}
    for key in dlambda_m.keys():
        dL[key] = -dloglikelihood[key] + dKL[key]

    return dL


@dataclass
class DirectVariationalState:
    """State container for direct variational GP.

    Stores all quantities needed for E-step, F-step, M-step in eigenspace.

    Attributes:
        m_b: Variational mean in eigenspace, shape (n_b,)
        V_b: Variational covariance in eigenspace, shape (n_b, n_b) - NOT diagonal!
        B: Eigenvector matrix, shape (M, n_b)
        eigvals_b: Kept eigenvalues, shape (n_b,)
        K_tilde_b: Inducing kernel in eigenspace - DIAGONAL, shape (n_b, n_b)
        K_b: Cross-kernel in eigenspace, shape (N, n_b)
        KKtilde_inv_b: K @ K_tilde_inv in eigenspace, shape (N, n_b)
        Kvec: Diagonal k(x_i, x_i), shape (N,)
        mask: Pixel mask from kernel (if use_mask=True), shape (n_pixels,) or None
    """
    m_b: torch.Tensor
    V_b: torch.Tensor
    B: torch.Tensor
    eigvals_b: torch.Tensor
    K_tilde_b: torch.Tensor
    K_b: torch.Tensor
    KKtilde_inv_b: torch.Tensor
    Kvec: torch.Tensor
    mask: Optional[torch.Tensor] = None


def compute_kernels_direct(
    kernel,
    X: torch.Tensor,
    X_tilde: torch.Tensor,
    eigval_tol: float = EIGVAL_TOL
) -> DirectVariationalState:
    """Compute initial kernel matrices with eigenspace projection.

    Creates the DirectVariationalState with all quantities needed for training.
    Initializes m_b = 0 and V_b = K_tilde_b (prior).

    Args:
        kernel: ArcCosineKernel instance (used as standalone calculator)
        X: Training inputs, shape (N, n_features)
        X_tilde: Inducing points, shape (M, n_features)
        eigval_tol: Eigenvalue threshold for projection

    Returns:
        DirectVariationalState with initialized quantities
    """
    # Compute kernel matrices using GPyTorch kernel
    # kernel(X1, X2).evaluate() returns the kernel matrix
    with torch.no_grad():
        K_tilde = kernel(X_tilde, X_tilde).evaluate()  # (M, M)
        K = kernel(X, X_tilde).evaluate()  # (N, M)
        Kvec = kernel(X, diag=True)  # (N,)

    # Get mask if kernel uses masking
    mask = kernel._cached_mask if hasattr(kernel, '_cached_mask') else None

    # Eigenspace projection
    B, eigvals_b, _ = compute_eigenspace(K_tilde, eigval_tol)
    n_b = len(eigvals_b)

    # Project K to eigenspace
    _, _, K_b = project_to_eigenspace(B, K=K)

    # K_tilde_b is diagonal in eigenspace
    K_tilde_b = compute_K_tilde_b_diagonal(eigvals_b)

    # Efficient K @ K_tilde_inv computation
    KKtilde_inv_b = compute_KKtilde_inv_b(K_b, eigvals_b)

    # Initialize variational parameters at prior
    m_b = torch.zeros(n_b, dtype=X.dtype, device=X.device)
    V_b = K_tilde_b.clone()  # Prior: V = K_tilde

    return DirectVariationalState(
        m_b=m_b,
        V_b=V_b,
        B=B,
        eigvals_b=eigvals_b,
        K_tilde_b=K_tilde_b,
        K_b=K_b,
        KKtilde_inv_b=KKtilde_inv_b,
        Kvec=Kvec,
        mask=mask,
    )


def recompute_kernels_after_mstep(
    kernel,
    X: torch.Tensor,
    X_tilde: torch.Tensor,
    state: DirectVariationalState,
    eigval_tol: float = EIGVAL_TOL
) -> DirectVariationalState:
    """Recompute kernels after M-step changes hyperparameters.

    When M-step modifies kernel hyperparameters, K_tilde changes, so the
    eigenspace changes. We recompute all kernel quantities and reproject
    m_b, V_b to the new eigenspace.

    Reference: utils.py lines 5593-5627

    Args:
        kernel: ArcCosineKernel instance (with updated hyperparameters)
        X: Training inputs, shape (N, n_features)
        X_tilde: Inducing points, shape (M, n_features)
        state: Current state with old eigenspace
        eigval_tol: Eigenvalue threshold for projection

    Returns:
        New DirectVariationalState with reprojected quantities
    """
    B_old = state.B
    m_b_old = state.m_b
    V_b_old = state.V_b

    # Recompute kernel matrices
    with torch.no_grad():
        K_tilde = kernel(X_tilde, X_tilde).evaluate()
        K = kernel(X, X_tilde).evaluate()
        Kvec = kernel(X, diag=True)

    mask = kernel._cached_mask if hasattr(kernel, '_cached_mask') else None

    # New eigenspace
    B_new, eigvals_b, _ = compute_eigenspace(K_tilde, eigval_tol)

    # Project K to new eigenspace
    _, _, K_b = project_to_eigenspace(B_new, K=K)

    # K_tilde_b diagonal in new eigenspace
    K_tilde_b = compute_K_tilde_b_diagonal(eigvals_b)

    # Efficient K @ K_tilde_inv
    KKtilde_inv_b = compute_KKtilde_inv_b(K_b, eigvals_b)

    # Reproject variational parameters to new eigenspace
    m_b_new, V_b_new = reproject_variational_params(B_old, B_new, m_b_old, V_b_old)

    return DirectVariationalState(
        m_b=m_b_new,
        V_b=V_b_new,
        B=B_new,
        eigvals_b=eigvals_b,
        K_tilde_b=K_tilde_b,
        K_b=K_b,
        KKtilde_inv_b=KKtilde_inv_b,
        Kvec=Kvec,
        mask=mask,
    )


def lambda_moments_eigenspace(state: DirectVariationalState) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute GP posterior moments using eigenspace quantities.

    Computes:
        lambda_m = K @ K_tilde_inv @ m = KKtilde_inv_b @ m_b
        lambda_var = Kvec + diag(a @ (V_b - K_tilde_b) @ a.T)
                   = Kvec + sum(a * (a @ (V_b - K_tilde_b)), dim=1)

    where a = KKtilde_inv_b.

    Reference: utils.py:lambda_moments()

    Args:
        state: DirectVariationalState with current quantities

    Returns:
        lambda_m: Posterior mean at training points, shape (N,)
        lambda_var: Posterior variance at training points, shape (N,)
    """
    a = state.KKtilde_inv_b  # (N, n_b)

    # Mean: lambda_m = a @ m_b
    lambda_m = a @ state.m_b  # (N,)

    # Variance: lambda_var = Kvec + diag(a @ (V - K) @ a.T)
    V_minus_K = state.V_b - state.K_tilde_b  # (n_b, n_b)
    aV = a @ V_minus_K  # (N, n_b)
    lambda_var = state.Kvec + (a * aV).sum(dim=1)  # (N,)

    # Clamp for numerical stability
    lambda_var = torch.clamp(lambda_var, min=1e-6)

    return lambda_m, lambda_var


def estep_eigenspace(
    state: DirectVariationalState,
    r: torch.Tensor,
    A: torch.Tensor,
    f_mean: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Newton update for variational parameters matching original varGP.

    This implementation matches utils.py:Estep() lines 4244-4256 exactly.

    The g_b and G_b computed here are the TRANSFORMED versions:
        g_b = K_tilde_inv @ g_standard
        G_b = K_tilde_inv @ G_standard @ K_tilde_inv

    The formulas below are correct for these transformed quantities.

    TODO: INVESTIGATE - The m_new formula here matches the old code but may have
    a mathematical discrepancy. See .claude/ESTEP_MATH_ANALYSIS.md for analysis.
    The correct formula would be: m_new = m + K_tilde @ solve(K_tilde + G, g - m)
    But the old code uses: m_new = V_new @ (G @ m + g)
    These differ unless K_tilde and G commute. Worth investigating if the
    "correct" formula improves results.

    Args:
        state: Current DirectVariationalState
        r: Spike counts, shape (N,)
        A: Gain parameter (scalar)
        f_mean: Expected firing rate exp(A*lambda_m + 0.5*A^2*lambda_var + lambda0),
                shape (N,)

    Returns:
        m_b_new: Updated variational mean, shape (n_b,)
        V_b_new: Updated variational covariance, shape (n_b, n_b)
    """
    a = state.KKtilde_inv_b  # (N, n_b) - this is K @ K_tilde_inv in eigenspace

    # Transformed gradient: g_b = A * a.T @ (r - f_mean)
    # This is K_tilde_inv @ g_standard
    g_b = A * (a.T @ (r - f_mean))  # (n_b,)

    # Transformed Hessian: G_b = A^2 * a.T @ diag(f_mean) @ a
    # This is K_tilde_inv @ G_standard @ K_tilde_inv
    G_b = (A * A) * (a.T @ (f_mean[:, None] * a))  # (n_b, n_b)

    # V update: V_new = solve(I + K_tilde @ G, K_tilde)
    # Matches utils.py line 4246
    n_b = state.K_tilde_b.shape[0]
    eye = torch.eye(n_b, dtype=state.K_tilde_b.dtype, device=state.K_tilde_b.device)
    V_b_new = torch.linalg.solve(eye + state.K_tilde_b @ G_b, state.K_tilde_b)

    # m update: m_new = V_new @ (G @ m + g)
    # Matches utils.py line 4247
    m_b_new = V_b_new @ (G_b @ state.m_b + g_b)

    # Symmetrize V for numerical stability
    V_b_new = (V_b_new + V_b_new.T) / 2

    return m_b_new, V_b_new


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


def compute_elbo_eigenspace(
    state: DirectVariationalState,
    r: torch.Tensor,
    lambda_m: torch.Tensor,
    lambda_var: torch.Tensor,
    A: torch.Tensor,
    lambda0: torch.Tensor
) -> torch.Tensor:
    """Compute ELBO = log_likelihood - KL_divergence.

    Log-likelihood:
        L = sum(r * (A * lambda_m + lambda0) - f_mean)
        where f_mean = exp(A * lambda_m + 0.5 * A^2 * lambda_var + lambda0)

    KL divergence for q(u) = N(m, V) vs p(u) = N(0, K_tilde):
        KL = 0.5 * (tr(K_tilde^-1 @ V) + m.T @ K_tilde^-1 @ m - n_b + log|K_tilde| - log|V|)

    In eigenspace with K_tilde_b diagonal:
        KL = 0.5 * (sum(V_b_diag / eigvals) + sum(m_b^2 / eigvals) - n_b
                   + sum(log(eigvals)) - log|V_b|)

    Args:
        state: Current DirectVariationalState
        r: Spike counts, shape (N,)
        lambda_m: Posterior mean, shape (N,)
        lambda_var: Posterior variance, shape (N,)
        A: Gain parameter (scalar)
        lambda0: Bias parameter (scalar)

    Returns:
        ELBO value (scalar, to be maximized)
    """
    # Log-likelihood
    f_mean = compute_f_mean(lambda_m, lambda_var, A, lambda0)
    log_lik = (r * (A * lambda_m + lambda0) - f_mean).sum()

    # KL divergence in eigenspace
    n_b = len(state.eigvals_b)
    eigvals = state.eigvals_b

    # tr(K_tilde^-1 @ V) = tr(diag(1/eigvals) @ V_b) = sum(V_b_diag / eigvals)
    V_diag = torch.diag(state.V_b)
    trace_term = (V_diag / eigvals).sum()

    # m.T @ K_tilde^-1 @ m = sum(m_b^2 / eigvals)
    quad_term = ((state.m_b ** 2) / eigvals).sum()

    # log|K_tilde| = sum(log(eigvals))
    log_det_K = torch.log(eigvals).sum()

    # log|V| - need full log determinant
    sign, log_det_V = torch.linalg.slogdet(state.V_b)
    if sign.item() <= 0:
        # V is not positive definite - this shouldn't happen
        warnings.warn("V_b is not positive definite in KL computation")
        log_det_V = torch.tensor(0.0, device=state.V_b.device, dtype=state.V_b.dtype)

    KL = 0.5 * (trace_term + quad_term - n_b + log_det_K - log_det_V)

    return log_lik - KL


def fstep_direct(
    likelihood,
    r: torch.Tensor,
    lambda_m: torch.Tensor,
    lambda_var: torch.Tensor,
    n_fstep: int,
    lr: float
):
    """F-step: Optimize A with LBFGS, lambda0 computed analytically.

    This is a simplified version that directly uses LBFGS on raw_A.

    Args:
        likelihood: PoissonLikelihood instance
        r: Spike counts, shape (N,)
        lambda_m: Posterior mean (held fixed), shape (N,)
        lambda_var: Posterior variance (held fixed), shape (N,)
        n_fstep: Number of LBFGS iterations
        lr: Learning rate for LBFGS
    """
    if n_fstep == 0:
        # Still update lambda0 analytically
        A = likelihood.A.squeeze()
        with torch.no_grad():
            new_lambda0 = lambda0_given_A(A, r, lambda_m, lambda_var)
            likelihood.lambda0.copy_(new_lambda0.reshape(likelihood.lambda0.shape))
        return

    # Initial lambda0 update
    A = likelihood.A.squeeze()
    with torch.no_grad():
        new_lambda0 = lambda0_given_A(A, r, lambda_m, lambda_var)
        likelihood.lambda0.copy_(new_lambda0.reshape(likelihood.lambda0.shape))

    optimizer = torch.optim.LBFGS(
        [likelihood.raw_A],
        lr=lr,
        max_iter=n_fstep,
        tolerance_change=1e-9,
        tolerance_grad=1e-7,
        history_size=n_fstep,
        line_search_fn='strong_wolfe'
    )

    def closure():
        optimizer.zero_grad()

        A = likelihood.A.squeeze()

        # Update lambda0 analytically
        with torch.no_grad():
            lambda0 = lambda0_given_A(A, r, lambda_m, lambda_var)
            likelihood.lambda0.copy_(lambda0.reshape(likelihood.lambda0.shape))

        # Compute f_mean
        f_mean = torch.exp(A * lambda_m + 0.5 * A * A * lambda_var + lambda0)

        # Stability check
        if f_mean.mean().item() > STABILITY_THRESHOLD or torch.any(torch.isnan(f_mean)):
            return torch.tensor(float('inf'), device=A.device, dtype=A.dtype)

        # Log-likelihood (negative for minimization)
        log_lik = (r * (A * lambda_m + lambda0) - f_mean).sum()

        # Compute gradient analytically for efficiency
        # dL/dA = r @ lambda_m - (lambda_m + A * lambda_var) @ f_mean
        # dL/d(logA) = A * dL/dA
        dL_dA = r @ lambda_m - torch.dot(lambda_m + A * lambda_var, f_mean)
        dL_dlogA = A * dL_dA

        likelihood.raw_A.grad = -dL_dlogA.reshape(likelihood.raw_A.shape)

        return -log_lik

    optimizer.step(closure)

    # Final lambda0 update
    with torch.no_grad():
        A = likelihood.A.squeeze()
        new_lambda0 = lambda0_given_A(A, r, lambda_m, lambda_var)
        likelihood.lambda0.copy_(new_lambda0.reshape(likelihood.lambda0.shape))


def mstep_lbfgs_autograd(
    kernel,
    likelihood,
    X: torch.Tensor,
    X_tilde: torch.Tensor,
    r: torch.Tensor,
    state: DirectVariationalState,
    n_mstep: int,
    lr: float
):
    """M-step: Optimize kernel hyperparameters with LBFGS using autograd.

    Uses LBFGS with PyTorch autograd for gradients (not analytical gradients).
    Kernel matrices are recomputed with gradients enabled during closure.

    Note: This does NOT use eigenspace projection during M-step optimization.
    The eigenspace is fixed during M-step; reprojection happens after.

    Args:
        kernel: ArcCosineKernel instance
        likelihood: PoissonLikelihood instance
        X: Training inputs, shape (N, n_features)
        X_tilde: Inducing points, shape (M, n_features)
        r: Spike counts, shape (N,)
        state: Current DirectVariationalState (m_b, V_b held fixed)
        n_mstep: Number of LBFGS iterations
        lr: Learning rate for LBFGS
    """
    if n_mstep == 0:
        return

    # Get kernel parameters
    kernel_params = list(kernel.parameters())

    optimizer = torch.optim.LBFGS(
        kernel_params,
        lr=lr,
        max_iter=n_mstep,
        tolerance_change=1e-9,
        tolerance_grad=1e-7,
        history_size=100,
        line_search_fn='strong_wolfe'
    )

    def closure():
        optimizer.zero_grad()

        # Compute kernels WITH gradients
        K_tilde = kernel(X_tilde, X_tilde).evaluate()
        K = kernel(X, X_tilde).evaluate()
        Kvec = kernel(X, diag=True)

        # Project into FIXED eigenspace (use state.B)
        # This is an approximation - we don't recompute eigenspace during M-step
        K_tilde_b = state.B.T @ K_tilde @ state.B  # (n_b, n_b)
        K_b = K @ state.B  # (N, n_b)

        # Check if K_tilde_b is still well-conditioned
        try:
            K_tilde_b_inv = torch.linalg.solve(K_tilde_b, torch.eye(K_tilde_b.shape[0],
                                               device=K_tilde_b.device, dtype=K_tilde_b.dtype))
        except RuntimeError:
            # Singular matrix - return inf to reject this step
            return torch.tensor(float('inf'), device=X.device, dtype=X.dtype)

        # Compute moments with fixed m_b, V_b
        a = K_b @ K_tilde_b_inv  # (N, n_b)
        lambda_m = a @ state.m_b  # (N,)

        V_minus_K = state.V_b - K_tilde_b
        aV = a @ V_minus_K
        lambda_var = Kvec + (a * aV).sum(dim=1)
        lambda_var = torch.clamp(lambda_var, min=1e-6)

        # Compute log-likelihood
        A = likelihood.A.squeeze()
        lambda0 = likelihood.lambda0.squeeze()
        f_mean = torch.exp(A * lambda_m + 0.5 * A * A * lambda_var + lambda0)

        if f_mean.mean().item() > STABILITY_THRESHOLD or torch.any(torch.isnan(f_mean)):
            return torch.tensor(float('inf'), device=X.device, dtype=X.dtype)

        log_lik = (r * (A * lambda_m + lambda0) - f_mean).sum()

        # Compute KL divergence
        n_b = len(state.eigvals_b)

        # Use projected K_tilde_b for KL
        V_diag = torch.diag(state.V_b)
        K_tilde_b_diag = torch.diag(K_tilde_b)
        trace_term = (V_diag / K_tilde_b_diag.clamp(min=1e-10)).sum()
        quad_term = (state.m_b @ K_tilde_b_inv @ state.m_b)

        sign_K, log_det_K = torch.linalg.slogdet(K_tilde_b)
        sign_V, log_det_V = torch.linalg.slogdet(state.V_b)

        if sign_K.item() <= 0 or sign_V.item() <= 0:
            return torch.tensor(float('inf'), device=X.device, dtype=X.dtype)

        KL = 0.5 * (trace_term + quad_term - n_b + log_det_K - log_det_V)

        # Negative ELBO (for minimization)
        loss = -log_lik + KL

        # Use autograd.grad to compute gradients (works with LBFGS line search)
        if loss.requires_grad:
            grads = torch.autograd.grad(loss, kernel_params, create_graph=False)
            for param, grad in zip(kernel_params, grads):
                param.grad = grad

        return loss

    optimizer.step(closure)
    kernel.clamp_hyperparameters()


def mstep_lbfgs_analytical(
    kernel,
    likelihood,
    X: torch.Tensor,
    X_tilde: torch.Tensor,
    r: torch.Tensor,
    state: DirectVariationalState,
    n_mstep: int,
    lr: float
):
    """M-step with analytical gradients (matching vargp_old).

    Computes dK/dθ matrices ONCE and caches them for LBFGS closure.
    Uses @torch.no_grad() closure for speed (no autograd graph construction).

    Implements all 8 numerical guardrails from original varGP.

    Args:
        kernel: ArcCosineKernel instance
        likelihood: PoissonLikelihood instance
        X: Training inputs, shape (N, n_features)
        X_tilde: Inducing points, shape (M, n_features)
        r: Spike counts, shape (N,)
        state: Current DirectVariationalState (m_b, V_b, B held fixed)
        n_mstep: Number of LBFGS iterations
        lr: Learning rate for LBFGS

    Reference:
        utils.py M-step closure, lines 5859-5963
    """
    if n_mstep == 0:
        return

    # Fixed variational params and eigenspace
    B = state.B
    m_b = state.m_b
    V_b = state.V_b
    eigvals_b = state.eigvals_b
    n_b = len(eigvals_b)

    # Get kernel parameter references
    kernel_params = list(kernel.parameters())

    optimizer = torch.optim.LBFGS(
        kernel_params,
        lr=lr,
        max_iter=n_mstep,
        tolerance_change=1e-9,
        tolerance_grad=1e-7,
        history_size=100,
        line_search_fn='strong_wolfe'
    )

    # Likelihood params (fixed during M-step)
    A = likelihood.A.squeeze().detach()
    lambda0 = likelihood.lambda0.squeeze().detach()

    @torch.no_grad()
    def closure():
        optimizer.zero_grad()

        # ===== Guardrail 1: Check hyperparameter bounds =====
        # Return inf to reject LBFGS step if out of bounds
        try:
            sigma_0_val = kernel.sigma_0.item()
            Amp_val = kernel.Amp.item()
            if sigma_0_val <= 0 or Amp_val <= 0 or Amp_val > kernel.AMP_MAX:
                return torch.tensor(float('inf'), device=X.device, dtype=X.dtype)

            if hasattr(kernel, 'raw_m2log2beta'):
                raw_beta = kernel.raw_m2log2beta.item()
                raw_rho = kernel.raw_mlog2rho2.item()
                if raw_beta < kernel.RAW_BETA_MIN or raw_beta > kernel.RAW_BETA_MAX:
                    return torch.tensor(float('inf'), device=X.device, dtype=X.dtype)
                if raw_rho < kernel.RAW_RHO_MIN or raw_rho > kernel.RAW_RHO_MAX:
                    return torch.tensor(float('inf'), device=X.device, dtype=X.dtype)

            if hasattr(kernel, 'eps_0x'):
                eps_x = kernel.eps_0x.item()
                eps_y = kernel.eps_0y.item()
                if eps_x < kernel.EPS_MIN or eps_x > kernel.EPS_MAX:
                    return torch.tensor(float('inf'), device=X.device, dtype=X.dtype)
                if eps_y < kernel.EPS_MIN or eps_y > kernel.EPS_MAX:
                    return torch.tensor(float('inf'), device=X.device, dtype=X.dtype)
        except Exception:
            return torch.tensor(float('inf'), device=X.device, dtype=X.dtype)

        # ===== 1. Compute C and dC (includes masking) =====
        C, mask, dC = compute_C_and_gradients(kernel)
        sigma_0 = kernel.sigma_0.squeeze()

        # Apply mask to inputs (mask from compute_C_and_gradients)
        X_masked = X[:, mask]
        X_tilde_masked = X_tilde[:, mask]

        # ===== 2. Compute K, K_tilde, Kvec and all gradients =====
        K_tilde, dK_tilde = compute_kernel_and_gradients(
            X_tilde_masked, X_tilde_masked, C, dC, sigma_0, diag=False
        )
        K, dK = compute_kernel_and_gradients(
            X_masked, X_tilde_masked, C, dC, sigma_0, diag=False
        )
        Kvec, dKvec = compute_kernel_and_gradients(
            X_masked, None, C, dC, sigma_0, diag=True
        )

        # ===== Guardrail 5: Symmetrize K_tilde =====
        K_tilde = (K_tilde + K_tilde.T) / 2

        # ===== 3. Project to eigenspace =====
        K_tilde_b = B.T @ K_tilde @ B
        K_b = K @ B

        # Symmetrize K_tilde_b
        K_tilde_b = (K_tilde_b + K_tilde_b.T) / 2

        # Project gradient matrices
        dK_tilde_b = {k: B.T @ v @ B for k, v in dK_tilde.items()}
        dK_b = {k: v @ B for k, v in dK.items()}

        # ===== 4. Compute moments and gradients =====
        lambda_m, lambda_var, dlambda_m, dlambda_var = compute_lambda_moments_and_gradients(
            K_b, K_tilde_b, Kvec, m_b, V_b,
            dK_b, dK_tilde_b, dKvec, eigvals_b
        )

        # Compute f_mean
        f_mean = torch.exp(A * lambda_m + 0.5 * A * A * lambda_var + lambda0)

        # ===== Guardrail 2: Firing rate check =====
        if f_mean.mean().item() > 100 or torch.any(torch.isnan(f_mean)):
            return torch.tensor(float('inf'), device=X.device, dtype=X.dtype)

        # ===== 5. Compute loss =====
        # Log-likelihood
        log_lik = (r * (A * lambda_m + lambda0) - f_mean).sum()

        # KL divergence
        K_tilde_inv_b_diag = 1.0 / eigvals_b
        V_diag = torch.diag(V_b)
        trace_term = (V_diag * K_tilde_inv_b_diag).sum()
        quad_term = (m_b * K_tilde_inv_b_diag) @ m_b
        log_det_K = torch.log(eigvals_b).sum()

        sign_V, log_det_V = torch.linalg.slogdet(V_b)
        if sign_V.item() <= 0:
            return torch.tensor(float('inf'), device=X.device, dtype=X.dtype)

        KL = 0.5 * (trace_term + quad_term - n_b + log_det_K - log_det_V)

        loss = -log_lik + KL

        # ===== Guardrail 7: Loss NaN/Inf detection =====
        if torch.isnan(loss) or torch.isinf(loss):
            return torch.tensor(float('inf'), device=X.device, dtype=X.dtype)

        # ===== 6. Compute analytical gradients =====
        dL = compute_loss_gradients(
            r, f_mean, A, m_b, V_b, K_tilde_b, eigvals_b,
            dlambda_m, dlambda_var, dK_tilde_b
        )

        # ===== Guardrail 8: Gradient NaN/Inf tracking =====
        nan_grad_count = 0
        for key, grad_val in dL.items():
            if torch.isnan(grad_val) or torch.isinf(grad_val):
                nan_grad_count += 1
        if nan_grad_count > 0:
            warnings.warn(f"M-step: {nan_grad_count} gradients contain NaN/Inf")

        # ===== 7. Set parameter gradients =====
        # Apply softplus transform correction for sigma_0 and Amp
        # softplus'(x) = sigmoid(x)
        kernel.raw_sigma_0.grad = dL['sigma_0'] * torch.sigmoid(kernel.raw_sigma_0)
        kernel.raw_Amp.grad = dL['Amp'] * torch.sigmoid(kernel.raw_Amp)

        # Direct gradients for other parameters
        kernel.eps_0x.grad = dL['eps_0x'].unsqueeze(0) if dL['eps_0x'].dim() == 0 else dL['eps_0x']
        kernel.eps_0y.grad = dL['eps_0y'].unsqueeze(0) if dL['eps_0y'].dim() == 0 else dL['eps_0y']
        kernel.raw_m2log2beta.grad = dL['raw_m2log2beta'].unsqueeze(0) if dL['raw_m2log2beta'].dim() == 0 else dL['raw_m2log2beta']
        kernel.raw_mlog2rho2.grad = dL['raw_mlog2rho2'].unsqueeze(0) if dL['raw_mlog2rho2'].dim() == 0 else dL['raw_mlog2rho2']

        return loss

    optimizer.step(closure)
    kernel.clamp_hyperparameters()


def train_vargp_direct(
    kernel,
    likelihood,
    X: torch.Tensor,
    X_tilde: torch.Tensor,
    r: torch.Tensor,
    n_iterations: int,
    n_estep: int,
    n_fstep: int,
    n_mstep: int,
    lr_f: float,
    lr_m: float,
    print_every: int = 10,
    eigval_tol: float = EIGVAL_TOL,
    verbose: bool = False,
    use_analytical_mstep: bool = False
) -> Dict:
    """Train using direct variational GP with eigenspace projection.

    Implements the original varGP loop structure:
    1. Kernel recomputation (after M-step)
    2. E-step: Newton updates on m_b, V_b
    3. F-step: LBFGS on A with analytical lambda0
    4. M-step: LBFGS on kernel hyperparameters

    Args:
        kernel: ArcCosineKernel instance
        likelihood: PoissonLikelihood instance
        X: Training inputs, shape (N, n_features)
        X_tilde: Inducing points, shape (M, n_features)
        r: Spike counts, shape (N,)
        n_iterations: Number of EM iterations
        n_estep: Number of E-step Newton iterations per EM iteration
        n_fstep: Number of F-step LBFGS iterations
        n_mstep: Number of M-step LBFGS iterations
        lr_f: Learning rate for F-step
        lr_m: Learning rate for M-step
        print_every: Print progress every N iterations
        eigval_tol: Eigenvalue tolerance for eigenspace projection
        verbose: Print detailed debugging info
        use_analytical_mstep: Use analytical gradients for M-step (faster, matches varGP)

    Returns:
        Dict with:
            'losses': List of ELBO values per iteration
            'state': Final DirectVariationalState
            'time_estep_total': Total E-step time (includes F-step)
            'time_mstep_total': Total M-step time
    """
    # Initialize
    state = compute_kernels_direct(kernel, X, X_tilde, eigval_tol)

    n_b = len(state.eigvals_b)
    M = X_tilde.shape[0]
    N = X.shape[0]
    print(f"Eigenspace dimension: n_b={n_b} (from M={M} inducing points)")

    time_estep_total = 0.0
    time_mstep_total = 0.0
    losses = []

    # Initial moments
    lambda_m, lambda_var = lambda_moments_eigenspace(state)
    A = likelihood.A.squeeze()
    lambda0 = likelihood.lambda0.squeeze()
    f_mean = compute_f_mean(lambda_m, lambda_var, A, lambda0)

    for iteration in range(1, n_iterations + 1):

        # ===== Kernel recomputation after M-step =====
        if n_mstep > 0 and iteration > 1:
            state = recompute_kernels_after_mstep(kernel, X, X_tilde, state, eigval_tol)
            # Recompute moments with new kernels
            lambda_m, lambda_var = lambda_moments_eigenspace(state)
            A = likelihood.A.squeeze()
            lambda0 = likelihood.lambda0.squeeze()
            f_mean = compute_f_mean(lambda_m, lambda_var, A, lambda0)

        # ===== E-step: Newton loop =====
        start_estep = time.time()

        for i_estep in range(n_estep):
            m_b_new, V_b_new = estep_eigenspace(state, r, A, f_mean)

            # Update state
            state.m_b = m_b_new
            state.V_b = V_b_new

            # Recompute moments
            lambda_m, lambda_var = lambda_moments_eigenspace(state)

            # Recompute f_mean
            f_mean = compute_f_mean(lambda_m, lambda_var, A, lambda0)

            # Stability check
            if f_mean.mean().item() > STABILITY_THRESHOLD:
                if verbose:
                    print(f"  E-step {i_estep}: f_mean unstable ({f_mean.mean().item():.1f})")
                break

            # Early stopping: check convergence
            # (Could add convergence criterion here)

        # ===== F-step: Optimize A =====
        fstep_direct(likelihood, r, lambda_m, lambda_var, n_fstep, lr_f)

        # Update A, lambda0 and recompute f_mean
        A = likelihood.A.squeeze()
        lambda0 = likelihood.lambda0.squeeze()
        f_mean = compute_f_mean(lambda_m, lambda_var, A, lambda0)

        time_estep_total += time.time() - start_estep

        # ===== M-step: Optimize kernel hyperparameters =====
        start_mstep = time.time()

        if n_mstep > 0 and iteration < n_iterations:
            if use_analytical_mstep:
                mstep_lbfgs_analytical(kernel, likelihood, X, X_tilde, r, state, n_mstep, lr_m)
            else:
                mstep_lbfgs_autograd(kernel, likelihood, X, X_tilde, r, state, n_mstep, lr_m)

        time_mstep_total += time.time() - start_mstep

        # ===== Compute and record loss =====
        elbo = compute_elbo_eigenspace(state, r, lambda_m, lambda_var, A, lambda0)
        loss = -elbo.item()  # Negative ELBO for consistency with other modes
        losses.append(loss)

        if iteration % print_every == 0 or iteration == 1:
            print(f"Iter {iteration}/{n_iterations}: loss={loss:.2f}, "
                  f"A={A.item():.4f}, lambda0={lambda0.item():.4f}, "
                  f"n_b={len(state.eigvals_b)}")

    return {
        'losses': losses,
        'state': state,
        'time_estep_total': time_estep_total,
        'time_mstep_total': time_mstep_total,
    }


def predict_direct(
    kernel,
    likelihood,
    state: DirectVariationalState,
    X_tilde: torch.Tensor,
    X_test: torch.Tensor
) -> Dict:
    """Predict at test points using trained direct variational GP.

    Computes posterior moments at test points and expected firing rates.

    Args:
        kernel: Trained ArcCosineKernel
        likelihood: Trained PoissonLikelihood
        state: Trained DirectVariationalState
        X_tilde: Inducing points, shape (M, n_features)
        X_test: Test inputs, shape (N_test, n_features)

    Returns:
        Dict with:
            'f_pred': Predicted firing rates, shape (N_test,)
            'lambda_m': Posterior mean at test points, shape (N_test,)
            'lambda_var': Posterior variance at test points, shape (N_test,)
    """
    with torch.no_grad():
        # Compute cross-kernel to inducing points
        K_test = kernel(X_test, X_tilde).evaluate()  # (N_test, M)
        Kvec_test = kernel(X_test, diag=True)  # (N_test,)

        # Project to eigenspace
        K_test_b = K_test @ state.B  # (N_test, n_b)

        # a = K_test @ K_tilde_inv = K_test_b @ diag(1/eigvals)
        a = K_test_b / state.eigvals_b.unsqueeze(0)  # (N_test, n_b)

        # Posterior mean: lambda_m = a @ m_b
        lambda_m = a @ state.m_b  # (N_test,)

        # Posterior variance
        V_minus_K = state.V_b - state.K_tilde_b
        aV = a @ V_minus_K
        lambda_var = Kvec_test + (a * aV).sum(dim=1)
        lambda_var = torch.clamp(lambda_var, min=1e-6)

        # Predicted firing rate
        A = likelihood.A.squeeze()
        lambda0 = likelihood.lambda0.squeeze()
        f_pred = torch.exp(A * lambda_m + 0.5 * A * A * lambda_var + lambda0)

    return {
        'f_pred': f_pred,
        'lambda_m': lambda_m,
        'lambda_var': lambda_var,
    }
