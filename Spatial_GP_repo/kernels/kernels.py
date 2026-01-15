"""
Clean kernel implementations with torch.autograd.Function support.

This module contains the new "clean" implementations of GP kernels that support
automatic differentiation via PyTorch's autograd system.

Contents:
- C_gradients_hyp: Analytical gradients of C matrix w.r.t. hyperparameters
- LocalkerCleanFunction: torch.autograd.Function for locality kernel
- localker_clean: User-facing wrapper for LocalkerCleanFunction
- acosker_clean: Arc-cosine kernel with optional analytical gradients
- AcoskerCleanFunction: torch.autograd.Function for arc-cosine kernel
- acosker_with_grad: Differentiable acosker_clean wrapper

Usage:
    from gaussian_processes.Spatial_GP_repo.kernels import localker_clean, acosker_clean
"""

import torch


# =============================================================================
# Locality Kernel (C matrix)
# =============================================================================

def C_gradients_hyp(C, theta_Amp, theta_2log2beta, logalpha, logCsmooth,
                    xcord, ycord, theta_eps0x, theta_eps0y):
    """
    Compute analytical gradients of C matrix w.r.t. hyperparameters.

    This is a dedicated function for computing dC/d{Amp, eps0x, eps0y, -2log2beta, -log2rho2}.
    It can be called either in forward (for speed) or backward (to save memory).

    Parameters
    ----------
    C : Tensor (n_masked, n_masked)
        The covariance matrix
    theta_Amp : Tensor (scalar)
        Amplitude hyperparameter
    theta_2log2beta : Tensor (scalar)
        Locality hyperparameter (-2*log(2*beta))
    logalpha : Tensor (n_masked,)
        Log of locality weights: -exp(-2log2beta) * dist_sq
    logCsmooth : Tensor (n_masked, n_masked)
        Log of smoothness kernel: -exp(-log2rho2) * dist_sq_smooth
    xcord, ycord : Tensor (n_masked,)
        Masked pixel coordinates
    theta_eps0x, theta_eps0y : Tensor (scalar)
        RF center position

    Returns
    -------
    dC_Amp : Tensor (n_masked, n_masked)
        dC/d(Amp)
    dC_eps0x : Tensor (n_masked, n_masked)
        dC/d(eps_0x)
    dC_eps0y : Tensor (n_masked, n_masked)
        dC/d(eps_0y)
    dC_logbeta : Tensor (n_masked, n_masked)
        dC/d(-2log2beta)
    dC_logrho : Tensor (n_masked, n_masked)
        dC/d(-log2rho2)
    """
    # dC/dAmp = C / Amp
    dC_Amp = C / theta_Amp

    # dC/deps0x = 2 * exp(-2log2beta) * C * (xcord[:, None] + xcord[None, :] - 2*eps0x)
    dC_eps0x = 2 * torch.exp(theta_2log2beta) * C * (xcord[:, None] + xcord[None, :] - 2 * theta_eps0x)

    # dC/deps0y = 2 * exp(-2log2beta) * C * (ycord[:, None] + ycord[None, :] - 2*eps0y)
    dC_eps0y = 2 * torch.exp(theta_2log2beta) * C * (ycord[:, None] + ycord[None, :] - 2 * theta_eps0y)

    # dC/d(-2log2beta) = C * (logalpha[:, None] + logalpha[None, :])
    dC_logbeta = C * (logalpha[:, None] + logalpha[None, :])

    # dC/d(-log2rho2) = C * logCsmooth
    dC_logrho = C * logCsmooth

    return dC_Amp, dC_eps0x, dC_eps0y, dC_logbeta, dC_logrho


class LocalkerCleanFunction(torch.autograd.Function):
    """
    Autograd wrapper for localker_clean with analytical backward pass.

    Forward: Computes C matrix from theta hyperparameters
    Backward: Uses analytical dC formulas for gradient w.r.t. theta

    The analytical gradients match those in localker(grad=True).

    Supports two modes via compute_grad parameter:
    - compute_grad=True (speed priority): Compute gradients in forward, save dC matrices
    - compute_grad=False (memory priority): Save intermediates, compute gradients in backward
    """

    @staticmethod
    def forward(ctx, theta_Amp, theta_2log2beta, theta_log2rho2,
                theta_eps0x, theta_eps0y,
                n_px_side, compute_grad):
        """
        Forward pass: compute C matrix.

        Parameters
        ----------
        theta_Amp : Tensor (scalar)
            Amplitude hyperparameter
        theta_2log2beta : Tensor (scalar)
            Locality hyperparameter (-2*log(2*beta))
        theta_log2rho2 : Tensor (scalar)
            Smoothness hyperparameter (-log(2*rho^2))
        theta_eps0x, theta_eps0y : Tensor (scalar)
            RF center position
        n_px_side : int
            Number of pixels per side
        compute_grad : bool
            If True, compute gradients in forward (faster backward, more memory)
            If False, compute gradients in backward (slower backward, less memory)

        Returns
        -------
        C : Tensor (n_masked, n_masked)
            Covariance matrix for masked pixels
        mask : Tensor (n_px_side^2,) bool
            Mask indicating active pixels
        """
        # Build coordinate grid
        ycord, xcord = torch.meshgrid(
            torch.linspace(-1, 1, n_px_side),
            torch.linspace(-1, 1, n_px_side),
            indexing='ij'
        )
        xcord = xcord.flatten().to(theta_Amp.device, dtype=theta_Amp.dtype)
        ycord = ycord.flatten().to(theta_Amp.device, dtype=theta_Amp.dtype)

        # Step 1: Compute mask with DETACHED theta (mask is structurally fixed)
        with torch.no_grad():
            dist_sq_detached = (xcord - theta_eps0x.detach())**2 + (ycord - theta_eps0y.detach())**2
            logalpha_for_mask = -torch.exp(theta_2log2beta.detach()) * dist_sq_detached
            alpha_for_mask = torch.exp(logalpha_for_mask)
            mask = alpha_for_mask >= 0.001

        # Step 2: Apply mask to coordinates
        xcord_masked = xcord[mask]
        ycord_masked = ycord[mask]

        # Step 3: Compute C WITH gradient tracking
        dist_sq = (xcord_masked - theta_eps0x)**2 + (ycord_masked - theta_eps0y)**2
        logalpha = -torch.exp(theta_2log2beta) * dist_sq
        alpha_local = torch.exp(logalpha)

        # Smooth prior
        dist_sq_smooth = (xcord_masked - xcord_masked[:, None])**2 + (ycord_masked - ycord_masked[:, None])**2
        logCsmooth = -torch.exp(theta_log2rho2) * dist_sq_smooth
        C_smooth = torch.exp(logCsmooth)

        # Combine: C = Amp * alpha_local * C_smooth * alpha_local^T
        C = theta_Amp * alpha_local[:, None] * C_smooth * alpha_local[None, :]

        # Symmetrize
        C = (C + C.T) / 2

        # Save for backward - two modes
        if compute_grad:
            # Speed priority: compute gradients now, save dC matrices
            dC_Amp, dC_eps0x, dC_eps0y, dC_logbeta, dC_logrho = C_gradients_hyp(
                C, theta_Amp, theta_2log2beta, logalpha, logCsmooth,
                xcord_masked, ycord_masked, theta_eps0x, theta_eps0y
            )
            ctx.save_for_backward(dC_Amp, dC_eps0x, dC_eps0y, dC_logbeta, dC_logrho)
            ctx.grad_precomputed = True
        else:
            # Memory priority: save intermediates, compute gradients in backward
            ctx.save_for_backward(C, theta_Amp, theta_2log2beta, logalpha, logCsmooth,
                                  xcord_masked, ycord_masked, theta_eps0x, theta_eps0y)
            ctx.grad_precomputed = False

        return C, mask

    @staticmethod
    def backward(ctx, grad_C, grad_mask):
        """
        Backward pass: compute gradients w.r.t. theta.

        Given: grad_C = ∂L/∂C
        Compute: ∂L/∂theta = sum(grad_C * dC/dtheta)

        If gradients were precomputed in forward, just apply chain rule.
        Otherwise, compute gradients now using C_gradients_hyp().
        """
        if ctx.grad_precomputed:
            # Speed mode: gradients already computed in forward
            dC_Amp, dC_eps0x, dC_eps0y, dC_logbeta, dC_logrho = ctx.saved_tensors
        else:
            # Memory mode: compute gradients now
            (C, theta_Amp, theta_2log2beta, logalpha, logCsmooth,
             xcord, ycord, eps0x, eps0y) = ctx.saved_tensors
            dC_Amp, dC_eps0x, dC_eps0y, dC_logbeta, dC_logrho = C_gradients_hyp(
                C, theta_Amp, theta_2log2beta, logalpha, logCsmooth,
                xcord, ycord, eps0x, eps0y
            )

        # Chain rule: ∂L/∂theta = sum(∂L/∂C * ∂C/∂theta)
        grad_Amp = (grad_C * dC_Amp).sum()
        grad_2log2beta = (grad_C * dC_logbeta).sum()
        grad_log2rho2 = (grad_C * dC_logrho).sum()
        grad_eps0x = (grad_C * dC_eps0x).sum()
        grad_eps0y = (grad_C * dC_eps0y).sum()

        # Return gradients for each input (same order as forward)
        # No gradient for n_px_side (int) and compute_grad (bool)
        return (grad_Amp, grad_2log2beta, grad_log2rho2,
                grad_eps0x, grad_eps0y,
                None, None)  # n_px_side, compute_grad


def localker_clean(theta, n_px_side, compute_grad=True):
    """
    Compute C matrix for arc-cosine kernel with autograd support.

    This is a user-facing wrapper around LocalkerCleanFunction.
    Unlike localker(), this function does NOT compute explicit dC gradients.
    Instead, gradients flow automatically through torch.autograd.

    Parameters
    ----------
    theta : dict
        Hyperparameters dictionary containing:
        - 'Amp': amplitude
        - '-2log2beta': locality parameter
        - '-log2rho2': smoothness parameter
        - 'eps_0x', 'eps_0y': RF center position
    n_px_side : int
        Number of pixels per side
    compute_grad : bool, optional (default=True)
        Controls speed/memory tradeoff for gradient computation:
        - True (default): Compute gradients in forward pass (faster backward, more memory)
        - False: Compute gradients in backward pass (slower backward, less memory)

    Returns
    -------
    C : Tensor (n_masked, n_masked)
        Covariance matrix for masked pixels
    mask : Tensor (n_px_side^2,) bool
        Mask indicating active pixels

    Notes
    -----
    - No theta limits checking (unlike localker)
    - No explicit dC computation - use .backward() for gradients
    - Mask is computed with detached theta (non-differentiable)
    - compute_grad=True is recommended unless memory is constrained
    """
    C, mask = LocalkerCleanFunction.apply(
        theta['Amp'],
        theta['-2log2beta'],
        theta['-log2rho2'],
        theta['eps_0x'],
        theta['eps_0y'],
        n_px_side,
        compute_grad
    )
    return C, mask


# =============================================================================
# Arc-Cosine Kernel
# =============================================================================

def acosker_clean(theta, X1, X2=None, C=None, diag=False, compute_grad=False):
    """
    Arc-cosine kernel with optional analytical gradient w.r.t. first argument.

    Clean implementation using native PyTorch shapes (n_points, n_features).
    No internal transposes. Gradient is always computed w.r.t. X1.

    Mathematical definition:
        K(x, x') = (1/π) · M · J(θ)

        where:
            v_x     = x^T C x + σ₀²
            v_{x'}  = x'^T C x' + σ₀²
            M       = √(v_x · v_{x'})
            cos(θ)  = (x^T C x' + σ₀²) / M
            J(θ)    = sin(θ) + (π - θ)cos(θ)

    Gradient formula (w.r.t. first argument x):
        ∇_x K(x, x') = (1/π) · [(π - θ) C x' + sin(θ) √(v_{x'}/v_x) C x]

    Parameters
    ----------
    theta : dict
        Hyperparameters dictionary containing 'sigma_0'
    X1 : Tensor, shape (n1, nx)
        First input points
    X2 : Tensor, shape (n2, nx) or None
        Second input points. If None, uses X1 (for diagonal case)
    C : Tensor, shape (nx, nx) or None
        Prior covariance matrix. If None, uses identity
    diag : bool
        If True, compute only diagonal K(x_i, x_i) for each point in X1
    compute_grad : bool
        If True, also return gradient w.r.t. X1

    Returns
    -------
    If compute_grad=False:
        K : Tensor
            - shape (n1, n2) if diag=False
            - shape (n1,) if diag=True

    If compute_grad=True:
        (K, dK_X1) : tuple
            - dK_X1 shape (n1, n2, nx) if diag=False
            - dK_X1 shape (n1, nx) if diag=True
    """
    sigma_0 = theta['sigma_0']
    sigma_0_sq = sigma_0 ** 2
    n1, nx = X1.shape

    # Default C to identity if not provided
    if C is None:
        C = torch.eye(nx, device=X1.device, dtype=X1.dtype)

    # Compute quadratic form: V[i] = X[i] @ C @ X[i].T + σ₀²
    # Using element-wise product and sum: (X @ C) * X, summed over features
    CX1 = X1 @ C  # (n1, nx)
    V1 = (CX1 * X1).sum(dim=1) + sigma_0_sq  # (n1,)

    if diag:
        # Diagonal case: K(x_i, x_i) = v_{x_i}
        K = V1  # (n1,)

        if compute_grad:
            # ∇_x K(x, x) = 2 C x
            dK_X1 = 2 * CX1  # (n1, nx)
            return K, dK_X1
        else:
            return K

    # Full matrix case
    if X2 is None:
        X2 = X1
        V2 = V1
        CX2 = CX1
    else:
        CX2 = X2 @ C  # (n2, nx)
        V2 = (CX2 * X2).sum(dim=1) + sigma_0_sq  # (n2,)

    n2 = X2.shape[0]

    # Cross-term: C12[i,j] = X1[i] @ C @ X2[j].T + σ₀²
    C12 = X1 @ C @ X2.T + sigma_0_sq  # (n1, n2)

    # Magnitude: M[i,j] = √(V1[i] · V2[j])
    M = torch.sqrt(V1[:, None] * V2[None, :])  # (n1, n2)

    # Normalized inner product (clamp for numerical stability with arccos)
    cos_theta = torch.clamp(C12 / M, -1.0, 1.0)  # (n1, n2)

    # Angle and sine
    theta_angle = torch.arccos(cos_theta)  # (n1, n2)
    sin_theta = torch.sqrt(1.0 - cos_theta ** 2)  # (n1, n2)

    # WARNING: When using torch.autograd.grad() with this function, gradients may
    # produce NaN if input points are very similar (cos_theta ≈ 1). This occurs because
    # arccos(x) has derivative -1/√(1-x²) → -∞ as x → 1, and similarly sqrt(1-x²) → 0.
    # The analytical gradient (compute_grad=True) does not have this issue.
    #
    # Proposed fix for autograd compatibility (not yet implemented):
    #   eps = 1e-7
    #   cos_theta = torch.clamp(C12 / M, -1.0 + eps, 1.0 - eps)
    #   sin_theta = torch.sqrt(torch.clamp(1.0 - cos_theta ** 2, min=eps))

    # Angular term: J(θ) = sin(θ) + (π - θ)cos(θ)
    J = sin_theta + (torch.pi - theta_angle) * cos_theta  # (n1, n2)

    # Kernel: K = M · J / π
    K = M * J / torch.pi  # (n1, n2)

    if compute_grad:
        # Gradient w.r.t. X1 (first argument):
        # ∇_x K(x, x') = (1/π) · [(π - θ) C x' + sin(θ) √(v_{x'}/v_x) C x]

        # Ratio √(V2[j] / V1[i]) for each (i,j) pair
        ratio = torch.sqrt(V2[None, :] / V1[:, None])  # (n1, n2)

        # Term 1: (π - θ) C x'  ->  broadcast to (n1, n2, nx)
        term1 = (torch.pi - theta_angle)[:, :, None] * CX2[None, :, :]  # (n1, n2, nx)

        # Term 2: sin(θ) √(v_{x'}/v_x) C x  ->  broadcast to (n1, n2, nx)
        term2 = (sin_theta * ratio)[:, :, None] * CX1[:, None, :]  # (n1, n2, nx)

        dK_X1 = (term1 + term2) / torch.pi  # (n1, n2, nx)

        return K, dK_X1
    else:
        return K


class AcoskerCleanFunction(torch.autograd.Function):
    """
    torch.autograd.Function wrapper for acosker_clean.
    Uses analytical gradient instead of autograd.
    """

    @staticmethod
    def forward(ctx, theta, X1, X2, C, diag):
        # Detach inputs for forward computation (no autograd needed)
        X1_detached = X1.detach()
        X2_detached = X2.detach() if X2 is not None else None

        K, dK_X1 = acosker_clean(theta, X1_detached, X2_detached, C=C, diag=diag, compute_grad=True)
        ctx.save_for_backward(dK_X1)
        ctx.diag = diag
        return K

    @staticmethod
    def backward(ctx, grad_output):
        dK_X1, = ctx.saved_tensors
        if ctx.diag:
            grad_X1 = grad_output[:, None] * dK_X1
        else:
            grad_X1 = torch.einsum('ij,ijk->ik', grad_output, dK_X1)
        # Return gradients for: theta, X1, X2, C, diag
        return None, grad_X1, None, None, None


def acosker_with_grad(theta, X1, X2=None, C=None, diag=False):
    """Differentiable acosker_clean using analytical gradient."""
    return AcoskerCleanFunction.apply(theta, X1, X2, C, diag)
