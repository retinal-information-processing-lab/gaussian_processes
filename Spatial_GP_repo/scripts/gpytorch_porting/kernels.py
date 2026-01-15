"""
Arc-Cosine Kernel for GPyTorch

This module implements the arc-cosine kernel as a GPyTorch kernel class.
The arc-cosine kernel is derived from infinite-width 2-layer ReLU neural networks.

Mathematical definition:
    K(x, x') = (1/π) · M · J(θ)

    where:
        v_x     = xᵀCx + σ₀²
        v_x'    = x'ᵀCx' + σ₀²
        M       = √(v_x · v_x')
        cos(θ)  = (xᵀCx' + σ₀²) / M
        J(θ)    = sin(θ) + (π - θ)cos(θ)

Stage 1 (C=I): Uses identity matrix, v_x = ‖x‖² + σ₀²
Stage 2 (RF structure): Computes C from receptive field parameters (β, ρ, ξ₀)

Reference: kernels/kernels.py:acosker_clean(), localker_clean()
"""

import numpy as np
import torch
import gpytorch
from gpytorch.kernels import Kernel
from gpytorch.constraints import Positive


class ArcCosineKernel(Kernel):
    """Arc-cosine kernel for GPyTorch.

    This kernel is non-stationary (depends on actual input values, not just distances).
    It's derived from infinite-width 2-layer neural networks with ReLU activations.

    Parameters
    ----------
    sigma_0 : float, optional
        Bias variance parameter (default: 1.0). Controls the kernel value
        when inputs are zero.
    C : Tensor, optional
        Structured covariance matrix (n_features, n_features).
        If None and n_px_side is None, uses identity matrix (Stage 1).
    n_px_side : int, optional
        Image dimension (e.g., 108 for PNAS data). If provided, C is computed
        from RF parameters (Stage 2). Mutually exclusive with C parameter.
    eps_0x : float, optional
        RF center x-coordinate on [-1, 1] grid (default: 0.0 = center)
    eps_0y : float, optional
        RF center y-coordinate on [-1, 1] grid (default: 0.0 = center)
    beta : float, optional
        RF size parameter (default: 0.1). Smaller = more localized RF.
    rho : float, optional
        Smoothness parameter (default: 0.1). Controls spatial correlation.
    use_mask : bool, optional
        If True and n_px_side is set, apply pixel masking based on RF parameters.
        Pixels with locality weight α >= 0.001 are included. This reduces C from
        (n_px, n_px) to (n_masked, n_masked), typically ~100x smaller.
        Default: True.

    Attributes
    ----------
    raw_sigma_0 : Parameter
        Unconstrained parameter for sigma_0
    sigma_0 : Property
        Constrained (positive) sigma_0 value
    """

    has_lengthscale = False  # Arc-cosine doesn't have a lengthscale
    MASK_THRESHOLD = 0.001  # Pixels with α >= threshold are included

    def __init__(self, sigma_0=1.0, C=None, n_px_side=None,
                 eps_0x=0.0, eps_0y=0.0, beta=0.1, rho=0.1,
                 use_mask=True, **kwargs):
        super().__init__(**kwargs)

        # Register sigma_0 parameter with dummy initial value
        self.register_parameter(
            name='raw_sigma_0',
            parameter=torch.nn.Parameter(torch.zeros(1))
        )
        # Register positivity constraint
        self.register_constraint('raw_sigma_0', Positive())

        # Now set the actual value via the property (applies inverse transform)
        self.sigma_0 = sigma_0

        # Store C matrix (None = identity for Stage 1)
        self.C = C

        # Stage 2: RF parameters
        self.n_px_side = n_px_side

        if n_px_side is not None:
            if C is not None:
                raise ValueError("Cannot specify both C and n_px_side")

            # RF center (unconstrained, range [-1, 1])
            self.register_parameter('eps_0x',
                torch.nn.Parameter(torch.tensor([eps_0x], dtype=torch.float64)))
            self.register_parameter('eps_0y',
                torch.nn.Parameter(torch.tensor([eps_0y], dtype=torch.float64)))

            # Log-space parameters for numerical stability
            # beta = 0.1 → raw_m2log2beta = -2 * log(2 * 0.1) ≈ 3.22
            # rho = 0.1 → raw_mlog2rho2 = -log(2 * 0.1^2) ≈ 3.91
            raw_m2log2beta = -2 * np.log(2 * beta)
            raw_mlog2rho2 = -np.log(2 * rho**2)
            self.register_parameter('raw_m2log2beta',
                torch.nn.Parameter(torch.tensor([raw_m2log2beta], dtype=torch.float64)))
            self.register_parameter('raw_mlog2rho2',
                torch.nn.Parameter(torch.tensor([raw_mlog2rho2], dtype=torch.float64)))

            # Setup pixel coordinate grid
            self._setup_pixel_coords()

        # Masking mode (only valid with n_px_side)
        self.use_mask = use_mask and (n_px_side is not None)
        # Cache for mask (computed on first forward pass)
        self._cached_mask = None

    @property
    def sigma_0(self):
        """Get the constrained sigma_0 value."""
        return self.raw_sigma_0_constraint.transform(self.raw_sigma_0)

    @sigma_0.setter
    def sigma_0(self, value):
        """Set sigma_0 value."""
        self._set_sigma_0(value)

    def _set_sigma_0(self, value):
        """Set sigma_0 via inverse transform."""
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(self.raw_sigma_0)
        self.initialize(raw_sigma_0=self.raw_sigma_0_constraint.inverse_transform(value))

    def _setup_pixel_coords(self):
        """Setup normalized pixel coordinate grid on [-1, 1] × [-1, 1].

        Creates buffers xcord and ycord containing flattened pixel coordinates.
        These are registered as buffers (not parameters) since they're fixed.
        """
        ycord, xcord = torch.meshgrid(
            torch.linspace(-1, 1, self.n_px_side, dtype=torch.float64),
            torch.linspace(-1, 1, self.n_px_side, dtype=torch.float64),
            indexing='ij'
        )
        self.register_buffer('xcord', xcord.flatten())
        self.register_buffer('ycord', ycord.flatten())

    def compute_mask(self):
        """Compute pixel mask based on current RF parameters.

        Mask is computed with DETACHED parameters (structural stability).
        Pixels with locality weight α >= MASK_THRESHOLD are included.

        Returns
        -------
        mask : Tensor, shape (n_px,), dtype=bool
            Boolean mask for active pixels
        """
        # Ensure coords are on same device as parameters
        xcord = self.xcord.to(self.eps_0x.device)
        ycord = self.ycord.to(self.eps_0y.device)

        # Compute mask with DETACHED theta (non-differentiable)
        with torch.no_grad():
            beta_detached = torch.exp(self.raw_m2log2beta.detach())
            dist_sq = (xcord - self.eps_0x.detach())**2 + (ycord - self.eps_0y.detach())**2
            logalpha = -beta_detached * dist_sq
            alpha = torch.exp(logalpha)
            mask = alpha >= self.MASK_THRESHOLD

        return mask

    def _compute_C_matrix(self, apply_mask=False):
        """Compute structured covariance matrix C encoding RF properties.

        C = α_local[:, None] · C_smooth · α_local[None, :]

        where:
            α_local[i] = exp(-β · ||ξᵢ - ξ₀||²)
            C_smooth[i,j] = exp(-ρ² · ||ξᵢ - ξⱼ||²)

        Parameters
        ----------
        apply_mask : bool
            If True, compute C only for masked pixels (smaller matrix).
            If False, compute full C matrix.

        Returns
        -------
        C : Tensor
            Structured covariance matrix.
            Shape (n_masked, n_masked) if apply_mask=True, else (n_px, n_px).
        mask : Tensor or None
            Boolean mask (n_px,) if apply_mask=True, else None.
        """
        # Transform log-space parameters
        beta = torch.exp(self.raw_m2log2beta)
        rho2 = torch.exp(self.raw_mlog2rho2)

        # Ensure coords are on same device as parameters
        xcord = self.xcord.to(self.eps_0x.device)
        ycord = self.ycord.to(self.eps_0y.device)

        # Compute mask if needed
        mask = None
        if apply_mask:
            mask = self.compute_mask()
            xcord = xcord[mask]
            ycord = ycord[mask]

        # Locality weights: distance from RF center
        dist_sq_center = (xcord - self.eps_0x)**2 + (ycord - self.eps_0y)**2
        logalpha = -beta * dist_sq_center
        alpha = torch.exp(logalpha)  # (n_px,) or (n_masked,)

        # Smoothness kernel: pairwise pixel distances
        dx = xcord[:, None] - xcord[None, :]
        dy = ycord[:, None] - ycord[None, :]
        dist_sq_pairwise = dx**2 + dy**2
        C_smooth = torch.exp(-rho2 * dist_sq_pairwise)

        # Full C matrix: outer product of alpha weighted by C_smooth
        C = alpha[:, None] * C_smooth * alpha[None, :]

        # Symmetrize for numerical stability
        C = (C + C.T) / 2

        return C, mask

    def forward(self, x1, x2, diag=False, **params):
        """Compute the arc-cosine kernel matrix.

        Parameters
        ----------
        x1 : Tensor, shape (..., n1, n_features)
            First input batch. When use_mask=True, inputs should be FULL images
            (n_features = n_px_side²) and masking is applied internally.
        x2 : Tensor, shape (..., n2, n_features)
            Second input batch
        diag : bool
            If True, return only the diagonal of the kernel matrix

        Returns
        -------
        Tensor
            Kernel matrix of shape (..., n1, n2) or (..., n1,) if diag=True
        """
        sigma_0_sq = self.sigma_0 ** 2

        # Determine C matrix source and handle masking
        mask = None
        if self.n_px_side is not None:
            # Stage 2: Compute C from RF parameters
            C, mask = self._compute_C_matrix(apply_mask=self.use_mask)
            # Cache mask for external access
            if mask is not None:
                self._cached_mask = mask
        elif self.C is not None:
            # Provided C matrix
            C = self.C.to(x1.device, x1.dtype)
        else:
            C = None  # Stage 1: C=I (identity)

        # Apply mask to inputs if using masking
        if mask is not None:
            x1 = x1[..., mask]
            x2 = x2[..., mask]

        # Handle C matrix (identity for Stage 1)
        if C is None:
            # C = I: quadratic form xᵀCx = xᵀx = ‖x‖²
            # V = ‖x‖² + σ₀²
            V1 = (x1 * x1).sum(dim=-1) + sigma_0_sq  # (..., n1)
            CX1 = x1  # When C=I, CX = X
        else:
            # General C: quadratic form xᵀCx
            C = C.to(x1.device, x1.dtype)
            CX1 = x1 @ C  # (..., n1, n_features)
            V1 = (CX1 * x1).sum(dim=-1) + sigma_0_sq  # (..., n1)

        if diag:
            # Diagonal case: K(x_i, x_i)
            # When x1 = x2, cos(θ) = 1, θ = 0, J(0) = π
            # K = M * π / π = M = √(v_x * v_x) = v_x
            return V1

        # Full matrix case
        if C is None:
            V2 = (x2 * x2).sum(dim=-1) + sigma_0_sq  # (..., n2)
            CX2 = x2
        else:
            CX2 = x2 @ C  # (..., n2, n_features)
            V2 = (CX2 * x2).sum(dim=-1) + sigma_0_sq  # (..., n2)

        # Cross-term: xᵀCx' (when C=I: xᵀx')
        # Shape: (..., n1, n2)
        if C is None:
            C12 = torch.matmul(x1, x2.transpose(-2, -1)) + sigma_0_sq
        else:
            C12 = torch.matmul(CX1, x2.transpose(-2, -1)) + sigma_0_sq

        # Magnitude: M = √(v_x · v_x')
        # Broadcast V1 (..., n1) and V2 (..., n2) to (..., n1, n2)
        M = torch.sqrt(V1.unsqueeze(-1) * V2.unsqueeze(-2))

        # Normalized inner product
        # Clamp for numerical stability (arccos domain is [-1, 1])
        eps = 1e-7
        cos_theta = torch.clamp(C12 / M, -1.0 + eps, 1.0 - eps)

        # Angle and sine
        theta_angle = torch.arccos(cos_theta)
        sin_theta = torch.sqrt(torch.clamp(1.0 - cos_theta ** 2, min=eps))

        # Angular term: J(θ) = sin(θ) + (π - θ)cos(θ)
        J = sin_theta + (torch.pi - theta_angle) * cos_theta

        # Kernel: K = M · J / π
        K = M * J / torch.pi

        return K


def test_kernel_matches_reference():
    """Test that ArcCosineKernel matches acosker_clean() output."""
    import sys
    from pathlib import Path

    # Add path to import acosker_clean from kernels/kernels.py
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from kernels.kernels import acosker_clean

    # Create test data
    torch.manual_seed(42)
    n1, n2, nx = 10, 8, 100
    X1 = torch.randn(n1, nx)
    X2 = torch.randn(n2, nx)
    sigma_0 = 1.5

    # Reference implementation
    theta = {'sigma_0': torch.tensor(sigma_0)}
    K_ref = acosker_clean(theta, X1, X2, C=None, diag=False)

    # GPyTorch implementation
    kernel = ArcCosineKernel(sigma_0=sigma_0)
    K_new = kernel(X1, X2).evaluate()

    # Compare
    max_diff = (K_ref - K_new).abs().max().item()
    print(f"Max absolute difference: {max_diff:.2e}")

    if max_diff < 1e-5:
        print("PASS: Kernels match!")
        return True
    else:
        print("FAIL: Kernels don't match!")
        return False


if __name__ == '__main__':
    test_kernel_matches_reference()
