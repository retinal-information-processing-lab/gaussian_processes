"""
Clean kernel implementations with torch.autograd.Function support.

This subpackage contains the new "clean" implementations of GP kernels
that support automatic differentiation via PyTorch's autograd system.

Usage:
    from gaussian_processes.Spatial_GP_repo.kernels import localker_clean, acosker_clean
"""

from .kernels import (
    # Locality kernel (C matrix)
    C_gradients_hyp,
    LocalkerCleanFunction,
    localker_clean,
    # Arc-cosine kernel
    acosker_clean,
    AcoskerCleanFunction,
    acosker_with_grad,
)

__all__ = [
    'C_gradients_hyp',
    'LocalkerCleanFunction',
    'localker_clean',
    'acosker_clean',
    'AcoskerCleanFunction',
    'acosker_with_grad',
]
