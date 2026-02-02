"""
DEPRECATED: E-step functions for vargp_style mode

This mode is no longer maintained. Use eigenspace or default_gpy instead.

This file imports deprecated functions from the parent estep.py module.
These functions will be removed from the main codebase in Phase 3.

Last working commit: 23c3856
"""

import sys
from pathlib import Path

# Add parent directory to path to import from main estep.py
sys.path.insert(0, str(Path(__file__).parent.parent))

from estep import (
    STABILITY_THRESHOLD,
    compute_kernel_cache,
    compute_L_K,
    compute_moments_from_kernel_cache,
    _newton_update,
    _estep_single,
    e_step_with_kernel_cache,
    e_step,
    e_step_explicit,
    e_step_loop,
    compute_moments,
)

__all__ = [
    'STABILITY_THRESHOLD',
    'compute_kernel_cache',
    'compute_L_K',
    'compute_moments_from_kernel_cache',
    '_newton_update',
    '_estep_single',
    'e_step_with_kernel_cache',
    'e_step',
    'e_step_explicit',
    'e_step_loop',
    'compute_moments',
]
