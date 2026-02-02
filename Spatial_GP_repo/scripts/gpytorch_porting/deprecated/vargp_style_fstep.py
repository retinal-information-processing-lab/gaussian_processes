"""
DEPRECATED: F-step functions for vargp_style mode

This mode is no longer maintained. Use eigenspace or default_gpy instead.

This file imports deprecated functions from the parent fstep.py module.
These functions will be removed from the main codebase in Phase 3.

Last working commit: 23c3856
"""

import sys
from pathlib import Path

# Add parent directory to path to import from main fstep.py
sys.path.insert(0, str(Path(__file__).parent.parent))

from fstep import (
    lambda0_given_A,
    f_step,
    f_step_lbfgs,
    compute_f_mean,
)

__all__ = [
    'lambda0_given_A',
    'f_step',
    'f_step_lbfgs',
    'compute_f_mean',
]
