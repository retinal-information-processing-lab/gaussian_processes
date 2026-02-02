"""
DEPRECATED: M-step function for vargp_style mode

This mode is no longer maintained. Use eigenspace or default_gpy instead.

This file imports deprecated function from the parent mstep.py module.
This function will be removed from the main codebase in Phase 3.

Last working commit: 23c3856
"""

import sys
from pathlib import Path

# Add parent directory to path to import from main mstep.py
sys.path.insert(0, str(Path(__file__).parent.parent))

from mstep import m_step

__all__ = ['m_step']
