"""
Test utilities for GPyTorch porting project.

This module provides common utilities for test scripts to ensure
consistent behavior across all tests.

Created: 2026-01-19
"""

import torch
import numpy as np
from pathlib import Path
from typing import Optional, Union

# Standard paths
GPYTORCH_PORTING_DIR = Path(__file__).parent.parent
DATA_PATH = GPYTORCH_PORTING_DIR.parent.parent / 'notebooks' / 'PNAS_paper_sorted_data.npz'


def set_reproducible_seed(seed: int = 42, device: Optional[Union[str, torch.device]] = 'cuda'):
    """Set random seed for reproducibility.

    IMPORTANT: This function initializes CUDA before setting the seed to ensure
    consistent random state regardless of when the GPU is first used in the script.

    Background (see HANDOFF_2026-01-18.md Section 20):
    - CUDA initialization can affect PyTorch's random state
    - If seed is set BEFORE CUDA init, the random sequence differs from
      setting seed AFTER CUDA init
    - This caused test_kernel_cache.py to fail while test_estep_pnas.py worked
      with identical code - the only difference was import order

    This function ensures CUDA is initialized first, then sets seeds, providing
    deterministic behavior regardless of import order or when GPU is first used.

    Args:
        seed: Random seed (default: 42)
        device: Target device. If 'cuda' or a CUDA device, triggers CUDA init.
                Use 'cpu' to skip CUDA initialization.

    Example:
        from test_utils import set_reproducible_seed
        set_reproducible_seed(42)  # Call before any data loading or model creation
    """
    # Determine if we should initialize CUDA
    init_cuda = False
    if device is not None:
        if isinstance(device, str):
            init_cuda = device == 'cuda' or device.startswith('cuda:')
        elif isinstance(device, torch.device):
            init_cuda = device.type == 'cuda'

    # Initialize CUDA BEFORE setting seed
    if init_cuda and torch.cuda.is_available():
        torch.cuda.init()

    # WORKAROUND: Replicate unexplained side effect from GP_utils.py line 49:
    #   torch.pi = torch.acos(torch.zeros(1)).item() * 2
    #
    # MYSTERY: The assignment to torch.pi itself appears to affect PyTorch's
    # random state in a way we don't understand. torch.pi is a built-in constant
    # since PyTorch 1.8, so this line in GP_utils is legacy/unnecessary code.
    # Yet without replicating it here, test_kernel_cache.py fails while
    # test_estep_pnas.py (which imports GP_utils) succeeds.
    #
    # TODO: Investigate why assigning to torch.pi affects random state.
    # See HANDOFF_2026-01-18.md Section 20 for details.
    torch.pi = torch.acos(torch.zeros(1)).item() * 2

    # Set seeds for all random number generators
    torch.manual_seed(seed)
    np.random.seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # For multi-GPU

    # Optional: Make CUDA operations deterministic (may reduce performance)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False


def get_device(prefer_cuda: bool = True) -> torch.device:
    """Get the appropriate device for testing.

    Args:
        prefer_cuda: If True, use CUDA if available

    Returns:
        torch.device for 'cuda' or 'cpu'
    """
    if prefer_cuda and torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')


def load_pnas_data(dtype: torch.dtype = torch.float64) -> dict:
    """Load PNAS dataset for testing.

    Returns:
        Dictionary with keys: X_train, X_val, X_test, R_train, R_val, R_test
        All tensors are on CPU with the specified dtype.
    """
    data = np.load(DATA_PATH)
    return {
        'X_train': torch.tensor(data['images_train'], dtype=dtype),
        'X_val': torch.tensor(data['images_val'], dtype=dtype),
        'X_test': torch.tensor(data['images_test'], dtype=dtype),
        'R_train': torch.tensor(data['responses_train'], dtype=dtype),
        'R_val': torch.tensor(data['responses_val'], dtype=dtype),
        'R_test': torch.tensor(data['responses_test'], dtype=dtype),
    }
