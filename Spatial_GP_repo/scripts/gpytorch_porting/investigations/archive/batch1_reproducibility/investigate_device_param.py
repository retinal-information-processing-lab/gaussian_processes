#!/usr/bin/env python3
"""
Investigation A2: Device Parameter Random Sequence Issue

Goal: Understand why `set_reproducible_seed(42, device='cuda')` produces
different random sequences than `set_reproducible_seed(42)` (without device param).

The device parameter defaults to 'cuda', so both calls SHOULD be equivalent.

Usage:
    python investigate_device_param.py
"""

import subprocess
import sys
import json

# ============================================================================
# Helper: Run code in isolated subprocess
# ============================================================================

def run_isolated_test(code_snippet: str, description: str) -> dict:
    """Run a code snippet in a fresh Python process and capture output."""

    # Full test code with imports and random sequence generation
    full_code = f'''
import torch
import numpy as np
from typing import Optional, Union

# === IMPLEMENTATION OF set_reproducible_seed (copied from test_utils.py) ===
def set_reproducible_seed(seed: int = 42, device: Optional[Union[str, torch.device]] = 'cuda'):
    init_cuda = False
    if device is not None:
        if isinstance(device, str):
            init_cuda = device == 'cuda' or device.startswith('cuda:')
        elif isinstance(device, torch.device):
            init_cuda = device.type == 'cuda'

    if init_cuda and torch.cuda.is_available():
        torch.cuda.init()

    torch.pi = torch.acos(torch.zeros(1)).item() * 2

    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

# === TEST CODE ===
{code_snippet}

# === GENERATE RANDOM SEQUENCE ===
device_for_gen = 'cuda' if torch.cuda.is_available() else 'cpu'

randn_vals = torch.randn(5, device=device_for_gen).tolist()
randperm_vals = torch.randperm(100, device=device_for_gen)[:5].tolist()
numpy_vals = np.random.rand(5).tolist()

# Output as JSON for parsing
import json
result = {{
    'randn': randn_vals,
    'randperm': randperm_vals,
    'numpy': numpy_vals,
    'device': device_for_gen
}}
print("RESULT_JSON:" + json.dumps(result))
'''

    try:
        result = subprocess.run(
            [sys.executable, '-c', full_code],
            capture_output=True,
            text=True,
            timeout=30
        )

        output = result.stdout + result.stderr

        # Parse JSON result
        for line in output.split('\n'):
            if line.startswith('RESULT_JSON:'):
                data = json.loads(line[12:])
                return {'success': True, 'data': data, 'output': output}

        return {'success': False, 'error': 'No result found', 'output': output}

    except Exception as e:
        return {'success': False, 'error': str(e), 'output': ''}


def compare_results(result1: dict, result2: dict) -> bool:
    """Compare two test results for equality."""
    if not result1['success'] or not result2['success']:
        return False

    d1, d2 = result1['data'], result2['data']

    randn_match = all(abs(a - b) < 1e-10 for a, b in zip(d1['randn'], d2['randn']))
    randperm_match = d1['randperm'] == d2['randperm']
    numpy_match = all(abs(a - b) < 1e-10 for a, b in zip(d1['numpy'], d2['numpy']))

    return randn_match and randperm_match and numpy_match


def format_sequence(data: dict) -> str:
    """Format a random sequence for display."""
    randn = [f'{v:.6f}' for v in data['randn'][:3]]
    return f"randn={randn}, randperm={data['randperm'][:3]}"


# ============================================================================
# Test: Basic calling patterns
# ============================================================================

def test_calling_patterns():
    """
    Test different ways of calling set_reproducible_seed.

    The key question: Does passing device explicitly produce different results
    than relying on the default?
    """
    print("\n" + "="*70)
    print("Test: Calling pattern differences")
    print("="*70)

    # Variant A: No args (use all defaults)
    code_a = "set_reproducible_seed(42)"
    result_a = run_isolated_test(code_a, "No device arg")

    # Variant B: Explicit device='cuda'
    code_b = "set_reproducible_seed(42, device='cuda')"
    result_b = run_isolated_test(code_b, "device='cuda'")

    # Variant C: device=None
    code_c = "set_reproducible_seed(42, device=None)"
    result_c = run_isolated_test(code_c, "device=None")

    # Variant D: torch.device object
    code_d = "set_reproducible_seed(42, device=torch.device('cuda'))"
    result_d = run_isolated_test(code_d, "device=torch.device('cuda')")

    print(f"\nVariant A (no device arg):     {format_sequence(result_a['data'])}")
    print(f"Variant B (device='cuda'):     {format_sequence(result_b['data'])}")
    print(f"Variant C (device=None):       {format_sequence(result_c['data'])}")
    print(f"Variant D (device=obj):        {format_sequence(result_d['data'])}")

    a_b = compare_results(result_a, result_b)
    a_c = compare_results(result_a, result_c)
    b_d = compare_results(result_b, result_d)

    print(f"\nA == B (default vs explicit): {a_b}")
    print(f"A == C (default vs None):     {a_c}")
    print(f"B == D (string vs object):    {b_d}")

    if not a_b:
        print("\n>>> CRITICAL: Default device produces DIFFERENT results than explicit 'cuda'!")
    else:
        print("\n>>> Default and explicit 'cuda' produce SAME results")

    if not a_c:
        print(">>> device=None produces DIFFERENT results than default")

    return result_a, result_b, result_c, result_d


# ============================================================================
# Test: Isolate the conditional logic
# ============================================================================

def test_isinstance_check():
    """
    Test if the isinstance() check itself triggers something.

    The current code does:
        if isinstance(device, str): ...
        elif isinstance(device, torch.device): ...

    Does this type checking have side effects?
    """
    print("\n" + "="*70)
    print("Test: isinstance() check side effects")
    print("="*70)

    # Re-implement without isinstance checks
    code_modified = '''
# Modified version: no isinstance checks, direct comparison
def set_reproducible_seed_modified(seed: int = 42, device='cuda'):
    init_cuda = False
    if device is not None:
        # Direct string comparison instead of isinstance
        if device == 'cuda' or (hasattr(device, 'startswith') and device.startswith('cuda:')):
            init_cuda = True
        elif hasattr(device, 'type') and device.type == 'cuda':
            init_cuda = True

    if init_cuda and torch.cuda.is_available():
        torch.cuda.init()

    torch.pi = torch.acos(torch.zeros(1)).item() * 2

    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

set_reproducible_seed_modified(42, device='cuda')
'''
    result_modified = run_isolated_test(code_modified, "Modified (no isinstance)")

    # Original with explicit device
    code_original = "set_reproducible_seed(42, device='cuda')"
    result_original = run_isolated_test(code_original, "Original with device")

    print(f"\nOriginal (isinstance):  {format_sequence(result_original['data'])}")
    print(f"Modified (no isinstance): {format_sequence(result_modified['data'])}")

    same = compare_results(result_original, result_modified)
    print(f"\nSame? {same}")

    if not same:
        print("\n>>> FINDING: isinstance() check changes random state!")
    else:
        print("\n>>> isinstance() is not the cause")

    return result_original, result_modified


# ============================================================================
# Test: Function parameter mechanics
# ============================================================================

def test_parameter_mechanics():
    """
    Test Python parameter passing mechanics.

    In Python, default args are evaluated once at function definition time.
    But 'cuda' is just a string literal, so this shouldn't matter...
    """
    print("\n" + "="*70)
    print("Test: Python parameter mechanics")
    print("="*70)

    # Test: Call with positional vs keyword
    code_positional = "set_reproducible_seed(42, 'cuda')"  # Positional
    code_keyword = "set_reproducible_seed(42, device='cuda')"  # Keyword

    result_positional = run_isolated_test(code_positional, "Positional arg")
    result_keyword = run_isolated_test(code_keyword, "Keyword arg")

    print(f"\nPositional:  {format_sequence(result_positional['data'])}")
    print(f"Keyword:     {format_sequence(result_keyword['data'])}")

    same = compare_results(result_positional, result_keyword)
    print(f"\nSame? {same}")

    return result_positional, result_keyword


# ============================================================================
# Test: Conditional cuda.init() logic
# ============================================================================

def test_cuda_init_conditional():
    """
    Test if the conditional cuda.init() logic matters.

    When device=None is passed, init_cuda stays False.
    When device='cuda' is passed (or default), init_cuda becomes True.
    """
    print("\n" + "="*70)
    print("Test: Conditional cuda.init() logic")
    print("="*70)

    # Version that ALWAYS calls cuda.init
    code_always_init = '''
def set_seed_always_init(seed: int = 42):
    # ALWAYS init CUDA, regardless of device param
    if torch.cuda.is_available():
        torch.cuda.init()

    torch.pi = torch.acos(torch.zeros(1)).item() * 2

    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

set_seed_always_init(42)
'''

    # Version that NEVER calls cuda.init
    code_never_init = '''
def set_seed_never_init(seed: int = 42):
    # NEVER init CUDA explicitly

    torch.pi = torch.acos(torch.zeros(1)).item() * 2

    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

set_seed_never_init(42)
'''

    result_always = run_isolated_test(code_always_init, "Always cuda.init")
    result_never = run_isolated_test(code_never_init, "Never cuda.init")

    # Original with default (which does cuda.init)
    code_original = "set_reproducible_seed(42)"
    result_original = run_isolated_test(code_original, "Original default")

    print(f"\nAlways init:   {format_sequence(result_always['data'])}")
    print(f"Never init:    {format_sequence(result_never['data'])}")
    print(f"Original:      {format_sequence(result_original['data'])}")

    always_never = compare_results(result_always, result_never)
    always_orig = compare_results(result_always, result_original)

    print(f"\nAlways == Never: {always_never}")
    print(f"Always == Original: {always_orig}")

    if not always_never:
        print("\n>>> FINDING: cuda.init() call DOES affect random state!")
    else:
        print("\n>>> cuda.init() does not affect random state")

    return result_always, result_never, result_original


# ============================================================================
# Test: Direct comparison with actual test_utils.py behavior
# ============================================================================

def test_actual_import_behavior():
    """
    Test the actual behavior when importing from test_utils.py vs calling directly.

    This replicates what was observed: run_single_mode.py vs inline tests.
    """
    print("\n" + "="*70)
    print("Test: Import behavior simulation")
    print("="*70)

    # Simulate run_single_mode.py: imports and uses device param
    code_with_import = '''
import sys
sys.path.insert(0, '/home/idv-eqs8-pza/IDV_code/ClosedLoopProject/gaussian_processes/Spatial_GP_repo/scripts/gpytorch_porting')
from tests.test_utils import set_reproducible_seed

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
set_reproducible_seed(42, device=device)
'''

    # Simulate inline test: just call without device param
    code_inline = '''
import sys
sys.path.insert(0, '/home/idv-eqs8-pza/IDV_code/ClosedLoopProject/gaussian_processes/Spatial_GP_repo/scripts/gpytorch_porting')
from tests.test_utils import set_reproducible_seed

set_reproducible_seed(42)
'''

    result_with_import = run_isolated_test(code_with_import, "With device param (like run_single_mode)")
    result_inline = run_isolated_test(code_inline, "Without device param")

    print(f"\nWith device param:    {format_sequence(result_with_import['data'])}")
    print(f"Without device param: {format_sequence(result_inline['data'])}")

    same = compare_results(result_with_import, result_inline)
    print(f"\nSame? {same}")

    if not same:
        print("\n>>> CONFIRMED: Device param creates different random sequence!")
        print("    This explains run_single_mode.py vs inline test differences")

    return result_with_import, result_inline


# ============================================================================
# Main
# ============================================================================

def main():
    print("="*70)
    print("Investigation A2: Device Parameter Random Sequence Issue")
    print("="*70)
    print("\nRunning tests in isolated subprocesses for clean state...")

    test_calling_patterns()
    test_isinstance_check()
    test_parameter_mechanics()
    test_cuda_init_conditional()
    test_actual_import_behavior()

    print("\n" + "="*70)
    print("Investigation complete. Check findings above.")
    print("="*70)


if __name__ == '__main__':
    main()
