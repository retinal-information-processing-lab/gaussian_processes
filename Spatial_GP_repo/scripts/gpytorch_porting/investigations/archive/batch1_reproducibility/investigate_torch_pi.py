#!/usr/bin/env python3
"""
Investigation A1: The torch.pi Mystery

Goal: Understand why `torch.pi = torch.acos(torch.zeros(1)).item() * 2`
affects PyTorch's random state.

IMPORTANT: This script must be run in a FRESH Python process for each hypothesis.
The subprocess module ensures isolation between tests.

Usage:
    python investigate_torch_pi.py          # Run all tests
    python investigate_torch_pi.py --test 1  # Run specific hypothesis test
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

# === TEST CODE ===
{code_snippet}

# === GENERATE RANDOM SEQUENCE ===
torch.manual_seed(42)
np.random.seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)
    torch.cuda.manual_seed_all(42)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Generate sequence
randn_vals = torch.randn(5, device=device).tolist()
randperm_vals = torch.randperm(100, device=device)[:5].tolist()
numpy_vals = np.random.rand(5).tolist()

# Output as JSON for parsing
import json
result = {{
    'randn': randn_vals,
    'randperm': randperm_vals,
    'numpy': numpy_vals,
    'torch_pi_value': getattr(torch, 'pi', None),
    'device': device
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

    # Compare randn values (allow small floating point differences)
    randn_match = all(abs(a - b) < 1e-10 for a, b in zip(d1['randn'], d2['randn']))
    randperm_match = d1['randperm'] == d2['randperm']
    numpy_match = all(abs(a - b) < 1e-10 for a, b in zip(d1['numpy'], d2['numpy']))

    return randn_match and randperm_match and numpy_match


def format_sequence(data: dict) -> str:
    """Format a random sequence for display."""
    randn = [f'{v:.6f}' for v in data['randn'][:3]]
    return f"randn={randn}, randperm={data['randperm'][:3]}"


# ============================================================================
# Hypothesis Tests
# ============================================================================

def test_hypothesis_1_1():
    """
    H1.1: Does torch.acos(torch.zeros(1)) trigger lazy PyTorch initialization?

    Test: Compare random sequences with and without the acos computation.
    """
    print("\n" + "="*70)
    print("H1.1: Does torch.acos(torch.zeros(1)) trigger lazy init?")
    print("="*70)

    # Variant A: No computation before seeding
    code_a = "# No computation"
    result_a = run_isolated_test(code_a, "No computation")

    # Variant B: Run the full torch.pi line
    code_b = "torch.pi = torch.acos(torch.zeros(1)).item() * 2"
    result_b = run_isolated_test(code_b, "Full torch.pi line")

    # Variant C: Just the acos computation (no assignment)
    code_c = "_ = torch.acos(torch.zeros(1)).item() * 2"
    result_c = run_isolated_test(code_c, "Acos computation only")

    print(f"\nVariant A (no computation):    {format_sequence(result_a['data'])}")
    print(f"Variant B (full torch.pi):     {format_sequence(result_b['data'])}")
    print(f"Variant C (acos, no assign):   {format_sequence(result_c['data'])}")

    same_a_b = compare_results(result_a, result_b)
    same_b_c = compare_results(result_b, result_c)
    same_a_c = compare_results(result_a, result_c)

    print(f"\nA == B: {same_a_b}")
    print(f"B == C: {same_b_c}")
    print(f"A == C: {same_a_c}")

    if same_b_c and not same_a_b:
        print("\n>>> FINDING: The acos computation (not the assignment) changes random state")
    elif not same_a_b and not same_b_c:
        print("\n>>> FINDING: Both the computation AND assignment matter")
    elif same_a_b and same_b_c:
        print("\n>>> FINDING: H1.1 REJECTED - no difference from torch.pi line")
    else:
        print("\n>>> FINDING: Unexpected pattern - needs more investigation")

    return result_a, result_b, result_c


def test_hypothesis_1_2():
    """
    H1.2: Does .item() conversion (tensor to Python float) matter?

    Test: Compare with and without .item() call.
    """
    print("\n" + "="*70)
    print("H1.2: Does .item() conversion matter?")
    print("="*70)

    # Variant A: Full line with .item()
    code_a = "torch.pi = torch.acos(torch.zeros(1)).item() * 2"
    result_a = run_isolated_test(code_a, "With .item()")

    # Variant B: Without .item() - keep as tensor
    code_b = "_ = torch.acos(torch.zeros(1)) * 2  # No .item()"
    result_b = run_isolated_test(code_b, "Without .item()")

    # Variant C: Using float() instead of .item()
    code_c = "torch.pi = float(torch.acos(torch.zeros(1))) * 2"
    result_c = run_isolated_test(code_c, "Using float()")

    print(f"\nVariant A (with .item()):      {format_sequence(result_a['data'])}")
    print(f"Variant B (no .item()):        {format_sequence(result_b['data'])}")
    print(f"Variant C (with float()):      {format_sequence(result_c['data'])}")

    same_a_b = compare_results(result_a, result_b)
    same_a_c = compare_results(result_a, result_c)

    print(f"\nA == B: {same_a_b}")
    print(f"A == C: {same_a_c}")

    if not same_a_b:
        print("\n>>> FINDING: .item() conversion DOES affect random state")
    else:
        print("\n>>> FINDING: .item() conversion does NOT matter")

    return result_a, result_b, result_c


def test_hypothesis_1_3():
    """
    H1.3: Does assignment to torch.pi specifically matter?

    Test: Compare assigning to torch.pi vs dummy variable vs no assignment.
    """
    print("\n" + "="*70)
    print("H1.3: Does assignment to torch.pi specifically matter?")
    print("="*70)

    # Variant A: Assign to torch.pi
    code_a = "torch.pi = torch.acos(torch.zeros(1)).item() * 2"
    result_a = run_isolated_test(code_a, "Assign to torch.pi")

    # Variant B: Assign to dummy variable
    code_b = "dummy = torch.acos(torch.zeros(1)).item() * 2"
    result_b = run_isolated_test(code_b, "Assign to dummy")

    # Variant C: No assignment at all
    code_c = "_ = torch.acos(torch.zeros(1)).item() * 2"
    result_c = run_isolated_test(code_c, "Discard result")

    print(f"\nVariant A (torch.pi =):        {format_sequence(result_a['data'])}")
    print(f"Variant B (dummy =):           {format_sequence(result_b['data'])}")
    print(f"Variant C (_ =):               {format_sequence(result_c['data'])}")

    same_a_b = compare_results(result_a, result_b)
    same_b_c = compare_results(result_b, result_c)

    print(f"\nA == B: {same_a_b}")
    print(f"B == C: {same_b_c}")

    if not same_a_b and same_b_c:
        print("\n>>> FINDING: Assignment to torch.pi specifically matters!")
    elif same_a_b:
        print("\n>>> FINDING: The assignment target does NOT matter")

    return result_a, result_b, result_c


def test_hypothesis_1_4():
    """
    H1.4: Is it the torch.zeros(1) tensor creation that matters?

    Test: Compare different tensor creation methods.
    """
    print("\n" + "="*70)
    print("H1.4: Is it the tensor creation that matters?")
    print("="*70)

    # Variant A: No tensor creation
    code_a = "# Nothing"
    result_a = run_isolated_test(code_a, "No tensor creation")

    # Variant B: Full torch.pi line
    code_b = "torch.pi = torch.acos(torch.zeros(1)).item() * 2"
    result_b = run_isolated_test(code_b, "Full line")

    # Variant C: Just create torch.zeros(1)
    code_c = "_ = torch.zeros(1)"
    result_c = run_isolated_test(code_c, "Just torch.zeros(1)")

    # Variant D: Create empty tensor
    code_d = "_ = torch.empty(1)"
    result_d = run_isolated_test(code_d, "Just torch.empty(1)")

    # Variant E: Create tensor([0.0])
    code_e = "_ = torch.tensor([0.0])"
    result_e = run_isolated_test(code_e, "torch.tensor([0.0])")

    print(f"\nVariant A (nothing):           {format_sequence(result_a['data'])}")
    print(f"Variant B (full line):         {format_sequence(result_b['data'])}")
    print(f"Variant C (zeros(1)):          {format_sequence(result_c['data'])}")
    print(f"Variant D (empty(1)):          {format_sequence(result_d['data'])}")
    print(f"Variant E (tensor([0.0])):     {format_sequence(result_e['data'])}")

    # Check which variants match the "nothing" baseline
    a_c = compare_results(result_a, result_c)
    a_d = compare_results(result_a, result_d)
    a_e = compare_results(result_a, result_e)
    b_c = compare_results(result_b, result_c)

    print(f"\nA == C (nothing vs zeros):     {a_c}")
    print(f"A == D (nothing vs empty):     {a_d}")
    print(f"A == E (nothing vs tensor):    {a_e}")
    print(f"B == C (full vs zeros):        {b_c}")

    if not a_c:
        print("\n>>> FINDING: torch.zeros(1) alone changes random state!")
    elif not a_d:
        print("\n>>> FINDING: torch.empty(1) changes random state!")
    elif not a_e:
        print("\n>>> FINDING: torch.tensor([0.0]) changes random state!")
    else:
        print("\n>>> FINDING: Tensor creation alone does NOT change random state")

    return result_a, result_b, result_c, result_d, result_e


def test_hypothesis_1_5():
    """
    H1.5: Does the order relative to cuda.init() matter?

    Test: Compare torch.pi before and after CUDA initialization.
    """
    print("\n" + "="*70)
    print("H1.5: Does order relative to cuda.init() matter?")
    print("="*70)

    # Variant A: torch.pi BEFORE cuda.init
    code_a = """
torch.pi = torch.acos(torch.zeros(1)).item() * 2
if torch.cuda.is_available():
    torch.cuda.init()
"""
    result_a = run_isolated_test(code_a, "torch.pi BEFORE cuda.init")

    # Variant B: torch.pi AFTER cuda.init
    code_b = """
if torch.cuda.is_available():
    torch.cuda.init()
torch.pi = torch.acos(torch.zeros(1)).item() * 2
"""
    result_b = run_isolated_test(code_b, "torch.pi AFTER cuda.init")

    # Variant C: No torch.pi, with cuda.init
    code_c = """
if torch.cuda.is_available():
    torch.cuda.init()
"""
    result_c = run_isolated_test(code_c, "No torch.pi, with cuda.init")

    # Variant D: No cuda.init, with torch.pi
    code_d = """
torch.pi = torch.acos(torch.zeros(1)).item() * 2
"""
    result_d = run_isolated_test(code_d, "torch.pi only")

    print(f"\nVariant A (pi BEFORE init):    {format_sequence(result_a['data'])}")
    print(f"Variant B (pi AFTER init):     {format_sequence(result_b['data'])}")
    print(f"Variant C (init, no pi):       {format_sequence(result_c['data'])}")
    print(f"Variant D (pi, no init):       {format_sequence(result_d['data'])}")

    a_b = compare_results(result_a, result_b)
    a_c = compare_results(result_a, result_c)
    b_c = compare_results(result_b, result_c)

    print(f"\nA == B (before vs after):      {a_b}")
    print(f"A == C (pi+init vs init):      {a_c}")
    print(f"B == C (pi+init vs init):      {b_c}")

    if not a_b:
        print("\n>>> FINDING: Order relative to cuda.init() MATTERS!")
    else:
        print("\n>>> FINDING: Order relative to cuda.init() does not matter")

    return result_a, result_b, result_c, result_d


def test_hypothesis_1_6():
    """
    H1.6: Check PyTorch version and built-in torch.pi behavior.

    Test: What is torch.pi's value before and after the assignment?
    """
    print("\n" + "="*70)
    print("H1.6: Check torch.pi value and PyTorch version")
    print("="*70)

    code = """
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")

# Check built-in torch.pi
builtin_pi = torch.pi
print(f"Built-in torch.pi: {builtin_pi}")
print(f"Built-in torch.pi type: {type(builtin_pi)}")

# Compute the value
computed_pi = torch.acos(torch.zeros(1)).item() * 2
print(f"Computed pi: {computed_pi}")
print(f"Computed pi type: {type(computed_pi)}")

# Are they equal?
print(f"Equal? {builtin_pi == computed_pi}")
print(f"Difference: {abs(builtin_pi - computed_pi)}")

# Check if assignment changes anything observable
torch.pi = computed_pi
print(f"After assignment torch.pi: {torch.pi}")
print(f"After assignment type: {type(torch.pi)}")
"""

    result = subprocess.run(
        [sys.executable, '-c', code],
        capture_output=True,
        text=True
    )

    print(result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr)


# ============================================================================
# Main
# ============================================================================

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Investigate torch.pi mystery')
    parser.add_argument('--test', type=int, help='Run specific hypothesis (1-6)')
    args = parser.parse_args()

    print("="*70)
    print("Investigation A1: The torch.pi Mystery")
    print("="*70)
    print("\nRunning tests in isolated subprocesses for clean state...")

    if args.test:
        tests = {
            1: test_hypothesis_1_1,
            2: test_hypothesis_1_2,
            3: test_hypothesis_1_3,
            4: test_hypothesis_1_4,
            5: test_hypothesis_1_5,
            6: test_hypothesis_1_6,
        }
        if args.test in tests:
            tests[args.test]()
        else:
            print(f"Unknown test: {args.test}. Valid tests: 1-6")
    else:
        # Run all tests
        test_hypothesis_1_6()  # Version info first
        test_hypothesis_1_1()
        test_hypothesis_1_2()
        test_hypothesis_1_3()
        test_hypothesis_1_4()
        test_hypothesis_1_5()

    print("\n" + "="*70)
    print("Investigation complete. Check findings above.")
    print("="*70)


if __name__ == '__main__':
    main()
