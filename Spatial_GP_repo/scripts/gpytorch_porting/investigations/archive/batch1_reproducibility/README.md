# Batch 1 Investigation: Reproducibility Issues

**Created**: 2026-01-23
**Issues**: A1 (torch.pi mystery), A2 (device parameter issue)
**Status**: In Progress

---

## Overview

This investigation examines two mysterious reproducibility issues in the GPyTorch porting codebase:

1. **A1 - torch.pi Mystery**: Why does `torch.pi = torch.acos(torch.zeros(1)).item() * 2` affect PyTorch's random state?

2. **A2 - Device Parameter Issue**: Why does `set_reproducible_seed(42, device='cuda')` produce different random sequences than `set_reproducible_seed(42)`?

## Files in This Investigation

| File | Purpose |
|------|---------|
| `README.md` | This overview |
| `investigate_torch_pi.py` | Test hypotheses for A1 |
| `investigate_device_param.py` | Test hypotheses for A2 |
| `minimal_reproduction.py` | Simplest reproduction case |
| `FINDINGS.md` | Results and conclusions (created after running tests) |

## How to Run

```bash
# Activate the correct environment first
conda activate pytorch_gpytorch

# Run each investigation script
cd /home/idv-eqs8-pza/IDV_code/ClosedLoopProject/gaussian_processes/Spatial_GP_repo/scripts/gpytorch_porting

python investigations/batch1_reproducibility/investigate_torch_pi.py
python investigations/batch1_reproducibility/investigate_device_param.py
python investigations/batch1_reproducibility/minimal_reproduction.py
```

## Background

The current `set_reproducible_seed()` function in `tests/test_utils.py` contains this line:

```python
torch.pi = torch.acos(torch.zeros(1)).item() * 2  # WHY DOES THIS MATTER?!
```

This line:
- Computes pi via arccos(0) * 2 = 3.141592...
- Assigns it to `torch.pi`, which is already a built-in constant since PyTorch 1.8
- The assignment SHOULD be a no-op, but somehow affects random state

Without this line, some tests fail with different random sequences.

## Hypotheses Being Tested

### For A1 (torch.pi):
- H1.1: Does `torch.acos(torch.zeros(1))` trigger lazy PyTorch initialization?
- H1.2: Does `.item()` (tensor-to-Python conversion) matter?
- H1.3: Does the assignment to `torch.pi` specifically matter?
- H1.4: Is it just the `torch.zeros(1)` tensor creation that matters?
- H1.5: Does the order relative to `cuda.init()` matter?

### For A2 (device param):
- H2.1: Does the `isinstance()` check trigger something?
- H2.2: String `'cuda'` vs `torch.device('cuda')` difference?
- H2.3: Default param vs explicit param evaluation?
- H2.4: Is it the conditional CUDA init logic?

## References

- Original torch.pi line: `utils.py:51`
- Current workaround: `tests/test_utils.py:68`
- Bug catalog: `investigations/VARIOUS_POSSIBLE_BUGS.md` (A1, A2)
- Archive analysis: `.claude/archive/ARCHIVE_2026-01-18_kernel_caching_and_whitening.md` (Section 17)
