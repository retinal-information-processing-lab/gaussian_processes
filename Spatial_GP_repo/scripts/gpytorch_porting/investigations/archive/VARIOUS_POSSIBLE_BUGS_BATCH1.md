# Batch 1 Investigation: Reproducibility Issues - WRAP-UP

**Investigation Date**: 2026-01-23
**Branch**: `pietro/investigate-reproducibility` (from `pietro/workingbranch`)
**Status**: INVESTIGATION COMPLETE - DECISION NEEDED
**PyTorch Version Tested**: 2.5.1+cu121

---

## What Was Investigated

Batch 1 from `VARIOUS_POSSIBLE_BUGS.md` contained two high-priority reproducibility issues:

| Issue | Description | Priority |
|-------|-------------|----------|
| **A1** | The `torch.pi` mystery - why does `torch.pi = torch.acos(torch.zeros(1)).item() * 2` affect random state? | HIGH |
| **A2** | Device parameter issue - why does `set_reproducible_seed(42, device='cuda')` differ from `set_reproducible_seed(42)`? | HIGH |

Both issues were documented as "cargo cult programming" - code that works but developers don't understand why.

---

## Key Finding

**NEITHER ISSUE COULD BE REPRODUCED** in PyTorch 2.5.1+cu121.

All 7 test scripts produced **identical random sequences** (checksum: 3.9199287802):

| Test | What It Tests | Result |
|------|---------------|--------|
| A | Baseline (nothing special) | Same |
| B | With `torch.pi = torch.acos(torch.zeros(1)).item() * 2` | Same |
| C | Just `torch.zeros(1)` tensor creation | Same |
| D | With explicit `cuda.init()` | Same |
| E | `set_reproducible_seed(42)` - no device param | Same |
| F | `set_reproducible_seed(42, device='cuda')` - with device param | Same |
| G | Import `utils.py` before seeding (has all side effects) | Same |

---

## Interpretation

### Possible Explanations

1. **Fixed in PyTorch 2.5.1**: The issues may have been caused by subtle PyTorch bugs that have since been fixed.

2. **Edge Case Not Reproduced**: The original failure involved training collapse at iteration 30-40, not just different random numbers. The issue might be:
   - Floating point accumulation over many iterations
   - GPU kernel caching differences
   - Specific data patterns triggering instability

3. **The Workaround Was Never Needed**: The `torch.pi` line may have been a coincidental correlation, not the actual fix.

### What We Know For Sure

- In fresh Python processes with PyTorch 2.5.1, neither the `torch.pi` line nor the device parameter affects random state
- The workaround code in `tests/test_utils.py` does not harm anything (it just sets values that are already the defaults)

---

## Decision Needed

**Three options for how to proceed:**

### Option 1: Keep Workaround (Conservative)
- Leave `tests/test_utils.py` as-is
- Keep the `torch.pi` line and `cuda.init()` logic
- **Pros**: No risk of breaking anything
- **Cons**: Keeps confusing "cargo cult" code with scary comments

### Option 2: Remove Workaround (Clean)
- Simplify `set_reproducible_seed()` to just set seeds
- Remove the mystery `torch.pi` line
- **Pros**: Clean, understandable code
- **Cons**: Small risk if edge case exists that we didn't find

### Option 3: Keep But Update Docs (Compromise)
- Keep the code as-is
- Update comments to note: "Could not reproduce in PyTorch 2.5.1 (2026-01-23), may be unnecessary"
- **Pros**: Safe, but with better documentation
- **Cons**: Still has confusing code

---

## Files Created (For Cleanup)

All investigation files are in `investigations/batch1_reproducibility/`:

### Test Scripts (keep for regression testing or delete)
```
investigations/batch1_reproducibility/
├── test_A_baseline.py           # Baseline - no special setup
├── test_B_with_torch_pi.py      # With the mystery torch.pi line
├── test_C_just_zeros.py         # Just torch.zeros(1) tensor creation
├── test_D_with_cuda_init.py     # With explicit cuda.init()
├── test_E_seed_no_device.py     # set_reproducible_seed(42) - no device
├── test_F_seed_with_device.py   # set_reproducible_seed(42, device='cuda')
├── test_G_import_utils.py       # Import utils.py before seeding
└── run_all_tests.sh             # Shell script runner (has line ending issues)
```

### Documentation
```
├── README.md                    # Investigation overview
└── FINDINGS.md                  # Detailed findings
```

### Older Scripts (can delete - use subprocess which may skew results)
```
├── investigate_torch_pi.py      # Original A1 investigation (subprocess-based)
└── investigate_device_param.py  # Original A2 investigation (subprocess-based)
```

---

## How to Continue This Investigation

### If You Want to Verify Further

Run the original failing test to see if it still works:
```bash
cd /home/idv-eqs8-pza/IDV_code/ClosedLoopProject/gaussian_processes/Spatial_GP_repo/scripts/gpytorch_porting
python tests/test_kernel_cache.py
```

Run multiple seeds to check consistency:
```bash
python run_single_mode.py --mode vargp_style --explicit-unwhitening --seed 42
python run_single_mode.py --mode vargp_style --explicit-unwhitening --seed 123
python run_single_mode.py --mode vargp_style --explicit-unwhitening --seed 456
```

### If You Choose Option 2 (Remove Workaround)

Simplified `tests/test_utils.py:set_reproducible_seed()`:
```python
def set_reproducible_seed(seed: int = 42):
    """Set random seed for reproducibility.

    Note: Previous versions had cuda.init() and torch.pi workarounds.
    Testing on 2026-01-23 (PyTorch 2.5.1) showed these were unnecessary.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
```

After making this change:
1. Run `python tests/test_kernel_cache.py` - should pass
2. Run `python run_single_mode.py --mode vargp_style --explicit-unwhitening` - should get test_r ~0.77
3. Run the canonical test suite

### If You Choose Option 3 (Update Docs)

Update `.claude/CLAUDE.md` section on "Known limitations" to say:
```
**HACKY WORKAROUND - Random State Reproducibility (January 2025):**
> NOTE (2026-01-23): Investigation could not reproduce this issue in PyTorch 2.5.1.
> The workaround is kept for safety but may be unnecessary.
> See investigations/VARIOUS_POSSIBLE_BUGS_BATCH1.md for details.
```

---

## Cleanup Commands

When done with investigation, to clean up:

```bash
# Option A: Delete all investigation files
rm -rf investigations/batch1_reproducibility/

# Option B: Keep only the findings document
cd investigations/batch1_reproducibility/
rm test_*.py investigate_*.py run_all_tests.sh README.md
# Keep FINDINGS.md
```

To merge back to main branch (if changes made):
```bash
git checkout pietro/workingbranch
git merge pietro/investigate-reproducibility
git branch -d pietro/investigate-reproducibility
```

---

## Related Documentation

- Original bug catalog: `investigations/VARIOUS_POSSIBLE_BUGS.md` (Issues A1, A2)
- Archive with original discovery: `.claude/archive/ARCHIVE_2026-01-18_kernel_caching_and_whitening.md` (Section 17)
- Current workaround location: `tests/test_utils.py:set_reproducible_seed()` (lines 20-76)
- Original torch.pi line: `utils.py:51`

---

## Summary for Quick Reference

| Question | Answer |
|----------|--------|
| What was investigated? | A1 (torch.pi mystery), A2 (device param issue) |
| Could issues be reproduced? | **NO** |
| PyTorch version? | 2.5.1+cu121 |
| Recommendation? | Option 3 (keep code, update docs) is safest |
| What needs decision? | Whether to keep/remove the workaround code |
| Branch with investigation? | `pietro/investigate-reproducibility` |
| Files to clean up? | 12 files in `investigations/batch1_reproducibility/` |
