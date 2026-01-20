# Handoff: Add UnwhitenedVariationalStrategy as Alternative

**Date**: 2026-01-20
**Status**: PLAN APPROVED, IMPLEMENTATION NOT STARTED
**Next Step**: Execute the plan

---

## PROMPT FOR NEXT SESSION

Copy this entire prompt to start the next session:

---

**CONTEXT**: We investigated the L_K whitening inconsistency problem in GPyTorch and determined that `UnwhitenedVariationalStrategy` is the solution for EM-style optimization. However, we want to ADD it as an ALTERNATIVE option, NOT replace the existing whitening code.

**READ THESE FILES FIRST** (in this order):
1. `.claude/WHITENING_INVESTIGATION_2026-01-20.md` - Full investigation of the whitening problem (comprehensive, ~600 lines)
2. `.claude/plans/mellow-chasing-wand.md` - The APPROVED implementation plan
3. `.claude/DECISION_LOG.md` - See Q26-Q27 for the decision rationale

**THE PROBLEM** (summary):
- GPyTorch's `VariationalStrategy` stores variational parameters in whitened form: `m_whitened = L_K⁻¹ @ m_natural`
- Our Newton E-step produces natural (m, V), we convert to whitened for storage
- When M-step changes kernel, L_K changes, but stored whitened params were computed with OLD L_K
- This causes ~8x corruption in predictive mean
- Solution: `UnwhitenedVariationalStrategy` stores natural params directly, no L_K involvement

**THE TASK**: Execute the approved plan to add `whitening` parameter to model.py that allows choosing between strategies:
- `whitening=True` (default): Use `VariationalStrategy` (current behavior, preserved)
- `whitening=False`: Use `UnwhitenedVariationalStrategy` (new option)

**FILES TO MODIFY**:

1. **`model.py`** (lines 11, 41, 48-56):
   - Add import: `UnwhitenedVariationalStrategy`
   - Add parameter: `whitening=True`
   - Add conditional: choose strategy based on parameter
   - Store: `self.whitening = whitening`

2. **`estep.py`** (lines ~77-132, ~748-942, ~1456-1605):
   - `compute_kernel_cache()`: Make L_K computation conditional on `model.whitening`
   - `e_step_loop()`: Change `use_whitening` default to `None`, auto-detect from `model.whitening`
   - `train_varGP_style()`: Same auto-detect pattern

3. **`test_estep_pnas.py`**:
   - Add `--unwhitened` flag
   - Pass `whitening=not args.unwhitened` to model creation

4. **`tests/test_estep_comparison.py`**:
   - Add `--unwhitened` flag
   - Add comparison mode

**VERIFICATION COMMANDS**:
```bash
# Existing behavior unchanged
conda run -n pytorch_gpytorch python test_estep_pnas.py --mode vargp_style --ntilde 50 --save-plot none

# New unwhitened path
conda run -n pytorch_gpytorch python test_estep_pnas.py --mode vargp_style --ntilde 50 --unwhitened --save-plot none
```

**CRITICAL CONSTRAINTS**:
- DO NOT delete any existing code
- Default behavior MUST remain unchanged (`whitening=True`)
- All existing tests MUST still pass

**KEY INSIGHT FROM INVESTIGATION**:
GPyTorch's `UnwhitenedVariationalStrategy` has identical API to `VariationalStrategy` (verified: both accept `jitter_val` parameter). The only difference is how it interprets stored variational parameters.

---

## Files Reference

| File | Purpose | Read Priority |
|------|---------|---------------|
| `.claude/WHITENING_INVESTIGATION_2026-01-20.md` | Full investigation with math, sources, contradictions examined | HIGH |
| `.claude/plans/mellow-chasing-wand.md` | Approved implementation plan | HIGH |
| `.claude/DECISION_LOG.md` | Q26-Q27 explain the decision | MEDIUM |
| `.claude/CLAUDE.md` | Project context and quick start | MEDIUM |
| `model.py` | Where VariationalStrategy is created | HIGH (modify) |
| `estep.py` | Where whitening conversions happen | HIGH (modify) |
| `test_estep_pnas.py` | Main test script | MEDIUM (modify) |
| `ARCHIVE_2026-01-18_kernel_caching_and_whitening.md` | Previous session's whitening analysis (historical) | LOW |

---

## What Was Accomplished This Session

1. Investigated the L_K inconsistency question thoroughly using 4 parallel subagents
2. Found that GPyTorch does NOT auto-adjust whitened params when kernel changes (by design)
3. Found that `UnwhitenedVariationalStrategy` is the clean solution for EM
4. Documented findings in `WHITENING_INVESTIGATION_2026-01-20.md` (comprehensive)
5. Added "Potential Contradictions Examined" section (Section 8b) to address subtle points
6. Updated `DECISION_LOG.md` with Q26-Q27
7. Updated `SESSION_LOG.md` with this session's entry
8. Created and got approval for implementation plan
9. Implementation NOT YET STARTED (context limit reached)

---

## Git Status at Session End

The following files were created/modified this session (documentation only, no code changes):
- `.claude/WHITENING_INVESTIGATION_2026-01-20.md` (NEW - comprehensive investigation)
- `.claude/DECISION_LOG.md` (MODIFIED - added Q26-Q27)
- `.claude/SESSION_LOG.md` (MODIFIED - added session entry)
- `.claude/plans/mellow-chasing-wand.md` (NEW - approved plan)
- `.claude/HANDOFF_2026-01-20_UNWHITENED.md` (NEW - this file)

Consider committing these before starting implementation.
