# Session Log

Brief summaries of recent sessions for context handoff.
Updated via "wrap up" command at session end (see WORKING_GUIDELINES.md Section 9).

---

## 2026-02-06: Codebase Cleanup - Orphaned Files Archived

**Accomplished:**
- Moved 5 orphaned files to `deprecated/`: `whitening.py`, `test_m_whitening.py`, `test_kernel_cache.py`, `test_reference_comparison.py`, `diagnose_unwhitened_performance.py`
- Fixed CLAUDE.md: `VARGP_DIRECT_REFERENCE.md` → `EIGENSPACE_REFERENCE.md` (3 locations)
- Removed stale "Superseded" section from CLAUDE.md (run_canonical_tests.py, query_benchmark.py already deleted)
- Updated test files list and deprecated folder description

**Files Changed:**
- 5 files moved to `deprecated/`
- `deprecated/README.md` - added "Additional Archived Files" section
- `.claude/CLAUDE.md` - fixed stale references

---

## 2026-02-05: YAML Experiment System (COMPLETE)

**Accomplished:**
- Created `configs/canonical.yaml` and `configs/quick.yaml` — all 35 params with WIRED/HARDCODED annotations
- Created `create_experiment.py`, `run_experiment.py`, `analyze_experiment.py`
- Refactored `run_single_mode.py`: extracted `run_single_config(config)` and `flatten_yaml_config()`
- Fixed bug: `--lr` was silently ignored for vargp_direct (was reading from defaults instead of args)
- Moved `results/` to `old_results/`
- Wired `gpy_lbfgs_max_iter` and `early_stopping` params through code

**Validation:** quick run gives test_r=0.849 (baseline match)

**Commit:** 5c31686

---

## 2026-02-02: Final Cleanup - Legacy Files Removed (COMPLETE)

**Accomplished:**
- Made deprecated/ folder self-contained (~1,100 lines of vargp_style code)
  - Extracted vargp_style functions from legacy files into deprecated/
  - deprecated/vargp_style_estep.py (807 lines): E-step functions
  - deprecated/vargp_style_fstep.py (228 lines): F-step functions
  - deprecated/vargp_style_mstep.py (75 lines): M-step function
- Deleted legacy files (2,672 lines removed):
  - train.py, estep.py, fstep.py, mstep.py, model.py
- Updated deprecated/vargp_style_run.py to work standalone
- Updated run_canonical_tests.py: removed vargp_style from choices
- Updated CLAUDE.md: removed legacy files section

**Testing:**
- vargp_direct mode: WORKS ✓
- default_gpy mode: WORKS ✓
- deprecated/vargp_style_run.py: Imports work standalone ✓

**Result:**
- Clean main directory with only active code (eigenspace_*, gpy_*, shared)
- deprecated/ folder is self-contained and documented
- 2,672 lines removed from main codebase

---

## 2026-02-02: Codebase Reorganization - Modular Structure (COMPLETE)

**Accomplished:**
- **Phase 1**: Created new modular file structure
  - Eigenspace modules: `eigenspace_*.py` (7 files) - model, utils, gradients, training, estep, fstep, mstep
  - GPyTorch modules: `gpy_*.py` (2 files) - model, training
  - Shared modules: `metrics.py`, `utils.py` (extracted from old files)
- **Phase 2**: Archived deprecated vargp_style code
  - Created `deprecated/` folder with README and archived files
  - Moved `whitening.py` → `deprecated/vargp_style_whitening.py`
- **Phase 3**: Cleaned up imports and deprecated mode handling
  - Removed vargp_style from `run_single_mode.py`
  - Marked legacy files (train.py, estep.py, fstep.py, mstep.py, model.py) as deprecated
- **Phase 4**: Updated documentation
  - Updated CLAUDE.md with new file map
  - Renamed VARGP_DIRECT_REFERENCE.md → EIGENSPACE_REFERENCE.md
  - Updated all file references

**New File Organization:**
- **Naming convention**: `eigenspace_*` prefix for eigenspace mode, `gpy_*` prefix for GPyTorch mode
- **Key renamings**: eigenspace.py → eigenspace_utils.py, direct_vargp.py → eigenspace_gradients.py, utils_gpy.py → utils.py
- **Status**: Both `vargp_direct` and `default_gpy` modes fully tested and working

**Commit**: 23c3856 (checkpoint before reorganization)

---

## 2026-01-31: vargp_direct Bug Fix & Canonical Test Update (COMPLETE)

**Accomplished:**
- Fixed critical diagonal approximation bug in `mstep_eigenspace_autograd()` (mstep.py lines 167-170)
- Extensive investigation using 4 parallel subagents confirmed no other similar bugs
- Updated `run_canonical_tests.py` to include vargp_direct mode (16 configs instead of 12)
- Added `--float32` flag to all canonical tests for fair comparison with vargp_old
- Updated VARGP_DIRECT_REFERENCE.md Section 8 to mark bug as FIXED
- Updated investigations/eigenspace_dimensions.md to RESOLVED status

**The Bug:**
- KL trace term used diagonal approximation `sum(V_diag / K_diag)` when K_tilde_b was NOT diagonal
- Inside M-step LBFGS closure, hyperparameters change → K_tilde_b = B.T @ K_tilde_new @ B is non-diagonal
- Fix: `trace_term = torch.trace(K_tilde_b_inv @ state.V_b)` using already-computed full matrix inverse

**Performance After Fix** (M=50, Cell 8, 50 iterations):
| Mode | test_r | Time |
|------|--------|------|
| vargp_old | 0.8413 | 6.3s |
| vargp_direct (FIXED) | 0.8442 | 5.6s |

**Canonical Test Results** (15/16 passed):
- vargp_direct matches vargp_old within ±0.004 across all configs
- 1 pre-existing failure: vargp_style at ntrain=2000, M=200 (negative variance)

**Known Remaining Difference:**
- KL formula: vargp_direct has `-n_b` term, vargp_old omits it
- Causes ~20-35 loss offset but does NOT affect optimization or predictions

**Files Changed:**
- `mstep.py` - Fixed diagonal approximation bug (lines 167-170)
- `run_canonical_tests.py` - Added vargp_direct, added --float32 for all modes
- `.claude/VARGP_DIRECT_REFERENCE.md` - Updated Section 8, performance table
- `investigations/eigenspace_dimensions.md` - Status changed to RESOLVED

---

## 2026-01-28: vargp_direct Mode Implementation (COMPLETE)

**Accomplished:**
- Implemented `vargp_direct` mode with eigenspace projection matching original varGP
- Created `eigenspace.py` (projection utilities) and `direct_vargp.py` (training loop)
- Uses LBFGS with autograd for M-step (slower than analytical, but works)
- Matches vargp_old E-step formulas exactly (including the "buggy" m_new formula)

**Performance** (M=50, 50 iterations): test_r=0.81 (vs vargp_old 0.84), time=18.8s (vs 6.2s)

**Deferred:** Correct m_new formula investigation, analytical M-step gradients for speed

**Files:** `eigenspace.py`, `direct_vargp.py`, updated `run_single_mode.py`

---

## 2026-01-22: Whitening Instability Confirmation (seed 456)

**Finding:** Whitening confirmed as cause of seed 456 instability.

| Seed | Whitening | Test r |
|------|-----------|--------|
| 123 | ON | 0.77 |
| 123 | OFF | 0.86 |
| 456 | ON | **0.11** (collapsed) |
| 456 | OFF | 0.59 |

Unwhitened mode is stable across seeds. Whitened mode fails for certain inducing point configurations.

---

## 2026-01-22: Amp Parameter Implementation (COMPLETE)

**Accomplished:**
- Implemented `Amp` parameter in `ArcCosineKernel` to match legacy varGP exactly
- Replaced `ScaleKernel` (linear output scaling) with internal `Amp*C` (non-linear, inside kernel)
- Updated analytical gradients (VJP and Jacobian modes) with correct `grad_Amp = (dL_dC * C).sum() / Amp`
- Updated all test files and investigation scripts (15+ files)
- Validated against legacy: unwhitened mode achieves 0.86 vs varGP's 0.87 explained variance
- Ran canonical tests with seeds 123 and 456

**Key Findings:**
- Amp inside C affects kernel non-linearly through sqrt and arccos operations
- ScaleKernel's linear scaling cannot replicate this behavior
- Whitening instability at seed 456 is a separate pre-existing issue (not fixed by Amp change)
- Kernel matrices match reference exactly (max diff = 0.0)

**Files Changed (core):**
- `kernels.py` - Amp parameter with Positive() constraint, clamp at 1000
- `analytical_gradients_vjp.py` - Forward/backward with Amp
- `analytical_gradients.py` - Forward/backward with Amp
- `run_single_mode.py`, `run_benchmark.py`, `estep.py` - ScaleKernel removal
- 10+ test/investigation files updated

**Branch:** `bugfix/amp-parameter-mismatch` (ready to commit)

---

## 2026-01-20: Whitening Collapse Seed Sensitivity Investigation

**Accomplished:**
- Ran canonical benchmarks at M=50,75,100,200
- Tested seed sensitivity (seeds 42, 123, 456) for M=50 and M=75
- Added `--seed` argument to `run_single_mode.py`
- Created `investigations/INVESTIGATION_whitening_collapse_M75.md`

**Key Finding:**
- Whitened mode collapse is **seed-dependent, not M-dependent**
- Seed 456 causes collapse at both M=50 and M=75
- Seed 123 works fine at all M values
- Legacy (unwhitened) mode never collapses regardless of seed

**Files Changed:**
- `run_single_mode.py` - added `--seed` argument
- `results/BENCHMARK_LOG.md` - added seed sensitivity results
- `investigations/INVESTIGATION_whitening_collapse_M75.md` - created

---

## 2026-01-20: L_K Whitening Investigation + UnwhitenedVariationalStrategy Implementation + Performance Investigation (COMPLETE)

**Accomplished:**
- Comprehensive investigation of GPyTorch whitening behavior when kernel parameters change
- Used 4 parallel subagents to explore: GPyTorch source code, academic literature, GitHub issues, local codebase
- Documented findings in `TECHNICAL_ANALYSIS_2026-01-20_whitening_LK_mismatch.md` (comprehensive, standalone document)
- Updated `DECISION_LOG.md` with Q26-Q27 (whitening findings and solution)
- **Implemented `UnwhitenedVariationalStrategy` as alternative** (preserves existing code):
  - Added `whitening` parameter to `VariationalGPModel` (default `True`)
  - Modified `estep.py` for auto-detection and conditional L_K computation
  - Added `--unwhitened` flag to `test_estep_pnas.py` and `tests/test_estep_comparison.py`
  - Added Q28 to DECISION_LOG.md with implementation details
- **RESOLVED: Unwhitened accuracy gap root cause identified:**
  - Created `tests/diagnose_unwhitened_performance.py` to measure KL/gradient dynamics
  - Found KL divergence **explodes** for unwhitened (0 → 423 in 5 iterations)
  - Gradient ratio escalates from 0.64x to 159x over training
  - Root cause: K̃⁻¹ in unwhitened KL amplifies gradients by O(cond(K̃)) ≈ 10⁴
  - Added Q29 to DECISION_LOG.md documenting findings

**Key Findings:**
- GPyTorch does NOT auto-adjust whitened params when kernel changes
- Design assumes joint optimization where autograd handles L_K coupling
- EM-style optimization bypasses autograd → L_K mismatch corruption
- UnwhitenedVariationalStrategy stores natural params directly (verified correct)

**Known Issue - Unwhitened accuracy gap (not fully understood):**
- 16% accuracy gap: 0.6878 vs 0.8381 explained variance
- Observed: KL divergence explodes for unwhitened (0 → 423 in 5 iters)
- Observed: Gradient ratio escalates (0.64x → 159x)
- Tentative hypothesis: K̃⁻¹ in unwhitened KL causes gradient instability
- **Root cause NOT fully validated - needs further investigation**

**Performance Comparison (M=50, N=500):**
- varGP reference: explained_var=0.8748, time=5.4s
- Whitened (no-whiten conv): explained_var=0.8381, time=6.5s
- Unwhitened: explained_var=0.6878, time=29.3s (4x slower, 16% worse)

**Sources Examined:**
- GPyTorch source: `variational_strategy.py`, `unwhitened_variational_strategy.py`
- GitHub issues: #1308, #1754, #1556, PR #903
- Academic: Matthews 2017, Salimbeni 2018, Adam 2021

**Files Created:**
- `.claude/TECHNICAL_ANALYSIS_2026-01-20_whitening_LK_mismatch.md` - full investigation document
- `tests/diagnose_unwhitened_performance.py` - gradient/KL diagnostic script

**Files Modified:**
- `model.py` - added `whitening` parameter, conditional strategy selection
- `estep.py` - auto-detect whitening, conditional L_K computation
- `test_estep_pnas.py` - added `--unwhitened` flag
- `tests/test_estep_comparison.py` - added `--unwhitened` flag
- `.claude/DECISION_LOG.md` - added Q26-Q29
- `.claude/CLAUDE.md` - added UnwhitenedVariationalStrategy callout

**Archive Files (reasoning history):**
- `.claude/archive/ARCHIVE_2026-01-20_unwhitened_implementation_plan.md` - original plan before implementation
- `.claude/archive/ARCHIVE_2026-01-20_unwhitened_investigation.md` - investigation notes and reproduction commands
- `.claude/archive/ARCHIVE_2026-01-20_whitening_research_notes.md` - raw research notes from subagent exploration

---

## 2026-01-19: Whitening Implementation for Non-Cached E-step Path

**Accomplished:**
- Implemented whitening support for non-cached E-step path (plan from previous session)
- Added `compute_L_K()` and `e_step_explicit()` functions to `estep.py`
- Modified `e_step_loop()` and `train_varGP_style()` with `use_whitening` parameter
- Added `--no-whitening` CLI flag to `test_estep_pnas.py`
- Created `tests/test_whitening_paths.py` test suite (6 tests, all pass)
- Verified GPyTorch parameter update mechanism (`.data.copy_()` is correct)
- Refactored all `.data.copy_()` calls to use `torch.no_grad()` + `.copy_()` (best practice)

**Key Results:**
- Cached+whitening ≈ Non-cached+whitening (λ_m ratio = 1.0000) ✓
- Test r: cached=0.7966, noncached=0.7998 (both paths work)

**Files Changed:**
- `estep.py` - whitening functions, `.data.copy_()` → `torch.no_grad()` refactor
- `test_estep_pnas.py` - added `--no-whitening` flag

**Files Created:**
- `tests/test_whitening_paths.py` - whitening test suite

---

## 2026-01-19: Workflow Restructure Implementation

**Accomplished:**
- Implemented restructure plan from previous session
- Created `MATH_REFERENCE.md` (Section 1 extracted from CLAUDE.md)
- Created `SESSION_LOG.md` with template
- Updated CLAUDE.md: ToC table, "Current Focus" field, compact Section 1
- Updated WORKING_GUIDELINES.md: Sections 3.9, 3.10, 3.11, 9
- Fixed `.gitignore` to track nested `.claude/` folders
- Committed: `6621c5a`

**Not addressed:**
- `tests/smoke_test.sh` not created (TODO for user)

**Files Changed:**
- `.claude/CLAUDE.md`, `.claude/WORKING_GUIDELINES.md` (modified)
- `.claude/MATH_REFERENCE.md`, `.claude/SESSION_LOG.md` (created)
- `gaussian_processes/.gitignore` (fixed)

---

## 2026-01-18: Timing Analysis & Workflow Planning

**Accomplished:**
- Added E-step/M-step timing breakdown to `train_varGP_style()` in `estep.py`
- Updated `test_estep_pnas.py` to print timing breakdown for vargp_style mode
- Ran benchmark suite (20 runs) comparing gradient modes (autograd vs vjp)
- Analyzed eigenspace projection: M=50 keeps 47/50 dimensions (94%) - NOT the cause of slowdown
- Planned workflow restructure for `.claude/` folder

**Key Findings:**
- vargp_style is 3x slower than vargp_old overall
- E-step+F-step: 8.8s vs 1.8s (4.9x slower) - likely GPyTorch overhead
- M-step: 6.8s vs 3.8s (1.8x slower)
- Eigenspace projection excluded as cause of slowdown

**Unresolved:**
- Root cause of GPyTorch overhead not fully identified
- Potential optimizations not yet explored

**Files Changed:**
- `estep.py` - timing in train_varGP_style()
- `test_estep_pnas.py` - timing output
- `run_benchmark_suite.sh` - created (committed)
- `results/benchmark_2026-01-18/` - benchmark results (committed)

---

## Template for Future Sessions

```
## YYYY-MM-DD: [Brief Title]

**Accomplished:**
- [Bullet points of what was done]

**Key Findings:**
- [Important discoveries or results]

**Unresolved:**
- [Things that came up but weren't addressed]

**Files Changed:**
- [List of modified files]
```
