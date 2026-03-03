# Investigation: Consolidate utility/ and utility_decompositions/ into one folder

**Branch**: `pietro/workingbranch`
**Date**: 2026-03-03
**Status**: Continuing
**Location**: `investigations/` (both `utility/` and `utility_decompositions/`)

---

## Problem Statement

Two investigation folders under `investigations/` contain related work on utility optimization. They were created at different times for different purposes, but now overlap conceptually. The user wants to consolidate them into a single folder while cleaning up redundant documentation.

The goal: **one folder, no redundant docs, all scripts still work.**

## Current Inventory

### `investigations/utility/` (13 files + __pycache__)

**Scripts** (4):
| File | Size | Purpose |
|------|------|---------|
| `explore_utility.py` | 30 KB | Workbench: trains model, explores utility across kernel types. **Provides `setup()` function imported by other scripts.** |
| `gradient.py` | 28 KB | LBFGS gradient ascent validation: confirms DA utility peaks at conditioning target |
| `entropy_landscape.py` | 23 KB | Entropy heatmap with adaptive r_max and MC comparison |
| `test_compute_H_MC.py` | 10 KB | Validates MC entropy estimator against Laplace approximation |

**Documentation** (6):
| File | Size | Purpose |
|------|------|---------|
| `REFERENCE.md` | 9 KB | Source of truth for DA utility definitions, scaling behavior, cross-kernel findings |
| `HANDOFF.md` | 3.7 KB | Session guide: what was deleted, re-implementation notes |
| `entropy_landscape.md` | 7.6 KB | Entropy heatmap analysis writeup |
| `proof_moments_and_conditioning.tex` | 12 KB | GP posterior formulas, conditioning, DA utility derivation |
| `proof_divergence_theorems.tex` | 20 KB | Formal theorems on norm scaling and divergence |
| `proof_kernel_solutions.tex` | 11 KB | Utility decomposition, normalized kernel solution, arc-sine saturation |
| `subspace_optimization_pca_ceigen_fourier.md` | 12 KB | **UNTRACKED** — appears to be a newer doc about PCA/C-eigen/Fourier subspace methods |

**PNGs** (6): explore_utility and gradient output plots for arc_cosine, arc_sine, rbf + entropy_landscape.png

### `investigations/utility_decompositions/` (4 files + .gitignore + __pycache__ + PNGs)

**Scripts** (2):
| File | Size | Purpose |
|------|------|---------|
| `subspace_optimization.py` | 61 KB | Main script: PCA, C-eigen, combined, pixel methods. Imports `setup()` from `utility/explore_utility.py` |
| `test_subspace_optimization.py` | 27 KB | 52-test suite for all subspace methods |

**Documentation** (1):
| File | Size | Purpose |
|------|------|---------|
| `HANDOFF_COMBINED_SUBSPACE.md` | 16 KB | Authoritative guide: all 3+1 methods, offset convention, optimization details, input warping idea |

**PNGs** (5): gitignored experiment output images

### Dependency between folders

```
utility_decompositions/subspace_optimization.py
  └── imports setup() from utility/explore_utility.py
```

This is the only cross-folder dependency. If consolidated, this import becomes a same-folder import.

## What Was Done This Session

### Merge of pca-utility-optimization branch
- **What**: Merged `pietro/pca-utility-optimization` (10 commits) into `pietro/workingbranch`
- **Result**: Clean merge, no conflicts. All investigation files now on workingbranch.
- **Verdict**: Complete

### Cleanup before merge
- **What**: Deleted `understanding_utility/` (2 orphan PNGs), deleted `HANDOFF_PCA.md` and `HANDOFF_C_EIGEN.md` (superseded), added `.gitignore` for PNGs in `utility_decompositions/`
- **Result**: Committed as `1031166`
- **Verdict**: Complete

### Test fix for unified offset convention
- **What**: Updated `test_subspace_optimization.py` to pass `data_mean_rf` to `compute_c_eigenspace()` calls and check offset = mu_rf instead of zeros
- **Result**: All 52 tests pass. Committed as `195aead`
- **Verdict**: Complete

## Key Findings

1. **CONFIRMED**: The two folders have NO code redundancy — different scripts doing different things. `utility/` = understanding utility behavior; `utility_decompositions/` = constrained optimization.

2. **CONFIRMED**: `understanding_utility/` was fully obsolete (only 2 PNGs, no code). Deleted.

3. **HYPOTHESIS**: Documentation across the two folders likely has significant overlap. There are 7 doc files total (not counting LaTeX proofs), and some probably cover the same concepts (DA utility definition, conditioning formulas, etc.). This was NOT verified — the docs were not read in detail this session.

4. **CONFIRMED**: The untracked file `utility/subspace_optimization_pca_ceigen_fourier.md` (12 KB) looks like it overlaps with `utility_decompositions/HANDOFF_COMBINED_SUBSPACE.md`. Needs reading to confirm.

## Why This Was Stopped

Context ran out. The merge/cleanup work is done. The doc consolidation requires carefully reading all 7+ doc files to identify overlaps, which is a fresh-context task.

## Things Noticed But Not Acted Upon

1. The untracked `subspace_optimization_pca_ceigen_fourier.md` in `utility/` was not committed and may be a draft or superseded doc — needs reading.
2. PNGs in `utility/` are tracked (unlike `utility_decompositions/` which has `.gitignore`). May want to gitignore those too during consolidation.
3. The `HANDOFF_COMBINED_SUBSPACE.md` line 8 flags that the C-eigenspace offset is not mathematically motivated — still open for investigation.
4. `investigations/input_warping/` and `investigations/diffusion/` also exist — not in scope for this consolidation but worth noting.

## Uncommitted Changes

None — working tree is clean (only untracked files: old exploratory experiments, PNGs in imgs/, the `subspace_optimization_pca_ceigen_fourier.md` doc, and `investigations/diffusion/`).

## Files Created

| File | Purpose | Keep/Delete |
|------|---------|-------------|
| This handoff file | Continuation guide | Keep until consolidation done |

## If Someone Revisits This

**What to do**:
1. Read ALL doc files in both folders carefully (REFERENCE.md, HANDOFF.md, entropy_landscape.md, HANDOFF_COMBINED_SUBSPACE.md, subspace_optimization_pca_ceigen_fourier.md)
2. Identify which content is duplicated vs unique
3. Propose a single-folder structure with one authoritative reference doc
4. Move scripts into unified folder, fix the `setup()` import path
5. Delete redundant docs, update CLAUDE.md file map
6. Run both test suites to verify nothing broke

**What NOT to do**:
- Don't merge the LaTeX proofs into markdown docs — they serve a different purpose (formal math vs operational reference)
- Don't rewrite scripts — just move them and fix imports
- Don't touch `acquisition.py` or other production code — this is purely an investigations/ cleanup

**Key question to resolve**: Should the unified folder keep both REFERENCE.md and HANDOFF_COMBINED_SUBSPACE.md as separate docs (one for utility theory, one for subspace optimization practice), or merge them into a single reference?

---

## Continuation Prompt

```
I am consolidating the utility investigation folders.

Read: investigations/HANDOFF_consolidate_utility_investigations.md

Two folders need to become one:
- investigations/utility/ (4 scripts, 6 docs, understanding utility behavior)
- investigations/utility_decompositions/ (2 scripts, 1 doc, subspace optimization)

Task: Read ALL doc files in both folders, identify redundancy, propose
a unified folder structure. Then execute the consolidation (move files,
fix imports, delete redundant docs, update CLAUDE.md).

The subspace script imports setup() from explore_utility.py — this import
path will need updating when folders merge.

Check git status and git branch before starting.
Do NOT modify production code outside investigations/.
```
