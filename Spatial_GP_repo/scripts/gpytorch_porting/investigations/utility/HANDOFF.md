# Investigation: Unified Utility Investigation Scripts

**Branch**: `pietro/workingbranch`
**Date**: 2026-02-20
**Status**: Complete (Phases 1-4)
**Location**: `investigations/utility/`

---

## What This Folder Contains

**Scripts:**
- `explore_utility.py` — Utility exploration workbench for all kernel types. Trains model, provides kernel/GP/utility helpers, generates DA utility landscape plots. Supports `--kernel-type {arc_cosine, arc_sine, rbf}`.
- `gradient.py` — LBFGS gradient ascent on DA utility across kernel types. Interpolation monotonicity check + gradient ascent with RF overlay visualization. Supports `--kernel-type {arc_cosine, arc_sine, rbf}`.
- `entropy_landscape.py` — Entropy heatmap H(mu_g, sigma2_g) with adaptive r_max and MC comparison.
- `test_compute_H_MC.py` — Validates MC entropy estimator against Laplace in safe region.

**Documentation:**
- `REFERENCE.md` — Source of truth for definitions, key results, and cross-kernel findings. Start here.
- `entropy_landscape.md` — Detailed findings from entropy heatmap investigation.
- This file (`HANDOFF.md`) — Session guide: what was deleted, deferred items, re-implementation guides.

**Proof files** (full LaTeX derivations, referenced from REFERENCE.md):
- `proof_moments_and_conditioning.tex` — GP posterior formulas, conditioning, log-firing-rate transform, DA utility derivation
- `proof_divergence_theorems.tex` — Formal theorems on norm scaling and divergence (Theorem 1, Corollaries, Proposition 1)
- `proof_kernel_solutions.tex` — Utility decomposition, normalized kernel solution, arc-sine saturation solution

These scripts replace the per-kernel scripts that were in `understanding_utility/`, `arcsine_kernel/`, `rbf_kernel/`, and `normalized_kernel/` (deleted, retrievable from git history).

---

## Deferred: distribution_gradient.py (Multi-Image DA Conditioning)

**Deleted** with the rbf_kernel/ folder. Retrievable from git history (`rbf_kernel/distribution_gradient.py`).

**Re-implementation guide** (if needed):
- Decompose utility into H_marg + MC loop over H_cond with per-sample backward for O(1) memory.
- Per-sample backward: `(-H_marg).backward()`, then `(H_cond_i / n_mc).backward()` per sample with fresh x_query each time.
- Must `.detach()` A_val/lam0_val to avoid double-backward through likelihood parameters.
- Key finding: gradients from N=50+ conditioning images cancel out, producing a flat utility landscape. The averaging across diverse images eliminates directional signal.

---

## Files Deleted (Retrievable from Git)

| Folder | Contents | Reason |
|--------|----------|--------|
| `investigations/normalized_kernel/` | validate_kernel.py, run_normalized.py, explore/gradient scripts, HANDOFFs, PNGs | Deprecated kernel, unified scripts cover arc_cosine/arc_sine/rbf |
| `investigations/arcsine_kernel/` | run_arcsine.py, explore/gradient scripts, HANDOFF, PNGs | Superseded by unified scripts + kernel selection in run_single_mode.py |
| `investigations/rbf_kernel/` | run_rbf.py, explore/gradient/distribution_gradient scripts, HANDOFF | Superseded by unified scripts + kernel selection in run_single_mode.py |
| `investigations/understanding_utility/explore_utility.py` | Arc-cosine explore script | Superseded by utility/explore_utility.py |
| `investigations/understanding_utility/gradient_unnormalized.py` | Arc-cosine gradient script | Superseded by utility/gradient.py |

`understanding_utility/` folder deleted. Reference material (entropy_landscape.*, test_compute_H_MC.py, proof *.tex files) moved to `utility/`. `key_facts.md` absorbed into `REFERENCE.md`.
