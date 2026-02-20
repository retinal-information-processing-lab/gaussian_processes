# Handoff: Unify Kernel Selection and Utility Investigation Scripts

**Branch**: `pietro/workingbranch`
**Date**: 2026-02-20
**Status**: Plan approved, ready for implementation
**Plan file**: `~/.claude/plans/cozy-humming-wadler.md`

---

## Motivation

The utility/acquisition investigation grew organically over several sessions (Feb 2026), with one investigation folder per kernel variant: understanding_utility (unnormalized arc-cosine), normalized_kernel, arcsine_kernel, rbf_kernel. Each folder contains its own copy of `run_single_mode.py` (~1200 lines duplicated per script with only the kernel class name swapped), plus explore_utility and gradient scripts that share ~80% of their code.

This duplication is now a maintenance burden — config changes to `run_single_mode.py` don't propagate, and the normalized kernel investigation is deprecated. The user wants to consolidate into: (1) kernel selection as a parameter in `run_single_mode.py`, and (2) a single set of utility investigation scripts that accept `--kernel-type` and work with any kernel. The stale per-kernel investigation folders get deleted (git history preserves them).

---

## Decisions and Rationale

### 1. Kernel types supported: arc_cosine, arc_sine, rbf (NOT normalized)

The normalized arc-cosine kernel (`ArcCosineKernelNormalized`) is excluded from the factory. The user explicitly declared it deprecated. It showed a ~25% test_r drop and its only advantage (eliminating norm-driven utility divergence) is shared by both arc-sine and RBF kernels. Keeping it would add complexity for no practical value.

### 2. Unified gradient script uses simple single-call pattern, NOT RBF's decomposed approach

The RBF gradient script (`gradient_rbf.py`) decomposed utility into H_marg + MC loop over H_cond with per-sample backward — this was needed for memory efficiency with many lambda samples. The user explicitly said "none of the MC sampling complications." The unified script calls `distribution_aware_utility()` once per LBFGS step (like unnormalized/arcsine scripts). The decomposed approach is documented as deferred.

Alternative rejected: Keeping both patterns with a flag. Too complex, the decomposed approach is experimental and its key finding (gradients from N=50+ images cancel out) suggests it's a dead end anyway.

### 3. Bounds: sigmoid + 2 modes (none, dataset)

The RBF script had 4 bounds modes (none/dataset/target/percentile). The user chose to keep only `none` (unconstrained) and `dataset` (per-pixel min/max from training images within RF). The `target` and `percentile` modes were experimental and not used in final analyses.

### 4. Hyperparams hardcoded at script level, NOT kernel-specific

N_STEPS, LR, LBFGS settings are single values in the unified gradient script, same for all kernels. The user explicitly rejected per-kernel defaults. Rationale: these are tuning knobs for the investigation, not model parameters. If a kernel needs different tuning, the user edits the script constant — explicit and visible.

### 5. RF center bounds (apply_rf_center_bounds) is a TRAINING concern

When the user said "RF stricter bounds," they meant `apply_rf_center_bounds()` in `run_single_mode.py` — constraining eps_0x/eps_0y during the M-step so the RF center can't wander far from the STA initial guess. This already exists in `run_single_mode.py` and all kernels inherit `set_center_bounds()` from `ArcCosineKernel`. No new code needed, just verify it works for non-arc-cosine kernels.

This was initially confused with pixel bounds in gradient optimization (different concept). Clarified via explicit question.

### 6. Keep understanding_utility/ folder (entropy landscape, math docs)

The `understanding_utility/` folder contains much more than just explore/gradient scripts: `entropy_landscape.py` (standalone entropy investigation with MC comparison), `test_compute_H_MC.py`, three LaTeX math reference documents, `key_facts.md`. Only `explore_utility.py` and `gradient_unnormalized.py` are superseded. The folder survives as reference material.

### 7. distribution_gradient.py — delete with reimplementation guide

The multi-image DA conditioning script is deleted. Its key finding (gradients cancel with many conditioning images, producing flat utility) is documented. A short reimplementation guide is left in the deferred items: decompose into H_marg + MC H_cond, per-sample backward for O(1) memory, reference git history.

### 8. Explore utility auto-detect right panel

The four explore scripts differ in what the right panel plots:
- arc_cosine: ||x||_C (norm grows with x)
- arc_sine: K(x,x) (self-value, saturates at 1)
- rbf: K(x*, x_cond) (kernel similarity)

The unified script auto-detects kernel type and picks the appropriate metric. This was chosen over "drop right panel" (loses diagnostic value) and "keep separate scripts" (defeats the purpose).

---

## Critical Subtleties

### LocalRBFKernel has an extra parameter: lengthscale

The RBF kernel requires a `lengthscale` parameter (default 100.0) that other kernels don't have. It must be added to `default_params.json` and YAML configs. The factory function passes it only when `kernel_type == 'rbf'`. If you forget to add it to `build_config_from_defaults()`, the factory will crash with KeyError on `config['lengthscale']` — which is the correct behavior per parameter discipline (crash, don't fall back).

### vargp_old mode only works with arc_cosine

The old codebase (`utils.py:varGP()`) only implements the arc-cosine kernel. The investigation scripts raise `NotImplementedError` for arc_sine/rbf + vargp_old. The unified `run_single_mode.py` must validate this upfront.

### Analytical gradient modes (vjp, jacobian) only work with arc_cosine

The VJP and Jacobian gradient implementations in `analytical_gradients.py` and `analytical_gradients_vjp.py` are hardcoded for the arc-cosine kernel. Other kernels must use `gradient_mode='autograd'`. The factory should validate this.

### importlib.util pattern is required for utils.py imports in investigation scripts

The playground's 1D code also has `utils.py` at repo root. Direct `import utils` would cache the wrong module. All investigation scripts use `importlib.util.spec_from_file_location()` to import the local `utils.py`. The unified scripts must maintain this pattern.

### Explore scripts import run_single_config from different sources

Currently: explore_utility.py imports from `run_single_mode.py`, explore_arcsine from `run_arcsine.py`, explore_rbf from `run_rbf.py`. After unification, all import from `run_single_mode.py` — the kernel-specific `run_*.py` scripts are deleted.

---

## Uncommitted Changes

The working tree has many uncommitted changes, mostly from previous sessions:
- Modified/deleted `.png` images in `imgs/` — old plots from deprecated modes
- Modified `investigations/arcsine_kernel/HANDOFF.md` — newer version from stash
- Modified `investigations/understanding_utility/gradient_unnormalized.py` — minor edit
- Modified `pietro_plan.md`
- Untracked exploratory experiment folders in `experiments/exploratory/`
- Untracked `.png` outputs in various investigation folders

These are pre-existing and unrelated to this planning session. No code was written in this session — it was purely discussion and plan creation.

---

## Files to Read First

1. **Plan file** (`~/.claude/plans/cozy-humming-wadler.md`) — the implementation plan with 4 phases
2. **This handoff** — decisions and rationale
3. `run_single_mode.py` — the main script to modify (kernel instantiation at ~lines 638, 832, 933; `build_config_from_defaults()` at ~line 270; `flatten_yaml_config()`)
4. `default_params.json` — config to extend with kernel.type and kernel.lengthscale
5. `kernels.py` — all kernel classes, their constructors, `set_center_bounds()` at line 377
6. `investigations/understanding_utility/explore_utility.py` — template for unified explore script (arc-cosine version)
7. `investigations/rbf_kernel/gradient_rbf.py` — reference for bounds modes (dataset mode implementation)
8. `investigations/understanding_utility/entropy_landscape.py` — must NOT be deleted

---

## Caveats and Open Questions

1. **Hyperparams may need kernel-specific tuning**: The plan hardcodes single N_STEPS/LR values. The arc-sine kernel previously needed 500 steps with LR=0.1 (10x more steps, 5x smaller LR) to converge. If the unified defaults (N_STEPS=50, LR=0.5) don't work well for arc-sine or RBF, the user will need to manually tune. This is a known tradeoff the user accepted.

2. **RBF lengthscale default (100.0)**: This value comes from `LocalRBFKernel`'s constructor default. It was used in the RBF investigation but hasn't been formally optimized. It may need tuning for different datasets or cell types.

3. **Existing uncommitted changes**: The working tree has many pre-existing uncommitted changes. Consider committing or stashing before starting implementation to keep the diff clean.

4. **YAML config immutability**: The plan adds new keys to `canonical.yaml` and `quick.yaml`. Per project rules, these configs are "immutable" — but adding new keys with default values (kernel.type=arc_cosine) doesn't change existing behavior. This should be safe, but worth noting.

5. **2D playground dependency**: `../2D_playground/utility_2d_rbf_base.py` imports from `kernels.py` and `acquisition.py`. Adding kernel types shouldn't break it, but verify after Phase 1.

---

## Continuation Prompt

```
I'm continuing work on unifying kernel selection and utility investigation scripts.

Read these files first:
1. .claude/handoffs/HANDOFF_2026-02-20_unify-kernel-selection-utility.md (decisions + rationale)
2. ~/.claude/plans/cozy-humming-wadler.md (implementation plan — 4 phases)
3. .claude/CLAUDE.md (project context)

The plan has 4 phases:
- Phase 1: Add --kernel-type to run_single_mode.py (factory + config + CLI)
- Phase 2: Unified investigations/utility/explore_utility.py (auto-detect kernel for right panel)
- Phase 3: Unified investigations/utility/gradient.py (single DA utility call, sigmoid bounds, 2 modes)
- Phase 4: Cleanup (delete per-kernel investigation duplicates, keep understanding_utility/ reference material)

Key constraints: no hardcoded params in run_single_mode.py, normalized kernel is deprecated (excluded),
no MC sampling in gradient script, entropy_landscape.py must be preserved.

Check git status and git branch before starting. There are pre-existing uncommitted changes — consider
committing or stashing first.

Start with Phase 1.
```
