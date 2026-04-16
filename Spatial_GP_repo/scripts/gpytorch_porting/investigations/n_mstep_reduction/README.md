# n_mstep Reduction Investigation

**Branch**: `pietro/investigate-nmstep`
**Started**: 2026-04-15
**Status**: Phase 0 (characterization) in progress
**Plan file**: `~/.claude/plans/elegant-meandering-pretzel.md`

## Question

For the canonical 64x64 / intl_fixAmp / ELBO-ES training config, what is the smallest fixed `n_mstep` that does not measurably degrade test_r, and how much wall time does that save?

## Motivation

The dominant cost in the training pipeline is the M-step LBFGS, which rebuilds the structured covariance `C` (~2500x2500 after masking) and `K_tilde = X^T C X` (M x M, built from 4096-pixel inputs) on every closure call. The eigendecomposition of K_tilde is sub-millisecond on GPU — it is not the bottleneck. Kernel-evaluation work inside the M-step is, and it scales linearly with the number of LBFGS inner iterations.

`n_mstep = 10` (default in `default_params.json:38`) has never been empirically validated. It may be:

- **Too high**: LBFGS converges in fewer iterations; the rest is wasted line-search work.
- **About right**: needed for correct kernel learning; reducing degrades test_r.
- **Already partially gated**: `tolerance_change=1e-9` and `tolerance_grad=1e-7` are set inside `torch.optim.LBFGS` (`eigenspace_mstep.py:62-63`), so LBFGS may already early-terminate for free — Phase 0 will measure this.

## Phase structure

### Phase 0 — Characterization (no sweep, no defaults changed)

Add optional instrumentation to `mstep_eigenspace_autograd` to record per-closure-call ELBO and wall time, closure call count, and LBFGS termination reason. Off by default.

Run on 5 cells x 3 seeds = 15 runs with the canonical config but `n_mstep=20` (deliberately above the default to see the convergence curve). Cells chosen to span the difficulty range:

- Cell 8: well-behaved baseline (test_r ~ 0.88)
- Cell 0: known stuck-near-init failure mode
- Cell 10: default_gpy struggles but vargp_direct handles it
- Cells 15, 22: mid-difficulty with STA edge artifact on 108x108 but clean on 64x64

Analyze:

1. Histogram of LBFGS iterations actually run before tolerance hit.
2. Per-outer-iteration ELBO trace inside the M-step (closure call index -> ELBO).
3. Wall-time breakdown (closure call time vs LBFGS overhead vs E-step).
4. Cell-to-cell variability.

Decision gate with user before Phase 1: grid range, acceptance criterion, whether the sweep is worth running at all.

### Phase 1 — Sweep (only after Phase 0 review)

41 cells x 3 seeds x N `n_mstep` values (grid TBD based on Phase 0). Template: `experiments/2026-04-06_es_sweeps_64x64/run_sweep_elbo_es_64x64.py`.

## Scope

- **In scope**: fixed `n_mstep` values, single-cell training, autograd M-step path only.
- **Out of scope**: adaptive (tolerance-based) stopping (deferred Phase 2), active loop benchmarking, analytical M-step path, structural changes to the M-step (Kronecker, caching).

## Files

### Phase 0

- `phase0_characterization/run_phase0.py` — runner
- `phase0_characterization/analyze_phase0.py` — analysis + plots
- `phase0_characterization/results.jsonl` — raw output

### Phase 1 (not yet created)

- `phase1_sweep/run_sweep.py`
- `phase1_sweep/analyze_phase1.py`
- `phase1_sweep/results.jsonl`

### Instrumentation (on this branch)

- `eigenspace_mstep.py` — `mstep_eigenspace_autograd` extended with optional diagnostics kwarg
- `eigenspace_training.py` — threads diagnostics flag, surfaces in `result['mstep_diagnostics']`

## Closure

On completion (either phase):

- Findings written to `FINDINGS.md`
- Merge back to `pietro/workingbranch` regardless of outcome
- Any change to `default_params.json` requires a separate planning discussion backed by Phase 1 evidence
