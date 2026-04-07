# Investigation: Paper Performance Gap (Goldin et al. 2023)

**Branch**: `pietro/investigate-paper-gap`
**Date**: 2026-03-31
**Status**: Continuing
**Location**: `investigations/paper_gap/`
**Worktree**: `gpytorch_porting_paper_gap/` (sibling of main `gpytorch_porting/`)

---

## Problem Statement

Our vargp_direct GP model underperformed the reference paper (Goldin et al. 2023
PNAS). The paper reported "adjusted R^2 > 0.8 for 36/41 cells." Our baseline
was avg adjusted_r2 = 0.720 with 13/41 > 0.8. We investigated why.

## What Was Found (21 Findings)

Full details: `investigations/paper_gap/INVESTIGATION_LOG.md`

### Gap A: vargp_direct vs vargp_old -- CLOSED

The 0.042 performance gap between our reimplementation (vargp_direct) and the
original code (vargp_old in utils.py) was entirely from confounds, not algorithm
bugs:

| Source | Contribution | Finding |
|--------|-------------|---------|
| Inducing point selection (pivoted vs random) | 0.028 | #18: run_single_mode.py line 672 silently gives vargp_old random IPs |
| Amp frozen vs free | 0.010 | fix_Amp=True freezes Amp; vargp_old ignores this flag |
| sigma_0 parameterization (exp vs direct) | 0.004 | #19: direct matches vargp_old exactly |
| **Total** | **0.042** | |

### Gap B: Our code vs paper -- Likely a metric mismatch

The paper's GitHub code (Jupyter notebook) differs from BOTH our implementations:
- **No Amp parameter** (C = alpha * Csmooth * alpha)
- **F-step interleaving** (damped Newton A/lambda0 at every E-step iteration)
- **No eigenspace projection** (full M x M space)
- **scipy L-BFGS-B + float64** for M-step
- **Different init**: beta_nat=0.0452, rho_nat=0.0821, A=1e-4, lambda0=-1

**Critical discovery (Finding 21)**: The paper likely reports the UNSQUARED metric
(`mean_accuracy / reliability`, our `explained_var`) despite calling it "adjusted R^2"
in Eq. 5. Evidence:
- Our best config gives **36/41 > 0.8 on explained_var** (exact match)
- Only 15/41 > 0.8 on the squared adjusted_r2
- Figure 2F caption says "explained variance", y-axis says "adjusted r^2"
- Figure 2F shows data points > 1.0 (much easier with unsquared formula)
- The only evaluation code in the repo computes accuracy/reliability (unsquared)

This is a HYPOTHESIS, not confirmed. See `METRICS_COMPARISON.md` for full analysis.

## Code Changes on This Branch

### Production code changes (all committed)

| File | Change | Notes |
|------|--------|-------|
| `kernels.py` | sigma_0: `Positive(transform=torch.exp, inv_transform=torch.log)` | Was softplus, then identity, now exp. Best overall (+0.008 adj_r2 vs identity). |
| `kernels.py` | Amp: kept as `Positive()` (softplus) | Exp caused runaway growth (16-284x). Softplus is stable. |
| `eigenspace_fstep.py` | Added `damped_newton_update_A_lambda0()` | Paper's updateA: 2x2 Hessian, alpha=0.25, max 100 iter, convergence at sum(\|g\|) < 1e-6. |
| `eigenspace_fstep.py` | F-step LBFGS: added mean>100 + max>500 thresholds | Matches vargp_old's conservative F-step. |
| `eigenspace_training.py` | Added `interleave_fstep` parameter | When True, calls damped Newton inside E-step loop. |
| `eigenspace_training.py` | Added `fix_Amp` parameter | When True, freezes raw_Amp (requires_grad=False). |
| `eigenspace_training.py` | E-step divergence: added mean>100 + max>500 | Was only max>1000. |
| `eigenspace_mstep.py` | LBFGS param filter: `[p for p in kernel.parameters() if p.requires_grad]` | Frozen params corrupted LBFGS Hessian. Both autograd and analytical paths. |
| `eigenspace_mstep.py` | sigma_0 gradient: `dL['sigma_0'] * kernel.sigma_0` | Chain rule for exp transform. |
| `run_single_mode.py` | Wire `interleave_fstep`, `fix_Amp` through config | `config.get('interleave_fstep', False)`, `config.get('fix_Amp', False)` |
| `run_single_mode.py` | vargp_old bug fix: `_model`/`_likelihood` = None for vargp_old | Was crashing with UnboundLocalError. |
| `run_single_mode.py` | `rf_init` config: 'ground_truth', 'sta', or 'center' | Auto-loads from `datasets/rf_centers_ground_truth.npz`. |
| `default_params.json` | `rf_init: "ground_truth"` | Default changed from STA to ground-truth ellipses. |
| `default_params.json` | `f_mean_max_threshold: 500`, `f_mean_mean_threshold: 100` | Replaces old `stability_threshold: 1000`. |
| `analytical_gradients_vjp.py` | sigma_0 exp + Amp softplus gradient corrections | Test comparisons updated. |
| `tests/test_mstep_analytical.py` | Same gradient corrections | |
| `tests/test_vargp_direct_match.py` | Same gradient corrections | |
| `.claude/rules/gradients.md` | Updated transform documentation | |
| `.claude/CLAUDE.md` | Paper gap findings, three-way comparison, file map, known issues | |

### Investigation rules (MUST follow for fair comparisons)

1. **ip_selection='random'** for all mode comparisons (pivoted silently differs for vargp_old)
2. **fix_Amp=True** for paper comparisons (paper has no Amp parameter)

### Key config flags

```python
config['interleave_fstep'] = True   # damped Newton A/lambda0 inside E-step
config['fix_Amp'] = True            # freeze Amp at 1.0 (paper has no Amp)
config['ip_selection'] = 'random'   # avoid pivoted/random confound
config['early_stop'] = False        # see full convergence
```

## Best Results

Stored in `experiments/2026-03-31_PNAS_paper_fits/`:

| File | sigma_0 | adj_r2 mean | expl_var n>0.8 |
|------|---------|-------------|----------------|
| `results.jsonl` | direct (identity) | 0.730 | **36/41** |
| `results_exp_sigma0.jsonl` | exp (current default) | **0.738** | 36/41 |

Config: broad+interleave (beta=0.1, A=1e-4, interleave_fstep=True, fix_Amp=True,
50/20/80, 108x108, M=250, n_train=3160, random IPs, ground-truth RF, lambda0=-1).

## Investigation Artifacts

| File | Purpose | Keep/Delete |
|------|---------|-------------|
| `INVESTIGATION_LOG.md` | Lab notebook, 21 findings | KEEP |
| `METRICS_COMPARISON.md` | Metric definitions + evidence | KEEP |
| `sweep_results.jsonl` | 738-run sweep (6 configs x 41 cells x 3 seeds) | KEEP |
| `sigma0_exp_results.jsonl` | 123-run exp sigma_0 test | KEEP |
| `run_sweep.py` | Sweep script (resume-safe) | KEEP |
| `plot_rf_diagnostic.py` | RF diagnostic plotter | KEEP |
| `sweep_summary.txt` | Human-readable sweep summary | KEEP |
| `experiment[1-5,7]_results.jsonl` | 5-cell subset intermediate results | DELETE (superseded by sweep) |
| `run_iteration_test.py` | Exp 1 script | DELETE |
| `run_diagnostic2.py` | Exp 2 script | DELETE |
| `run_paper_config.py` | Exp 3 script | DELETE |
| `run_paper_init.py` | Exp 4 script | DELETE |
| `run_paper_init_fixed_amp.py` | Exp 5 script | DELETE |
| `test_interleave_fstep.py` | Exp 7 script | DELETE |
| `sweep.log`, `sweep.pid`, `sigma0_exp_test.log` | Logs | DELETE |
| `bad_cells_rf_diagnostic.png`, `rf_diagnostic.png` | Diagnostic plots | DELETE |
| `__pycache__/` | Cache | DELETE |

## Why This Was Stopped

Context running out after a very long session. The investigation achieved its
main goals (Gap A closed, Gap B likely explained by metric mismatch). The next
session should clean up artifacts, run a few more structured sweeps (to be
discussed with the user), and merge the branch.

## Things Noticed But Not Acted Upon

1. **Tests not verified**: The gradient tests (test_mstep_analytical, test_vargp_direct_match,
   analytical_gradients_vjp) were updated for the new sigma_0 exp + Amp softplus
   corrections but never run to verify they pass.

2. **64x64 + new features not tested**: Our previous best on 64x64 was adj_r2=0.720
   (M=2910, old defaults). We never ran 64x64 with interleaving + exp sigma_0 + fix_Amp.
   Could be better since 64x64 avoids the STA edge artifact for 6 cells.

3. **M=2910 + interleaving not tested**: All interleaving tests used M=250. With
   M=2910 (no sparse approximation), interleaving + exp sigma_0 could give the
   best absolute numbers.

4. **Variance-based FVE metric**: The paper's CNN code also computes
   `(var_total - MSE) / (var_total - var_noise)`. We never computed this for our
   GP fits. Could be another metric the paper reports.

5. **run_inference.py and train_all_cells.py only support default_gpy**: To share
   vargp_direct results with proper plots (STA + RF overlay + scatter), these
   scripts need extending. Currently results are only in JSONL format.

6. **The damped Newton interleaving is unstable with A=0.01**: The alpha=0.25
   damping is too conservative for larger A. Only works well with A=1e-4.
   A higher alpha or adaptive damping could help.

7. **gpy_training.py appears in the diff** but the change wasn't documented in
   this session. Check what changed.

## If Someone Continues This

**What to do next (in order):**
1. Clean up investigation artifacts (delete intermediate scripts/results/plots/logs per table above)
2. Run gradient tests to verify they pass with current parameterization
3. Discuss with user which additional sweeps to run (64x64? M=2910? free Amp comparison?)
4. Run agreed sweeps
5. Merge branch into `pietro/workingbranch`

**What NOT to try:**
- Pivoted Cholesky IP selection for mode comparisons (confounded, Finding 18)
- EIGVAL_TOL tuning (no effect, Finding 16)
- More EM iterations alone (diminishing returns past 80, Finding 20)
- A=0.01 + interleave_fstep (unstable, damped Newton too conservative)

**Key files to read first:**
- `investigations/paper_gap/INVESTIGATION_LOG.md` (21 findings, compounding factors table)
- `experiments/2026-03-31_PNAS_paper_fits/README.md` (best results + reproduction)
- `experiments/2026-03-31_PNAS_paper_fits/METRICS_COMPARISON.md` (metric definitions)

---

## Continuation Prompt

```
I'm continuing the paper gap investigation on branch pietro/investigate-paper-gap.
Working directory: gpytorch_porting_paper_gap/ (a git worktree).

Read the handoff at investigations/paper_gap/HANDOFF.md for full context.
Also read INVESTIGATION_LOG.md (21 findings) and
experiments/2026-03-31_PNAS_paper_fits/README.md.

Summary: Gap A (vargp_direct vs vargp_old) is closed -- the implementations
match within 0.002. Gap B (our code vs paper) is likely a metric mismatch --
we get 36/41 > 0.8 on explained_var, matching the paper's claim exactly.

This session's goals:
1. Clean up investigation artifacts (delete intermediate scripts per HANDOFF table)
2. Run gradient tests to verify they pass
3. Discuss and run additional sweeps (64x64? M=2910? other configs?)
4. Merge branch into pietro/workingbranch when ready

Check git branch and git status before starting.
```
