# Paper Gap Investigation

**Goal**: Understand why our vargp_direct GP underperformed Goldin et al. 2023 PNAS
(paper reports "adjusted R^2 > 0.8 for 36/41 cells"; our baseline: 13/41).

**Outcome**: Gap fully explained. See INVESTIGATION_LOG.md for all 21 findings.
- Gap A (vargp_direct vs vargp_old): closed -- 3 confounds totaling 0.042.
- Gap B (our code vs paper): metric mismatch -- we get 36/41 on explained_var.

**Best results**: `experiments/2026-03-31_PNAS_paper_fits/` (all 41 cells, 3 seeds).

---

## Experiments (chronological)

All experiments below use cells [9, 14, 18, 28, 39] (5-cell subset), seed=1,
vargp_direct mode, 108x108 images unless noted. Each experiment isolates one
factor from the compounding factors table in INVESTIGATION_LOG.md.

### Experiment 1: Iteration counts
**Script**: `run_iteration_test.py` | **Results**: `experiment1_results.jsonl` (15 runs)
**Question**: Does matching the paper's inner iteration counts help?
**Configs**: baseline (10/10/50 estep/mstep/iter), paper_like (50/20/50), extended (50/20/150)
**Data**: 64x64, M=n_train=2910, ground-truth RF centers
**Finding**: +0.016 avg adj_r2 from more iterations. Moderate effect.

### Experiment 2: Cross-mode and resolution
**Script**: `run_diagnostic2.py` | **Results**: `experiment2_results.jsonl` (5 runs)
**Question**: Does vargp_old beat vargp_direct? Does 108x108 help?
**Configs**: vargp_old on 64x64 vs vargp_direct on 108x108; M=n_train=2910, 10/10 inner iters
**Finding**: Established baseline cross-mode comparison.

### Experiment 3: Paper training schedule
**Script**: `run_paper_config.py` | **Results**: `experiment3_results.jsonl` (10 runs)
**Question**: Does the paper's exact training setup (M=250, n_train=3160, 50/20/50) close the gap?
**Configs**: vargp_direct vs vargp_old, both with M=250, n_train=3160, nEstep=50, nMstep=20
**Finding**: Compared modes under paper config. Gap persisted -- not just iteration counts.

### Experiment 4: Paper initialization
**Script**: `run_paper_init.py` | **Results**: `experiment4_results.jsonl` (10 runs)
**Question**: Do the paper's init values matter? (beta=0.0452, rho=0.0821, A=1e-4, lambda0=-1)
**Configs**: vargp_direct vs vargp_old, paper init, M=250, n_train=3160, 80 iterations
**Finding**: Paper init helps specific cells (9, 14, 18). A=1e-4 creates chicken-and-egg with F-step.

### Experiment 5: Fixed Amp
**Script**: `run_paper_init_fixed_amp.py` | **Results**: `experiment5_results.jsonl` (5 runs)
**Question**: The paper has no Amp parameter. Does freezing Amp at 1.0 help?
**Configs**: Paper init + fix_Amp=True (vargp_direct and vargp_old)
**Finding**: Confirmed free Amp absorbs scale from A (grows 5-284x). fix_Amp stabilizes optimization.

### Experiment 7: Interleaved F-step
**Script**: `test_interleave_fstep.py` | **Results**: `experiment7_results.jsonl` (25 runs)
**Question**: The paper updates A/lambda0 inside each E-step iteration. Does this help?
**Configs**: 5 configs varying interleaving (on/off) and A init (1e-4 vs 0.01)
**Finding**: Interleaving + A=1e-4 is critical. Without interleaving, A=1e-4 can't bootstrap.

### Full sweep (final)
**Script**: `run_sweep.py` | **Results**: `sweep_results.jsonl` (738 runs)
**Scope**: All 41 cells x 3 seeds x 6 configs. Mandatory: ip_selection='random', fix_Amp=True.
**Configs**: our_defaults, our+paper_inner, paper_config, paper_init, broad+interleave, broad+A01
**Summary**: `sweep_summary.txt`
**Finding**: broad+interleave is best (avg adj_r2=0.730, 36/41 explained_var > 0.8).

### Exp sigma_0 comparison
**Results**: `sigma0_exp_results.jsonl` (123 runs = 41 cells x 3 seeds)
**Config**: Same as broad+interleave but sigma_0 uses exp parameterization instead of direct.
**Finding**: +0.008 mean adj_r2 vs direct. Exp is now the default.

---

## Known confound: 64x64 n_train (FIXED 2026-03-31)

**All 64x64 experiments before this fix used n_train=2910 instead of 3160.**

The 64x64 dataset has the same train/val split as 108x108 (2910 train + 250 val
= 3160 total). The 108x108 sweep correctly used n_train=3160 (train+val combined),
but the 64x64 sweep script hardcoded n_train=2910 (train only). This means all
64x64 results trained on 250 fewer images than the 108x108 results, confounding
any resolution comparison.

**Affected results**: `sweep_64x64_results.jsonl` (all 246 runs: configs
`64_free_amp_no_intl` and `64_free_amp_no_intl_A01`). Also any prior 64x64 runs
in `experiments/` that used n_train=2910.

**Fix**: Changed `run_sweep_64x64.py` to use n_train=3160. Results after the fix
are in `sweep_64x64_results_ntrain3160.jsonl`.

---

## Diagnostic tools

| File | Purpose |
|------|---------|
| `plot_rf_diagnostic.py` | Plot STA + init RF + learned RF per cell. Reads sweep_results.jsonl. Usage: `python plot_rf_diagnostic.py --cells 0 5 39` |
| `rf_diagnostic.png` | Example output from plot_rf_diagnostic.py (all cells) |
| `bad_cells_rf_diagnostic.png` | Diagnostic for cells with poor performance |

---

## Key documents

| File | Content |
|------|---------|
| `INVESTIGATION_LOG.md` | Full lab notebook: 21 findings, compounding factors table |
| `METRICS_COMPARISON.md` | Metric definitions (adj_r2 vs explained_var) and paper evidence |
| `HANDOFF.md` | Session handoff with code changes, caveats, and next steps |

---

## Disposable files

| File | Reason |
|------|--------|
| `sweep.pid` | Stale PID from background sweep process |
