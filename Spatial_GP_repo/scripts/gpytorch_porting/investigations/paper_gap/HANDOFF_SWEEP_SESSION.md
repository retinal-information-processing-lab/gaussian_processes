# Investigation: Parameter Sweep — Amp, Interleaving, Resolution, A_init

**Branch**: `pietro/investigate-paper-gap`
**Date**: 2026-04-02
**Status**: Continuing
**Location**: `investigations/paper_gap/`
**Worktree**: `/home/idv-eqs8-pza/IDV_code/ClosedLoopProject/gaussian_processes/Spatial_GP_repo/scripts/gpytorch_porting_paper_gap/`

---

## Problem Statement

After closing Gap A (vargp_direct == vargp_old) and identifying that `broad+interleave` is the best 108x108 config (36/41 cells > 0.8 explained_var), we needed to:
1. Test on 64x64 images (cheaper, avoids STA edge artifact for 6 cells)
2. Isolate which factors matter: Amp (free vs fixed), interleaving, A_init, resolution
3. Build a complete parameter grid for future model tuning decisions

## What Was Tried

### 64x64 sweeps with corrected n_train
- **What**: Ran 4 configs on 64x64, 41 cells x 3 seeds each, with n_train=3160 (matching 108x108)
- **Result**: See summary table below
- **Verdict**: Complete, clean data

### Timeout reruns
- **What**: 12 cells timed out (>600s) in the `64_intl_freeAmp_n3160` config due to GPU contention with 3 workers. Reran them sequentially with 1800s timeout.
- **Result**: All 12 completed successfully (167-597s each). Full 123/123 runs now available.

## Key Findings

### Complete Results Table

**108x108 (all fixed Amp, n=3160, exp sigma_0):**

| Config | Amp | Intl | A_init | beta | E/M/outer | test_r | expl_var | >0.8 | s/cell |
|--------|-----|------|--------|------|-----------|--------|----------|------|--------|
| our_defaults | fixed | no | 0.01 | 0.1 | 10/10/50 | 0.776 | 0.833 | 30/41 | 6 |
| our+paper_inner | fixed | no | 0.01 | 0.1 | 50/20/80 | 0.790 | 0.847 | 31/41 | 57 |
| paper_init | fixed | no | 1e-4 | 0.0452 | 50/20/80 | 0.806 | 0.865 | 32/41 | 64 |
| paper+interleave | fixed | yes | 1e-4 | 0.0452 | 50/20/80 | 0.818 | 0.878 | 33/41 | 129 |
| **broad+interleave** | **fixed** | **yes** | **1e-4** | **0.1** | **50/20/80** | **0.829** | **0.889** | **36/41** | **145** |
| broad+A01+intl | fixed | yes | 0.01 | 0.1 | 50/20/80 | 0.690 | 0.744 | 27/41 | 116 |

**64x64 (n=3160, exp sigma_0):**

| Config | Amp | Intl | A_init | beta | E/M/outer | test_r | expl_var | >0.8 | s/cell |
|--------|-----|------|--------|------|-----------|--------|----------|------|--------|
| 64_A01_free_no_intl_n3160 | free | no | 0.01 | 0.1 | 10/10/50 | 0.827 | 0.888 | 34/41 | 91 |
| **64_intl_fixAmp_n3160** | **fixed** | **yes** | **1e-4** | **0.1** | **50/20/80** | **0.838** | **0.899** | **36/41** | **519** |
| 64_intl_freeAmp_n3160 | free | yes | 1e-4 | 0.1 | 50/20/80 | 0.835 | 0.897 | 36/41 | 482 |

**NOT RUN (gaps in the grid):**
- 108x108 with free Amp (any combination) -- `run_sweep.py` hardcodes `fix_Amp=True` on line 125
- 64x64 with fixed Amp, no interleaving
- 64x64 with free Amp, A_init=1e-4, no interleaving (only n=2910 version exists, confounded)

### Findings (numbered)

1. CONFIRMED: **Interleaving is the dominant factor**. On 64x64: +0.011 test_r (0.827 -> 0.838). On 108x108: +0.053 (0.776 -> 0.829). Costs 5-6x compute time.

2. CONFIRMED: **Free vs fixed Amp makes almost no difference** when interleaving is on. 64x64: 0.835 (free) vs 0.838 (fixed). Both hit 36/41 cells > 0.8.

3. CONFIRMED: **A_init=0.01 + interleaving is catastrophic on 108x108** (0.690, worst config). Works fine without interleaving. The interleaved damped Newton overshoots when A starts too high.

4. CONFIRMED: **64x64 matches or slightly beats 108x108** when algorithm is identical (0.838 vs 0.829). Resolution reduction from 108->64 is not harmful. Likely because: fewer noisy pixels, no STA edge artifact, faster fitting.

5. CONFIRMED: **n_train=2910 vs 3160 confound was negligible**. Comparing `64_free_amp_no_intl_A01` (n=2910) vs `64_A01_free_no_intl_n3160` (n=3160): 0.828 vs 0.827 test_r. Nearly identical.

6. CONFIRMED: **Paper's 36/41 > 0.8 metric is explained_var** (mean_accuracy / reliability), not adjusted_r2. Our best configs match this number exactly.

## Gotchas and Tricky Details

### 1. `run_sweep.py` hardcodes `fix_Amp=True` (line 125)
The 108x108 sweep script sets `config['fix_Amp'] = True` OUTSIDE the per-config dict, overriding any per-config setting. This means ALL 108x108 results have fixed Amp regardless of what the config dict says. The `run_sweep_64x64.py` script reads `fix_Amp` from the config dict (line 97: `config['fix_Amp'] = cfg['fix_Amp']`), so 64x64 configs can actually vary Amp.

### 2. `run_sweep.py` also hardcodes `ip_selection='random'` and `early_stop=False`
Lines 124-126 of `run_sweep.py`. Same pattern: set outside per-config dicts. The 64x64 script does the same (lines 95-96) but reads `fix_Amp` from config.

### 3. Iteration counts differ between non-interleaved configs
`our_defaults` and `64_A01_free_no_intl_n3160` use 10/10/50 (E/M/outer). All interleaved configs use 50/20/80 (paper schedule). This means comparisons between interleaved and non-interleaved also confound iteration count. The `our+paper_inner` config (50/20/80, no interleaving) partially controls for this but is 108x108 only.

### 4. `run_sweep_64x64.py` currently has only `64_intl_freeAmp_n3160` in CONFIGS
The CONFIGS list was edited between runs. To rerun other configs, you need to edit it back. The resume logic skips already-completed (config_name, cell, seed) tuples, so adding old config dicts won't re-run completed cells.

### 5. Timing is confounded by parallelization
The 64x64 interleaved runs (~519s/cell with 3 workers) are much slower than 108x108 interleaved (~145s/cell sequential). This is GPU contention, not algorithmic. Single-cell timing for 64x64 interleaved is ~130s.

### 6. Result files structure
- `sweep_results.jsonl`: ALL 108x108 results (738 runs, 6 configs)
- `sweep_64x64_results.jsonl`: OLD 64x64 results with n_train=2910 confound (246 runs, 2 configs). Keep for reference only.
- `sweep_64x64_results_ntrain3160.jsonl`: CLEAN 64x64 results (369 runs, 3 configs). This is the authoritative 64x64 file.

### 7. Key metric names in JSONL
- `test_r` (Pearson r on test set)
- `explained_var` (test_r^2 / reliability, per Goldin et al.)
- `adjusted_r2` (test_r^2, adjusted for chance)
- `train_time` (seconds)
- `final_Amp` (1.0 = fixed, >1.0 = free was active)
- `config_name` (matches tables above)

## Why This Was Stopped

Context ran out. The sweep data is complete and the main findings are clear. Continuing to explore remaining grid cells and decide on final model configuration.

## Things Noticed But Not Acted Upon

1. The `our_defaults` config uses 10/10/50 iterations while all interleaved configs use 50/20/80. A clean comparison would need a 50/20/80 non-interleaved config on 64x64 (only `our+paper_inner` exists, on 108x108).

2. The 108x108 free Amp grid is completely empty. If free Amp matters more on 108x108 (where the input space is larger), we'd miss it.

3. `64_intl_freeAmp_n3160` final_Amp range is 0.76-6.87, much tighter than non-interleaved free Amp (1.97-22.43). Interleaving constrains Amp more.

4. Cells 0, 1, 2 consistently time out or run slowest across all interleaved configs. These are known difficult cells (STA edge artifact on 108x108; on 64x64 they still take longer).

## Uncommitted Changes

```
modified:   .claude/CLAUDE.md               # Added PNAS datasets table + n_train warning
modified:   investigations/paper_gap/run_sweep.py  # Added notes to config dicts
deleted:    investigations/paper_gap/sweep.pid      # Stale PID file
modified:   run_single_mode.py              # Added 'gpu' field to result dict

Untracked:
  investigations/paper_gap/README.md
  investigations/paper_gap/run_sweep_64x64.py
  investigations/paper_gap/sweep_64x64_results.jsonl          # n=2910 confounded
  investigations/paper_gap/sweep_64x64_results_ntrain3160.jsonl  # clean results
```

Plus log files (`.log`) that don't need committing.

## Files Created

| File | Purpose | Keep/Delete |
|------|---------|-------------|
| `investigations/paper_gap/run_sweep_64x64.py` | Parallelized 64x64 sweep script | Keep |
| `investigations/paper_gap/README.md` | Documentation of all experiments | Keep |
| `investigations/paper_gap/sweep_64x64_results.jsonl` | Old 64x64 results (n=2910 confound) | Keep (reference) |
| `investigations/paper_gap/sweep_64x64_results_ntrain3160.jsonl` | Clean 64x64 results (369 runs) | Keep |
| `investigations/paper_gap/sweep_64x64_freeAmp_intl.log` | Log from freeAmp sweep | Delete OK |
| `investigations/paper_gap/sweep_64x64_intl.log` | Log from fixAmp sweep | Delete OK |
| `investigations/paper_gap/sweep_64x64.log` | Log from early 64x64 sweep | Delete OK |
| `investigations/paper_gap/sweep_64x64_n3160.log` | Log from n3160 sweep | Delete OK |
| `investigations/paper_gap/sweep_64x64_timeout_rerun.log` | Log from timeout reruns | Delete OK |

## If Someone Revisits This

**Most promising next steps:**
1. Run the missing grid cells: 108x108 with free Amp (requires editing `run_sweep.py` line 125 to read from config dict instead of hardcoding `True`). Also 64x64 fixed Amp no-interleaving.
2. Run 64x64 non-interleaved with 50/20/80 iterations to isolate interleaving effect from iteration count.
3. Decide on final production config: `64_intl_fixAmp` is best but 5-6x slower. `64_A01_free_no_intl` is 91% as good at 1/6 the compute.

**What NOT to try:**
- A_init=0.01 with interleaving: catastrophic on 108x108, no reason to expect better elsewhere.
- n_train=2910 vs 3160 comparison: already confirmed negligible difference.

**Tricky things to remember:**
- `run_sweep.py` line 125 hardcodes `fix_Amp=True` for ALL configs. Edit this line if you want free Amp on 108x108.
- `run_sweep_64x64.py` CONFIGS list changes between runs. Check what's currently in it before launching.
- Interleaved configs use 50/20/80 iterations; non-interleaved use 10/10/50. This is a confound in direct comparisons.

---

## Continuation Prompt

```
I'm continuing the paper gap parameter sweep investigation.

Branch: pietro/investigate-paper-gap
Worktree: /home/idv-eqs8-pza/IDV_code/ClosedLoopProject/gaussian_processes/Spatial_GP_repo/scripts/gpytorch_porting_paper_gap/

Read the handoff first:
  investigations/paper_gap/HANDOFF_SWEEP_SESSION.md

Key context:
- We ran 10 configs across 108x108 and 64x64 (1230+ total runs)
- Best config: 64_intl_fixAmp_n3160 (test_r=0.838, 36/41 > 0.8 expl_var)
- Free vs fixed Amp: negligible difference
- Interleaving is the dominant factor but costs 5-6x compute
- GOTCHA: run_sweep.py line 125 hardcodes fix_Amp=True for ALL 108x108 configs
- GOTCHA: non-interleaved configs use 10/10/50 iters, interleaved use 50/20/80
- There are uncommitted changes -- check git status and git branch before starting

Results files:
  investigations/paper_gap/sweep_results.jsonl (108x108, 738 runs)
  investigations/paper_gap/sweep_64x64_results_ntrain3160.jsonl (64x64, 369 runs)
```
