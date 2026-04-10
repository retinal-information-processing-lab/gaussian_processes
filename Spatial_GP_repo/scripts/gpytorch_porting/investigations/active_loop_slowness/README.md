# Investigation: Active loop slowness, OOMs, and kernel parameter instability

**Date opened**: 2026-04-09
**Branch**: `pietro/utility_optimization`
**HEAD at investigation opening**: `0a625c1` (Add torch.no_grad() to argmax utility evaluation)
**Status**: OPEN — failure modes characterized, root causes not yet fixed

---

## TL;DR

A background batch of 10 cells x 7 seeds x 2 strategies (argmax + random) was launched and killed early (14 of 140 runs completed) after observing:

1. Runs were **5-7x slower than projected** (30-60 min/run vs an 8-10 min estimate).
2. **3 of 14 runs OOM'd** with CUDA out of memory during test-set prediction, well into the loop.
3. **6 of 14 runs produced near-zero test_r** despite completing normally.

The batch had only processed cell 0 and the beginning of cell 1. The problems are likely cell-specific but may not be unique to cell 0. Before running a large batch, the three distinct failure modes below need to be understood and fixed, and a quick audit across all 41 cells needs to tell us whether cell 0 is an outlier or representative.

All existing data is preserved at:
`results/active_loop/2026-04-08_first10cells_7seeds_n450/cell_00/seed_{0..6}/{argmax,random}/`

Each run directory has `results.jsonl` (per-iteration metrics including beta, rho, A, lambda0, Amp, sigma_0, test_r, n_b, wall_time_s), `run.log` (captured subprocess stdout/stderr, including tracebacks for failures), `config.json` (frozen config snapshot), and `checkpoints/iter_NNN.pt`.

---

## Context

### What the batch was trying to do
Simulated active learning loop: start with 50 inducing points, grow to 500 by adding one image at a time and retraining. Two strategies being compared: argmax (utility-guided) vs random (baseline). Phase 1 uses pivoted-Cholesky IP selection; Phase 2 adds images via `extend_model_with_new_point` (rank-1 extension) and runs 5 EM iterations per addition.

Launch command (killed at run 15/140):
```
python run_active_loop_batch.py \
    --cells 0 1 2 3 4 5 6 7 8 9 \
    --n-seeds 7 \
    --n-active 450 \
    --output-dir results/active_loop/2026-04-08_first10cells_7seeds_n450
```

### What triggered this investigation
We had previously launched a smaller test (cell 8, seed 42, M 50->300) in `results/active_loop/2026-04-08_cell8_seed42_M50_n250/` that completed cleanly. Based on that, we estimated ~78 min for a 450-iter argmax run on 108x108. The actual runs on cell 0 took 30-60 min per 450-iter run **and most of them failed in various ways**, suggesting cell 8 was atypical.

### Relevant recent changes
- `0a625c1` — Added `torch.no_grad()` to the argmax utility evaluation. This fixed a major autograd overhead (roughly 3-4x speedup at M=50-70), but was NOT the main cause of the slowness — see Finding 5 below.
- `fc27eb8` — Bumped `n_active_iterations` default to 250; added `assert M == n_train` inside the active loop. No known effect on these failures.

---

## Key findings

### Finding 1: There is no jitter added to K_tilde anywhere in the vargp_direct path.

Initial hypothesis was that a jitter term (e.g., `K_tilde + eps * I`) before the eigendecomposition was shifting every eigenvalue above the threshold, making the eigenspace filtering never activate. This was verified false.

**Checked**:
- `eigenspace_utils.py::eigendecompose_K_tilde` — calls `torch.linalg.eigh(K_tilde, UPLO='L')` directly, no modification.
- `eigenspace_model.py::_compute_eigenspace_quantities` (line ~110) — K_tilde = `kernel(X_tilde, X_tilde).to_dense()` with no post-processing.
- `kernels.py` — grep for `jitter|add_jitter|eye|.diagonal.add` returned empty.
- Cross-reference: a subagent exploration of the original `utils.py::varGP()` (the vargp_old reference implementation) confirmed that IT also passes `K_tilde` directly to `eigh` at 7 call sites (lines 425, 1543, 4477, 5171, 5435, 5594, 6061) with the **identical** threshold formula `max(eigvals.max() * EIGVAL_TOL, EIGVAL_TOL)` and the **identical** default value `EIGVAL_TOL = 1e-4`.
- GPyTorch's own `jitter_val` / `variational_cholesky_jitter` mechanism is only active on the `default_gpy` code path (through `VariationalStrategy`), not on vargp_direct. See `.claude/rules/jitter.md` for the detailed GPyTorch internals trace.

**Conclusion**: The eigenspace machinery is a faithful port of the original. The `eigval_tol = 1e-4` default is consistent across `default_params.json`, `_constants.py`, all YAML configs (`canonical`, `quick`, `massive_allcells_{48,64,108}`, `smoke_test_*`), all historical experiment configs (2026-03-20, 2026-03-24), and vargp_old. All 64x64 massive-sweep runs used 1e-4.

### Finding 2: The eigenvalue threshold DOES filter eigenvalues when the kernel is healthy.

We initially thought `n_b = M` always because we looked at runs where the kernel was in a pathological wide state (high beta -> smooth C -> every inducing point adds rank). On a healthy argmax run, the filter drops a substantial fraction of eigenvalues at high M.

Evidence from `cell_00/seed_0/argmax` (healthy run, test_r=0.647):
```
iter    M   n_b   ratio   beta_final   Amp_final
   0   50    50   1.000    0.121         2.47
 100  150   141   0.940    0.0712        6.17
 150  200   182   0.910    0.0654        6.19
 200  250   209   0.836    0.0619        6.19
 250  300   207   0.690    0.0581        6.19
 350  400   215   0.538    0.0562        6.19
 450  500   210   0.420    0.0539        6.20
```

n_b saturates around 210 while M grows to 500 — every new inducing point beyond iter ~250 contributes an eigenvalue just below threshold. For comparison, on the pathological `cell_00/seed_0/random` run, `n_b = M` all the way to the crash point because the drifted kernel (beta=0.45) is so smooth that inducing points are almost independent in kernel space — they all add rank, but it is rank in a useless basis.

This also explains why the earlier cell 8 timing probe (which had n_b ~ M) was misleading: cell 8 never triggers strong eigenvalue compression.

### Finding 3: Three distinct failure modes on cell 0

All 14 cell 0 runs from the killed batch:

| run | last iter | status | test_r  | beta_final | Amp_final | mode |
|---|---|---|---|---|---|---|
| seed_0/argmax | 450 | DONE | **0.6471** | 0.0539 | 6.196 | HEALTHY |
| seed_0/random | 435 | FAILED | — (OOM) | **0.4480** | **12.365** | **Beta explosion** |
| seed_1/argmax | 450 | DONE | 0.0538 | **0.1025** | **1.016** | Stuck-near-init |
| seed_1/random | 450 | DONE | 0.1790 | **0.1026** | **1.016** | Stuck-near-init |
| seed_2/argmax | 450 | DONE | 0.1059 | **0.1025** | **1.014** | Stuck-near-init |
| seed_2/random | 450 | DONE | 0.1436 | **0.1025** | **1.014** | Stuck-near-init |
| seed_3/argmax | 450 | DONE | 0.2310 | **0.1044** | **1.028** | Stuck-near-init |
| seed_3/random | 450 | DONE | 0.0611 | **0.1043** | **1.028** | Stuck-near-init |
| seed_4/argmax | 450 | DONE | **0.6540** | 0.0520 | 0.933 | HEALTHY |
| seed_4/random | 447 | FAILED | — (OOM) | 0.0457 | **0.244** | **Amp collapse** (or unexplained OOM) |
| seed_5/argmax | 412 | FAILED | — (OOM) | 0.0618 | 3.089 | **Unexplained OOM** (normal params) |
| seed_5/random | 450 | DONE | -0.0280 | **0.1036** | **1.034** | Stuck-near-init |
| seed_6/argmax | 450 | DONE | **0.6354** | 0.0491 | 1.407 | HEALTHY |
| seed_6/random | 450 | DONE | 0.4964 | 0.0323 | 1.413 | Moderate |

**Failure mode A: Beta explosion** (1 run — seed_0/random). Beta drifts upward from 0.121 -> 0.448; Amp climbs from 2.47 -> 12.37; test_r collapses from ~0.35 at iter 100 to <0 from iter 150 onward. Because `beta_code = 1 / (4 * beta_nat^2)` the RF mask radius scales inversely with beta_nat. At beta=0.448 the mask cutoff distance (d where alpha >= 1e-3) exceeds the image diagonal, so the mask covers **all 11664 pixels**. The kernel's internal C matrix (`alpha @ C_smooth @ alpha`) is then `11664 x 11664 = 519 MB in float32`. Over hundreds of active iterations this almost certainly contributes to the OOM. Full trajectory:
```
iter    M   beta     rho      A         lambda0    sigma_0   Amp      test_r
   0   50   0.121    0.072    0.0166   -3.684     0.9997   2.472    -0.087
  10   60   0.119    0.073    0.0130   -3.906     0.9997   4.103    -0.088
  50  100   0.094    0.060    0.0150   -2.285     0.9997   4.310    -0.058
 100  150   0.106    0.062    0.0214   -2.885     0.9994   4.400     0.352
 150  200   0.328    0.088    0.0036   -3.085     0.9994   4.870    -0.106   <- DRIFT
 200  250   0.363    0.068    0.0034   -2.880     0.9991   5.933    -0.112
 250  300   0.395    0.053    0.0034   -2.973     0.9988   6.870    -0.100
 300  350   0.435    0.038    0.0032   -2.850     0.9982   8.883    -0.088
 350  400   0.451    0.029    0.0029   -2.663     0.9967  11.862    -0.086
 400  450   0.448    0.039    0.0037   -3.565     0.9966  12.365    -0.009
 435  485   0.448    0.039    0.0038   -3.743     0.9966  12.365    -0.025   <- LAST LOGGED
```

**Failure mode B: Amp collapse** (1 run — seed_4/random). Amp shrinks to 0.244 by iter 447; beta stays at 0.046 (fine). test_r unknown because the run died before iter 450. This is NOT a beta-driven OOM — parameters are in healthy ranges — so the OOM has a different cause than mode A. Needs investigating. (Hypothesis: Amp is the gain on log-firing-rate `g = A * lambda + lambda0`. Very small Amp collapses the output range, which might trigger other numerical issues.)

**Failure mode C: Unexplained OOM with normal parameters** (1 run — seed_5/argmax). beta=0.062, Amp=3.089 — both healthy. Crashed at iter 412/450. This does NOT fit the beta-explosion hypothesis at all. Suggests there is a separate memory issue that accumulates independently of kernel parameters. Could be GPyTorch LazyTensor cache retention, could be checkpoint save leak, could be utility evaluation graph retention in a branch we did not test. **The highest-priority unexplained failure.**

**Failure mode D: Stuck-near-init** (6 runs — seeds 1/2/3 both strategies + seed 5 random). These 6 runs converged to suspiciously identical parameters: beta in [0.1025, 0.1044], Amp in [1.014, 1.034]. The initial values are close to beta=0.121, Amp=2.47 (phase 1 output), so "stuck" is slightly generous — beta moved a little, Amp moved a lot (2.47 -> 1.01). But the clustering is way too tight to be coincidence. test_r stays near zero (range 0.054 to 0.231). The identical-looking final values across 6 runs with different seeds suggest the model is stuck at some structural minimum that does not depend on the data ordering. **Possibly a bounds clamp? Possibly a zero gradient region? Possibly a degenerate RF center initialization for cell 0?**

### Finding 4: OOMs happen during test-set prediction, not during the main training loop

All three crashes' tracebacks end in:
```
File ".../eigenspace_training.py", line 609, in predict_eigenspace
    K_test = kernel(X_test, X_tilde).to_dense()  # (N_test, M)
...
File ".../kernels.py", line 540, in _compute_C_matrix
    C = (C + C.T) / 2
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 520.00 MiB. GPU 0 has a total capacity of 23.64 GiB of which 504.19 MiB is free. Including non-PyTorch memory, this process has 22.90 GiB memory in use.
```

At the crash point, **22.9 GB** of GPU memory is already in use. This is way more than a single forward pass should need, even with a full-image C matrix (~520 MB). Something is holding many copies or large LazyTensor caches. One iteration of `predict_eigenspace` should NOT need 22 GB. This strongly suggests cached lazy kernel evaluations accumulating across the 400+ active iterations.

Full crash log for mode-A (beta explosion): `results/active_loop/2026-04-08_first10cells_7seeds_n450/cell_00/seed_0/random/run.log`.
Full crash log for mode-C (unexplained): `results/active_loop/2026-04-08_first10cells_7seeds_n450/cell_00/seed_5/argmax/run.log`.

### Finding 5: The no_grad fix was real but NOT the dominant speed cost

Earlier this session we added `torch.no_grad()` to `compute_utility_and_select` in `run_active_loop.py`. Before the fix, per-iteration time at M=50-70 was 1.2-2.7s; after, it dropped to 0.4-0.8s (~3-4x). That fix is correct and should be kept.

BUT the projected speedup from that fix alone (from the 18.2-min actual cell 8 run pre-fix) did NOT materialize at scale in the real batch. Per-run times on cell 0 were 30-60 min — substantially worse than what either the healthy cell 8 extrapolation OR the post-fix probe predicted. The kernel-drift and stuck-near-init modes likely drive the slowness: a model stuck at beta=0.1025, Amp=1.01 probably has much larger intermediate kernel computations than a healthy one at beta=0.054.

**Implication**: per-iteration wall time is dominated by kernel evaluation cost, which is dominated by mask size, which is dominated by beta. Any speed analysis must account for cell-specific kernel convergence.

### Finding 6: Cell 8 is the known-healthy reference cell

`results/active_loop/2026-04-08_cell8_seed42_M50_n250/argmax/results.jsonl` (and the matching random run) completed cleanly at n_active=250 with test_r 0.62-0.67 and no parameter drift. Kernel parameters stayed in the beta ~ 0.06-0.09 range throughout. Use cell 8 as the reference comparison point for any future change.

---

## What was ruled out (don't redo)

1. **Jitter added to K_tilde before eigendecomposition** (Finding 1). Not happening anywhere. Do not go hunting for it again.
2. **Wrong `eigval_tol` default** (Finding 1). 1e-4 everywhere, identical to vargp_old. Do not re-check the configs.
3. **The eigenvalue threshold is broken** (Finding 2). It is working correctly. Cell 8 just doesn't need much compression; the healthy cell 0 argmax shows strong compression.
4. **The `no_grad` was the dominant cost** (Finding 5). It was a real fix and is committed (`0a625c1`) — do not revert — but it does NOT explain the 5-7x slowness vs projection.
5. **A straightforward memory leak in checkpoint saving**. The checkpoints are small (~500 KB at M=500) and cell 8 ran 250 iters with no issue.
6. **The `debugging.md` entry about `eigval_tol=1e-4` being "too strict"**. Different symptom ("test_r degrades after iteration 20"). Not our issue.

---

## Reproduction recipes

### A. Inspect existing data without re-running (fastest path)

All the failure-mode data is already on disk. A new session can analyze it directly:

```bash
cd /home/idv-eqs8-pza/IDV_code/ClosedLoopProject/gaussian_processes/Spatial_GP_repo/scripts/gpytorch_porting
ls results/active_loop/2026-04-08_first10cells_7seeds_n450/cell_00/
```

For any run, the full per-iteration trajectory is in `results.jsonl`. Each row has `iteration, n_training, n_b, selected_idx, spike_count, utility, train_loss, train_log_lik, train_kl, train_r, final_A, final_lambda0, final_beta, final_rho, final_sigma_0, final_eps_0x, final_eps_0y, final_Amp, test_r, adjusted_r2, explained_var, reliability, wall_time_s, timestamp`. Checkpoints are in `checkpoints/iter_NNN.pt` (one per iteration for most runs; FAILED runs stop at the crash iteration). `run.log` has the full subprocess stdout/stderr including the crash traceback.

### B. Reproduce a specific failure from scratch

Each of these is a single-run command that bypasses the batch script. Each takes 30-60 min on GPU at n_active=450. The n_active=200 variant is a faster reproduction that at least reaches the drift onset for mode A.

**Mode A (beta explosion, cell 0 seed 0 random):**
```bash
# Full reproduction (~45 min, crashes around iter 435)
python run_active_loop.py \
    --cell 0 --seed 0 --strategy random \
    --n-active 450 \
    --output-dir /tmp/repro_modeA_full

# Fast reproduction of drift onset only (~15-20 min, stops well before crash)
# The drift starts around iter 150 in the original run.
python run_active_loop.py \
    --cell 0 --seed 0 --strategy random \
    --n-active 200 \
    --output-dir /tmp/repro_modeA_fast
```
Check: `tail results.jsonl` and look for `final_beta > 0.3` around iter 150-200.

**Mode B (Amp collapse, cell 0 seed 4 random):**
```bash
python run_active_loop.py \
    --cell 0 --seed 4 --strategy random \
    --n-active 450 \
    --output-dir /tmp/repro_modeB
```
Check: `final_Amp` decreasing over iterations; expect OOM near iter 447.

**Mode C (unexplained OOM, cell 0 seed 5 argmax):**
```bash
python run_active_loop.py \
    --cell 0 --seed 5 --strategy argmax \
    --n-active 450 \
    --output-dir /tmp/repro_modeC
```
Check: parameters stay healthy (beta~0.06, Amp~3), but OOM near iter 412. **This is the most important one to understand** — start here.

**Mode D (stuck-near-init, cell 0 seed 1 argmax):**
```bash
python run_active_loop.py \
    --cell 0 --seed 1 --strategy argmax \
    --n-active 450 \
    --output-dir /tmp/repro_modeD
```
Check: final beta ≈ 0.1025, final Amp ≈ 1.016, test_r ~ 0.05. To verify the degeneracy: compare `final_beta`/`final_Amp` at iter 50, 100, 200 — they should all be ≈ same cluster values.

### C. Healthy reference reproductions

**Reference healthy cell 0 (seed 0, argmax):**
```bash
python run_active_loop.py \
    --cell 0 --seed 0 --strategy argmax \
    --n-active 450 \
    --output-dir /tmp/repro_healthy_cell0
```
Expected: test_r ≈ 0.647, beta ≈ 0.054, Amp ≈ 6.2. The SAME cell/seed pair where random fails catastrophically. Good A/B comparison.

**Reference healthy cell 8 (seed 42, argmax):**
```bash
python run_active_loop.py \
    --cell 8 --seed 42 --strategy argmax \
    --n-active 250 \
    --output-dir /tmp/repro_healthy_cell8
```
Expected: test_r ≈ 0.624, completes in ~18 min (pre-no_grad-fix) or ~6-7 min (post-fix). No parameter drift. This is the main sanity check — if cell 8 ever starts failing, the investigation scope has expanded.

### D. Quick cell-level screening (to find out if cell 0 is special)

Before running a large batch, audit all 41 cells with a short n_active to see which failure modes appear where. This is not implemented yet; the new session should probably write it. Suggested shape:

```
for cell in 0..40:
    python run_active_loop.py --cell $cell --seed 0 --strategy random --n-active 100 \
        --output-dir screen/cell_${cell}
```

Then check the distribution of (final_beta, final_Amp, test_r) across cells. If only cell 0 shows stuck-near-init and drift, it is a cell-specific RF init issue. If many cells show it, there is a systemic training-stability bug.

---

## Hypotheses to test next (ranked)

### H1 (highest priority): Mode C — unexplained OOM with healthy parameters
cell 0 seed 5 argmax crashed at iter 412 with beta=0.062 and Amp=3.089. There is no parameter-driven explanation — the C matrix should be ~2 MB at that beta. Something else is accumulating 22 GB of GPU memory over 412 iterations. Candidates:
- **GPyTorch LazyTensor cache retention**: `predict_eigenspace` uses `kernel(X_test, X_tilde).to_dense()`. LazyTensors have an internal memoize cache (`gpytorch.utils.memoize`) that may not be cleared between iterations. **Check**: add `gpytorch.settings.memoize_cache_size(0)` at the top of the loop, or explicitly clear the cache between iterations.
- **Checkpoint tensor retention**: `save_eigenspace_checkpoint` is called every iteration; does it inadvertently hold GPU references?
- **Rank-1 extension leak**: `extend_model_with_new_point` creates a new `DirectVGPModel` each iteration; does the old model get garbage-collected? Does the kernel copy?
- **Per-iteration utility call**: we added `torch.no_grad()` but the result dict (`result['utility']`, `result['mu_g']`) might still retain GPU-side references across iterations if not properly detached or released.

Proposed diagnostic: instrument `run_active_loop.py` to log `torch.cuda.memory_allocated() / 1024**3` each iteration AND call `gc.collect(); torch.cuda.empty_cache()` every N iterations. Rerun cell 0 seed 5 argmax and plot memory over iterations. If memory grows monotonically, it is a leak; if it oscillates, it is a peak-usage issue.

### H2: Mode D — stuck-near-init
6 of 14 cell 0 runs converged to (beta, Amp) ≈ (0.1025, 1.016) regardless of seed or strategy. Candidates:
- **Bad RF center init** for cell 0: `compute_rf_center_from_sta()` uses a smoothed argmax over the STA. Cell 0 may have a spurious STA peak (cf. the documented `investigations/sta_edge_artifact/` issue for 108x108 and cells 0, 5, 6, 15, 22, 39). **Check**: `cell 0` is explicitly listed as one of the 6 cells affected by the STA edge artifact in 108x108 data. This is very likely the underlying cause.
- **Zero gradient at init**: if the RF center is outside any real signal, the M-step gradient w.r.t. beta is zero and it stays put.
- **Clamp hitting**: check `kernels.py::params_in_bounds()` and `clamp_hyperparameters()`. The values 0.1025 and 1.016 don't look like standard bound values, but worth checking.

Proposed diagnostic: re-run cell 0 with `rf_init='center'` or `rf_init='ground_truth'` (see `datasets/rf_centers_ground_truth.npz` per CLAUDE.md) instead of the default STA-based init. If the stuck-near-init mode disappears, we've identified cell 0 as an STA-edge victim. The 64x64 center-cropped dataset is another workaround cited in CLAUDE.md: "Center crops (48x48, 64x64) avoid this because edge pixels are excluded from the STA computation."

### H3: Mode A — beta explosion in M-step
Cell 0 seed 0 random: beta drifts 0.121 -> 0.448. Candidates:
- **Missing or too-loose upper bound on beta** in `kernels.py::params_in_bounds()`. Check what the actual bounds are.
- **LBFGS line search overshooting**: the M-step uses LBFGS with strong Wolfe. If the gradient is dominated by a single direction, a long step can push beta into a pathological region. The `params_in_bounds()` guard rejects out-of-bounds trial steps, but if the bound is wider than the "safe" region, drift can still happen.
- **Random points providing no locality signal**: argmax keeps picking points that constrain the kernel, whereas random allows beta to drift because the likelihood landscape is flat w.r.t. beta when the data is uninformative. This is a deeper scientific issue.

Proposed diagnostic: print beta at each M-step iteration (not just per-active-iteration), and check `kernels.py::params_in_bounds()` for the beta bounds.

### H4: Mode B — Amp collapse in F-step
Cell 0 seed 4 random: Amp drops to 0.244. Amp controls the log-firing-rate gain. Check `likelihoods.py::params_in_bounds()` / `clamp_params()` for Amp bounds, and check the F-step (`eigenspace_fstep.py::damped_newton_update_A_lambda0` or the LBFGS closure) for Amp trajectory over iterations.

---

## Files and locations relevant to this investigation

### Where the data is
```
results/active_loop/2026-04-08_first10cells_7seeds_n450/   <- killed batch (14 runs)
    cell_00/seed_{0..6}/{argmax,random}/
        results.jsonl          <- per-iter metrics
        run.log                <- stdout/stderr (tracebacks for failures)
        config.json            <- frozen config + git commit + integrity tags
        checkpoints/iter_NNN.pt
results/active_loop/2026-04-08_cell8_seed42_M50_n250/      <- known-healthy reference
    {argmax,random}/
```

Batch-level log: `results/active_loop/batch_first10cells.log` (one line per run: DONE/FAILED, final test_r, wall time).

### Source files to inspect for the M/F-step bounds and training stability

```
run_active_loop.py                       # main active loop, compute_utility_and_select
eigenspace_training.py                   # train_eigenspace(), predict_eigenspace()
    - predict_eigenspace (~line 609): where the OOMs crash
eigenspace_model.py                      # DirectVGPModel, _compute_eigenspace_quantities
eigenspace_mstep.py                      # M-step LBFGS closure for kernel params
eigenspace_fstep.py                      # F-step (A, lambda0); damped Newton variant
eigenspace_estep.py                      # E-step Newton update for (m, V)
rank1_update.py                          # extend_model_with_new_point
kernels.py                               # ArcCosineKernel, params_in_bounds, clamp_hyperparameters
likelihoods.py                           # PoissonLikelihood, params_in_bounds, clamp_params
utils.py                                 # compute_rf_center_from_sta (smoothed argmax)
```

### Relevant config and documentation

```
default_params.json                      # all defaults. Model: eigval_tol=1e-4, jitter=1e-4,
                                         # cholesky_max_tries=3. Kernel: sigma_0, Amp, beta,
                                         # rho init values. Active learning: phase2 n_estep=10,
                                         # n_fstep=10, n_mstep=10.
.claude/rules/jitter.md                  # jitter mechanism details (GPyTorch internals)
.claude/rules/debugging.md               # known issues index (don't trust the eigval_tol entry
                                         # — it's a different symptom)
.claude/rules/critical_short_rules.md    # parameter discipline, no_grad context, etc.
CLAUDE.md "STA edge artifact" section    # lists cells 0, 5, 6, 15, 22, 39 as affected on 108x108
datasets/rf_centers_ground_truth.npz     # ground-truth RF centers for all 41 cells (workaround
                                         # for STA edge artifact) — see datasets/README.md
```

### Dataset
The batch used `PNAS_108x108_original.npz` (the default in `default_params.json`). Cell 0 is explicitly listed in CLAUDE.md as one of the 6 cells affected by the STA edge artifact on 108x108 images. This is relevant to Hypothesis H2.

---

## What NOT to do

1. **Do not revert the `torch.no_grad()` fix** (commit `0a625c1`). It is correct and removes a real ~3-4x overhead for the argmax-evaluation step. There is a comment next to it saying to remove it if switching to gradient-based utility optimization — respect that.
2. **Do not touch `eigval_tol`.** It matches vargp_old exactly. It is NOT the bug.
3. **Do not add jitter to K_tilde.** There is no jitter there by design; the original vargp_old does not have it either. Adding jitter would shift eigenvalues and change the threshold behavior.
4. **Do not try to "fix" the M-step bounds by tightening them without understanding why beta drifts in the first place.** A tight clamp would just mask the underlying stability issue.
5. **Do not delete any of the existing run directories under `results/active_loop/2026-04-08_first10cells_7seeds_n450/`.** They are the entire failure dataset and need to stay around for the new session.
6. **Do not launch a big batch before the investigation completes.** Sequential single-run reproductions with logging are the right tool; a parallel batch will re-burn GPU time on the same failures.

---

## Handoff notes for the new session

- Read this file first, then skim `run_active_loop.py`, `eigenspace_training.py:predict_eigenspace`, and `rank1_update.py`.
- The **most tractable** bug to start with is probably H2 (stuck-near-init, cell 0 STA edge artifact) because it has a clear candidate cause and a cheap test (swap to `rf_init='ground_truth'`).
- The **most important** bug is H1 (unexplained OOM with healthy parameters) because it affects the whole infrastructure, not just cell 0. Start here if scoping for impact.
- H3 and H4 are parameter-stability issues that may share a common cause with H2 (if cell 0 has a bad RF init, the F/M-step gradient landscape is malformed, which can produce drift OR collapse depending on which direction the local gradient happens to point).
- The killed batch's resume logic (via `run_active_loop_batch.py`'s line-count check) will skip any run with a complete `results.jsonl` — so if you re-launch the batch after fixing the bugs, it will only re-run the failed/missing ones. **Remember to delete** the partial `results.jsonl` and `checkpoints/` directories for the 3 FAILED runs before re-launching, or the resume logic will re-run them regardless (which is what we want here).
