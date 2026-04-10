# Active Loop Investigation Findings (2026-04-10)

**Session scope**: Fix and optimize the active learning loop (`run_active_loop.py`).
Continuation from the 2026-04-09 session (HANDOFF.md).

---

## Task 1: Efficient rank-1 K_tilde column append (DONE)

**Commit**: `bb27094`

Replaced full O(M^2) kernel recomputation in `extend_model_with_new_point()` with O(M) column-append approach ported from `utils.py:add_one_img_to_kernel`.

Changes:
- Added `K_tilde` field to `DirectVariationalState` (stored after each eigenspace computation)
- Added `_precomputed_state` bypass to `DirectVGPModel.__init__`
- Rewrote `extend_model_with_new_point` to compute only new column of K_tilde

Verification (cell 8, seed 42, argmax, n_active=50, 64x64):
- Phase 1: bit-identical
- First 7 active iterations: identical selections
- Divergence at iter 8 from float32 near-tie (relative error ~7e-8 in K_tilde)
- n_b trajectory: identical
- Wall time: 39.5s -> 37.8s (modest at M=50-100; savings scale with M^2)

---

## Task 2: Memory profiling (DONE)

**Commit**: `ff9b6a7`

Added per-iteration `gpu_mem_gb` field to `results.jsonl` (gc.collect() + empty_cache() for clean floor).

**Finding: Memory leak appears FIXED by the column-append optimization + gc.**

Cell 0 seed 5 argmax on 108x108, 162+ iterations:
```
Linear fit: gpu_GB = 6.77 MB/iter + 0.160 GB (R^2 = 0.98)
Projected at iter 412: 2.95 GB
Original OOM was at:   22.9 GB after 412 iters (OLD code)
```

The memory growth is purely linear (expected from model expansion), not superlinear. The old code's `DirectVGPModel(kernel, likelihood, X_tilde_new, X_tilde_new, eigval_tol)` called `_compute_eigenspace_quantities()` which computed K_tilde (M^2 entries), K (another M^2 entries), and Kvec via three separate kernel calls with intermediate GPU caches. These caches accumulated across iterations. The column append avoids two of those three kernel calls.

---

## Task 3: Stuck-near-init diagnosis (DONE)

**Commit**: `452bc11`

### Root cause: A (gain parameter) collapses to ~0.0001 during Phase 1

Phase 1 output for cell 0 across seeds:
```
Seed  Label   test_r   beta     A       Amp    stopped_at
  0  HEALTHY  -0.087   0.121   0.0166   2.47   iter 49 (no ES)
  1    STUCK  -0.029   0.103   0.0001   1.02   iter 19 (ES fired)
  4  HEALTHY   0.294   0.084   0.1018   0.67   iter 49 (no ES)
```

When A ~ 0, firing rate is constant: f(x) = exp(0.0001 * lambda + lambda0) ~ exp(lambda0).
Gradient of likelihood w.r.t. kernel params vanishes -> M-step cannot learn.

### Interventions tested (cell 0 seed 1, 64x64)

| Intervention | A | beta | test_r | Verdict |
|---|---|---|---|---|
| Baseline (M=50, 50 iters) | 0.0001 | 0.103 | 0.054 | STUCK |
| 200 EM iters, no ES | 0.0003 | 0.105 | 0.012 | STILL STUCK |
| M=100 (more data) | 0.043 | 0.105 | 0.133 | A recovers, kernel stuck |

- More EM iterations: A stays collapsed. The trap is structural (zero gradient), not just ES stopping too early.
- More data (M=100): A recovers to 0.043 (40x larger). But kernel stays at init because 100 data points with 90% zeros is still too few to constrain 5+ kernel params.

### Initial training set comparison

All seeds have similar zero-response fraction (84-92%). The A collapse is NOT driven by data composition differences — it's the stochastic dynamics of the first few E/F-step iterations.

---

## Task 4: Beta explosion safeguard (DONE)

**Commit**: `eaa072e`

- Tightened `BETA_MAX` from 1.0 to 0.3 in `ArcCosineKernel`
- At beta=0.3: RF sigma = 23 px on 108x108 (diameter ~46 px), still generous
- The pathological cell 0 seed 0 random (beta=0.448 -> OOM) would now be clamped
- Added mask coverage warning when mask covers >50% of pixels
- The warning fires at beta ~0.19 on 64x64 (72% coverage), as observed during testing

---

## Summary of remaining issues

1. **Stuck-near-init (H2)**: Structural problem in Phase 1 optimization for cells with sparse responses. Possible mitigation: larger initial set (M=100), A lower bound, or interleaved F-step (damped Newton). None fully solve it yet.

2. **Memory growth**: Now linear and well-behaved (6.8 MB/iter). The original 22.9 GB OOM is likely fixed by the column-append optimization + gc.collect. Needs verification on a healthy-params run (not stuck-near-init) since the stuck model has lower memory footprint.
