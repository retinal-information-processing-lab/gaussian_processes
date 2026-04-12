# Active Loop Investigation Findings (2026-04-12)

**Session scope**: Fix and optimize the active learning loop (`run_active_loop.py`).
Continuation from the 2026-04-09 session (HANDOFF.md).

---

## Task 1: Efficient rank-1 K_tilde column append (DONE — no speedup on 64x64)

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

**Full 400-iter benchmark (cell 8, seed 42, argmax, 64x64, M 50->450):**
- OLD (full recompute): 45.8 min
- NEW (column append): 45.7 min
- **Speedup: 1.00x** — no measurable difference

The extend step is a negligible fraction of per-iteration time. Training (E/M/F steps)
and utility evaluation dominate. The column-append optimization is correct but does not
help on 64x64 where individual kernel evaluations are cheap (~microseconds for 4096 px).

**DEFERRED**: Investigate whether the speedup materializes on 108x108 (11664 pixels,
~6x larger C matrix per kernel call) or at M>500. Low priority since the bottleneck
was actually the autograd leak (Task 2).

---

## Task 2: Autograd memory leak — root cause found and fixed (15x speedup)

**Commits**: `d771c90` (main fix), `ff9b6a7` (memory logging instrumentation)

### The bug

The active loop reached **19.9 GB GPU memory at M=419**, while a single fit at the same
M=419 with the same indices used **0.087 GB** (230x excess). After 50 EM iterations in
Phase 1 alone, **8,760 CUDA tensors** were alive (should be ~50).

### Root cause

The E-step reads `A = model.likelihood.A.squeeze()`, where `A` has `requires_grad=True`
(exp transform of `raw_A`). All Newton computations (`g_b = A * ...`, `G_b = A*A * ...`,
`V_b_new`, `m_b_new`) carried autograd computation graphs. These were stored in model
state via `update_variational_params(m_b_new, V_b_new)`. The next E-step read the stored
m_b/V_b, creating a LONGER graph that referenced the previous one. Over 500 Newton steps
per EM cycle x 50 EM iterations x 369 active iterations, the chained graphs accumulated
hundreds of thousands of nodes holding intermediate GPU tensors.

### The fix (4 changes)

1. **`eigenspace_model.py`**: `.detach()` in `update_variational_params()` — breaks the
   chain where m_b/V_b carry grad_fn into the next E-step. m_b and V_b are Newton-updated
   (closed-form), never gradient-descended; detaching is semantically correct.

2. **`eigenspace_estep.py`**: Wrap entire E-step body in `torch.no_grad()` — prevents
   building ~20 graph nodes per Newton step. Without this, the nodes are built and
   immediately discarded by the detach — pure waste.

3. **`eigenspace_training.py`**: Detach `A`/`lambda0` at all 5 read sites in the training
   loop + wrap `model(X_train)` posterior calls in `no_grad()` at 4 call sites. These
   feed E-step, metrics, and ELBO — none need backprop. The F-step and M-step read from
   `model.likelihood` directly inside their own closures, unaffected.

4. **`run_active_loop.py`**: Null `.grad` on deepcopied kernel/likelihood params after
   `copy.deepcopy()`. Deepcopy preserves `.grad` tensors from the previous LBFGS, which
   reference old computation graphs.

### Results

| Metric | Before fix | After fix |
|---|---|---|
| Phase 1 CUDA tensors | 8,760 | 53 |
| Tensor growth per active iter | ~550 | ~9 (model state only) |
| GPU memory at M=450 | 19.9 GB (OOM risk) | 0.122 GB |
| **400-iter wall time** | **45.7 min** | **3.0 min (15x)** |

Per-M-range speedup (cell 8, seed 42, argmax, 64x64):
```
M=[ 51,100]:  2.7x
M=[101,200]: 10.2x
M=[201,300]: 18.5x
M=[301,420]: 22.3x
```

The autograd graph construction and GC overhead was the dominant cost, not the kernel
evaluations. Training equivalence verified: test_r=0.6150 (exact match at M=50).

### Future-proofing note

The user plans to compute gradients of utility/prediction w.r.t. input images (x*) in
the future. All fixes are safe for this: m_b, V_b, A, lambda0 are constants w.r.t. x*
and should be detached. The utility code path (`acquisition.py`) is separate from the
training loop and unaffected.

---

## Task 3: Stuck-near-init diagnosis (root cause found, no fix yet)

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

- More EM iterations: A stays collapsed. The trap is structural (zero gradient), not
  just ES stopping too early. Re-verified after memory fix — same result.
- More data (M=100): A recovers to 0.043 (40x larger). But kernel stays at init because
  100 data points with 90% zeros is still too few to constrain 5+ kernel params.

### Initial training set comparison

All seeds have similar zero-response fraction (84-92%). The A collapse is NOT driven
by data composition differences — it's the stochastic dynamics of the first few
E/F-step iterations.

### Status: OPEN — no fix implemented

Possible mitigations not yet tested: A lower bound, different init strategy, interleaved
F-step (damped Newton) during Phase 1.

---

## Task 4: Beta explosion safeguard (DONE)

**Commit**: `eaa072e`

- Tightened `BETA_MAX` from 1.0 to 0.3 in `ArcCosineKernel`
- At beta=0.3: RF sigma = 23 px on 108x108 (diameter ~46 px), still generous
- The pathological cell 0 seed 0 random (beta=0.448 -> OOM) would now be clamped
- Added mask coverage warning when mask covers >50% of pixels
- The warning fires at beta ~0.19 on 64x64 (72% coverage), as observed during testing
