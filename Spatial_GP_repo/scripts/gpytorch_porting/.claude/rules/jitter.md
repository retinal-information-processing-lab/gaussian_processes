# Jitter & Cholesky Stability in GPyTorch

**Purpose**: Documents how jitter flows through GPyTorch's Cholesky pipeline and how we configure it. Read this before modifying anything jitter-related.

---

## Architecture: Two Jitter Layers

### Layer 1 (pre-Cholesky): `jitter_val`

GPyTorch's `VariationalStrategy.forward()` adds jitter to K_uu before Cholesky:

```
variational_strategy.py:196
    induc_induc_covar = full_covar[:M, :M].add_jitter(self.jitter_val)
```

- `jitter_val` = our `model.jitter` (default 1e-4), passed via `gpy_model.py:63`
- Also adds jitter to K_XX at lines 226/231 (predictive covariance)
- If `jitter_val` is None, defaults to `gpytorch.settings.variational_cholesky_jitter` (1e-4 for float32, 1e-6 for float64)

**Our model's `forward()` does NOT add jitter** — GPyTorch handles it internally.

### Layer 2 (retry): `psd_safe_cholesky`

If Cholesky fails after Layer 1, GPyTorch retries with escalating jitter:

```
variational_strategy.py:104
    L = psd_safe_cholesky(to_dense(induc_induc_covar).type(torch.double))
                                                           ^^^^^^^^^^^^
                                                  K_uu promoted to float64
```

```
linear_operator/utils/cholesky.py:12-47
    for i in range(max_tries):
        jitter_new = jitter * (10**i)    # exponential schedule
        add jitter_new to diagonal
        retry Cholesky
```

We override two settings via context managers in `gpy_training.py`:

| Setting | GPyTorch default | Our override | Effect |
|---------|-----------------|--------------|--------|
| `cholesky_jitter` (float64) | 1e-8 | `model.jitter` (1e-4) | Retry starts at our jitter, not 1e-8 |
| `cholesky_max_tries` | 3 | configurable (default 3) | Number of retry decades |

With jitter=1e-4 and max_tries=3, retries add: 1e-4, 1e-3, 1e-2.

---

## Config Wiring

```
default_params.json          gpy_model.py              gpy_training.py
  model.jitter=1e-4  ──────> jitter_val  ──────────>  Layer 1 (GPyTorch internal)
                       └───>  cholesky_jitter override ──> Layer 2 retry start
  model.cholesky_max_tries=3 ──────────────────────>  Layer 2 retry count
```

CLI: `--jitter 1e-4`, `--cholesky-max-tries 3`
YAML: `numerical.jitter`, `numerical.cholesky_max_tries`

---

## Why Float64 Promotion?

GPyTorch promotes K_uu to float64 before Cholesky (`_linalg_dtype_cholesky = torch.double`). This is **intentional and beneficial**: Cholesky involves sequential subtractions that can cause catastrophic cancellation in float32. Float64 gives ~15 digits vs ~7, preventing the subtraction from producing negative diagonal elements. Cost is negligible for M=50-200.

---

## GPyTorch Source Locations

| What | File | Lines |
|------|------|-------|
| `jitter_val` property | `gpytorch/variational/_variational_strategy.py` | 98-102 |
| `.add_jitter(jitter_val)` on K_uu | `gpytorch/variational/variational_strategy.py` | 196 |
| `_cholesky_factor` — calls psd_safe_cholesky | `gpytorch/variational/variational_strategy.py` | 102-105 |
| Float64 promotion setting | `linear_operator/settings.py` | 189-190 |
| `cholesky_jitter` defaults (1e-6 f32, 1e-8 f64) | `linear_operator/settings.py` | 193-202 |
| `cholesky_max_tries` default (3) | `linear_operator/settings.py` | 205-210 |
| `psd_safe_cholesky` retry loop | `linear_operator/utils/cholesky.py` | 12-47 |
| `variational_cholesky_jitter` (1e-4 f32) | `gpytorch/settings.py` | 389-402 |

All GPyTorch files at: `/home/idv-eqs8-pza/anaconda3/envs/pytorch_gpytorch/lib/python3.12/site-packages/`

---

## Common Mistakes

1. **Adding jitter in `forward()`**: Redundant — GPyTorch already handles K_uu and K_XX jitter via `jitter_val`. Creates confusing double-jitter.

2. **Setting `cholesky_max_tries` without `cholesky_jitter`**: Retry starts at 1e-8 (float64 default). First ~4 retries are negligible. Always override both together.

3. **Thinking the error "up to 1.0e-06" means float32 jitter**: It means float64 jitter (1e-8 * 10^2 = 1e-6 after 3 tries). The matrix was promoted to float64.