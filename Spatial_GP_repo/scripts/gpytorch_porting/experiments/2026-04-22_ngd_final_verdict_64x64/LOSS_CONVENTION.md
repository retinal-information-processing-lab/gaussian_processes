# Loss-curve convention note (STAMP — Phase 3C sweep)

**Sweep**: Phase 3C final-verdict NGD sweep.
**JSONL files**: `results.jsonl`, `results.backup_*.jsonl`.
**Affected fields**: `train_loss_curve`, `A_curve` is fine, `beta_curve` is fine — only the loss curve has this issue. (Final scalars like `test_r`, `explained_var`, `final_A` are ALSO UNAFFECTED; only the per-iter `train_loss_curve` carries the bias.)

## What's "wrong" with these curves

This sweep was produced BEFORE the 2026-04-22 audit fix to `ngd_training.py`
(commit TBD). At that time the training loop logged `current_loss` via a
no_grad recompute placed AFTER the optimizer step:

```python
# (pre-fix code)
output = model(X_train)
loss = -mll(output, r_train)
loss.backward()
ngd_opt.step()      # q_i -> q_{i+1}
adam_opt.step()     # theta_i -> theta_{i+1}
clamp(...)
with torch.no_grad():
    ell_full = likelihood.expected_log_prob(r_train, output)
    #   ^ output.mean/var are PRE-step (cached), but likelihood.A and
    #     likelihood.lambda0 are POST-step (just updated by Adam).
    kl_full = model.variational_strategy.kl_divergence()
    #   ^ fully POST-step (uses just-updated q_{i+1} and theta_{i+1}).
    current_loss = (-ell_full + kl_full).item()
```

So `current_loss` at iter i is neither the pre-step nor the post-step
ELBO — it mixes `output` from (theta_i, q_i) with live A/lambda0/q/theta
from the POST-step state. Early in training, KL(q_{i+1}) > KL(q_i) as
q moves away from the prior; the logged loss is therefore systematically
biased upward by roughly the per-iter KL change. The bias is largest in
the first ~200 iters and shrinks as KL saturates.

## What's NOT wrong

The **optimizer trajectory is unaffected**. The backward() used the pre-step
`loss`, which is computed correctly by `-mll(output, r_train)` BEFORE any
`.step()` call. The gradients, NGD step, Adam step, and clamps are all
identical to what the fixed code produces.

Therefore:
- `test_r` — unaffected (computed after training via `predict()`)
- `explained_var`, `adjusted_r2`, `reliability` — unaffected
- `final_A`, `final_lambda0`, `final_beta`, etc. — unaffected
- `A_curve`, `beta_curve`, `rho_curve`, `lambda0_curve`, etc. —
  unaffected (these read `likelihood.A.item()` / `kernel.beta.item()`
  at end-of-iter, a coherent POST-step read; they're stylistically
  post-step but self-consistent)
- `nat_vec_norm_curve`, `nat_tril_offdiag_norm_curve` — unaffected
  (read of variational distribution state at end-of-iter)

**What IS affected:**
- `train_loss_curve` — semantic mix, biased high early
- `train_log_lik_curve` (if stored) — biased (uses post-step A with pre-step output)
- `train_kl_curve` (if stored) — purely post-step KL, doesn't match the loss decomposition
- `best_iteration` — argmax is computed on `-current_loss` which was mixed;
  the picked "best" iter may differ from the true pre-step-ELBO argmax by
  a few iters (ES patience=200 is much larger than this error)

## What "Phase 3D fix" changes

Post-fix, `train_loss_curve`, `train_log_lik_curve`, and `train_kl_curve`
are all cleanly pre-step values at each iter i — they describe the ELBO
at (theta_i, q_i) before the optimizer produces (theta_{i+1}, q_{i+1}).
This matches the convention used by `gpy_training.py` and
`eigenspace_training.py` curves, so cross-mode plots line up properly.

## Implications for this sweep's verdict

The Phase 3C verdict ("NO GAP, mean Δ = +0.0065") is based on `test_r`,
which is unaffected by the logging bug. The verdict stands.

If you need to compare NGD's per-iter loss trajectory against vargp or
default_gpy curves for a plot, one of the following is the right move:

1. Re-run the sweep with the fixed code (the fit is ~27 min, tractable).
2. Apply a correction: subtract the per-iter KL-delta from the logged
   curve to recover the pre-step convention. (Only works approximately;
   the post-step KL read is influenced by both variational and kernel
   updates.)
3. Plot only the tail of the curve, where the bias has shrunk below the
   resolution of interest.

Option (1) is the cleanest.

## Where to find the fixed code

`ngd_training.py` at project root, post 2026-04-22 commit. Module
docstring has a "Logging convention" block referring to this note.
SCRAPBOOK.md §47 has the full derivation.
