# Loss-curve convention note (STAMP — Phase 3 and 3B sweep results)

**Affected files**:
- `ngd_results.jsonl` (Phase 3B: 16 cells × 3 seeds, ES on)
- `ngd_results.noES_1000iter.jsonl` (Phase 3: same cells, ES off)
- `ngd_results.backup_*.jsonl` (historical prototype runs)

**Affected fields**: `train_loss`, `train_log_lik`, `train_kl` in the per-iter
`curves.*` lists.

## Summary

All JSONL files in this folder were produced BEFORE the 2026-04-22 audit
fix to `ngd_training.py` (commit TBD). Their `train_loss` / `train_log_lik`
/ `train_kl` per-iter curves were logged under a mixed pre/post-step
convention and are systematically biased (upward for `train_loss`, downward
for `train_log_lik`, neither-pre-nor-post for `train_kl`). The bias is
largest in the first ~200 iters where KL(q) is moving fast, and shrinks
as training plateaus.

The `test_r`, `explained_var`, `final_A`, `final_beta`, etc. scalars are
NOT affected — they were computed after training by independent
`predict()` calls that bypass the flawed logging code. Also unaffected:
`A_curve`, `beta_curve`, `rho_curve`, `lambda0_curve`, `nat_vec_norm_curve`,
`nat_tril_offdiag_norm_curve` (these were coherent POST-step reads of
parameter state, not a mix of pre/post with cached `output`).

For the full derivation and an analogous stamp on the Phase 3C sweep,
see:
- `experiments/2026-04-22_ngd_final_verdict_64x64/LOSS_CONVENTION.md`
- `SCRAPBOOK.md §47`

## Impact on Phase 3B conclusions

Phase 3B's headline numbers (|mean Δ(NGD − vargp_direct)| = 0.012; 40%
wall-time reduction; 3/5 outlier cells recovered) are based on `test_r`
and wall-clock, both unaffected by the logging bug. The verdict stands.

If you need to compare NGD's ELBO trajectory to vargp or default_gpy for
a plot that bridges modes, re-run with the fixed `ngd_training.py` — the
48-run Phase 3B sweep reruns in ~10 min.
