# NGD+LBFGS investigation (Phase 3E) — DEFERRED (2026-04-23)

**Status**: deferred. Five LBFGS variants tested, none passed the §40
prototype gate. User decision: "nothing came out of it, adam seems
enough". NGD+Adam (Phase 3C verdict sweep) remains the recommended
GPyTorch-native training mode.

**Start here**: `../SCRAPBOOK.md` §51–59 is the canonical write-up.

- §51: current status (read first)
- §54: results tables for V0/V1/V2/V3
- §55: per-variant mechanism analysis
- §56: synthesis — the dominant failure mode (A→0 in Poisson-exp
  likelihood) is structural, not a tuning issue
- §57: decision matrix — stop/write-up vs full sweep vs try V4/V5
- §58: artifact map
- §59: re-entry summary

**File naming in this folder**:
- `prototype_V{0,1,2,3}_*.jsonl` — per-variant prototype results
- `log_V{0,1,2,3}_*.log` — run logs
- `prototype_results.jsonl` — the most-recent run output (currently
  V3, will be overwritten by the next `run_prototype.py` invocation)
- `ngd_lbfgs_training.py` — the training loop; defaults are the
  current variant being tested. Flip `separate_likelihood` / `n_warmup`
  in the function signature to switch variants.
- `run_prototype.py` — 5 cells × 3 seeds runner
- `run_sweep.py` — 41 × 3 runner (written, never executed)

**Do not change production code** without explicit user approval. The
investigation's conclusions do not currently support promoting
NGD+LBFGS over NGD+Adam.
