# Early exploration — FIXED low M = 50 (SUPERSEDED)

**Status: DONE / early exploration. Superseded by the M = n_train approach.**

These were the first figures from this investigation. They used a **fixed, low number
of inducing points M = 50** (a sparse GP approximation) while varying
`n_train ∈ {50, 100, 200, 300}`. So at n_train = 300 the model summarized 300 training
images with only 50 inducing points — a coarse sparse approximation.

**Why superseded:** we now use **M = n_train at all times** (every training point is an
inducing point → the full, non-sparse variational GP at each training size). This is the
cleaner methodology and gives slightly better, more stable models (e.g. cell 13 at
n_train=300: test_r 0.96 with M=n_train vs 0.94 with M=50). All current/active figures use
M = n_train.

**Kept for traceability** (not deleted) per the project's figure-folder rule. Do not treat
these as current results.

## Files (M=50, n_train ∈ {50,100,200,300})
- `fig_useful_natural.{png,svg}` — 4-cell grid, natural-start optimized images.
- `fig_useful_rf.{png,svg}` — preferred stimulus on neutral gray.
- `fig_rf_difference.{png,svg}` — RF-localized change.
- `fig_hero_cell{1,3,11,12,13}.{png,svg}` — per-cell 3-view summaries.

Produced by `run_grid.py` + `make_figure.py` when those used a fixed M=50 (they have since
been updated to M=n_train).

## diagnostics/
Raw per-run diagnostic PNGs from the M=50 exploration (8-panel per-(cell,n_train) from
`run_grid.py`, 6-panel single-run from `optimize_image.py`). Regenerable and gitignored —
kept here for traceability, not part of the figure set.
