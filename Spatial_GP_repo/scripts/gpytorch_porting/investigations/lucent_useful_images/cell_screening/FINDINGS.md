# FINDINGS: PNAS testbed-cell screen for the lucent image pipeline

**Date**: 2026-06-18 · **Branch**: `pietro/lucent-useful-images` (gaussian_processes submodule)
**Script**: `screen_testbed.py` · **Cache**: `cache/testbed_ladders.{pkl,csv}` · **Figure**: `testbed_ladders.png`
**Engine**: pinned `75b207a`, `default_gpy`, M = n_train, seed 42 (no engine edits).

## Goal + agreed methodology

Find a few testbed cells whose `default_gpy` `test_r` rises **cleanly with training-set
size**, to demonstrate the lucent "image sharpens as the model improves with data" pipeline.

Agreed with the user (2026-06-18):
- **Metric**: `test_r` = Pearson r on the 30-image test set (what the pipeline uses).
- **Grid**: n_train = M = `{50,100,150,200,250,300}` (step 50), seed 42, one seed.
- **Pool**: all 41 PNAS cells minus the 6 STA-edge cells (0,5,6,15,22,39) minus the
  default_gpy-hard cell 10 → **34 cells**.
- **Selection (user's words)**: `test_r` must be **increasing**, must **NOT already be
  high (>0.80) at n=50** (`START_MAX=0.80`, hard eligibility filter), ideally a wide
  **~0.3 → ~0.9** climb. RF position/size is **not** a criterion.
- **Scoring**: rank eligible cells by net gain, report all metrics, **human-pick 3** —
  no auto-thresholding (one seed can inject a spurious dip; see caveats).

Runtime: **13.3 min** for 33 cells (cell 13 was cached from the timing run), 0 failures.

## Headline finding (important — read before picking)

**No PNAS cell shows a smooth, gradual ~0.3 → ~0.9 climb across all six points.** The
`default_gpy` fits fall into three shapes, and you cannot have all of "positive ~0.3
start", "reaches ~0.9", and "gradual" at once:

1. **Gradual but plateaus below ~0.85** — e.g. cell 3 (0.18→0.83), cell 36 (0.31→0.73).
   The improvement is genuinely spread across n_train (best for the *demo narrative*), but
   the ceiling is ~0.7–0.85, not 0.9.
2. **Reaches ~0.9+ but via a single jump at n=100** (then a high plateau) — e.g. cell 13
   (0.37→0.96), cell 33 (−0.02→0.97). Hits the top, but most of the gain is one step.
3. **Reaches ~0.9+ only from a broken (negative) start / late jump** — e.g. cell 1
   (−0.15→0.98), cell 20 (−0.40→0.85), cell 14 (−0.31→0.82). Big "gain", but the start is
   broken, not low.

Concretely: **cell 13 is essentially the *only* cell that goes from a low-but-positive
start (~0.37) to ~0.9+ with a monotonic, dip-free curve.** Every other cell that reaches
0.9 either starts negative (broken), starts >0.80 (ineligible), or has a big mid-curve dip.
This is consistent with the documented `default_gpy` low-n instability (the fit "catches"
and jumps) and the high-n degradation (`.claude/rules/debugging.md` 3.6/3.7).

## Recommended 3 (all eligible, all increasing, all positive ~0.2–0.4 starts)

| pick | cell | ladder (n=50…300) | start | final | max-drop | Spearman | shape |
|---|---|---|---|---|---|---|---|
| 1 | **3**  | 0.18 0.74 0.78 0.80 0.82 0.83 | 0.18 | 0.83 | **0.000** | **1.00** | strictly monotonic; improves across the whole ladder |
| 2 | **13** | 0.37 0.94 0.93 0.95 0.96 0.96 | 0.37 | **0.96** | 0.011 | 0.89 | the literal "0.3→0.9"; clean & high (jump at n=100) |
| 3 | **36** | 0.31 0.30 0.60 0.65 0.73 0.73 | 0.31 | 0.73 | 0.015 | 0.94 | most gradual ramp; start exactly ~0.3 |

Why this trio: it is the set of **cleanly-increasing, positive-start (~0.3), eligible**
cells, spanning the trade-off — cell 3 is the only strictly-monotonic climber that reaches
a good level, cell 13 is the unique clean 0.3→0.9, cell 36 is the most gradual ramp. For
the lucent demo, 3 and 36 show *progressive* sharpening (model improves at every step),
while 13 shows it reaching a confident, high-quality RF.

**Decision (locked 2026-06-18):** picks **3, 13, 36** confirmed by the user under *relaxed*
criteria — reaching ~0.9 is NOT required and strict monotonicity is NOT required; single
seed 42, no multi-seed rerun. So cell 36's 0.73 ceiling and minor single-step dips are
acceptable; the trio stands as the lucent testbed.

### Swap options, by which trade-off you prefer
- **Want all three to reach ~0.9** (accept a step / ~0 start): swap cell 36 → **cell 33**
  (−0.02→0.97, clean high plateau, jump at n=100). Then the trio reaches 0.83/0.96/0.97.
- **Want maximum dynamic range** (accept a *broken negative* start): **cell 20** (−0.40→0.85,
  Spearman 0.94) or **cell 14** (−0.31→0.82, max-drop 0.005). Big visible jump, but the
  start is broken rather than "low".
- **Want the most-gradual shapes** (accept lower ceilings): cell 3 + cell 36 + **cell 29**
  (0.01→0.47, Spearman 0.83) or **cell 28** (−0.11→0.58, strictly monotonic).

## Full per-cell table (34 cells; M=n_train, seed 42)

`MONO` = max single-step drop ≤ 0.03 AND net gain > 0. `start>0.80` = ineligible
(already high at n=50). Sorted by eligibility then net gain.

| cell | n50 | n100 | n150 | n200 | n250 | n300 | start | final | gain | maxdrop | spear | flags |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 25 | −0.54 | 0.85 | 0.83 | 0.84 | 0.84 | 0.81 | −0.54 | 0.81 | 1.35 | 0.036 | 0.03 | |
| 20 | −0.40 | 0.82 | 0.79 | 0.84 | 0.85 | 0.85 | −0.40 | 0.85 | 1.24 | 0.030 | 0.94 | MONO |
| 14 | −0.31 | 0.24 | 0.80 | 0.81 | 0.81 | 0.82 | −0.31 | 0.82 | 1.13 | 0.005 | 0.94 | MONO |
| 1  | −0.15 | −0.24 | 0.06 | 0.97 | 0.97 | 0.98 | −0.15 | 0.98 | 1.13 | 0.091 | 0.94 | |
| 33 | −0.02 | 0.88 | 0.95 | 0.93 | 0.95 | 0.97 | −0.02 | 0.97 | 0.99 | 0.021 | 0.94 | MONO |
| 11 | −0.08 | −0.09 | 0.14 | 0.01 | 0.88 | 0.89 | −0.08 | 0.89 | 0.98 | 0.127 | 0.89 | |
| 2  | −0.11 | 0.93 | −0.07 | 0.91 | 0.91 | 0.85 | −0.11 | 0.85 | 0.95 | 1.009 | 0.26 | |
| 21 | −0.24 | 0.67 | 0.53 | 0.68 | 0.52 | 0.67 | −0.24 | 0.67 | 0.91 | 0.167 | 0.26 | |
| 12 | −0.07 | 0.40 | 0.80 | 0.80 | 0.77 | 0.79 | −0.07 | 0.79 | 0.86 | 0.029 | 0.49 | MONO |
| 16 | 0.07 | 0.82 | 0.81 | 0.84 | 0.83 | 0.82 | 0.07 | 0.82 | 0.75 | 0.011 | 0.71 | MONO |
| 28 | −0.11 | −0.05 | 0.26 | 0.37 | 0.55 | 0.58 | −0.11 | 0.58 | 0.69 | 0.000 | 1.00 | MONO |
| 40 | −0.24 | 0.74 | 0.86 | 0.54 | 0.43 | 0.43 | −0.24 | 0.43 | 0.67 | 0.321 | −0.09 | |
| **3** | 0.18 | 0.74 | 0.78 | 0.80 | 0.82 | 0.83 | 0.18 | 0.83 | 0.65 | **0.000** | **1.00** | MONO ← pick |
| 34 | −0.01 | 0.77 | 0.72 | 0.70 | 0.71 | 0.63 | −0.01 | 0.63 | 0.64 | 0.081 | −0.09 | |
| 19 | −0.09 | 0.53 | 0.50 | 0.56 | 0.60 | 0.53 | −0.09 | 0.53 | 0.62 | 0.069 | 0.54 | |
| 38 | −0.02 | 0.48 | 0.57 | 0.54 | 0.63 | 0.60 | −0.02 | 0.60 | 0.62 | 0.035 | 0.89 | |
| 32 | 0.17 | −0.16 | 0.68 | 0.40 | 0.16 | 0.77 | 0.17 | 0.77 | 0.60 | 0.334 | 0.49 | |
| **13** | 0.37 | 0.94 | 0.93 | 0.95 | 0.96 | 0.96 | 0.37 | 0.96 | 0.59 | 0.011 | 0.89 | MONO ← pick |
| 37 | −0.14 | 0.04 | 0.52 | 0.11 | 0.72 | 0.44 | −0.14 | 0.44 | 0.58 | 0.417 | 0.71 | |
| 18 | 0.40 | −0.19 | 0.59 | 0.90 | 0.90 | 0.92 | 0.40 | 0.92 | 0.52 | 0.590 | 0.89 | |
| 4  | −0.10 | −0.01 | −0.04 | −0.00 | 0.28 | 0.39 | −0.10 | 0.39 | 0.49 | 0.029 | 0.94 | MONO |
| 29 | 0.01 | 0.39 | 0.36 | 0.39 | 0.42 | 0.47 | 0.01 | 0.47 | 0.47 | 0.032 | 0.83 | |
| **36** | 0.31 | 0.30 | 0.60 | 0.65 | 0.73 | 0.73 | 0.31 | 0.73 | 0.42 | 0.015 | 0.94 | MONO ← pick |
| 17 | 0.43 | 0.77 | 0.90 | 0.88 | 0.84 | 0.81 | 0.43 | 0.81 | 0.38 | 0.038 | 0.43 | |
| 8  | 0.36 | 0.74 | 0.73 | 0.72 | 0.70 | 0.72 | 0.36 | 0.72 | 0.36 | 0.015 | −0.09 | MONO |
| 31 | −0.13 | −0.24 | −0.27 | −0.29 | −0.18 | 0.06 | −0.13 | 0.06 | 0.19 | 0.105 | 0.20 | |
| 23 | 0.00 | −0.17 | −0.14 | 0.08 | 0.12 | 0.09 | 0.00 | 0.09 | 0.09 | 0.166 | 0.77 | |
| 26 | 0.79 | 0.61 | 0.77 | 0.75 | 0.79 | 0.81 | 0.79 | 0.81 | 0.01 | 0.182 | 0.37 | |
| 35 | 0.11 | 0.09 | 0.01 | −0.10 | 0.05 | 0.11 | 0.11 | 0.11 | −0.00 | 0.112 | −0.26 | |
| 9  | 0.75 | 0.76 | 0.71 | 0.74 | 0.74 | 0.75 | 0.75 | 0.75 | −0.01 | 0.049 | −0.37 | |
| 7  | 0.59 | 0.43 | 0.61 | 0.56 | 0.58 | 0.57 | 0.59 | 0.57 | −0.02 | 0.165 | −0.14 | |
| 24 | 0.84 | 0.91 | 0.91 | 0.92 | 0.94 | 0.94 | 0.84 | 0.94 | 0.10 | 0.005 | 0.94 | start>0.80 |
| 27 | 0.93 | 0.86 | 0.91 | 0.94 | 0.95 | 0.94 | 0.93 | 0.94 | 0.01 | 0.073 | 0.66 | start>0.80 |
| 30 | 0.80 | 0.77 | 0.77 | 0.73 | 0.71 | 0.67 | 0.80 | 0.67 | −0.13 | 0.041 | −1.00 | start>0.80 |

Note: cells 24 and 27 have the cleanest high ladders overall but are **ineligible** — they
are already excellent at n=50 (0.84 / 0.93), so they cannot show "improvement with data".

## Caveats

- **One seed.** A single unlucky variational fit can inject a spurious dip (e.g. cell 2's
  0.93→−0.07→0.91 at n=150, or cell 18's 0.40→−0.19 at n=100). Treated as soft penalties,
  not auto-disqualifiers. The recommended 3 are robust (clean curves), but if a pick later
  misbehaves in the pipeline, re-run that cell at 2–3 seeds before discarding it.
- **Stale-engine confound (real but small — checked the log).** This screen runs on the
  pinned engine `75b207a`; `pietro/workingbranch` is **106 commits** ahead. Two of those
  target the instabilities seen here, but NEITHER is a clean fix: the A-prior + adaptive
  A_init (`559e09f`) is a **feature flag, default OFF**, and the validation sweep (`5cb2b2b`)
  found it a **partial fix only** ("σ=0.5 A-prior alone insufficient"); the high-n test_r
  decline was **investigated and closed as a known limit** (attempted fix reverted —
  `bdd2658` / `13507f5`). So the low-n breakage and high-n decline ARE partly engine
  artifacts, but rebasing would offer only an *opt-in, partial* low-n mitigation — it would
  not be expected to turn the broken-start cells into smooth 0.3→0.9 climbers. We stay
  pinned to match the pipeline/analysis; rebasing is the user's call and unlikely to change
  the picks.
- **April-2026 M=50 roster superseded.** The README "Cell roster" table (cells 13,3,1,11,12)
  was the early **fixed-M=50**, 4-point screen. This is the full **M=n_train**, step-50,
  34-cell re-screen the memory `default-gpy-low-ntrain-unstable` asked for. Under M=n_train
  cell 13 saturates faster (jump by n=100 vs the gradual 0.37→0.73→0.94 it showed at M=50).

## Reproduce

```bash
PY=/home/idv-eqs8-pza/anaconda3/envs/pytorch_gpytorch/bin/python
cd .../investigations/lucent_useful_images
$PY cell_screening/screen_testbed.py                       # full 34-cell screen (~13 min)
$PY cell_screening/screen_testbed.py --rank-only           # re-print ranking from cache
$PY cell_screening/screen_testbed.py --figure --highlight 3 13 36
```

Cache is incremental/resumable (skips cached cells; `--force` to recompute). Figure +
`cache/*` are gitignored (regenerable); the committed deliverables are `screen_testbed.py`
and this `FINDINGS.md`.
