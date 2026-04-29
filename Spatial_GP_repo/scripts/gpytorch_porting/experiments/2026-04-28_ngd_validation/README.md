# NGD validation pipeline (B → C → A)

**Date**: 2026-04-28 (start) → 2026-04-29 (end)
**Status**: COMPLETE. Three follow-up tests after the M-sweep.
**Branch**: `pietro/investigate-default-gpy`

## Why this exists

After the M-sweep showed NGD ≥ vargp at production n_train=3160, we
needed three more validations before considering NGD as a default:
1. Does NGD work with `fix_Amp=False` (Amp trainable)? Phase 3C and the
   M-sweep both froze Amp.
2. Does NGD scale to 108×108 (the larger production dataset)? All
   prior NGD validation was 64×64 only.
3. How does NGD behave in the data-starved active-learning regime
   (M = n_train very small)? No prior data here.

## Pipeline

`run_pipeline.py` runs three experiments back-to-back, each crash-safe.
Total wall time: 9.3h (62 min B + 6.3 h C + 2 h A).

### Exp B — NGD with free Amp on 64×64

- **Question**: does NGD work when Amp is trainable (vs frozen at 1.0)?
- **Config**: 41 cells × seed=1, M=250, n_train=3160, `fix_Amp=False`
- **Reference**: existing vargp baseline `64_elbo_intl_freeAmp_Amp1` from
  `experiments/2026-04-06_es_sweeps_64x64/sweep_64x64_elbo_es_results.jsonl`
  (seed=1)
- **Result**: 41/41 paired. NGD mean_r=0.848, vargp 0.838. Δ = **+0.011**.
- **Verdict**: ✓ NGD with free Amp works. Amp values in [1.07, 2.37]
  — Adam trains it correctly, no pathology.
- **Output**: `results_B.jsonl`

### Exp C — Low n_train (M = n_train), both modes

- **Question**: how do NGD vs vargp compare in the data-starved regime?
- **Config**: 41 cells × seed=1 × {(M=50, n=50), (M=150, n=150),
  (M=300, n=300)} = 246 runs. Both modes from scratch, no existing
  baseline.
- **Result**: NGD trails vargp at every (M, n_train):

  | M=n | vargp | NGD | Δ NGD−vargp | NGD disasters | vargp disasters |
  |-----|-------|-----|-------------|---------------|-----------------|
  | 50  | 0.512 | 0.394 | −0.118 | 15/41 | 10/41 |
  | 150 | 0.545 | 0.462 | −0.083 | 14/41 | 8/41 |
  | 300 | 0.637 | 0.573 | −0.064 | 6/41 | 5/41 |

- **Verdict**: ⚠️ NGD genuinely worse at low n_train. Investigated
  thoroughly in `investigations/ngd_low_ntrain/` — first hypothesis
  (config/ES tuning) was falsified by a 41-cell follow-up sweep with
  vargp-scale ES, which made the gap *wider*. Open investigation.
- **Output**: `results_C.jsonl`

### Exp A — NGD on 108×108

- **Question**: does NGD scale to the larger 108×108 dataset?
- **Config**: 41 cells × seed=1, M=300, n_train=2910, `fix_Amp=False`
  (matches existing vargp baseline)
- **Reference**: `experiments/2026-03-20_massive_allcells_108/results.jsonl`,
  vargp_direct seed=1, M=300. (vargp's final_Amp ≈ 2.1 there → free Amp
  config; we matched.)
- **Result**: 41/41 paired. NGD mean_r=0.773, vargp 0.729. Δ = **+0.043**.
  Crucially, NGD std is also lower (0.17 vs vargp 0.29) — more stable.
- **Verdict**: ✓ NGD beats vargp on 108×108. The 6 cells with vargp's
  STA edge artefact (cells 0, 5, 6, 15, 22, 39 — see CLAUDE.md
  "STA edge artifact") are handled better by NGD.
- **Output**: `results_A.jsonl`

## Summary across all four NGD validation runs

| Test | NGD vs vargp Δ | Confidence | Notes |
|------|---------------|------------|-------|
| Phase 3C (M=250, n=3160, fix_Amp=True) | +0.007 | n=123 | core verdict |
| M-sweep (M ∈ {50..1500}, n=3160) | +0.000 to +0.015 | n=1107 | NGD ≥ vargp at all M ≥ 200 |
| Exp B (M=250, n=3160, fix_Amp=False) | +0.011 | n=41 | free Amp OK |
| Exp A (108×108, M=300, n=2910) | +0.043 | n=41 | NGD wins on harder dataset |
| Exp C (M=n=50,150,300) | −0.118, −0.083, −0.064 | n=41 each | NGD trails low n_train |

## Files

- `run_pipeline.py` — sequential B → C → A driver
- `run_B_freeamp_64x64.py` / `results_B.jsonl`
- `run_C_lowntrain_64x64.py` / `results_C.jsonl`
- `run_A_108x108.py` / `results_A.jsonl`
- `pipeline.log` — full stdout (huge, mostly clamp warnings — grep for
  `test_r=` or `Exp [ABC] done` for milestones)

## Implications for the GPyTorch-default decision

**For**: NGD wins on accuracy at all production points (Phase 3C,
M-sweep, Exp A, Exp B). On 108×108 it beats vargp by +0.043 with
half the variance. It also handles the STA-edge-artifact cells better.

**Against**: NGD has a genuine and unexplained deficit at low n_train
(Exp C). This matters for the active-learning use case which starts
with n_train ≈ 50–250 and grows. The simple "config tuning" fix
doesn't work (see `investigations/ngd_low_ntrain/`).

**Recommendation as of 2026-04-29**: NGD as default works for the
"large pre-trained" use case but not for the "small initial training
set" use case. Either:
- Mode-switch at runtime based on n_train, or
- Document the limitation and recommend vargp for n_train < ~500.

The default `run.mode` in `default_params.json` is still `default_gpy`
pending a clean resolution of the low-n_train question.
