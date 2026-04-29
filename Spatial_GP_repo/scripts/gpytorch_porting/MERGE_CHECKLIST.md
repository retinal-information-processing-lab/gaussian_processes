# Merge checklist — `pietro/investigate-default-gpy` → `pietro/workingbranch`

**For**: a future Claude Code session asked to "merge the NGD work into main".
**Read first**: `investigations/default_gpy_gap_v2/SCRAPBOOK.md` §62–68
(canonical recap of all post-Phase-3D work).

---

## TL;DR — what to do, what NOT to do

✅ **Safe to merge** the production code (additive, backward-compatible,
audited). Existing `default_gpy` and `vargp_direct` modes are untouched.

❌ **Do NOT** flip `default_params.json["run"]["mode"]` from `"default_gpy"`
to `"ngd"` as part of this merge. NGD is not yet validated for the
active-learning low-n_train regime — see "Known limitations" below.

❌ **Do NOT** silently delete investigation/experiment folders. They are
the reproducible record of how this work was validated.

---

## 1. What this branch changes (production code)

7 production files, all additive and backward-compatible:

| file | change | risk |
|------|--------|------|
| `ngd_training.py` | NEW (348 lines) — the NGD training loop | Zero — new file |
| `run_single_mode.py` | added `mode='ngd'` branch (+ alternating_fstep wiring for default_gpy) | Zero — existing branches untouched |
| `gpy_model.py` | added `variational_distribution_cls` kwarg, default `'cholesky'` | Zero — old default preserved |
| `gpy_training.py` | added `alternating_fstep` kwarg, default `False` | Zero — old default preserved |
| `default_params.json` | added top-level `"ngd"` block | Zero — no existing key changed |
| `_constants.py` | added 8 NGD-specific exports | Zero — no existing export changed |
| `tests/test_gpy_alternating_fstep.py` | NEW test | N/A |

`/simplify` audit passed (3-agent review at commit `f900a2c`). Issues
flagged then fixed before that commit.

## 2. What the branch validates

Five independent tests, all on real PNAS dataset:

| test | runs | result | folder |
|------|------|--------|--------|
| Phase 3C verdict sweep | 41 cells × 3 seeds = 123 | NGD vs vargp Δ = +0.007 (NO GAP) | `experiments/2026-04-22_ngd_final_verdict_64x64/` |
| M-sweep | 41 × 3 × 9 M = 1107 | NGD ≥ vargp at all M ≥ 200 | `experiments/2026-04-23_ngd_M_sweep_64x64/` |
| Exp B (free Amp 64×64) | 41 | Δ = +0.011 ✓ | `experiments/2026-04-28_ngd_validation/` |
| Exp A (108×108) | 41 | Δ = +0.043 ✓ (NGD lower variance too) | same as above |
| Exp C (low n_train) | 246 | Δ = −0.06 to −0.12 ✗ | same as above |

See `experiments/<folder>/README.md` for each.

## 3. Known limitations to surface in the PR description

These are honest gaps you should mention in the PR, not hide:

- **Low n_train regime untrained**: at M = n_train ∈ {50, 150, 300}
  NGD trails vargp by 0.06–0.12 mean test_r and is 2× slower wall-time.
  Investigated in `investigations/ngd_low_ntrain/` — first hypothesis
  (ES tuning) was falsified by a 41-cell follow-up. Investigation is
  **OPEN**. Active-learning users that start with small training sets
  would regress if NGD becomes default.
- **Phase 3E LBFGS investigation**: deferred negative result. See
  `investigations/default_gpy_gap_v2/ngd_lbfgs/`. Five LBFGS variants
  failed (cell-30 A→0 attractor in Poisson-exp likelihood). No
  production code from Phase 3E enters this merge.
- **NGD only validated with `fix_Amp=True` or `fix_Amp=False`** at
  fixed-Amp value 1.0. Not validated with non-default Amp_init.

## 4. What this branch does NOT update (and why)

- `default_params.json["run"]["mode"]` stays `"default_gpy"` —
  see TL;DR.
- **CLAUDE.md is NOT updated**: Training Modes table, Parameter
  Matching Table, "Authoritative Sources" don't mention NGD or the
  M-sweep / validation folders. Also: the "UNDER INVESTIGATION
  default_gpy A explosion" note is stale (NGD resolves it but the
  note still says open).
  → A merge PR should update CLAUDE.md to:
    1. Add NGD to Training Modes table
    2. Add NGD row to Parameter Matching Table
    3. Add three rows to Authoritative Sources (M-sweep, validation,
       low-n_train investigation)
    4. Update or remove the stale `default_gpy` A-explosion note
       (it points to `investigations/default_gpy_gap/` v1 — the
       resolution is in v2's Phase 3 [NGD as default_gpy alternative]
       and Phase 3E [LBFGS attempt deferred])
- `configs/canonical.yaml` does NOT include `mode='ngd'`. Adding it
  is a separate decision: do you want NGD in the regression test
  matrix? If yes, add `'ngd'` to `experiment.modes`. If no, document
  why.
- The validated NGD config uses `ngd_n_iterations=1500` explicitly,
  but `default_params.json["ngd"]["n_iterations"]=1000` (the default
  from Phase 3D wiring). Discrepancy:
  - Phase 3C verdict sweep override: 1500 (validated)
  - default_params.json: 1000 (untested at scale)
  - **Recommend**: bump default_params.json to 1500 to match the
    validated config, OR document the discrepancy. Currently it's a
    silent footgun.

## 5. Pre-merge verification (recommended)

Run these before opening the PR; report results in the PR description.

```bash
# 1. Working tree clean
git status

# 2. Existing tests still pass
pytest tests/ -x

# 3. NGD smoke test (Phase 3C single-cell, ~14s on clean GPU)
python run_single_mode.py --mode ngd \
    --data-path datasets/PNAS_64x64_center_crop_no_renorm.npz \
    --ntilde 250 --n-train 3160 --cell 16 --seed 1 --ip-selection random
# Expected: test_r ≈ 0.93–0.97, no crashes

# 4. default_gpy smoke (regression check — NGD changes shouldn't affect it)
python run_single_mode.py --mode default_gpy \
    --ntilde 50 --seed 123
# Expected: same numbers as on workingbranch (small perturbation OK due
# to alternating_fstep code-path additions, but should be tiny)

# 5. vargp_direct smoke (regression check)
python run_single_mode.py --mode vargp_direct \
    --ntilde 50 --seed 123
# Expected: same numbers as on workingbranch
```

## 5b. Data files that live ONLY on `dgx3` (not in repo)

Per `.gitignore` and CRITICAL RULE 9 (CLAUDE.md): the full M-sweep
result file with per-iter trajectories is too large for git (80 MB)
and lives only on the dgx3 machine.

| file | size | path on dgx3 | regenerate cost |
|------|------|--------------|-----------------|
| `results.jsonl` (M-sweep, full curves) | 80 MB | `experiments/2026-04-23_ngd_M_sweep_64x64/results.jsonl` | ~17h GPU |

The committed summary `results.summary.jsonl` (824 KB) covers the
headline numbers but lacks per-iter curves. If the merge target
machine doesn't have access to dgx3, document this when handing off.

**Do NOT delete the dgx3 copy** unless the user explicitly approves —
17 hours of compute is well past the CRITICAL RULE 9 threshold.

## 6. Specific gotchas

These bit me during this work; flag them in the PR or commit messages
so future devs see them:

1. **`build_config_from_defaults` defaults to 108×108 dataset.** Any
   64×64 sweep MUST pass `data_path=…64x64…` explicitly. Already
   documented in `experiments/2026-04-23_ngd_M_sweep_64x64/README.md`
   §"Two explicit non-default overrides".

2. **NGD's `ngd_n_iterations`**: see §4 above. Validated at 1500;
   default at 1000.

3. **Loss logging convention** (SCRAPBOOK §47–50): pre-2026-04-22
   sweeps logged a "Frankenstein" mixed-step ELBO. The current code
   uses pre-step convention. Old JSONLs (Phase 3B sweep) are stamped
   with a `LOSS_CONVENTION.md` note in their folder. Don't compare
   loss curves across the convention boundary without reading those.

4. **Alternating F-step in default_gpy** (`alternating_fstep=True`): is
   a feature flag wired through `gpy_training.py`. Default `False`
   (matches old behavior). It's enabled by setting the kwarg or via
   `default_gpy_alternating_fstep` mode in `run_single_mode.py`. This
   was added during the v1 default_gpy_gap investigation (commit
   `e4fae78`) before the NGD pivot. Phase 2 §6.2 found it makes things
   *worse* on most cells; it's available but not recommended.

5. **The Phase 3E LBFGS investigation produced
   `investigations/default_gpy_gap_v2/ngd_lbfgs/ngd_lbfgs_training.py`** —
   this is NOT production code. It's an investigation artifact
   (deferred result). Don't accidentally promote it.

6. **`run.mode` flag name confusion**: the canonical key is
   `default_params.json["run"]["mode"]`. `run_single_mode.py` accepts
   `--mode` as CLI override. Don't confuse with kernel modes (e.g.
   `kernel_type='arc_cosine'`).

## 7. Suggested PR structure

If the merge is via PR (not direct merge), a clean structure:

- **Title**: "Add NGD as first-class GPyTorch-native training mode"
- **Body**: paste TL;DR + Validations table + Known limitations from
  this file. Link to SCRAPBOOK §62–68 for the full recap.
- **Reviewers**: ask for explicit review of:
  - The NGD training loop (`ngd_training.py`)
  - The `mode='ngd'` branch in `run_single_mode.py`
  - The `alternating_fstep` plumbing in `gpy_training.py` (added during
    the same investigation arc but is separate functionality)
- **Testing**: paste output of the §5 verification commands.

## 8. Branch hygiene before merge

- Branch has 3 commits ahead of `pietro/workingbranch` at the time of
  this writeup: `e4fae78`, `d27c4eb`, `f900a2c`. Each is a clean
  logical unit; merge as-is or squash by judgment.
- `pietro/workingbranch` may have moved since. Rebase or merge
  workingbranch first; resolve any conflicts in the touched
  production files (rare — most changes are additive).

## 9. After merge

- Open issue or note in the project tracker:
  "Resolve `investigations/ngd_low_ntrain/` before promoting NGD to
  default mode."
- If `configs/canonical.yaml` was updated to add NGD, run the
  canonical regression sweep on workingbranch and verify the new
  baselines.

---

## Appendix: where everything lives

```
gpytorch_porting/
├── ngd_training.py                          [NEW prod] — NGD training loop
├── gpy_training.py                          [MODIFIED] — alternating_fstep
├── gpy_model.py                             [MODIFIED] — variational_distribution_cls
├── run_single_mode.py                       [MODIFIED] — mode='ngd', alternating_fstep
├── default_params.json                      [MODIFIED] — "ngd" block added
├── _constants.py                            [MODIFIED] — NGD constants added
├── tests/test_gpy_alternating_fstep.py      [NEW test]
├── experiments/
│   ├── 2026-04-22_ngd_final_verdict_64x64/  [Phase 3C: 123 runs, headline +0.007]
│   ├── 2026-04-23_ngd_M_sweep_64x64/        [M-sweep: 1107 runs]
│   └── 2026-04-28_ngd_validation/           [B+C+A pipeline: 328 runs]
├── investigations/
│   ├── default_gpy_gap_v2/
│   │   ├── SCRAPBOOK.md                     [§62–68 = canonical recap]
│   │   └── ngd_lbfgs/                       [Phase 3E deferred — NOT prod code]
│   └── ngd_low_ntrain/                      [OPEN investigation]
└── MERGE_CHECKLIST.md                       [THIS FILE]
```
