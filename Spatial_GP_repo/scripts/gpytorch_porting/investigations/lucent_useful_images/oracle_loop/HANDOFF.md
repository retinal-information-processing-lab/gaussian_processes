# Investigation Plan: in-silico closed loop — ceiling GP as "oracle cell" + generated-image acquisition

**Status**: PLANNED / not started (design captured 2026-07-04, no code yet)
**Discussed on branch**: `pietro/lucent-useful-images` (gaussian_processes submodule)
**Doc home**: `investigations/lucent_useful_images/oracle_loop/` (placed with the image generator;
but see "Structural question" — Part 1 is really analysis-side and the work may straddle two repos/branches)

---

## The idea (one paragraph)

We built a lucent image generator that produces bounded, utility-maximizing images
(`../FINDINGS_gray_start.md`). The next step is to use it as an **acquisition strategy** inside
a **simulated closed loop** and compare it against the current pool-based selection. The neuron
is stood in for by an **oracle = the full-data "ceiling" GP** for a cell (the best model of that
cell; the model behind
`analysis/cross_sessions/results/v0.7/pooled_explained_variance_no_greedy.svg`). We cannot run
real cells yet, so the oracle plays the cell: we query it for responses. A **student** GP learns
online; at each step it picks the next stimulus; we measure how fast the student converges to
the oracle (explained-variance-vs-ceiling, the exact axis of that figure).

## SPLIT INTO TWO PARTS (do them in order — user's call)

Two genuinely new ingredients are being introduced (the ceiling-as-oracle, and the image
generator). Validate them one at a time.

### Part 1 — oracle simulation, NO generation. Reproduce the linked figure in silico.
Use the ceiling model as the cell. Run the **active + random** selection over the natural image
pool (same as the real experiment), but replace the real spike responses with **oracle-model
responses**. Grow the student's training set image-by-image and measure explained-variance-vs-
ceiling as #images grows. **Success = the in-silico curves reproduce the shape/ordering of
`pooled_explained_variance_no_greedy.svg` (active > random, rising with #images).** This proves
the oracle-as-cell machinery behaves like the real experiment BEFORE any generation is involved.
If it does not reproduce, the simulation is wrong and Part 2 would be built on sand.

### Part 2 — add the generator as a third arm.
Same loop, add **generate**: the next stimulus is synthesized by following the utility gradient
(lucent), not picked from the pool. Compare **generate vs active vs random** on the same EV axis.
Expected question answered: does lifting the "must be a real dataset image" constraint learn the
oracle faster?

## Decisions already made (the "why", so a new session doesn't re-litigate)

- **Objective = maximize information gain (utility)**, exactly as the current active case — the
  generator just removes the pool constraint by following the gradient. If a generated image does
  not drive the cell, that is an empirical **finding** (a method result to report), NOT a design
  blocker to pre-solve.
- **Compare random / active / generated like-for-like** (same oracle, same student, same metric).
  The inductive-bias overlap (student can represent an oracle of the same GP family, which
  flatters absolute numbers) affects all three arms equally, so the *comparison* is fair. Known
  limit, accepted.
- **Accepted, unfixable-without-real-cells limit:** the oracle is a GP fit on *natural* images;
  its responses to *generated* (off-manifold) images are extrapolations we cannot validate. This
  bounds every claim to "learns the **oracle** faster", NOT "learns the **cell** faster". We
  recognize it and move on; do not spend effort trying to fix it in silico.

## Critical subtleties / where a new session MUST read more

1. **Dataset + engine bridge (resolve FIRST).** The ceiling models live in the **closed-loop
   world** (16 cells, April+August, the analysis-pipeline GP, real-experiment images). The lucent
   generator is **PNAS + default_gpy**, and DA-utility generation needs `default_gpy`'s covariance
   matrix. These are not the same engine. Two options:
   - (a) refit the ceilings as `default_gpy` on the closed-loop images, or
   - (b) port DA-utility generation to the analysis/eigenspace GP (the deferred "augmented-matrix"
     port, see `../../../.claude/rules/acquisition.md` deferred items).
   CONFIRM the exact ceiling engine/kernel before choosing. Read: `analysis/CLAUDE.md`,
   `analysis/cross_sessions/README.md`, and the ceiling artifacts (`ceiling_summary.json`,
   `ceiling_model.pt` per cell).
2. **The EV metric + ceiling definition.** Read `analysis/cross_sessions/CONSUMER_DESIGN.md`,
   `SCHEMA_DRAFT.md`, and the figure producers `analysis/cross_sessions/code/replot_perf_no_greedy.py`
   + `consumer.py`. Explained variance is student-vs-ceiling; `_no_greedy` = active/random only.
3. **The real active loop mechanics** (what "active" means, warm-start rank-1 refit, the spike
   window, standard vs DA utility). Read the mother-repo `.claude/CLAUDE.md` "Experiment flow" +
   "Sequence modes", and `standalone_linux/main_loop.py`. The real loop **warm-starts** (rank-1
   model update), it does NOT refit from scratch — the in-silico loop should mirror that (the
   lucent panels refit from scratch, which is fine for a snapshot but wrong for a trajectory).
4. **The generator + utility.** `../FINDINGS_gray_start.md`, `../optimize_image.py`,
   `../acquisition.py` (`distribution_aware_utility`, needs default_gpy), and the lucent
   caveat: naturalness comes from the parameterization + start, not the objective.

## Open questions for the new session to settle with the user

- Which bridge, (a) refit ceilings as default_gpy or (b) port DA utility to the analysis GP?
- Response model: oracle **mean** (deterministic, reproducible) or **Poisson-sampled** (realistic
  noise)? Start deterministic.
- Which cells (the 16 closed-loop cells? a subset?) and which #images grid / warm-start scheme, to
  match the real trajectory.
- Metric: explained-variance-vs-ceiling (to match the SVG) and/or LSTA-vs-ceiling correlation.
- (Part 2 only, optional) whether to keep the GP ceiling as oracle or move to a richer oracle
  (CNN / the repo's diffusion response model) to make the generated-image arm less self-consistent
  — the user's call; Part 1 uses the GP ceiling regardless.

## Structural question (flag to the user early)

Part 1 (reproduce EV with the oracle) is naturally **analysis-side** (ceiling models + EV
machinery live on `analysis/april26`, superproject). Part 2 (generation) is **lucent/gpytorch-
side** (`pietro/lucent-useful-images`, submodule). These are different repos/branches. The new
session must decide WHERE this investigation lives and on which branch before writing code. This
doc sits in the lucent folder because that is the thread it grew from; the actual work may need an
analysis-side home. Do not assume — confirm with the user.

## What to read first (ordered)

1. This doc.
2. `../FINDINGS_gray_start.md` + `../README.md` (the generator: what it does, the DA utility, the
   sample_lambda / n_mc conclusions).
3. `analysis/cross_sessions/README.md` + `analysis/CLAUDE.md` (the pipeline, the ceiling, the EV).
4. The figure producers: `analysis/cross_sessions/code/{replot_perf_no_greedy.py, consumer.py}`.
5. Mother-repo `.claude/CLAUDE.md` "Experiment flow" + "Sequence modes" (the real active loop).
6. `gpytorch_porting/acquisition.py` + `.claude/rules/acquisition.md` (the utility, deferred
   vargp_direct DA port).
7. Project memory: `lucent-useful-images`.

---

## Continuation Prompt (paste block for a future session)

```
Start a NEW investigation: an in-silico closed loop that uses the full-data "ceiling" GP as an
"oracle cell" and (later) the lucent image generator as an acquisition strategy. Read the plan
first: investigations/lucent_useful_images/oracle_loop/HANDOFF.md. DO NOT code before settling
the open questions with the user.

Two-part plan (do in order):
  PART 1 (no generation): reproduce analysis/cross_sessions/results/v0.7/
    pooled_explained_variance_no_greedy.svg IN SILICO -- ceiling model as the cell, run
    active + random selection over the natural pool with oracle-model responses instead of real
    spikes, measure explained-variance-vs-ceiling as #images grows. Success = reproduces the
    active>random rising shape. Validates the oracle-as-cell machinery.
  PART 2 (add generation): add a third arm where the next stimulus is lucent-generated (follow
    the utility gradient, not pick from the pool). Compare generate vs active vs random.

Decisions already locked (don't re-litigate): objective = info-gain via the gradient (a
non-driving generated image is a finding, not a blocker); compare all arms like-for-like (fair
even if GP-on-GP flatters absolutes); the oracle is only valid on natural images -> claims are
"learns the oracle", not "learns the cell" (accepted limit).

FIRST resolve with the user: (1) the dataset/engine bridge -- ceilings are closed-loop/analysis-
engine, the generator is PNAS/default_gpy and DA utility needs default_gpy covariance: refit
ceilings as default_gpy, or port DA utility to the analysis GP? (2) where this investigation
lives / which branch (Part 1 is analysis-side on analysis/april26; Part 2 is lucent-side on
pietro/lucent-useful-images). Check git branch + git status before touching anything;
additive files only; no engine edits.
```
