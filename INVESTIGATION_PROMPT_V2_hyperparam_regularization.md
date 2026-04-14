# Investigation V2 — Principled Fix for M Degradation and A Initialization

You are continuing an investigation started on another machine. The first phase
diagnosed the problem; your job is to design and implement a **principled,
universal fix**. Read this entire document before taking any action.

---

## 0. First — verify the environment (do this before anything else)

Run these checks and stop if any fail. Do not proceed until all four are correct.

```bash
cd <repo root on this machine>
git fetch origin
git checkout pietro/investigate-M-degradation
git pull --ff-only origin pietro/investigate-M-degradation
git branch --show-current   # must print: pietro/investigate-M-degradation
git log --oneline -1        # must be commit 7e4b7f1 or a descendant of it
pwd                         # note the path for all subsequent commands
```

If any check fails, stop and tell the user. Do not guess or work around it.

---

## 1. Safety constraints (session runs with `--dangerously-skip-permissions`)

You have broad permissions. In exchange, follow these rules strictly:

- **Stay on `pietro/investigate-M-degradation`.** NEVER run `git checkout`, `git
  switch`, `git worktree add`, or any command that changes the branch.
- **NEVER `git push --force`, `git push origin main`, or push to any branch
  other than `pietro/investigate-M-degradation`.** Regular `git push origin
  pietro/investigate-M-degradation` is fine.
- **NEVER modify `default_params.json`.** It is shared with other branches and
  changing it would affect unrelated work. If you need different default values,
  pass them explicitly in your config-building code.
- **NEVER delete files that are already tracked in git.** If you think a file
  should be removed, propose it to the user and wait.
- **Scope your work to `Spatial_GP_repo/scripts/gpytorch_porting/`.** Do not
  edit anything outside this subtree (no `.claude/` outside it, no parent
  READMEs, no `SETUP.md`).
- **Commit incrementally.** Not one giant commit at the end. After each
  coherent unit of work (analysis doc, code change, validation run), commit
  with a clear message. The other machine's session already pushed commits
  `0561d73` and `7e4b7f1`; you build on top of those.
- **Before any commit, run `git branch --show-current` again** to confirm you
  are still on the right branch.

If you find yourself wanting to do something outside these rules, stop and
ask the user.

---

## 2. What the problem is

We have a variational GP with Poisson likelihood (model of retinal ganglion
cell responses to natural images). The training algorithm is an EM-like loop
with interleaved F-step — see `investigations/M_degradation/FINDINGS.md` for
references to the math and code paths.

Two coupled problems were identified in the previous session:

### Problem 1: A_init is not principled

The E-step Newton gradient for the variational mean scales as
`A * N_train * max(r)`. If `A_init` is too large, the first Newton step
overshoots, pushing `μ` to extreme values; the interleaved F-step then
amplifies `A` further, causing E-step divergence or A-collapse.

Current workaround: hardcoded `A_init = 1e-4`. This was empirically tuned for
`N_train = 3160`. It has no justification for other N (e.g. `N = 50` in the
active learning loop). Data-adaptive formulas were sketched in `ToDo.md`
under "Data-adaptive A initialization for interleaved F-step stability".

### Problem 2: M degradation — hyperparameter overfitting at large M

A 41-cell × 8-M × 3-seed sweep (984 runs total, in
`experiments/2026-04-13_M_sweep_64x64/`) showed that **9 of 41 cells show
monotonic test_r degradation as M grows**. Every degrading cell shows the
classic overfitting signature: `train_r` rises while `test_r` falls. Cell 39
is the worst (-0.199 from M=50 to M=1500); Cell 35 is -0.057.

### The common root cause

**The M-step optimizes hyperparameters (A, β, ρ, σ₀) to maximize the ELBO.
The ELBO has no regularizer on hyperparameters** — only on the variational
distribution `q(λ̃)` via the KL term. With more inducing points, the ELBO
becomes a tighter bound on the marginal likelihood, revealing finer structure
in the training data including noise. The optimizer uses the extra capacity
to drift `A` upward, amplifying predictions to better fit single-trial
training responses at the cost of generalization.

This is a **type-II ML overfitting** problem, well known in ML: joint
MLE of hyperparameters over-adapts when model capacity exceeds the effective
signal in the data. The standard fix is a prior on hyperparameters (MAP
instead of MLE), or explicit regularization.

### Evidence

You should verify the chain of evidence before designing a fix:

- `investigations/M_degradation/FINDINGS.md` — full investigation story.
- `experiments/2026-04-13_M_sweep_64x64/README.md` — canonical results with
  per-cell trends table.
- `experiments/2026-04-13_M_sweep_64x64/M_sweep_results.jsonl` — 984 raw
  records with full per-iteration curves.

Key tests already performed (you do NOT need to redo them):
- **Multiple metrics confirm overfitting**: test_r, adjusted_r2, and held-out
  val_r (carved from training data) all drop together for Cell 35.
- **Warm-init experiment**: starting M=1500 training from Cell 35's
  M=50-optimal hyperparameters, training STILL drifts A upward and loses
  test_r. This rules out "local optimum in hyperparameter space" — the
  ELBO genuinely prefers the higher-A solution at large M.
- **Noise ceiling correlation**: degrading cells tend to be near their
  explained-variance ceiling already at M=50, so additional capacity has no
  real signal left to learn, only noise.

Cell 35 response statistics: `mean(r)=0.21`, `frac_zero=0.88`,
`test_reliability=0.80`, `explained_var at M=50 = 0.94`.

---

## 3. Your mission

Design and implement a **principled, universal** fix that:

1. **Prevents the M degradation for the 9 affected cells** (or substantially
   reduces it — specifically: Cells 39, 35 should no longer degrade
   monotonically with M).
2. **Does not hurt the 27 improving cells or the 5 flat ones.** Universal
   means universal. A fix that saves degraders by hurting improvers is a
   failure.
3. **Principled, not ad-hoc.** Prefer approaches grounded in Bayesian
   reasoning (priors, MAP estimation, evidence-based regularization) over
   heuristics (arbitrary thresholds, cell-specific rules).
4. **Minimal.** Do not rewrite the training loop. The smallest code change
   that works is the best one. No per-cell tuning. No if-cell-is-sparse-then
   logic.
5. **Solves A_init too, ideally.** A Bayesian prior on A naturally addresses
   both Problem 1 (initialization) and Problem 2 (drift during training).
   An integrated fix is preferred over two separate patches.

Framing: you are a machine learning engineer being handed a system where
hyperparameter MLE is overfitting. Think from first principles about what a
competent ML engineer would do.

---

## 4. Design principles — do not violate

- **No per-cell tuning.** The fix must not use cell-specific thresholds,
  priors, or exceptions. One global set of choices.
- **No data-peeking.** The fix cannot use the test set to tune itself. Can
  use training data response statistics (`mean(r)`, `sum(r²)`, etc.) because
  those are legitimately available.
- **No cheating via capacity caps.** Artificially capping M, capping `n_b`,
  or raising the eigenvalue threshold would "fix" the problem by preventing
  the model from expressing itself. That's not principled — it's a bandaid.
  Capacity caps may be discussed but are a last resort.
- **Keep `--float32`.** Do not switch to float64.
- **Keep the existing mode (`vargp_direct`).** Do not port to `default_gpy`
  or invent a new mode.
- **Respect existing library conventions.** Hyperparameter bounds live in
  `kernels.py`; the F-step lives in `eigenspace_fstep.py`; the training
  loop is in `eigenspace_training.py`. Follow these patterns.

---

## 5. Candidate approaches (non-exhaustive — think freely)

These are a starting point. You are free to propose others.

### A. Hyperparameter prior (most principled, my recommendation to investigate first)

Add a penalty to the ELBO that regularizes A:
```
ELBO_reg = ELBO - 0.5 * (log(A) - log(A_0))² / σ_A²
```
This is equivalent to a log-normal prior on A. Questions to work through:
- What should `A_0` be? Data-driven (e.g. derived from response statistics),
  fixed constant, or learnable?
- What should `σ_A` be? Tight enough to prevent drift, loose enough to let
  cells with strong signal find their optimum. One global value.
- Should there also be a prior on β (RF size)? A weak one might prevent
  the β-drift seen in Cell 35.
- Where does this hook in? The F-step's damped Newton needs to include the
  prior gradient. The M-step's LBFGS closure needs to include the prior in
  the loss.

### B. Data-adaptive A_init (addresses Problem 1 only)

From `ToDo.md`, three candidate formulas:
- `A_init = c / (N_train * mean(r))`
- `A_init = c / sqrt(N_train * sum(r²))`
- `A_init = c / (N_train * mean(r) + max(r))`

The analysis in ToDo.md shows formula 1 is too aggressive for sparse cells
(mean(r) underestimates the tail). Formula 2 or 3 are more robust. This
would replace the hardcoded `1e-4` but does NOT solve Problem 2 on its own.

### C. Validation-based M-step early stopping

Stop the M-step when a held-out validation metric stops improving. This
directly prevents hyperparameter overfitting but adds complexity (needs
carved validation data, interacts with the existing ELBO-based ES).

### D. Hybrid: data-adaptive A_init + weak prior on A

A+B combined. The data-adaptive formula sets a reasonable starting point
and also defines `A_0` for the prior. Could be the cleanest unified solution.

---

## 6. Working process — the user wants one check-in

Structure your work as:

### Phase 1 — Deep analysis and proposal (autonomous)

Read the evidence files (see Section 2). Verify the mechanism yourself by
loading `M_sweep_results.jsonl` and inspecting a few degrading cells. Then
write a proposal document at:

`investigations/M_degradation/REGULARIZATION_PROPOSAL.md`

containing:
- Your understanding of the problem (brief — show you read the docs)
- Your proposed approach(es), with reasoning for why this is principled
- Specific formulas / constants / code changes, with justification for each
  number or formula shape
- The validation protocol you plan to run (see Section 7 for the budget)
- Any concerns, alternatives considered, or open questions

**Then STOP and ask the user to review.** Do not proceed to implementation
without the user's OK. This is the one check-in. The `bewary.md` rule in
`.claude/rules/bewary.md` makes this mandatory — formulas and tolerance
values must be discussed before implementation.

### Phase 2 — Implementation and validation (autonomous after approval)

After the user approves an approach:
1. Implement the minimal code change(s).
2. Validate using the protocol below.
3. If it works, commit and update `FINDINGS.md` + the experiment README.
4. If it doesn't, iterate (you may need another check-in if the approach
   changes substantially).

You can commit intermediate progress as often as makes sense. Descriptive
messages only — no Co-Authored-By lines.

---

## 7. Validation protocol

Budget: small enough to iterate quickly, large enough to be conclusive.

**Validation cells (11 total):**
- 9 degraders: `[39, 35, 16, 13, 10, 33, 15, 14, 27]` — all of them, because
  universality is the goal
- 2 controls (sanity check we didn't hurt good cells): pick one improver
  with large gain (e.g., Cell 8, +0.121) and one flat cell at ceiling
  (e.g., Cell 1, 0.98). Justify your choice.

**M grid**: 3-4 values covering small/medium/large. Suggestion:
`[50, 300, 1500]` or `[100, 500, 1500]`. Pick one and justify.

**Seeds**: 3 seeds (same as main sweep: 0, 1, 2).

Total runs per candidate config: ~100. At ~60-120s per run, that's 2-3
hours per candidate. Fine for iterating.

**Success criteria** — a candidate "works" if:
- **All 9 degraders** show trend ≥ flat (|delta M=small→M=large| < 0.005)
  OR at minimum Cell 39 and Cell 35 are saved. Perfect save is ideal.
- **Both control cells** are within 0.005 of their baseline numbers in
  `M_sweep_results.jsonl`.
- Grand mean across the 11 cells at the largest M is not worse than the
  baseline.

Store results in a new experiment folder under
`experiments/YYYY-MM-DD_<descriptive_name>/` following the convention
documented in `experiments/2026-04-06_es_sweeps_64x64/README.md` and
`experiments/2026-04-13_M_sweep_64x64/README.md`.

**IMPORTANT — save the final trained model for every run** (critical rule #9
in `.claude/CLAUDE.md`). One `.pt` per run, the final trained state — NOT
per-iteration snapshots. The M-sweep from April 13 did NOT save any models
and that was a costly mistake. Use
`eigenspace_checkpoint.save_eigenspace_checkpoint()` per run, naming the
files `<experiment_dir>/models/cell_XX_M<M>_seed<S>.pt`. The pattern is in
`train_ceiling_models.py`. The JSONL alone is not sufficient for a colleague
to run inference on these models later.

**Reference baseline** (for comparison, already on disk):

| Cell | M=50 | M=300 | M=1500 |
|------|------|-------|--------|
| 1 | 0.981 | 0.983 | 0.982 |
| 8 | 0.750 | 0.862 | 0.871 |
| 10 | 0.905 | 0.899 | 0.890 |
| 13 | 0.951 | 0.931 | 0.934 |
| 14 | 0.909 | 0.909 | 0.896 |
| 15 | 0.704 | 0.698 | 0.691 |
| 16 | 0.955 | 0.955 | 0.927 |
| 27 | 0.905 | 0.897 | 0.898 |
| 33 | 0.969 | 0.962 | 0.955 |
| 35 | 0.806 | 0.779 | 0.749 |
| 39 | 0.699 | 0.649 | 0.501 |

(Mean of 3 seeds, from `M_sweep_results.jsonl`.)

---

## 8. Expected deliverables

At the end of the investigation (whenever that is — you set the pace):

1. **A working fix, committed.** Code changes in `Spatial_GP_repo/scripts/
   gpytorch_porting/`. Tests pass (or you explain why not).
2. **A new experiment folder** with README + JSONL documenting the
   validation runs, following the convention in
   `experiments/2026-04-13_M_sweep_64x64/`.
3. **Updated `investigations/M_degradation/FINDINGS.md`** with a new section
   describing the fix, the reasoning, and the validation results.
4. **Updated `ToDo.md`** — mark the "Data-adaptive A initialization" entry
   as resolved (or scope remaining work).
5. **Updated `.claude/CLAUDE.md`** — the Authoritative Sources table should
   point to the new experiment folder and any new rules/config.
6. **`REGULARIZATION_PROPOSAL.md`** with the approved approach recorded
   (from Phase 1).

---

## 9. Things the first session wanted you to know

- `build_config_from_defaults()` does NOT produce a working config for
  interleaved training. See `experiments/2026-04-13_M_sweep_64x64/README.md`
  for the required overrides. Always pass `A_init=1e-4, lambda0_init=-1.0,
  n_estep=50, n_mstep=20, n_iterations=80` explicitly unless you're changing
  them intentionally.
- The previous session found that the April 9 commits (`97bc88f`, `0f4dfff`)
  changed the seed → IP mapping but do NOT change performance for matched
  configs. This is not a code bug.
- The E-step loss decomposition (ELL + KL from `curves_train_loss` and
  `curves_train_log_lik`) is stored per iteration in the JSONL, so you can
  study the KL trajectory without re-running anything.
- Cells 12 and 30 were thought to be "permanently failing" but succeed fine
  with the correct config. Do not assume any cell is "broken" without
  verification.

---

## 10. What NOT to do

- Do not increase `EIGVAL_TOL` or otherwise shrink `n_b` as the fix. That's
  a capacity cap, not regularization.
- Do not switch modes, switch kernels, or change the likelihood.
- Do not implement something for Cell 35 specifically that you wouldn't
  also want active for Cell 1. One set of rules.
- Do not spend compute on the 108×108 dataset. Stay on 64×64.
- Do not write 500-line abstractions. A config flag plus 20-30 lines of
  code change is a reasonable target.
- Do not skip the Phase 1 proposal. The user wants to see the reasoning
  before you modify library code.
- Do not re-run the full 984-run sweep to validate. Use the small
  validation protocol in Section 7.
- Do not merge `pietro/investigate-M-degradation` into any other branch.
  Stay on the investigation branch.

---

## 11. Continuation prompt for starting the session

When you start, after the environment checks in Section 0:

> I've read INVESTIGATION_PROMPT_V2_hyperparam_regularization.md and the
> background files it references. I'm going to start Phase 1 by verifying
> the mechanism myself and drafting the proposal.

Then proceed autonomously until the Phase 1 proposal is written. Stop
there and wait for user review.
