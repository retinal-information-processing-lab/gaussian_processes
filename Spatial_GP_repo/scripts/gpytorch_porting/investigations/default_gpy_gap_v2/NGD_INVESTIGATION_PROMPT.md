# Hand-off prompt: Investigate GPyTorch Natural Gradient Descent (NGD) as a fix for joint-LBFGS instability on non-conjugate SVGP

**Read this entire document before doing anything.** It is a hand-off from a
previous session. You are highly autonomous, but autonomy does not mean
jumping to implementation. Follow the process in §7 below.

---

## 1. Environment

- **Working directory**: `/home/pietro/GP/gaussian_processes/Spatial_GP_repo/scripts/gpytorch_porting/`
- **Python interpreter** (DO NOT USE the system python): `/home/pietro/conda-envs/gp_neural/bin/python`
- **GPyTorch version**: 1.15.2 (confirmed). Installed at:
  `/home/pietro/conda-envs/gp_neural/lib/python3.11/site-packages/gpytorch/`
- **GPU**: CUDA available. All training uses float32 (see CLAUDE.md CRITICAL RULE 3).
- **Git branch**: `pietro/investigate-default-gpy` (verify at session start with `git branch --show-current`).
- **Git status at hand-off**: working tree has multiple new files in
  `investigations/default_gpy_gap_v2/`; nothing under training code paths
  is modified. Run `git status --porcelain` and raise to the user if
  anything besides `investigations/default_gpy_gap_v2/` is dirty.

## 2. Required reading (do this first)

In strict order — do not skip, do not skim:

1. `investigations/default_gpy_gap_v2/PROMPT.md` — the v2 investigation
   charter. The original question the sweep was designed to answer.
2. `investigations/default_gpy_gap_v2/SCRAPBOOK.md` — ALL previous findings
   (P2-1 through P2-5). In particular: the GAP decision (§6.3), the outlier
   diagnostics (§ "Phase 2 Finding P2-1"), the beta_init=0.2 partial fix
   (P2-3 through P2-5).
3. `.claude/CLAUDE.md` (the one in `gpytorch_porting/.claude/`) — project
   conventions, CRITICAL RULES, parameter tables, file map, known issues.
4. `.claude/rules/bewary.md` — **hard rules about hardcoding, asking for
   permission, and deferring to the user**. These apply even under
   permissions-skip. Read carefully.
5. `.claude/rules/critical_short_rules.md` — short gotchas, including
   "No hidden hardcoded parameters" and "Use `_constants.py`".
6. `.claude/PORTING_LESSONS.md` — dead ends. § "Whitening Kernel Mismatch"
   explains why we cannot use a custom E-step inside GPyTorch's
   VariationalStrategy (whitened L_K becomes corrupt across M-steps).
   This is the exact path that was ruled out before NGD was proposed.
7. `deprecated/vargp_style_*.py` — the abandoned EM-in-GPyTorch code.
   Look for 2 minutes to internalize what failed — then move on.

Memory files (apply throughout the session):
- `/home/pietro/.claude/projects/-home-pietro-GP-gaussian-processes/memory/MEMORY.md`
- `feedback_motivate_experiment_choices.md` — motivate each experiment
  BEFORE running it, with stated hypothesis, design, and known confounds.
- `feedback_tentative_conclusions.md` — do NOT write "supported",
  "confirmed", "root cause found" until the user has seen the result and
  weighed in. Use "consistent with", "appears to", "needs replication".

## 3. Context: why this session exists

We are investigating a gap between `vargp_direct` (our custom eigenspace
EM implementation) and `default_gpy` (standard GPyTorch SVGP with joint
LBFGS). Decision from the v2 sweep at (M=300, n_train=1500, 64x64,
arc_cosine, ground_truth RF, ELBO ES on):

```
mean_Δ(vargp_direct − default_gpy) = +0.0782 over 48 (cell × seed) pairs
GAP EXISTS per the pre-registered 0.02 threshold.
```

The gap is heavy-tailed — mostly driven by 2 outlier cells (40 and 38)
where `default_gpy`'s joint-LBFGS commits to a bad local optimum in 5-14
outer iterations and stays there. Diagnostics show two failure modes:

- Cell 40: joint-LBFGS pushes A from 0.01 to 0.4+ in 3 outer iterations,
  beta collapses to 0.011 (vs vargp_direct's 0.11), ELBO plateaus at 459
  (vs 387).
- Cell 38: LBFGS freezes at iter 5 at a bad plateau (loss 1030 vs
  vargp_direct's 661). IDENTICAL loss for 15 iters → ES fires.

A 3-seed intervention with `beta_init=0.2` closed ~63% of the aggregate
gap but did not eliminate it — it's a basin-assignment trick, not a fix.

## 4. The hypothesis you are testing

The user asked an excellent ML-engineering question: why does GPyTorch's
canonical SVGP struggle where a custom method succeeds? The answer, based
on literature + inspection of both implementations, is that
**`vargp_direct`'s Newton E-step is mathematically a form of natural
gradient descent on the variational parameters**, and joint LBFGS on the
whitened-Cholesky parameterization is known to be ill-conditioned for
non-conjugate SVGP.

GPyTorch **does** ship a canonical solution for this case. We just
haven't tried it yet.

Reference: Salimbeni, Eleftheriadis, Hensman (2018). "Natural Gradients
in Practice: Non-Conjugate Variational Inference in Gaussian Process
Models." AISTATS 2018. arxiv.org/abs/1803.09151. The paper argues
natural gradients in the η-parameterization dramatically accelerate
convergence for non-conjugate SVGP, particularly in ill-conditioned
posteriors where ordinary gradients are "unusable."

## 5. GPyTorch NGD API (from local installation)

Path: `/home/pietro/conda-envs/gp_neural/lib/python3.11/site-packages/gpytorch/`

### NaturalVariationalDistribution
- File: `variational/natural_variational_distribution.py`
- Signature: `__init__(num_inducing_points, batch_shape=torch.Size([]), mean_init_std=1e-3, **kwargs)`
- Docstring: "A multivariate normal `_VariationalDistribution`, parameterized by **natural** parameters."
- Warning (verbatim): "The `NaturalVariationalDistribution` can only be
  used with `gpytorch.optim.NGD`, or other optimizers that follow exactly
  the gradient direction. Failure to do so will cause the natural matrix
  Θ_mat to stop being positive definite, and a RuntimeError will be raised."
- Converges faster but is less numerically stable.

### TrilNaturalVariationalDistribution
- File: `variational/tril_natural_variational_distribution.py`
- Same signature as above.
- Parameterizes via a triangular decomposition of the natural matrix.
- Docstring note: "more numerically stable ... at the cost of needing
  more iterations to make variational regression converge."

### gpytorch.optim.NGD
- File: `optim/ngd.py`
- Signature: `NGD(params, num_data, lr=0.1)`
- Step: `p += -lr * num_data * p.grad` under `@torch.no_grad()`.
- Docstring: "Implements a natural gradient descent step. It **can only**
  be used in conjunction with a `_NaturalVariationalDistribution`."

### Official tutorial
- File: `examples/04_Variational_and_Approximate_GPs/Natural_Gradient_Descent.ipynb`
- Pattern: two optimizers. NGD for variational params
  (`model.variational_parameters()`), Adam for hyperparameters + likelihood.
- Within each step: `ngd.zero_grad(); adam.zero_grad(); loss.backward(); ngd.step(); adam.step()`
- Tutorial quotes: "Use a large learning rate for the variational
  optimizer. Typically, 0.1 is a good learning rate." / "You must use
  `gpytorch.optim.NGD` as the variational NGD optimizer! Adaptive gradient
  algorithms will mess up the natural gradient steps."

## 6. Defaults you may use (with user's explicit permission, if motivated)

These are the values the previous session recommended after reading the
docs and tutorial. **You are free to change them** if you find a
principled reason to — but if you do, write the motivation in the
scrapbook before running. You are NOT free to treat them as permanent
defaults if you see anything suspicious — flag it.

| # | Decision | Default | Rationale | Alternatives allowed |
|---|---|---|---|---|
| 1 | Variational distribution | `TrilNaturalVariationalDistribution` | stability over speed (float32, Poisson likelihood) | `NaturalVariationalDistribution` if Tril converges too slowly |
| 2 | NGD lr | 0.1 | tutorial default, paper's recommendation | smaller if instability |
| 3 | Adam lr | 0.01 | tutorial default for hyperparams | adjust with justification |
| 4 | Epochs | 200 | first-order optimizer needs many more steps than LBFGS's O(1000) grad evals | go longer if convergence unclear |
| 5 | Hyperparam clamp | apply `clamp_hyperparameters()` + `clamp_params()` after each Adam step | consistent with existing LBFGS 3-layer defense | none |
| 6 | Batching | full-batch | n_train=1500 fits easily, clean comparison with existing sweep | minibatches if noise matters |
| 7 | Early stopping | OFF for prototype | want full trajectory visible | re-enable after prototype with same ELBO-ES machinery |
| 8 | Kernel/likelihood init | match main sweep (beta=0.1, A=0.01, lambda0=1.0, ground_truth RF) | clean comparison | do NOT silently change |

**All of these are hardcoded numerical values from the bewary.md
perspective.** Do not silently decide they are "fine". Flag any you plan
to deviate from, and flag any you plan to use as-is but want the user to
endorse explicitly. The user has already been told these defaults and
said "proceed" — so repeating each one individually is redundant — but
any DEVIATION needs an explicit flag.

## 7. Process you must follow

**DO NOT jump to coding. DO NOT jump to a full sweep.**

1. **Read (§2).** Confirm to the user you've read the required files and
   raise any questions about the charter, past findings, or design defaults.
2. **Independent research (optional but encouraged).** Use a subagent of
   type `general-purpose` (or `Explore`) to:
   - Re-verify the GPyTorch NGD API against the installed version (in case
     something changed).
   - Search online for any _recent_ (post-Salimbeni 2018) work on NGD with
     Poisson / count-likelihood SVGP. What do people actually use?
   - Check GPyTorch GitHub issues for known NGD failures or caveats.
   - Look for better alternatives — e.g., PASGD, TyXe, Pyro's SVGP with
     natural gradients, structured black-box VI. If you find something
     better-motivated than NGD, raise it BEFORE running NGD.
3. **Enter plan mode.** Write a concrete plan:
   - Where the new code will live (prefer
     `investigations/default_gpy_gap_v2/ngd_prototype.py` — or a new
     subfolder `investigations/ngd_investigation/` if you prefer).
   - Exactly which cells and seeds for the prototype.
   - Exact metrics you will report and exact decision criterion for
     moving from prototype to full sweep.
   - Known confounds.
4. **Exit plan mode autonomously** once the plan is self-consistent.
5. **Prototype run**: cheap — single seed, 4–6 cells (list below).
6. **Analyze.** Update SCRAPBOOK.md (new section "Phase 3 — NGD
   investigation"). Do NOT write "NGD works" or "NGD doesn't work".
   Write "consistent with" / "not consistent with" / "needs more seeds".
7. **Hand back to user** with the prototype result, analysis, and a
   proposal for the next step. Do NOT launch the full sweep on your own.
   The user must see the prototype first.

## 8. Cells and seeds for comparability

The main sweep used 16 cells (`cells_used.json`) at 3 seeds: {42, 123, 789}.
The 14.6-minute sweep produced `results.jsonl` (112 records). The
beta_init=0.2 3-seed sweep produced `step3_results.jsonl` + `step4_results.jsonl`.

**For the NGD prototype, use cells that maximize comparability with what
we have:**

- **Known-failures (default_gpy at baseline)**: cells 40, 38 — both
  systematically bad under baseline default_gpy.
- **Known-healthy (default_gpy at baseline)**: cells 16, 30 — both
  consistently good under baseline default_gpy across seeds.
- **Interesting (seed-sensitive)**: cell 29 — fine at baseline, breaks at
  seed 42 under beta_init=0.2 (the "LBFGS-freezes-at-bad-plateau" mode).
- **User-added**: cell 8 — optional for this prototype; can be skipped.

**Suggested prototype scope**: 5 cells x 1 seed (42) = 5 runs. If NGD
converges cleanly and matches or beats `vargp_direct` on cells 40 and 38,
propose seed-replication (seeds 123, 789) as the NEXT step, NOT as part
of the same run.

Baselines for comparison (pull from existing JSONL files):
- `vargp_direct` seed 42 test_r per cell (target to match)
- `default_gpy` baseline seed 42 test_r per cell (current floor)
- `default_gpy beta_init=0.2` seed 42 (partial-fix baseline for context)

## 9. Hard constraints

- **DO NOT modify `default_params.json`** without user approval.
- **DO NOT modify** `gpy_model.py`, `gpy_training.py`, `kernels.py`,
  `likelihoods.py`, `eigenspace_*.py`, `run_single_mode.py` unless the
  user explicitly approves. The NGD prototype should be **self-contained
  in the investigation folder**, importing existing components where
  possible but NOT editing them.
- **DO NOT hardcode any numerical tolerance / threshold / step-size**
  without flagging it in SCRAPBOOK.md AND raising to the user.
- **DO NOT touch the `deprecated/` folder** (it is the record of what
  failed before). You can read it for context.
- **DO NOT run the full 16x3-seed sweep before prototype results are
  user-approved.**
- **DO NOT declare NGD a fix** based on 5 cells x 1 seed.

## 10. Known-hazardous specifics

- GPyTorch's `VariationalStrategy` promotes K_uu to float64 for Cholesky
  internally (`.claude/rules/jitter.md`). Our `model.jitter=1e-4` feeds
  both the pre-Cholesky jitter and the retry-jitter-start. NGD does NOT
  change this machinery — the Cholesky stack is untouched — but if you
  see "NotPSDError: up to 1e-02" errors, check whether you've changed
  `jitter_val`.
- Our `ArcCosineKernel` has bound `BETA_MAX=0.3` (beta explosion
  safeguard). `PoissonLikelihood` has `A_MAX=10` and `lambda0 ∈ [-50, 50]`.
  Adam with lr=0.01 should not push these, but `clamp_hyperparameters()` +
  `clamp_params()` are your safety net. Apply them **after** each
  `adam.step()`, not inside a closure (Adam doesn't use closures).
- `NaturalVariationalDistribution` comes with a runtime check at
  `natural_variational_distribution.py:103-107` that re-raises cholesky
  failures with message "You probably updated it using an optimizer other
  than gpytorch.optim.NGD (such as Adam). This is not supported." If you
  see this error, check that Adam's parameter group does NOT include the
  variational distribution's parameters (natural parameters). Use
  `model.variational_parameters()` vs `model.hyperparameters()` +
  `likelihood.parameters()` per the GPyTorch tutorial.
- Our `VariationalGPModel` in `gpy_model.py:16` uses
  `CholeskyVariationalDistribution`. The NGD prototype needs a different
  class (Tril or plain Natural). Either subclass with a constructor
  override, or write a fresh minimal model class in the investigation
  folder. Keep the latter simpler.

## 11. Subagents encouraged

The `Agent` tool with `subagent_type` can parallelize investigation work.
Suggested uses:

- `Explore` or `general-purpose`: codebase spelunking (e.g., "Find all
  places `CholeskyVariationalDistribution` is used in this repo and
  document the API surface we'd need to swap.")
- `general-purpose`: live web/GitHub research (e.g., "Find any GPyTorch
  GitHub issues or blog posts about NGD instability on non-conjugate
  likelihoods. Report with URLs.")
- `Plan`: software architecture for the prototype (e.g., "Design a
  self-contained `ngd_training.py` function that takes model, likelihood,
  data, and trains via NGD+Adam. Return step-by-step plan.")

Use them in parallel with a single multi-tool message when work is
independent.

## 12. Scrapbook protocol

- Add a new top-level section to
  `investigations/default_gpy_gap_v2/SCRAPBOOK.md` titled
  "**Phase 3 — NGD investigation**".
- Write a Charter subsection first (before any code), using the plan you
  produced.
- After each prototype run, add a "Finding P3-n" subsection with the
  result, tables, and tentative interpretation.
- List any files you create in the artifacts / cleanup inventory at the
  bottom of SCRAPBOOK.md (there's an existing template to match).

## 13. Baseline values to compare against (already computed — do not re-run)

At seed 42, M=300, n_train=1500, 64x64:

| cell | vargp_direct test_r | default_gpy baseline test_r | default_gpy beta=0.2 test_r |
|---|---|---|---|
| 40 | 0.8553 | 0.3663 | 0.8224 |
| 38 | 0.7776 | 0.3106 | 0.5795 |
| 16 | 0.9581 | 0.8794 | 0.8766 |
| 30 | 0.7469 | 0.7918 | 0.7973 |
| 29 | 0.7287 | 0.7527 | 0.5183 |
| 8  | 0.8551 | 0.8142 | 0.6948 |

Full records in `investigations/default_gpy_gap_v2/results.jsonl` and
`step3_results.jsonl`. Read these directly from disk rather than
retraining.

## 14. What "success" would look like

Tentative markers of a positive signal (NOT a fix):

- NGD converges without PSD/NaN errors on the 5-cell prototype.
- On cells 40 and 38, NGD's test_r ≥ vargp_direct's (0.85 and 0.78 respectively).
- On healthy cells (16, 30, 8), NGD does not regress below default_gpy baseline.
- ELBO curve is smooth, monotone-ish, no "freeze-forever" pattern.

If any of these fail, the finding is "NGD prototype did not converge
cleanly on 5-cell prototype — diagnose before scaling." That is still a
valuable result; it answers the question.

## 15. Final exhortations

- You have `--dangerously-skip-permissions`. This does NOT mean you can
  skip thinking, planning, or asking. It means the tool layer trusts you;
  the USER still expects you to follow this hand-off's process.
- **Think like an ML engineer**: be skeptical of first results, prefer
  cheap diagnostic runs over expensive sweeps, keep the user in the loop
  with short summaries, and surface trade-offs before committing.
- The goal of this session is NOT "ship NGD as the new default." The
  goal is to answer "**does NGD eliminate, reduce, or have no effect on
  the `vargp_direct` vs `default_gpy` gap on our data?**"
- If the answer is "no effect" or "negative", that is a completely valid
  finding. Report it honestly and stop.

Good luck.
