# Experiment YAML configs

This folder holds YAML configs for `run_experiment.py`. There are two active
templates that track the current project defaults, and six legacy configs
that were frozen for historical experiments.

## Active templates (follow current defaults)

| File | Purpose |
|------|---------|
| `canonical.yaml` | Full test matrix template for `create_experiment.py` (modes x M x seeds x cells). All 35+ params, annotated with HARDCODED/WIRED tags. |
| `quick.yaml` | Single-point defaults for `run_experiment.py --quick` exploratory runs. |

These two files are kept in sync with the project's current default choices
(see `default_params.json` and `.claude/CLAUDE.md`). As of April 2026:

- `es_metric: elbo` (only 'elbo' or 'none' are valid)
- `n_val_split: 0` (no validation carving, all 3160 training images used)
- Val metrics (`val_log_lik`, `val_r`, `val_rho`) are NOT used for early
  stopping. They are still computed as diagnostic curves when
  `n_val_split > 0`.

See `.claude/DECISION_LOG.md` Q32 for the ELBO ES decision rationale and
`experiments/2026-04-06_es_sweeps_64x64/README.md` for the empirical
comparison.

## Legacy configs (pre-April-2026, frozen)

| File | Originally ran with |
|------|---------------------|
| `massive_allcells_108.yaml` | `es_metric: val_ll`, `n_val_split: 250` (implicit — keys were absent, fell back to the old defaults of the time) |
| `massive_allcells_64.yaml` | same |
| `massive_allcells_48.yaml` | same |
| `smoke_test_108.yaml` | same |
| `smoke_test_64.yaml` | same |
| `smoke_test_48.yaml` | same |

These configs predate the addition of `es_metric` and `n_val_split` keys
to the YAML schema. They were run in March 2026 with `val_ll` ES and
`n_val_split=250`, which were the project defaults at the time.

**If you re-run them as-is**, the `flatten_yaml_config()` fallback in
`run_single_mode.py` (`es.get('es_metric', 'elbo')`,
`dat.get('n_val_split', 0)`) will apply the **current** defaults:
ELBO ES and no validation carving. The behaviour will differ from the
original runs.

**If you need to reproduce the original runs** exactly:
1. `val_ll` ES is no longer a valid choice — adding `es_metric: val_ll`
   to these files will raise a `ValueError` at training time.
2. Checkout the commit that was current when they were originally run
   (see the experiment folders under `experiments/` for recorded git
   commits in each record's metadata) before re-running.

The legacy configs are kept in-place rather than moved to an archive
because the experiment folders under `experiments/` reference them by
relative path. Historical records (`experiments/2026-03-*/`) expect
them here.
