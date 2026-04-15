# Task — Train GP models at M=300 and assemble a standalone inference package

You are a Claude Code session running on a machine that has:
- A GPU available for use (unlike the other machine, which can't use its GPU right now)
- A local clone of the `gaussian_processes` repo
- The `pytorch_gpytorch` conda environment installed
- Data files (`PNAS_64x64_center_crop_no_renorm.npz` and `rf_centers_ground_truth.npz`) already present at `Spatial_GP_repo/scripts/gpytorch_porting/datasets/`

Read this entire document before taking any action. It contains everything you need.

---

## 0. First — verify the environment

Run these checks in order and stop if any fail.

```bash
cd <repo root on this machine>        # gaussian_processes/ — adjust to your path
git fetch origin
git checkout pietro/investigate-M-degradation
git pull --ff-only origin pietro/investigate-M-degradation
git branch --show-current              # must print: pietro/investigate-M-degradation
git log --oneline -1                   # must print: 2af6f6b Add train_and_save_M300.py...
pwd

# Verify the training script exists on disk
ls Spatial_GP_repo/scripts/gpytorch_porting/experiments/2026-04-13_M_sweep_64x64/train_and_save_M300.py

# Verify data files are present (they should NOT be gitignored-but-missing)
ls -la Spatial_GP_repo/scripts/gpytorch_porting/datasets/PNAS_64x64_center_crop_no_renorm.npz
ls -la Spatial_GP_repo/scripts/gpytorch_porting/datasets/rf_centers_ground_truth.npz

# Verify conda env
/home/*/anaconda3/envs/pytorch_gpytorch/bin/python -c "import torch; print('torch', torch.__version__, 'CUDA', torch.cuda.is_available())"
# Expected: torch 2.5.x+cu121, CUDA True
```

If any check fails, stop and tell the user. Do not guess or work around it.

---

## 1. Safety constraints (session runs with `--dangerously-skip-permissions`)

You have broad permissions. In exchange, follow these rules strictly:

- **Stay on `pietro/investigate-M-degradation`.** NEVER run `git checkout`, `git switch`, or `git worktree add`. The one exception is the initial checkout in Section 0 to get ON this branch, which must land exactly there.
- **NEVER `git push --force`, `git push origin main`, or push to any branch other than `pietro/investigate-M-degradation`.** Regular `git push origin pietro/investigate-M-degradation` is fine.
- **Scope your work to `Spatial_GP_repo/scripts/gpytorch_porting/` and the new deliverable folder you create.** Do not edit anything outside this subtree (no parent READMEs, no `SETUP.md`, no `.claude/` outside the gpytorch_porting tree).
- **Do not change existing values in `default_params.json`**. Adding new keys is allowed; changing existing ones breaks other branches.
- **NEVER delete files that are already tracked in git.** If you think a file should be removed, propose and wait.
- **Commit incrementally.** After training completes, after the package folder is assembled, after the README is written. Not one giant commit at the end.
- **Before any commit, run `git branch --show-current`** to confirm the branch.
- **Do not run any investigation, sweep, or exploratory training beyond what this prompt describes.** The task is narrow and specified.

If you find yourself wanting to do something outside these rules, stop and ask the user.

---

## 2. What this task IS (and what it isn't)

This is a **delivery task**, not an investigation. You are not debugging, exploring, or designing a fix. You are:

1. Running an existing, already-written training script that trains 41 GP models and saves `.pt` checkpoints.
2. Assembling the trained models, the required code, the dataset, and a README into a single self-contained folder that a non-technical colleague can use for inference without needing this repo.

**Do not modify the training config.** The config is already correct and was chosen after extensive investigation — see the background in Section 4. Your job is to execute, not redesign.

**Do not re-run the training if it already succeeded** — check first.

---

## 3. The task, step by step

### Step A: Run the training script in the background

The script `Spatial_GP_repo/scripts/gpytorch_porting/experiments/2026-04-13_M_sweep_64x64/train_and_save_M300.py` already exists on this branch (committed as `2af6f6b`). It:

- Trains 41 cells (cells 0 through 40) at M=300, seed=0, on the 64×64 PNAS dataset.
- Uses the `vargp_direct` mode with the full ES config: `A_init=1e-4, lambda0_init=-1.0, n_estep=50, n_mstep=20, n_iterations=80, interleave_fstep=True, fix_Amp=True`.
- Saves one `.pt` file per cell via `save_eigenspace_checkpoint()` (the correct checkpoint format for `vargp_direct`).
- Writes a summary `ceiling_results.json` and per-run `sweep_results.jsonl`.
- Is resume-safe: if interrupted, re-running picks up where it left off.

Output directory: `Spatial_GP_repo/scripts/gpytorch_porting/checkpoints/64x64_M300_es_intl_fixAmp/`

#### How to launch

**Important**: a previous session on the other machine launched this with `nohup python ...` and the subshell did not inherit the conda env, so `import torch` failed silently. You must use the full path to the conda env's python. Example:

```bash
cd Spatial_GP_repo/scripts/gpytorch_porting

PY=/home/idv-eqs8-pza/anaconda3/envs/pytorch_gpytorch/bin/python
# (adjust this path for the current machine — do `which python` with the env active, or
# `ls /home/*/anaconda3/envs/pytorch_gpytorch/bin/python` or equivalent for miniconda3/opt/conda)

nohup $PY experiments/2026-04-13_M_sweep_64x64/train_and_save_M300.py \
  > experiments/2026-04-13_M_sweep_64x64/train_M300.log 2>&1 &
echo "PID=$!"
```

After a few seconds, confirm the process is still running:
```bash
ps -p <PID> -o pid,etime,cmd
head -20 experiments/2026-04-13_M_sweep_64x64/train_M300.log
```

The log should show "M=300 ES-config all-cells training + checkpoint save" and start processing Cell 0. If it says `ModuleNotFoundError: No module named 'torch'`, you used the wrong python — kill the process and re-launch with the correct absolute path.

#### Expected runtime

About 40-60 minutes for all 41 cells. You can either:

- **Monitor periodically** with `tail -20 experiments/2026-04-13_M_sweep_64x64/train_M300.log` and `ls Spatial_GP_repo/scripts/gpytorch_porting/checkpoints/64x64_M300_es_intl_fixAmp/cell_*.pt | wc -l` — do NOT poll more than once every 2-3 minutes.
- **Proceed to drafting the package structure / README** in parallel (Steps B-C) while training runs — just do NOT commit anything training-related until the run finishes.

#### Expected results (to verify when training completes)

Mean `test_r` across the 41 cells should be **approximately 0.838** (± very small numerical noise). This matches the reference data in `Spatial_GP_repo/scripts/gpytorch_porting/experiments/2026-04-13_M_sweep_64x64/M_sweep_results.jsonl` for the same (cell, M=300, seed=0) tuples — same code, same config, same seed → deterministic result.

Quick verification snippet after training:

```python
import json
from pathlib import Path
base = Path("Spatial_GP_repo/scripts/gpytorch_porting")
summary = json.load(open(base / "checkpoints/64x64_M300_es_intl_fixAmp/ceiling_results.json"))
test_rs = [v["test_r"] for v in summary.values() if v.get("test_r") is not None]
print(f"{len(test_rs)} cells, mean test_r = {sum(test_rs)/len(test_rs):.4f}")
# Expected: 41 cells, mean test_r ~ 0.838
```

If the mean is more than 0.005 away from 0.838, something is off — stop and tell the user.

### Step B: Assemble the shareable package

Create a new folder `Spatial_GP_repo/scripts/gpytorch_porting/deliverables/2026-XX-XX_vargp_direct_M300/` (replace `XX-XX` with today's date, UTC). Inside, assemble the following **self-contained** structure:

```
deliverables/2026-XX-XX_vargp_direct_M300/
├── README.md                              # colleague-facing, see Step C
├── environment.yml                        # conda env spec (copy from Spatial_GP_repo/scripts/gpytorch_porting/package/environment.yml and verify it's current)
├── data/
│   ├── PNAS_64x64_center_crop_no_renorm.npz   # copy the real file (dereference the symlink if it is one; the colleague must receive the actual NPZ, not a broken symlink)
│   └── rf_centers_ground_truth.npz
├── checkpoints/
│   ├── cell_00.pt ... cell_40.pt          # the 41 .pt files produced in Step A
│   └── ceiling_results.json               # the summary produced in Step A
├── code/                                  # the minimum set of modules required to load a checkpoint and run inference
│   ├── default_params.json
│   ├── _constants.py
│   ├── kernels.py
│   ├── likelihoods.py
│   ├── metrics.py
│   ├── utils.py
│   ├── eigenspace_model.py
│   ├── eigenspace_utils.py
│   ├── eigenspace_estep.py                # only if imported transitively — verify
│   ├── eigenspace_fstep.py                # only if imported transitively — verify
│   ├── eigenspace_mstep.py                # only if imported transitively — verify
│   ├── eigenspace_training.py             # needed for `predict_eigenspace`, which is the inference entry point
│   ├── eigenspace_gradients.py            # only if imported transitively — verify
│   ├── eigenspace_checkpoint.py           # provides `load_eigenspace_checkpoint`
│   ├── analytical_gradients.py            # only if imported transitively — verify
│   ├── analytical_gradients_vjp.py        # only if imported transitively — verify
│   ├── inference.py                       # NEW FILE you write — see below
│   └── ...                                # any other local dependency that the above import (follow imports to make sure nothing is missing)
└── results/                               # (OPTIONAL — populate only if you have compute budget left after training)
    ├── summary.csv                        # per-cell test_r, explained_var, hyperparameters
    └── plots/cell_XX.png                  # STA + RF overlay + test scatter, one per cell
```

**Important notes on the `code/` folder:**

- **Do NOT copy `run_inference.py` directly.** As of commit `2af6f6b`, `run_inference.py` uses `from checkpoint import load_checkpoint` — which is the `default_gpy` path. Our checkpoints are eigenspace format and must be loaded with `load_eigenspace_checkpoint` from `eigenspace_checkpoint.py`. `run_inference.py` would need adaptation to handle eigenspace checkpoints, and that is more scope than this task should take. Instead:
- **Write a new `inference.py`** inside `code/` that:
  - Takes `--data <path to PNAS_64x64 npz>` and `--checkpoints <path to checkpoints folder>` and optional `--cells 1 8 10`
  - For each cell: loads the checkpoint via `load_eigenspace_checkpoint`, runs prediction via `predict_eigenspace` (in `eigenspace_training.py`), computes test_r and explained_var via functions in `metrics.py`, writes a row to `summary.csv` and generates a per-cell plot (STA + scatter).
  - Keep it simple — do NOT over-engineer. Follow the structure of `run_inference.py` but swap the checkpoint loading and prediction paths.

- **Figuring out the minimum file set**: start with `eigenspace_checkpoint.py` and `eigenspace_training.py` as entry points, walk their imports (read the `import` lines at the top), and recursively include every local module they touch. Only local files (same-directory imports) count — third-party packages come from the conda env. When unsure, include the file; it's better to have an unused file than a missing one. If you find yourself needing a file outside `Spatial_GP_repo/scripts/gpytorch_porting/`, stop — that's a sign of a scope miss.

- **Dereference symlinks when copying**. The data files may be symlinks. Use `cp -L` or `rsync -L`, NOT plain `cp`, to make sure the deliverable folder contains real files, not dangling symlinks. The dataset files are about ~53 MB (64x64 NPZ) and 8 KB (rf_centers).

- **`default_params.json`** is used by `build_config_from_defaults()` and also by `_constants.py` at import time — it's required. Copy it verbatim.

- **Do NOT include** `investigations/`, `experiments/`, `checkpoints/` (beyond what we're delivering), old `1D_playground`, `2D_playground`, `package/`, or any historical scripts. The deliverable must be lean and self-contained.

### Step C: Write the README.md

The README is the most important file for the colleague. Write it assuming:

- The colleague has basic Python/conda knowledge but does NOT know our project history.
- They want to load the models and get predictions on the test set, NOT retrain.
- They will copy the folder to their machine (which has a CUDA GPU) and run inference.

**Required sections:**

1. **What this is.** One-paragraph summary: "Pre-trained variational GP models for 41 retinal ganglion cells on 64×64 natural images, one model per cell, M=300 inducing points, trained with (config summary)."

2. **Folder layout.** Show the directory tree (exactly as assembled in Step B) with a one-line description per file/folder.

3. **Setup.** `conda env create -f environment.yml`, `conda activate gp_neural` (or whatever the env is named in environment.yml — use the real name), then verify with `python -c "import torch; print(torch.cuda.is_available())"`.

4. **Run inference.** Exact command to run (using the new `code/inference.py`), and what outputs to expect. Show a small example command and describe the outputs (`summary.csv`, `plots/`).

5. **Model summary.** A small table showing per-cell `test_r`, `explained_var`, number of inducing points (constant at M=300), and final hyperparameters (A, beta, etc.). Generate this from the `ceiling_results.json` file — don't hardcode numbers. Target ~5-10 rows of the best cells plus a mean line; full data is in the JSON.

6. **Scientific context.** 3-4 sentences: what the model predicts (firing rate for a natural image input), what the training set was (2910 train + 250 val = 3160 training pool, 30 test images × 30 repetitions), and what "test_r" means (Pearson correlation between predicted firing rate and mean-over-30-repeats test response).

7. **Limitations / caveats.** Brief: "Trained on 64×64 center crops of PNAS natural images. Out-of-distribution images (very different statistics) may give unreliable predictions. Cells 0, 12, 30 are known to be harder to fit; their test_r is lower than the median." Mention that models are frozen — changing hyperparameters would require retraining from scratch (which is out of scope for this package).

8. **Technical details (appendix).** Config used for training (all five ES overrides + the fixed choices: mode=vargp_direct, M=300, seed=0, n_train=3160, interleave_fstep=True, fix_Amp=True, ip_selection='random', rf_init='ground_truth', ELBO early stopping with patience=15). File format: `load_eigenspace_checkpoint(<path>, X_pool, pool_sum_tolerance)`. Mention the integrity check: the dataset must be bit-identical to what was used in training — the checkpoint load will fail otherwise.

**Style notes:**
- Plain markdown. No emojis.
- Code blocks for every command.
- Keep it under ~200 lines total.
- Use the real paths in the deliverable folder, not the repo paths.

### Step D: Commit and push

When Steps A-C are done and verified:

```bash
git add Spatial_GP_repo/scripts/gpytorch_porting/experiments/2026-04-13_M_sweep_64x64/train_M300.log
git add Spatial_GP_repo/scripts/gpytorch_porting/checkpoints/64x64_M300_es_intl_fixAmp/
git add Spatial_GP_repo/scripts/gpytorch_porting/deliverables/
git status    # review — ensure nothing else sneaks in
git commit -m "Train M=300 ES-config checkpoints; assemble standalone inference package

41 cells × seed=0 × M=300 with (A_init=1e-4, lambda0_init=-1, n_estep=50,
n_mstep=20, n_iterations=80, interleave+fixAmp). Mean test_r ~0.838.
Deliverable folder at Spatial_GP_repo/scripts/gpytorch_porting/deliverables/
is self-contained (data + checkpoints + inference code + README) and can
be zipped and shared with the colleague."
git push origin pietro/investigate-M-degradation
```

If the `.pt` files push is too large (~2-10 MB each × 41 = up to 400 MB), check the remote push limits. If GitHub rejects the push for file size:
- Stop.
- Tell the user, with the exact file sizes, before doing anything else.
- Do NOT attempt to force-push, split commits, or rewrite history.

---

## 4. Background context (read once, don't modify anything based on it)

This is context so you understand WHY the config is what it is. You are NOT being asked to verify or refine it.

**Problem**: a variational GP with Poisson likelihood for retinal cell responses to natural images. The training algorithm is an EM-like loop (E-step updates variational params, M-step updates kernel hyperparameters, F-step updates likelihood params). An interleaved F-step variant (damped Newton on A, λ₀ at every E-step iteration) was found to work best.

**Config rationale**:

- `A_init=1e-4`: with interleaved F-step, the first E-step Newton gradient scales as `A * N_train * max(r)`. Default `A_init=0.01` causes overshoot and A-collapse for sparse cells. `1e-4` is empirically stable for `N_train=3160`.
- `n_estep=50, n_mstep=20, n_iterations=80`: more inner iterations improve convergence. Established in the ES sweep experiment (`experiments/2026-04-06_es_sweeps_64x64/`).
- `interleave_fstep=True, fix_Amp=True`: matches the paper's algorithm (Goldin et al. 2023). Best known config from the April 2026 sweep.
- `M=300`: population plateau from the 984-run M sweep (`experiments/2026-04-13_M_sweep_64x64/`). Larger M marginally helps most cells but overfits 9 of 41 — M=300 is a safe universal choice.
- `seed=0`: one single seed for a clean per-cell deliverable.
- `n_train=3160`: full training pool (train + val, no val carving) since ELBO-based early stopping does not require held-out validation data.
- `ELBO ES patience=15`: current default in `default_params.json`, matches the April 2026 sweep.

All of these are baked into the script `train_and_save_M300.py`. Do not change them.

**Investigation history** (read only if curious, not needed for the task):

- `investigations/M_degradation/FINDINGS.md` — the story of how we discovered the config-mismatch problem and the hyperparameter overfitting at large M.
- `experiments/2026-04-13_M_sweep_64x64/README.md` — the 984-run sweep that justifies M=300.
- `experiments/2026-04-06_es_sweeps_64x64/README.md` — the original ES investigation that established A_init=1e-4.

---

## 5. Things to double-check before declaring done

- [ ] `git branch --show-current` returns `pietro/investigate-M-degradation`
- [ ] `checkpoints/64x64_M300_es_intl_fixAmp/` has 41 `.pt` files (one per cell) and a `ceiling_results.json`
- [ ] Mean test_r from `ceiling_results.json` is within ±0.005 of 0.838 across all 41 cells
- [ ] `deliverables/2026-XX-XX_vargp_direct_M300/` is self-contained: `data/`, `checkpoints/`, `code/`, `README.md`, `environment.yml`
- [ ] The data files in `deliverables/.../data/` are real files, not dangling symlinks (check with `file <path>` or `stat <path>`)
- [ ] `code/inference.py` runs end-to-end on at least one cell (don't do a full 41-cell inference unless you have time; a 1-cell smoke test is enough to validate the package structure)
- [ ] The README has all 8 required sections and explains the layout clearly
- [ ] Commit is on `pietro/investigate-M-degradation`, pushed to origin
- [ ] No modifications to files outside `Spatial_GP_repo/scripts/gpytorch_porting/`
- [ ] No modifications to `default_params.json` existing values
- [ ] No unrelated files in the commit (`git show --stat HEAD` should only list training outputs + deliverable folder + your new `inference.py`)

---

## 6. What NOT to do

- Do not launch any other training run, sweep, or experiment.
- Do not edit `train_and_save_M300.py` — it is already correct.
- Do not edit any library file (kernels.py, eigenspace_*.py, etc.) except to copy them into the deliverable `code/` folder.
- Do not put yourself as co-author in commit messages.
- Do not push to any branch other than `pietro/investigate-M-degradation`.
- Do not run `git checkout`, `git switch`, `git merge`, `git rebase`, or `git worktree`.
- Do not delete the training log or intermediate files — commit them so the run is traceable.
- Do not ship a deliverable folder that depends on files outside itself. Self-contained means self-contained.
- Do not skip the verification step in Section 5. An unverified deliverable is not done.

---

## 7. If anything unexpected happens

- Training crashes mid-run: check `train_M300.log`, report the error to the user, do NOT silently restart. Re-running the script is resume-safe (it skips cells already in `sweep_results.jsonl`) but only restart after the user gives the go-ahead and you have diagnosed the cause.
- Mean `test_r` significantly off 0.838: something is different between this machine's environment and the one the reference data came from. Stop and report.
- Can't find a local module that something imports: stop, ask the user. Do NOT stub it out.
- GitHub rejects the push due to size: stop, report sizes to user.
- Any edge case not covered here: stop, describe what you see, ask.

---

## 8. Continuation prompt for starting the session

When you open the session, after the checks in Section 0:

> I've read TRAIN_AND_PACKAGE_PROMPT_M300.md. I'm going to verify the environment, launch the training in the background, and begin assembling the deliverable folder (code/ + README) while training runs. I'll report when training completes with the verification numbers.

Then proceed autonomously through Steps A-D. Check in with the user:
- Once training is launched and confirmed running (with PID and first few log lines).
- Once training completes, with the mean test_r and any failures.
- Before pushing, with a summary of what's in the commit.
