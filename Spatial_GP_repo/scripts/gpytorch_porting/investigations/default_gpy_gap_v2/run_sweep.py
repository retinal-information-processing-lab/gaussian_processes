"""Pre-registered sweep for default_gpy_gap_v2.

Runs the 112-run matrix locked in PROMPT.md:
  - 16 cells (15 stratified + user-added cell 8; see cells_used.json)
  - seeds {42, 123, 789}    for vargp_direct and default_gpy (joint)
  - seed  {42}              for default_gpy_alt (alternating_fstep=True)
  - M=300, n_train=1500, 64x64 dataset, arc_cosine kernel, ground-truth RF init,
    ELBO early stopping on, ip_selection='random', all other params from
    default_params.json via build_config_from_defaults().

Writes one JSONL record per run to results.jsonl. If results.jsonl already
exists and is non-empty, the old file is moved to results_<timestamp>.jsonl
and a fresh one is started (charter "Do NOT delete or overwrite" rule).

Writes a sweep_metadata.json at launch with git commit, start time, planned
run count, cell list, per-mode seed plan. This is the reference for the
SCRAPBOOK.md "Sweep execution" section.

Usage (from gpytorch_porting/ root):
    python investigations/default_gpy_gap_v2/run_sweep.py --dry-run  # show plan only
    python investigations/default_gpy_gap_v2/run_sweep.py            # launch

Per charter: no CLI knobs for M / n_train / cells / seeds / modes. Any change
to the locked config is a pre-registration violation and must be edited into
this file explicitly after user discussion.
"""

import argparse
import datetime
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

INVESTIGATION_DIR = Path(__file__).parent
ROOT = INVESTIGATION_DIR.parent.parent  # gpytorch_porting/

sys.path.insert(0, str(ROOT))
from run_single_mode import build_config_from_defaults, run_single_config  # noqa: E402

# ---------------------------------------------------------------------------
# Locked config (PROMPT.md "Experimental design"). Do not edit without
# user agreement; edits here are pre-registration deviations.
# ---------------------------------------------------------------------------
DATA_PATH = 'datasets/PNAS_64x64_center_crop_no_renorm.npz'
M = 300
N_TRAIN = 1500
SEEDS_MAIN = [42, 123, 789]          # vargp_direct, default_gpy
SEEDS_ALT = [42]                      # default_gpy_alt sanity
MODES_MAIN = ['vargp_direct', 'default_gpy']
MODE_ALT = 'default_gpy_alt'

CELLS_USED_PATH = INVESTIGATION_DIR / 'cells_used.json'
RESULTS_PATH = INVESTIGATION_DIR / 'results.jsonl'
SWEEP_META_PATH = INVESTIGATION_DIR / 'sweep_metadata.json'


def load_cells():
    if not CELLS_USED_PATH.exists():
        raise FileNotFoundError(
            f"{CELLS_USED_PATH} does not exist. Run select_cells.py first."
        )
    with open(CELLS_USED_PATH) as f:
        payload = json.load(f)
    return [s['cell_id'] for s in payload['cells']], payload


def plan_runs(cell_ids):
    """Build the list of (mode, cell, seed) triples in execution order.

    Execution order: cell outer, seed middle, mode inner (within each
    (cell, seed) we run both main modes back-to-back). default_gpy_alt
    seed-42 runs are appended after all main runs.
    """
    plan = []
    for cell in cell_ids:
        for seed in SEEDS_MAIN:
            for mode in MODES_MAIN:
                plan.append((mode, cell, seed))
    for cell in cell_ids:
        for seed in SEEDS_ALT:
            plan.append((MODE_ALT, cell, seed))
    return plan


def git_head_commit():
    try:
        h = subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True
        ).strip()
        return h
    except Exception:
        return None


def git_working_tree_dirty():
    try:
        out = subprocess.check_output(
            ['git', 'status', '--porcelain'], cwd=ROOT, text=True
        )
        return out.strip() != ''
    except Exception:
        return None


def make_config(mode, cell, seed):
    """Build a config dict for one run. The mode string is logical
    ('default_gpy_alt'); we translate to the actual mode + flag here."""
    alternating = False
    actual_mode = mode
    if mode == 'default_gpy_alt':
        actual_mode = 'default_gpy'
        alternating = True
    return build_config_from_defaults(
        mode=actual_mode,
        data_path=DATA_PATH,
        M=M,
        n_train=N_TRAIN,
        cell=cell,
        seed=seed,
        ip_selection='random',
        alternating_fstep=alternating,
    )


def snapshot_fields(config):
    """Fields from config that we archive per run for reproducibility.

    Not the whole config (too big). These are the fields that are either
    overridden by this sweep or that matter for interpreting a run.
    """
    keys = [
        'mode', 'M', 'n_train', 'cell', 'seed',
        'n_iterations',
        'kernel_type', 'rf_init',
        'early_stop', 'patience', 'min_delta_rel', 'min_iterations',
        'restore_best', 'es_metric',
        'jitter', 'cholesky_max_tries', 'eigval_tol',
        'lambda_var_clamp', 'gpy_lbfgs_max_iter',
        'ip_selection', 'alternating_fstep',
        'dtype', 'device', 'data_path',
    ]
    return {k: config.get(k) for k in keys}


def run_one(mode, cell, seed, run_index, total):
    """Execute a single run and return the record dict."""
    config = make_config(mode, cell, seed)
    t0 = time.time()
    started_at = datetime.datetime.now().isoformat(timespec='seconds')

    banner = (
        f"[{run_index:3d}/{total}] mode={mode:<18} cell={cell:>3} "
        f"seed={seed:>4}  started={started_at}"
    )
    print('=' * len(banner))
    print(banner)
    print('=' * len(banner))

    try:
        result = run_single_config(config)
        err = None
    except Exception as e:
        result = None
        err = repr(e)
        print(f"  ERROR: {err}")

    elapsed = time.time() - t0

    if result is None:
        record = {
            'mode': mode,
            'cell': cell,
            'seed': seed,
            'status': 'failed',
            'error': err,
            'wall_time_s': elapsed,
            'started_at': started_at,
            'ended_at': datetime.datetime.now().isoformat(timespec='seconds'),
            'config_snapshot': snapshot_fields(config),
        }
    else:
        curves = result.get('curves') or {}
        record = {
            'mode': mode,
            'cell': cell,
            'seed': seed,
            'status': result.get('status', 'unknown'),
            'wall_time_s': elapsed,
            'started_at': started_at,
            'ended_at': datetime.datetime.now().isoformat(timespec='seconds'),
            'test_r': result.get('test_r'),
            'train_r': result.get('train_r'),
            'explained_var': result.get('explained_var'),
            'final_loss': result.get('final_loss'),
            'train_time': result.get('train_time'),
            'n_iterations_run': result.get('n_iterations_run'),
            'stopped_early': result.get('stopped_early'),
            'best_iteration': result.get('best_iteration'),
            'final_A': result.get('final_A'),
            'final_lambda0': result.get('final_lambda0'),
            'final_beta': result.get('final_beta'),
            'final_rho': result.get('final_rho'),
            'final_eps_0x': result.get('final_eps_0x'),
            'final_eps_0y': result.get('final_eps_0y'),
            # Per-iteration curves (ELBO and A are the charter-relevant ones)
            'train_loss_curve': curves.get('train_loss'),
            'A_curve': curves.get('A'),
            'config_snapshot': snapshot_fields(config),
        }

    with open(RESULTS_PATH, 'a') as f:
        f.write(json.dumps(record) + '\n')

    tag_test_r = record.get('test_r')
    tag_iters = record.get('n_iterations_run')
    if tag_test_r is None:
        print(f"  FAILED after {elapsed:.1f}s")
    else:
        print(f"  test_r={tag_test_r:.4f}  iters_run={tag_iters}  "
              f"final_A={record.get('final_A'):.4g}  "
              f"wall={elapsed:.1f}s")
    return record


def rotate_results_if_present():
    if RESULTS_PATH.exists() and RESULTS_PATH.stat().st_size > 0:
        stamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        backup = RESULTS_PATH.with_name(f'results_{stamp}.jsonl')
        shutil.move(str(RESULTS_PATH), str(backup))
        print(f"Rotated existing results.jsonl -> {backup.name}")


def write_sweep_metadata(cell_ids, plan, commit, dirty):
    meta = {
        'investigation': 'default_gpy_gap_v2',
        'charter': 'investigations/default_gpy_gap_v2/PROMPT.md',
        'started_at': datetime.datetime.now().isoformat(timespec='seconds'),
        'git_commit': commit,
        'git_working_tree_dirty_at_launch': dirty,
        'config_locked': {
            'data_path': DATA_PATH,
            'M': M,
            'n_train': N_TRAIN,
            'seeds_main': SEEDS_MAIN,
            'seeds_alt': SEEDS_ALT,
            'modes_main': MODES_MAIN,
            'mode_alt': MODE_ALT,
        },
        'cells': cell_ids,
        'planned_run_count': len(plan),
    }
    with open(SWEEP_META_PATH, 'w') as f:
        json.dump(meta, f, indent=2)


def print_plan_summary(cell_ids, plan, commit, dirty):
    print()
    print("=" * 72)
    print("default_gpy_gap_v2 — sweep plan")
    print("=" * 72)
    print(f"Charter:         investigations/default_gpy_gap_v2/PROMPT.md")
    print(f"Git commit:      {commit}")
    print(f"Working tree:    {'DIRTY' if dirty else 'clean'}")
    print(f"Cells ({len(cell_ids)}):      {cell_ids}")
    print(f"Config:          data={DATA_PATH}")
    print(f"                 M={M}, n_train={N_TRAIN}, arc_cosine, "
          f"ground_truth RF, ELBO ES on, ip_selection=random")
    print(f"Seeds (main):    {SEEDS_MAIN}  modes: {MODES_MAIN}")
    print(f"Seeds (alt):     {SEEDS_ALT}   mode:  {MODE_ALT}")
    print(f"Planned runs:    {len(plan)}")
    print(f"Output:          {RESULTS_PATH}")
    print(f"Sweep metadata:  {SWEEP_META_PATH}")
    print()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dry-run', action='store_true',
                        help='Print the plan and exit without running anything.')
    args = parser.parse_args()

    os.chdir(ROOT)  # data paths are relative to gpytorch_porting/

    cell_ids, cells_payload = load_cells()
    plan = plan_runs(cell_ids)
    commit = git_head_commit()
    dirty = git_working_tree_dirty()

    print_plan_summary(cell_ids, plan, commit, dirty)

    if args.dry_run:
        print("Dry run — not launching.")
        return

    rotate_results_if_present()
    write_sweep_metadata(cell_ids, plan, commit, dirty)

    t_start = time.time()
    records = []
    for i, (mode, cell, seed) in enumerate(plan, start=1):
        rec = run_one(mode, cell, seed, run_index=i, total=len(plan))
        records.append(rec)

    elapsed = time.time() - t_start
    print()
    print("=" * 72)
    print(f"Sweep complete. {len(records)} runs in {elapsed/60:.1f} min.")
    print(f"Results: {RESULTS_PATH}")
    print(f"Metadata: {SWEEP_META_PATH}")
    print("=" * 72)


if __name__ == '__main__':
    main()
