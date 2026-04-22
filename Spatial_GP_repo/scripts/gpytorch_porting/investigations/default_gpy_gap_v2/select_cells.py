"""Stratified cell selection for the default_gpy_gap_v2 investigation.

Implements the pre-registered procedure from PROMPT.md:

  1. Load the training pool from PNAS_64x64_center_crop_no_renorm.npz.
     Combine responses_train + responses_val to get the full 3160-response
     pool (matches how run_single_mode.py:652-655 assembles the pool).
  2. For each cell in 0..40, compute firing_rate = R_pool[:, cell].mean().
  3. Sort cells by firing rate ascending.
  4. Split into 3 buckets: low = sorted-indices 0..13, mid = 14..27,
     high = 28..40 (14+14+13 = 41 cells).
  5. Within each bucket, pick cells at local positions 1, 3, 6, 9, 12.
  6. Print the 15 selected cells with firing rates and bucket labels.
  7. Prompt for user confirmation before writing cells_used.json.

Usage (from gpytorch_porting root):
    python investigations/default_gpy_gap_v2/select_cells.py

The script is deterministic — running it twice gives the same 15 cells.
"""

import json
import sys
from pathlib import Path

import numpy as np

# Defaults from the charter (investigations/default_gpy_gap_v2/PROMPT.md,
# "Cell selection procedure"). Not runtime knobs — changing them would
# violate the pre-registration.
DATA_PATH = 'datasets/PNAS_64x64_center_crop_no_renorm.npz'
BUCKET_BOUNDARIES = [0, 14, 28, 41]  # low: 0..13, mid: 14..27, high: 28..40
BUCKET_LABELS = ['low', 'mid', 'high']
POSITIONS_WITHIN_BUCKET = [1, 3, 6, 9, 12]

# Deviation from charter pre-registration: user explicitly requested that cell 8
# (the v1 focus cell) be included in addition to the 15 stratified cells.
# Each entry is a cell_id; firing rate / bucket are filled in automatically.
USER_ADDED_CELLS = [8]

OUTPUT_PATH = Path(__file__).parent / 'cells_used.json'


def select_cells():
    """Run the stratified selection. Returns (selected, firing_rates_all)."""
    root = Path(__file__).parent.parent.parent
    data_file = root / DATA_PATH
    if not data_file.exists():
        raise FileNotFoundError(f"Dataset not found at {data_file}")

    data = np.load(data_file)
    # Match run_single_mode.py:652-653: concat train + val into a 3160-pool.
    R_pool = np.concatenate([data['responses_train'], data['responses_val']], axis=0)
    if R_pool.shape != (3160, 41):
        raise ValueError(
            f"Expected R_pool shape (3160, 41), got {R_pool.shape}. "
            "Dataset format may have changed."
        )

    firing_rates = R_pool.mean(axis=0)  # (41,)
    sorted_cells = np.argsort(firing_rates, kind='stable')  # ascending, deterministic

    selected = []
    for bi, (lo, hi, label) in enumerate(zip(
        BUCKET_BOUNDARIES[:-1], BUCKET_BOUNDARIES[1:], BUCKET_LABELS
    )):
        bucket_cells = sorted_cells[lo:hi]
        bucket_size = len(bucket_cells)
        for pos in POSITIONS_WITHIN_BUCKET:
            if pos >= bucket_size:
                raise IndexError(
                    f"Position {pos} out of range for '{label}' bucket "
                    f"(size {bucket_size})"
                )
            cell_id = int(bucket_cells[pos])
            selected.append({
                'cell_id': cell_id,
                'firing_rate': float(firing_rates[cell_id]),
                'bucket': label,
                'bucket_pos': pos,
                'global_rank_ascending': int(lo + pos),
                'user_added': False,
            })

    # Append user-requested cells (outside the stratification).
    for cell_id in USER_ADDED_CELLS:
        if any(s['cell_id'] == cell_id for s in selected):
            continue  # already covered by stratified selection
        rank = int(np.where(sorted_cells == cell_id)[0][0])
        # Determine bucket label from rank.
        for lo, hi, label in zip(
            BUCKET_BOUNDARIES[:-1], BUCKET_BOUNDARIES[1:], BUCKET_LABELS
        ):
            if lo <= rank < hi:
                bucket_label = label
                bucket_lo = lo
                break
        selected.append({
            'cell_id': cell_id,
            'firing_rate': float(firing_rates[cell_id]),
            'bucket': bucket_label,
            'bucket_pos': rank - bucket_lo,
            'global_rank_ascending': rank,
            'user_added': True,
        })

    return selected, firing_rates


def print_table(selected, firing_rates):
    """Print a table of selected cells plus bucket boundary summary."""
    print()
    print("=" * 72)
    print("Stratified cell selection for default_gpy_gap_v2")
    print("=" * 72)
    print(f"Dataset: {DATA_PATH}")
    print(f"Pool: responses_train + responses_val = (3160, 41)")
    print()

    fr_sorted = np.sort(firing_rates)
    print("Firing-rate bucket boundaries (sorted ascending):")
    for lo, hi, label in zip(
        BUCKET_BOUNDARIES[:-1], BUCKET_BOUNDARIES[1:], BUCKET_LABELS
    ):
        print(f"  {label:>4}: rank {lo:2d}..{hi-1:2d}  fr range "
              f"[{fr_sorted[lo]:.4f}, {fr_sorted[hi-1]:.4f}]")

    print()
    n_stratified = sum(1 for s in selected if not s['user_added'])
    n_added = sum(1 for s in selected if s['user_added'])
    print(
        f"Selected {len(selected)} cells: {n_stratified} stratified "
        f"(5 per bucket, positions {POSITIONS_WITHIN_BUCKET}) + "
        f"{n_added} user-added: {USER_ADDED_CELLS}"
    )
    print()
    print(f"  {'bucket':>6} {'pos':>4} {'rank':>5} {'cell':>5} "
          f"{'firing_rate':>14} {'user_added':>11}")
    print(f"  {'-'*6} {'-'*4} {'-'*5} {'-'*5} {'-'*14} {'-'*11}")
    for s in selected:
        print(f"  {s['bucket']:>6} {s['bucket_pos']:>4} "
              f"{s['global_rank_ascending']:>5} {s['cell_id']:>5} "
              f"{s['firing_rate']:>14.6f} {str(s['user_added']):>11}")
    print()


def write_cells_used(selected):
    """Write cells_used.json after user confirmation."""
    payload = {
        'procedure': (
            'stratified by firing rate, 3 buckets x 5 positions; '
            'plus user-added cells (deviation from charter pre-registration)'
        ),
        'data_path': DATA_PATH,
        'pool_source': 'responses_train + responses_val (3160 rows)',
        'bucket_boundaries': BUCKET_BOUNDARIES,
        'bucket_labels': BUCKET_LABELS,
        'positions_within_bucket': POSITIONS_WITHIN_BUCKET,
        'user_added_cells': USER_ADDED_CELLS,
        'cells': selected,
    }
    with open(OUTPUT_PATH, 'w') as f:
        json.dump(payload, f, indent=2)
    print(f"Wrote {OUTPUT_PATH}")


def main():
    selected, firing_rates = select_cells()
    print_table(selected, firing_rates)

    # List form, for quick copy into a run script.
    cell_ids = [s['cell_id'] for s in selected]
    print(f"cell_ids = {cell_ids}")
    print()

    if OUTPUT_PATH.exists():
        print(f"Note: {OUTPUT_PATH} already exists. Will overwrite if you confirm.")

    ans = input("Write cells_used.json? [y/N]: ").strip().lower()
    if ans == 'y':
        write_cells_used(selected)
    else:
        print("Not writing cells_used.json. Re-run after user confirmation.")
        sys.exit(1)


if __name__ == '__main__':
    main()
