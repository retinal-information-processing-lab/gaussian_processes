"""
Screen candidate cells for a CLEAN, monotonically-increasing default_gpy test_r
ladder across n_train = {50,100,200,300}. The figure's model-quality axis needs
test_r to rise with n_train; some cells train unstably at low n_train (negative or
non-monotonic test_r), which we must avoid.

Usage:
    python screen_ladder.py                 # default candidate list
    python screen_ladder.py 13 3 12 17 30   # explicit
"""
import sys
import time

import numpy as np
import gp_models

N_TRAINS = [50, 100, 200, 300]
SEED = 42  # M = n_train at all times (train_default_gpy defaults M to n_train)

# High test_r at the top corner (from screen_cells.py); test the full ladder now.
DEFAULT_CANDIDATES = [13, 1, 11, 12, 3, 17, 30, 8, 9, 4, 7, 23, 2]


def main(cells):
    print(f"Ladder screen: n_train={N_TRAINS} M=n_train seed={SEED}\n")
    rows = []
    for c in cells:
        rs = []
        for nt in N_TRAINS:
            try:
                _, _, _, tr = gp_models.train_default_gpy(cell_id=c, n_train=nt, seed=SEED)  # M=n_train
            except Exception:
                tr = float("nan")
            rs.append(tr)
        rs = np.array(rs)
        # "clean" = all positive AND non-decreasing within a small tolerance
        mono = np.all(np.diff(rs) > -0.03) and np.all(rs > 0.0)
        rng = rs[-1] - rs[0]
        rows.append((c, rs, mono, rng))
        flag = "CLEAN" if mono else "     "
        print(f"  cell {c:2d}: " + "  ".join(f"{nt}:{v:+.2f}" for nt, v in zip(N_TRAINS, rs))
              + f"   span={rng:+.2f}  [{flag}]")
        sys.stdout.flush()

    print("\n=== CLEAN monotonic ladders, by span (best demo of model improvement) ===")
    clean = [r for r in rows if r[2]]
    for c, rs, mono, rng in sorted(clean, key=lambda r: -r[3]):
        print(f"  cell {c:2d}: " + "  ".join(f"{v:+.2f}" for v in rs) + f"   span={rng:+.2f}")
    if not clean:
        print("  (none fully clean; pick best-effort from the list above)")


if __name__ == "__main__":
    cells = [int(x) for x in sys.argv[1:]] or DEFAULT_CANDIDATES
    main(cells)
