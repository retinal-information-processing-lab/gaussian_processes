"""
Screen candidate cells for default_gpy learnability at the high-quality corner
(n_train=300, M=n_train=300, seed=42), so we pick cells where the model is actually
good before running the full quality-ladder optimization.

Excludes the 6 STA-edge-artifact cells (0,5,6,15,22,39) per gpytorch CLAUDE.md.

Usage:
    python screen_cells.py                # default candidate list
    python screen_cells.py 1 2 3 8 11     # explicit cell ids
"""
import sys
import time

import gp_models

# Candidate cells: a spread across the 41, excluding STA-edge-artifact cells
# (0,5,6,15,22,39) and the known default_gpy-hard cell 10 (debugging.md 3.6).
DEFAULT_CANDIDATES = [1, 2, 3, 4, 7, 8, 9, 11, 12, 13, 14, 17, 23, 30, 40]

N_TRAIN_SCREEN = 300
SEED = 42  # M = n_train (full inducing set) -- train_default_gpy defaults M to n_train


def main(cells):
    print(f"Screening {len(cells)} cells at n_train={N_TRAIN_SCREEN}, M=n_train, seed={SEED}")
    print(f"cells: {cells}\n")
    results = []
    for c in cells:
        t0 = time.time()
        try:
            model, lik, idx, test_r = gp_models.train_default_gpy(
                cell_id=c, n_train=N_TRAIN_SCREEN, seed=SEED)  # M=n_train
            dt = time.time() - t0
            results.append((c, test_r, float(lik.A), float(lik.lambda0), dt))
            print(f"  cell {c:2d}: test_r={test_r:+.4f}  A={float(lik.A):.4f}  "
                  f"lambda0={float(lik.lambda0):+.3f}  ({dt:.1f}s)")
        except Exception as e:
            print(f"  cell {c:2d}: FAILED ({e})")
        sys.stdout.flush()

    print("\n=== sorted by test_r (best first) ===")
    for c, tr, A, l0, dt in sorted(results, key=lambda r: -r[1]):
        print(f"  cell {c:2d}: test_r={tr:+.4f}")


if __name__ == "__main__":
    cells = [int(x) for x in sys.argv[1:]] or DEFAULT_CANDIDATES
    main(cells)
