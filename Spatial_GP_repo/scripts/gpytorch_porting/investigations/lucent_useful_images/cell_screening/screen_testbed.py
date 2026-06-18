"""
Full PNAS-cell testbed screen for the lucent image-generation pipeline.

Trains the `default_gpy` GP at M = n_train, seed = 42, across the step-50 ladder
n_train = {50,100,150,200,250,300} for every candidate cell, records the test_r
ladder, and ranks cells by how cleanly + widely test_r climbs with dataset size.

WHY: the lucent demo shows "as the model improves with more data, the synthesized
optimal image sharpens." That needs testbed cells whose test_r actually RISES with
n_train (not noisy, not already-saturated).

Selection priority (set by the user, 2026-06-18):
  - test_r must INCREASE with n_train (monotonic, no big single-step dips), AND
  - must NOT already be high at n=50: test_r[50] <= START_MAX (=0.80). A cell that
    starts good leaves nothing to demonstrate.
  - ideal profile: ~0.3 at n=50 climbing to ~0.9 at n=300 (wide dynamic range).
  RF position / size is explicitly NOT a criterion.
The script does NOT auto-pick: it ranks eligible cells (start<=START_MAX) by net
gain and prints every metric, for a human pick of the 3 smoothest 0.3->0.9 climbers.

POOL: all 41 PNAS cells (0..40) minus
  - the 6 STA-edge-artifact cells (0,5,6,15,22,39)  [gpytorch_porting CLAUDE.md]
  - the known default_gpy-hard cell 10               [.claude/rules/debugging.md 3.6]
  -> 34 candidate cells.

PARAMS: n_train grid, seed, and M=n_train are the experiment's chosen axes (see
gp_models.train_default_gpy; M defaults to n_train). START_MAX is the user's
eligibility threshold above. All GP hyperparameters come from default_params.json
via gp_models -> build_config_from_defaults (no hardcoding here).

CACHE: cell_screening/cache/testbed_ladders.pkl (+ a sorted .csv), written
incrementally after every cell so a crash or a re-rank needs no recompute. A
re-run skips cells already cached unless --force.

Env: /home/idv-eqs8-pza/anaconda3/envs/pytorch_gpytorch/bin/python  (shared GPU --
nvidia-smi first). Reproducible: seed fixed, no random elements.

Usage:
    python screen_testbed.py                 # full 34-cell pool (background-friendly)
    python screen_testbed.py --cells 13      # explicit subset (e.g. one-cell timing)
    python screen_testbed.py --force         # recompute even if cached
    python screen_testbed.py --rank-only     # re-print ranking from cache, no fits
    python screen_testbed.py --figure        # render test_r-vs-n_train figure from cache
"""
import argparse
import gc
import os
import pickle
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PARENT = os.path.dirname(HERE)  # .../lucent_useful_images
if PARENT not in sys.path:
    sys.path.insert(0, PARENT)

import gp_models  # noqa: E402  (adds the GP engine to sys.path internally)

# --- experiment axes (documented overrides; not hidden magic numbers) ----------
N_TRAINS = [50, 100, 150, 200, 250, 300]  # M = n_train at every point
SEED = 42                                  # train_default_gpy default
# Pool = all 41 minus STA-edge artifacts minus default_gpy-hard cell 10.
STA_EDGE_CELLS = [0, 5, 6, 15, 22, 39]     # gpytorch_porting CLAUDE.md
DEFAULT_GPY_HARD = [10]                     # debugging.md 3.6
ALL_CELLS = list(range(41))
POOL = [c for c in ALL_CELLS if c not in set(STA_EDGE_CELLS + DEFAULT_GPY_HARD)]

# --- user selection threshold (2026-06-18) -------------------------------------
START_MAX = 0.80  # a cell with test_r[50] above this is "already good" -> ineligible

CACHE_DIR = os.path.join(HERE, "cache")
CACHE_PKL = os.path.join(CACHE_DIR, "testbed_ladders.pkl")
CACHE_CSV = os.path.join(CACHE_DIR, "testbed_ladders.csv")
FIG_PNG = os.path.join(HERE, "testbed_ladders.png")


# ------------------------------------------------------------------------------
# cache I/O
# ------------------------------------------------------------------------------
def load_cache():
    if os.path.exists(CACHE_PKL):
        with open(CACHE_PKL, "rb") as f:
            return pickle.load(f)
    return {}


def save_cache(cache):
    os.makedirs(CACHE_DIR, exist_ok=True)
    tmp = CACHE_PKL + ".tmp"
    with open(tmp, "wb") as f:
        pickle.dump(cache, f)
    os.replace(tmp, CACHE_PKL)  # atomic; a crash mid-write cannot corrupt the cache


# ------------------------------------------------------------------------------
# one cell: full ladder
# ------------------------------------------------------------------------------
def screen_cell(cell_id):
    """Train the cell across the full n_train ladder; return ladder dict."""
    ladder, fit_times = [], []
    for nt in N_TRAINS:
        t0 = time.time()
        try:
            model, lik, idx, tr = gp_models.train_default_gpy(
                cell_id=cell_id, n_train=nt, seed=SEED)  # M = n_train
            dt = time.time() - t0
            # keep only the scalar; free everything GPU-side before the next fit
            del model, lik, idx
            import torch
            gc.collect()
            torch.cuda.empty_cache()
        except Exception as e:
            dt = time.time() - t0
            tr = float("nan")
            print(f"    cell {cell_id:2d} n_train={nt}: FAILED ({e})")
        ladder.append(float(tr))
        fit_times.append(float(dt))
        print(f"    cell {cell_id:2d} n_train={nt:3d}: test_r={tr:+.4f}  ({dt:.1f}s)")
        sys.stdout.flush()
    return {"n_trains": list(N_TRAINS), "test_r": ladder, "fit_times": fit_times,
            "seed": SEED, "M_equals_n_train": True}


# ------------------------------------------------------------------------------
# metrics + ranking
# ------------------------------------------------------------------------------
def cell_metrics(rec):
    """Derived quality metrics for one cell's ladder. NaN-safe."""
    from scipy.stats import spearmanr
    r = np.asarray(rec["test_r"], dtype=float)
    nts = np.asarray(rec["n_trains"], dtype=float)
    failed = bool(np.any(~np.isfinite(r)))
    start = float(r[0])
    final = float(r[-1])
    net_gain = final - start
    diffs = np.diff(r)
    max_drop = float(max(0.0, -np.min(diffs))) if diffs.size else 0.0  # largest single-step decrease
    spearman = float(spearmanr(nts, r).correlation) if not failed else float("nan")
    eligible = (not failed) and (start <= START_MAX)
    # "clean monotonic" annotation (not an auto-selector): rises overall, no big dip
    mono = (not failed) and (max_drop <= 0.03) and (net_gain > 0.0)
    return dict(start=start, final=final, net_gain=net_gain, max_drop=max_drop,
                spearman=spearman, eligible=eligible, mono=mono, failed=failed)


def print_ranking(cache):
    rows = []
    for c in sorted(cache):
        m = cell_metrics(cache[c])
        rows.append((c, cache[c]["test_r"], m))

    def fmt_ladder(r):
        return " ".join(f"{v:+.2f}" for v in r)

    print("\n" + "=" * 100)
    print(f"LADDERS  (n_train = {N_TRAINS};  M = n_train;  seed = {SEED})")
    print("=" * 100)
    hdr = f"{'cell':>4}  " + "  ".join(f"n{nt}" for nt in N_TRAINS) + \
          f"   {'start':>6} {'final':>6} {'gain':>6} {'maxdrop':>7} {'spear':>6}  flags"
    print(hdr)
    for c, r, m in rows:
        ladder = "  ".join(f"{v:+.2f}" for v in r)
        flags = []
        if m["failed"]:
            flags.append("FAILED")
        if not m["eligible"] and not m["failed"]:
            flags.append(f"start>{START_MAX:.2f}")
        if m["mono"]:
            flags.append("MONO")
        print(f"{c:>4}  {ladder}   {m['start']:+6.2f} {m['final']:+6.2f} "
              f"{m['net_gain']:+6.2f} {m['max_drop']:7.3f} {m['spearman']:+6.2f}  {' '.join(flags)}")

    # ranking among ELIGIBLE cells (start <= START_MAX), by net gain
    elig = [(c, r, m) for c, r, m in rows if m["eligible"]]
    elig.sort(key=lambda t: -t[2]["net_gain"])
    print("\n" + "=" * 100)
    print(f"ELIGIBLE cells (test_r[50] <= {START_MAX:.2f}), ranked by net gain "
          f"(want a wide, smooth ~0.3 -> ~0.9 climb):")
    print("=" * 100)
    print(f"{'rank':>4} {'cell':>4}  {'start':>6} {'final':>6} {'gain':>6} "
          f"{'maxdrop':>7} {'spear':>6}   ladder")
    for i, (c, r, m) in enumerate(elig, 1):
        print(f"{i:>4} {c:>4}  {m['start']:+6.2f} {m['final']:+6.2f} {m['net_gain']:+6.2f} "
              f"{m['max_drop']:7.3f} {m['spearman']:+6.2f}   {fmt_ladder(r)}")
    if not elig:
        print("  (no eligible cell -- every screened cell already starts above "
              f"{START_MAX:.2f} or failed. Raise to the user.)")
    return rows, elig


def write_csv(cache):
    rows = []
    for c in sorted(cache):
        m = cell_metrics(cache[c])
        rows.append((c, cache[c]["test_r"], m))
    rows.sort(key=lambda t: (not t[2]["eligible"], -t[2]["net_gain"]))
    os.makedirs(CACHE_DIR, exist_ok=True)
    with open(CACHE_CSV, "w") as f:
        f.write("cell," + ",".join(f"r_n{nt}" for nt in N_TRAINS) +
                ",start,final,net_gain,max_drop,spearman,eligible,mono,failed\n")
        for c, r, m in rows:
            f.write(f"{c}," + ",".join(f"{v:.4f}" for v in r) +
                    f",{m['start']:.4f},{m['final']:.4f},{m['net_gain']:.4f},"
                    f"{m['max_drop']:.4f},{m['spearman']:.4f},"
                    f"{int(m['eligible'])},{int(m['mono'])},{int(m['failed'])}\n")
    print(f"\n[csv] {CACHE_CSV}")


# ------------------------------------------------------------------------------
# figure
# ------------------------------------------------------------------------------
def make_figure(cache, highlight=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["Liberation Sans", "Arial", "Helvetica", "DejaVu Sans"]
    plt.rcParams["font.family"] = "sans-serif"

    highlight = highlight or []
    fig, ax = plt.subplots(figsize=(7, 5))
    # all cells, faded
    for c in sorted(cache):
        r = cache[c]["test_r"]
        nts = cache[c]["n_trains"]
        ax.plot(nts, r, color="0.75", lw=1.0, zorder=1)
    # highlighted picks, bold + colored
    cmap = plt.get_cmap("tab10")
    for i, c in enumerate(highlight):
        if c not in cache:
            continue
        r = cache[c]["test_r"]
        nts = cache[c]["n_trains"]
        ax.plot(nts, r, color=cmap(i % 10), lw=2.4, marker="o", ms=5,
                zorder=3, label=f"cell {c}")
    ax.axhline(START_MAX, color="crimson", ls="--", lw=1.0, alpha=0.7,
               label=f"start cap {START_MAX:.2f}")
    ax.set_xlabel("n_train  (= M, inducing points)")
    ax.set_ylabel("test_r  (Pearson r, 30-image test set)")
    ax.set_title("default_gpy test_r vs training-set size (PNAS cells, seed 42)")
    ax.set_xticks(N_TRAINS)
    ax.set_ylim(-0.6, 1.0)
    ax.grid(True, alpha=0.3)
    if highlight:
        ax.legend(loc="lower right", fontsize=9, framealpha=0.9)
    fig.tight_layout()
    fig.savefig(FIG_PNG, dpi=150)
    print(f"[fig] {FIG_PNG}")


# ------------------------------------------------------------------------------
# main
# ------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cells", type=int, nargs="*", default=None,
                    help="explicit cell ids to screen (default: full 34-cell pool)")
    ap.add_argument("--force", action="store_true",
                    help="recompute even cells already in the cache")
    ap.add_argument("--rank-only", action="store_true",
                    help="print ranking from the cache, run no fits")
    ap.add_argument("--figure", action="store_true",
                    help="render the test_r-vs-n_train figure from the cache")
    ap.add_argument("--highlight", type=int, nargs="*", default=None,
                    help="cell ids to highlight in the figure")
    args = ap.parse_args()

    cache = load_cache()

    if args.rank_only or args.figure:
        if not cache:
            print("cache is empty; run the screen first.")
            return
        print_ranking(cache)
        write_csv(cache)
        if args.figure:
            make_figure(cache, highlight=args.highlight)
        return

    cells = args.cells if args.cells is not None else POOL
    todo = [c for c in cells if args.force or c not in cache]
    print(f"Testbed screen: pool={len(cells)} cells, {len(todo)} to fit "
          f"({len(cells) - len(todo)} cached).")
    print(f"n_train={N_TRAINS}  M=n_train  seed={SEED}  START_MAX={START_MAX}")
    print(f"cells to fit: {todo}\n")

    t_start = time.time()
    for k, c in enumerate(todo, 1):
        print(f"[{k}/{len(todo)}] cell {c}")
        cache[c] = screen_cell(c)
        save_cache(cache)  # incremental: persist after every cell
    dt = time.time() - t_start
    print(f"\nfit loop done: {len(todo)} cells in {dt/60:.1f} min")

    print_ranking(cache)
    write_csv(cache)
    make_figure(cache)


if __name__ == "__main__":
    main()
