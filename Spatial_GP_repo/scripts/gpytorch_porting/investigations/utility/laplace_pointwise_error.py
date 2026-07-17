"""
Pointwise Laplace-error measurement for log p(r).

Measures the error of the Laplace (saddle-point) approximation of log p(r) at
FIXED single spike counts r, with NO sum over r, by comparing against an
adaptive-quadrature reference. Fixing r and never summing isolates the Laplace
approximation error (L) from the sum-truncation error (T): two independent
failure modes that previous investigations tangled together.

    p(r | mu_g, sigma2_g) = INTEGRAL  Poisson(r; e^g) * N(g; mu_g, sigma2_g) dg

The Laplace approximation of log p(r) is the code under test
(utils.py:_diff_laplace_log_probs) -- the ONLY torch in this script, since
measuring its error is the whole point. The reference uses no torch.

Reference: scipy.integrate.quad (adaptive QUADPACK, self-reporting error).
No PyTorch reference (it served no purpose: this is an offline, 1D, smooth,
non-differentiated integral -- no GPU and no autograd benefit). torchquad was
also ruled out: not installed, and its only adaptive rule is Monte-Carlo VEGAS,
which we reject. The integrand peak g* is located INDEPENDENTLY by scipy brentq
on the saddle equation (no reuse of the code under test), and quad integrates a
peak-factored integrand on an informed finite window whose edges are verified to
sit deep in the tail. Validated three independent ways:
  (a) interval-independence: agree when the window is widened (edge-drop 40 vs 80),
  (b) at small sigma2 it must match the exact Poisson log-pmf,
  (c) sum_r p_ref(r) ~ 1 (the true density is normalised, unlike the Laplace
      approximation, whose sum drifts from 1 -- that drift IS (L)).

Design (grid / panels / sweep) agreed with the user on 2026-05-25:
  - mu_g panels  : 10 values from -10 to 8 (the user operating range; -1.7 = production)
  - r per panel  : 14 counts (see _r_grid_for_mu): {0,1,2,3,4,5} plus 8 log-spaced
                   up to r_hi = clip(round(e^{mu_g}), 30, 300) -- dense low (biggest
                   error), thinning out; capped at 300 (counts beyond are unphysical)
  - sigma2_g     : log-spaced 1e-8 .. 100, crossing the 1e-6 exact-Poisson branch
                   of _diff_laplace_log_probs (so sigma2 < 1e-6 tests that branch,
                   not Laplace)
  - error metric : signed  Delta = log p_Laplace - log p_ref  (nats), primary;
                   relative p(r) error  e^Delta - 1  (percent), secondary
  - precision    : float64 reference; Laplace run in BOTH float32 and float64
                   (float64 = pure approximation error; the f32-f64 gap = rounding)

Runs entirely on CPU.

Output (PNG is gitignored by design):
  laplace_pointwise_error.png   signed error vs sigma2_g, one line per spike count
                                r, one panel per mu_g (cosmetic style borrowed from
                                analysis/figures/utility_landscape)
  + a printed threshold table (sigma2_g where |Delta| crosses 0.01 and 0.1 nats).

DETERMINISTIC: no random elements. Same execution -> identical results.

Usage (env: pytorch_gpytorch, torch 2.5.1):
    /home/idv-eqs8-pza/anaconda3/envs/pytorch_gpytorch/bin/python \
        investigations/utility/laplace_pointwise_error.py
"""

import math
import time
import argparse
from pathlib import Path

import torch
import numpy as np
from scipy.optimize import brentq
from scipy.integrate import quad

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# ---------------------------------------------------------------------------
# Load the local, side-effect-free utils.py (matches entropy_landscape.py:31-40).
# Only _diff_laplace_log_probs is needed -- it is the code under test.
# ---------------------------------------------------------------------------
_script_dir = Path(__file__).resolve().parent          # investigations/utility
_gpytorch_dir = _script_dir.parent.parent              # gpytorch_porting

import importlib.util
_local_utils_path = _gpytorch_dir / 'utils.py'
_spec = importlib.util.spec_from_file_location("gpytorch_porting_utils", str(_local_utils_path))
_local_utils = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_local_utils)
_diff_laplace_log_probs = _local_utils._diff_laplace_log_probs

# ---------------------------------------------------------------------------
# Experiment design (agreed with user 2026-05-25). Not hidden magic numbers:
# these define this measurement and are documented in the module docstring.
# ---------------------------------------------------------------------------
MU_G_PANELS = [-10.0, -8.0, -5.0, -3.0, -1.7, 0.0, 2.0, 4.0, 6.0, 8.0]  # user operating range; -1.7 = production
SIGMA2_MIN_DEFAULT = 1e-8             # below the 1e-6 exact-Poisson branch
SIGMA2_MAX_DEFAULT = 100.0            # stability analysis up to sigma2=100 (nothing discarded below)
N_SIGMA2_DEFAULT = 60                 # log-spaced points (10 decades, 1e-8..100)
POISSON_BRANCH = 1e-6                 # _diff_laplace_log_probs switches here (utils.py:476)
ERR_THRESHOLDS = (0.01, 0.05, 0.1)    # nats ~ 1% / 5% / 10% error in p(r) -- where Laplace bends
R_PHYSICAL_MAX = 300                  # cap on tested spike count: ~3x production r_max=100;
                                      # counts beyond are unphysical (>~850 Hz in 350ms) and never summed by the live code

# Reference quadrature controls (algorithmic, not experiment parameters).
REF_HALF_WIDTH_STD = 20.0   # window half-width in units of the local curvature std at the peak
REF_EDGE_DROP = 40.0        # widen the window until log-integrand at edges < peak - 40
REF_MAX_WIDEN = 8           # max widenings if the edge check is not yet satisfied
_G_MAX = 700.0              # cap on g so e^g stays finite in float64 (e^700 ~ 1e304)


# ---------------------------------------------------------------------------
# The 1D log-integrand  L(g) = log[ Poisson(r; e^g) * N(g; mu, sigma2) ]
#   = r*g - e^g - log(r!) - (g-mu)^2 / (2 sigma2) - 0.5 log(2 pi sigma2)
# ---------------------------------------------------------------------------
def _log_integrand(g, mu, sigma2, r):
    """log of the (un-integrated) integrand. g: float or np.ndarray; rest floats."""
    return (r * g - np.exp(g) - math.lgamma(r + 1.0)
            - (g - mu) ** 2 / (2.0 * sigma2)
            - 0.5 * math.log(2.0 * math.pi * sigma2))


def _find_peak(mu, sigma2, r):
    """Locate the integrand peak g* via brentq on the saddle equation
        h(g) = e^g + (g - mu)/sigma2 - r = 0
    h is strictly increasing (h' = e^g + 1/sigma2 > 0) -> unique root.
    Independent of the code under test (no Lambert-W reuse)."""
    def h(g):
        return math.exp(g) + (g - mu) / sigma2 - r
    g_lo = mu - 60.0
    g_hi = math.log(r + 1.0) + 60.0
    return brentq(h, g_lo, g_hi, xtol=1e-13, rtol=1e-15, maxiter=200)


def _window(mu, sigma2, r, width_mult=1.0):
    """Informed finite integration window [g_lo, g_hi] centred on the peak.

    Half-width = REF_HALF_WIDTH_STD * local_std, where local_std = |L''(g*)|^-1/2
    is the curvature std AT the peak. We deliberately use the local curvature std,
    NOT the prior sigma: curvature = e^{g*} + 1/sigma2 >= 1/sigma2, so local_std
    <= sigma always, and local_std -> sigma in the low-rate limit (where the prior
    tail is what needs covering). For sharp high-rate peaks local_std is tiny and
    the window stays tight -- crucial so quad does not sample over a span far wider
    than the spike and miss it (the bug an earlier 'max(..., 1.0)' floor caused).
    The loop still widens if the edge check is not met; width_mult lets the caller
    probe window-independence (1x vs 2x)."""
    g_star = _find_peak(mu, sigma2, r)
    curv = math.exp(min(g_star, _G_MAX)) + 1.0 / sigma2     # |L''(g*)|
    local_std = curv ** -0.5
    W = REF_HALF_WIDTH_STD * local_std * width_mult
    g_lo = g_hi = g_star
    min_drop = -math.inf
    for _ in range(REF_MAX_WIDEN):
        g_lo = g_star - W
        g_hi = min(g_star + W, _G_MAX)
        Ls = _log_integrand(np.array([g_lo, g_star, g_hi]), mu, sigma2, r)
        min_drop = float(min(Ls[1] - Ls[0], Ls[1] - Ls[2]))
        if min_drop > REF_EDGE_DROP:
            break
        W *= 2.0
    return g_star, g_lo, g_hi, min_drop


def reference_log_p(mu, sigma2, r, width_mult=1.0):
    """High-accuracy log p(r) via scipy.integrate.quad (adaptive QUADPACK), on an
    informed finite window, with the peak factored out so the integrand is in
    (0, 1]. The integral is SPLIT at the peak g* (peak on an endpoint of each
    half) so a sharp spike is never missed. Returns (log_p, rel_abserr, edge_drop)."""
    g_star, g_lo, g_hi, min_drop = _window(mu, sigma2, r, width_mult)
    L_pk = float(_log_integrand(g_star, mu, sigma2, r))

    def f(g):
        return math.exp(float(_log_integrand(g, mu, sigma2, r)) - L_pk)

    I_lo, e_lo = quad(f, g_lo, g_star, limit=400, epsabs=1e-13, epsrel=1e-13)
    I_hi, e_hi = quad(f, g_star, g_hi, limit=400, epsabs=1e-13, epsrel=1e-13)
    I = I_lo + I_hi
    return L_pk + math.log(I), (e_lo + e_hi) / max(I, 1e-300), min_drop


# ---------------------------------------------------------------------------
# r-grid: mode-anchored hybrid (agreed 2026-05-25).
# ---------------------------------------------------------------------------
def _r_grid_for_mu(mu_g):
    """14 spike counts per panel, placed where the error is informative.

    {0,1,2,3,4,5} always -- the low counts carry the biggest error (~1/(12 r))
    and are where the mass piles up at high sigma2. Plus 8 log-spaced integers
    from 6 up to r_hi = clip(round(e^{mu_g}), 30, R_PHYSICAL_MAX):
      - round(e^{mu_g}) is the left-edge (small-sigma2) mode -- the highest the
        peak ever reaches; covers the peak's full downward travel for the
        high-firing panels (mu_g=5 -> 148).
      - 30 is the floor for the low-firing panels (where the mode never exceeds
        1): the per-count error 1/(12 r) is already ~0.3% by r=30, so there is
        no point chasing the heavy, near-error-free high-sigma2 tail past it.
      - R_PHYSICAL_MAX caps the top: above ~300 spikes the count is unphysical
        (mu_g=8,10 have e^{mu_g}=2981,22026) and never summed by the live code.
    """
    r_hi = min(max(int(round(math.exp(mu_g))), 30), R_PHYSICAL_MAX)
    low = [0, 1, 2, 3, 4, 5]
    fill = {int(round(x)) for x in np.geomspace(6, r_hi, 8)}
    return sorted(set(low) | fill)


# ---------------------------------------------------------------------------
# Validation of the reference (runs before the sweep).
# ---------------------------------------------------------------------------
def validate_reference():
    print("=" * 78)
    print("REFERENCE VALIDATION (scipy.quad)")
    print("=" * 78)

    cases = [(-1.7, 1e-8, 1), (0.0, 1e-7, 5), (5.0, 1e-6, 148),  # sharp peaks (small sigma2)
             (-1.7, 0.05, 0), (-1.7, 0.1, 1), (-1.7, 0.19, 2),
             (0.0, 0.5, 1), (0.0, 2.0, 3),
             (2.0, 1.0, 7), (2.0, 5.0, 20),
             (5.0, 0.5, 148), (5.0, 5.0, 296), (5.0, 10.0, 2)]

    # (a) window-independence: doubling the window half-width must not change log p.
    print("\n(a) window-independence: quad on 1x vs 2x window half-width (log p, nats):")
    print(f"  {'mu_g':>6} {'sigma2':>9} {'r':>5} {'logp(1x)':>14} {'logp(2x)':>14} "
          f"{'|diff|':>10} {'relabserr':>10}")
    max_diff = 0.0
    worst_abserr = 0.0
    for mu, s2, r in cases:
        lp1, e1, _ = reference_log_p(mu, s2, r, width_mult=1.0)
        lp2, e2, _ = reference_log_p(mu, s2, r, width_mult=2.0)
        d = abs(lp1 - lp2)
        max_diff = max(max_diff, d)
        worst_abserr = max(worst_abserr, e1, e2)
        print(f"  {mu:>6.2f} {s2:>9.2e} {r:>5d} {lp1:>14.8f} {lp2:>14.8f} "
              f"{d:>10.2e} {max(e1, e2):>10.1e}")
    print(f"  -> max |1x vs 2x| = {max_diff:.2e} nats "
          f"({'PASS' if max_diff < 1e-9 else 'FAIL'}, tol 1e-9); "
          f"worst quad rel-abserr = {worst_abserr:.1e}")

    # (b) small-sigma2 -> exact Poisson log-pmf.
    print("\n(b) small sigma2 vs exact Poisson log-pmf  log p = r*mu - e^mu - log r!:")
    print(f"  {'mu_g':>6} {'sigma2':>9} {'r':>5} {'reference':>14} {'poisson':>14} {'|diff|':>10}")
    max_pois = 0.0
    for mu in (-1.7, 0.0, 2.0):
        for s2 in (1e-8, 1e-7):
            for r in (0, 1, 2, 5):
                lp_ref, _, _ = reference_log_p(mu, s2, r)
                lp_pois = r * mu - math.exp(mu) - math.lgamma(r + 1.0)
                d = abs(lp_ref - lp_pois)
                max_pois = max(max_pois, d)
                print(f"  {mu:>6.2f} {s2:>9.2e} {r:>5d} {lp_ref:>14.8f} "
                      f"{lp_pois:>14.8f} {d:>10.2e}")
    print(f"  -> max |reference - Poisson| = {max_pois:.2e} nats "
          f"({'PASS' if max_pois < 1e-4 else 'FAIL'}, tol 1e-4; residual is the O(sigma2) correction)")

    # (c) sum_r p_ref(r) ~ 1 (true density normalised) vs Laplace sum (drifts).
    print("\n(c) sum_r p(r): reference (true) ~ 1, Laplace drifts (this drift IS (L)):")
    for mu, s2, rmax in [(0.0, 0.1, 60), (2.0, 1.0, 300), (5.0, 1.0, 1200)]:
        rs = torch.arange(0, rmax + 1, dtype=torch.float64)
        sum_ref = sum(math.exp(reference_log_p(mu, s2, int(r))[0]) for r in rs)
        _, log_p_lap = _diff_laplace_log_probs(
            torch.tensor([mu], dtype=torch.float64),
            torch.tensor([s2], dtype=torch.float64), rs)
        sum_lap = torch.exp(log_p_lap).sum().item()
        print(f"  mu_g={mu:>5.2f} sigma2={s2:>5.2f} (r=0..{rmax}):  "
              f"sum_ref={sum_ref:.6f}   sum_Laplace={sum_lap:.6f}   "
              f"(Laplace off by {100*(sum_lap-1):+.3f}%)")
    print()
    return max_diff, max_pois, worst_abserr


# ---------------------------------------------------------------------------
# Main sweep + plotting.
# ---------------------------------------------------------------------------
def run_sweep(sigma2_grid):
    """For each mu_g panel: Laplace log p(r) (float32 & float64) and reference
    log p(r), over the sigma2 sweep, for the panel's r-grid. CPU only."""
    out = {}
    for mu in MU_G_PANELS:
        r_list = _r_grid_for_mu(mu)
        r_t64 = torch.tensor(r_list, dtype=torch.float64)
        r_t32 = torch.tensor(r_list, dtype=torch.float32)
        n_s = len(sigma2_grid)

        # Laplace path (code under test), vectorised: (n_s, R). CPU tensors.
        mu64 = torch.full((n_s,), mu, dtype=torch.float64)
        s264 = torch.tensor(sigma2_grid, dtype=torch.float64)
        _, logp_lap64 = _diff_laplace_log_probs(mu64, s264, r_t64)
        _, logp_lap32 = _diff_laplace_log_probs(mu64.float(), s264.float(), r_t32)
        logp_lap64 = logp_lap64.numpy()
        logp_lap32 = logp_lap32.numpy()

        # Reference (scipy.quad), looped per (sigma2, r).
        logp_ref = np.empty((n_s, len(r_list)))
        worst_abserr = 0.0
        worst_edge = math.inf
        for i, s2 in enumerate(sigma2_grid):
            for j, r in enumerate(r_list):
                lp, rel_abserr, ed = reference_log_p(mu, float(s2), int(r))
                logp_ref[i, j] = lp
                worst_abserr = max(worst_abserr, rel_abserr)
                worst_edge = min(worst_edge, ed)

        out[mu] = {
            'r_list': r_list,
            'logp_ref': logp_ref,
            'logp_lap64': logp_lap64,
            'logp_lap32': logp_lap32,
            'err64': logp_lap64 - logp_ref,           # signed nats (primary)
            'err32': logp_lap32 - logp_ref,
            'worst_abserr': worst_abserr,
            'worst_edge': worst_edge,
        }
    return out


def _first_crossing(sigma2_grid, abs_err, level):
    """Smallest sigma2 where |error| first exceeds `level` (scanning small->large)."""
    for s2, e in zip(sigma2_grid, abs_err):
        if e > level:
            return s2
    return None


def print_summary(sigma2_grid, results):
    print("=" * 78)
    print("THRESHOLD SUMMARY: sigma2_g where |log p(r) Laplace error| first exceeds ...")
    print("=" * 78)
    for mu in MU_G_PANELS:
        d = results[mu]
        print(f"\nmu_g = {mu:+.2f}   (E[R] ~ {math.exp(mu):.3g} at small sigma2)   "
              f"[ref: worst quad rel-abserr={d['worst_abserr']:.1e}, "
              f"worst edge_drop={d['worst_edge']:.0f}]")
        print(f"  {'r':>5} {'@0.01nat':>11} {'@0.05nat':>11} {'@0.1nat':>11} "
              f"{'max|err64|':>12} {'f32-f64':>10}")
        for j, r in enumerate(d['r_list']):
            abs64 = np.abs(d['err64'][:, j])
            ss = [(f"{c:.2e}" if c is not None else ">100")
                  for c in (_first_crossing(sigma2_grid, abs64, t) for t in ERR_THRESHOLDS)]
            gap = np.abs(d['err32'][:, j] - d['err64'][:, j]).max()
            print(f"  {r:>5d} {ss[0]:>11} {ss[1]:>11} {ss[2]:>11} {abs64.max():>12.3e} {gap:>10.2e}")

    print()


def _decade_ticks(s2_min, s2_max):
    """Powers of ten within [s2_min, s2_max] -- the 1e-k points on the log x-axis."""
    k_lo = int(math.ceil(math.log10(s2_min)))
    k_hi = int(math.floor(math.log10(s2_max)))
    return [10.0 ** k for k in range(k_lo, k_hi + 1)]


def exact_peak_mode(mu, sigma2, r_search_max):
    """Exact peak = most likely spike count = argmax over r of the TRUE p(r)
    (evaluated with the quadrature reference, not the Laplace approximation)."""
    logps = [reference_log_p(mu, sigma2, r)[0] for r in range(r_search_max + 1)]
    return int(np.argmax(logps))


def compute_peak_modes(decades):
    """Exact peak count at each sigma2_g decade, per mu_g panel. Search range is
    round(e^{mu_g}) + 10, capped at R_PHYSICAL_MAX (counts above that are
    physically meaningless); a capped result reads as >R_PHYSICAL_MAX."""
    out = {}
    for mu in MU_G_PANELS:
        r_search_max = min(int(math.ceil(math.exp(mu))) + 10, R_PHYSICAL_MAX)
        out[mu] = [exact_peak_mode(mu, float(s2), r_search_max) for s2 in decades]
    return out


def print_peak_modes(decades, peak_modes):
    print("=" * 78)
    print("EXACT PEAK = most likely spike count (mode of p(r)) at each sigma2_g decade")
    print("=" * 78)
    print("  mu_g  " + "".join(f"{s2:>9.0e}" for s2 in decades))
    for mu in MU_G_PANELS:
        cells = "".join((f"{'>'+str(R_PHYSICAL_MAX):>9}" if m >= R_PHYSICAL_MAX
                         else f"{m:>9d}") for m in peak_modes[mu])
        print(f"  {mu:+5.2f} " + cells)
    print()


def make_plots(sigma2_grid, results, outdir, decades, peak_modes):
    """Single figure: 5x2 grid (10 mu_g panels), signed Laplace error (nats) vs sigma2_g,
    one line per spike count r (coloured along a viridis ramp, small r dark ->
    large r yellow). The TOP axis of each panel shows the EXACT peak count (mode
    of p(r)) at each sigma2_g decade -- it slides toward 0 as sigma2_g grows.
    Cosmetic conventions borrowed from analysis/figures/utility_landscape.
    The data is 1D slices per r, so this stays line plots rather than a heatmap."""
    fig, axes = plt.subplots(5, 2, figsize=(13, 23), sharex=True, constrained_layout=True)
    for idx, (ax, mu) in enumerate(zip(axes.ravel(), MU_G_PANELS)):
        d = results[mu]
        colors = plt.cm.viridis(np.linspace(0.12, 0.92, len(d['r_list'])))
        for j, r in enumerate(d['r_list']):
            ax.plot(sigma2_grid, d['err64'][:, j], color=colors[j],
                    marker='.', ms=3, lw=1.3, label=f"{r}", zorder=4)
        ax.set_xscale('log')
        ax.xaxis.set_major_locator(mticker.FixedLocator([1e-8, 1e-6, 1e-4, 1e-2, 1e0, 1e2]))
        ax.xaxis.set_major_formatter(mticker.LogFormatterMathtext(base=10.0))
        ax.xaxis.set_minor_locator(mticker.LogLocator(base=10.0, subs=(1.0,), numticks=20))
        ax.xaxis.set_minor_formatter(mticker.NullFormatter())
        ax.set_yscale('symlog', linthresh=1e-4)
        ax.axvspan(sigma2_grid[0], POISSON_BRANCH, color='0.6', alpha=0.15, zorder=0)
        ax.axvline(POISSON_BRANCH, color='0.4', ls=':', lw=1.0, zorder=1)
        ax.axhline(0.0, color='0.3', lw=0.6, zorder=1)
        for lvl in ERR_THRESHOLDS:
            ax.axhline(lvl, color='0.6', ls='--', lw=0.7, zorder=1)
            ax.axhline(-lvl, color='0.6', ls='--', lw=0.7, zorder=1)
        ax.set_title(rf'$\mu_g = {mu:+.2f}$', fontsize=10)
        ax.tick_params(axis='both', labelsize=7)
        ax.grid(True, which='major', alpha=0.30)
        ax.grid(True, which='minor', alpha=0.15)
        ax.legend(title=r'spike count $r$', fontsize=6, title_fontsize=7,
                  ncol=3, loc='upper left', framealpha=0.85)
        # Top axis: EXACT peak count (mode of p(r)) at each sigma2_g decade.
        secax = ax.secondary_xaxis('top')
        secax.set_xticks(decades)
        secax.set_xticklabels([(f'>{R_PHYSICAL_MAX}' if m >= R_PHYSICAL_MAX else str(m))
                               for m in peak_modes[mu]])
        secax.tick_params(axis='x', labelsize=6, pad=1)
        if idx < 2:  # explanatory label on the top row only
            secax.set_xlabel('exact peak count (mode of p(r)) at each decade', fontsize=7)
    for ax in axes[-1, :]:
        ax.set_xlabel(r'$\sigma^2_g = A^2\,\mathrm{var}(\lambda)$', fontsize=9)
    for ax in axes[:, 0]:
        ax.set_ylabel(r'signed error  $\log p_\mathrm{Lap}(r) - \log p_\mathrm{ref}(r)$  [nats]',
                      fontsize=9)
    # One legend-of-markings box, bottom-right panel (least crowded by lines there).
    axes[-1, -1].text(0.98, 0.02,
                    r'gray band ($\sigma^2_g<10^{-6}$): exact-Poisson, not Laplace' '\n'
                    r'dotted = $10^{-6}$ branch    dashed = $\pm0.01,\pm0.05,\pm0.1$ nats',
                    transform=axes[-1, -1].transAxes, fontsize=7, va='bottom', ha='right',
                    bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                              alpha=0.85, edgecolor='gray'))
    fig.suptitle(
        r"Pointwise Laplace error of $\log p(r)$ at fixed spike count $r$ "
        r"(no sum over $r$ $\Rightarrow$ truncation excluded)"
        "\nreference: scipy.quad (adaptive QUADPACK); error computed in float64",
        fontsize=11)
    path = outdir / "laplace_pointwise_error.png"
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"[saved] {path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--n-sigma2', type=int, default=N_SIGMA2_DEFAULT)
    ap.add_argument('--sigma2-min', type=float, default=SIGMA2_MIN_DEFAULT)
    ap.add_argument('--sigma2-max', type=float, default=SIGMA2_MAX_DEFAULT)
    ap.add_argument('--outdir', type=str, default=str(_script_dir))
    args = ap.parse_args()

    torch.set_grad_enabled(False)
    outdir = Path(args.outdir)
    sigma2_grid = np.logspace(math.log10(args.sigma2_min),
                              math.log10(args.sigma2_max), args.n_sigma2)

    t0 = time.time()
    print("r-grids per panel:")
    for mu in MU_G_PANELS:
        print(f"  mu_g={mu:+.2f} (r0={int(round(math.exp(mu)))}): {_r_grid_for_mu(mu)}")
    print(f"sigma2 sweep: {args.n_sigma2} pts log-spaced "
          f"[{args.sigma2_min:.1e}, {args.sigma2_max:.1e}]\n")

    validate_reference()
    results = run_sweep(sigma2_grid)
    print_summary(sigma2_grid, results)

    decades = _decade_ticks(args.sigma2_min, args.sigma2_max)
    peak_modes = compute_peak_modes(decades)
    print_peak_modes(decades, peak_modes)

    make_plots(sigma2_grid, results, outdir, decades, peak_modes)
    print(f"\nTotal elapsed: {time.time() - t0:.1f} s")


if __name__ == '__main__':
    main()
