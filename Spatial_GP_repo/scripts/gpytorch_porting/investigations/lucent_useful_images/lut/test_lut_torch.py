"""Correctness gate: the torch LUT twin (lut_utility.lut_U) must match the vendored numpy
reference (lut_select.LUTUtility) and have correct gradients.

Standalone: imports ONLY the vendored lut_select (numpy/scipy) + lut_utility (torch) from
this folder -- no GP engine, no GPU, no dataset. Deterministic (fixed seed; gradcheck is
exact), so re-running gives identical numbers.

Checks (all in float64 unless noted; gate = 1e-5 per the plan):
  1. node reproduction  -- every one of the 2,501 grid nodes returns its stored U.
  2. in-domain random   -- mu in [-6,6], sigma2 in [0,6] (uniform + log-sampled small s2).
  3. sigma2-OOB random  -- sigma2 in (6,50]: the continuity-corrected high-rate fallback.
  4. mu-clamp random    -- mu in [-12,-6) U (6,12]: mu clamped to the LUT edge.
  5. combined OOB        -- mu out of range AND sigma2 > 6 together.
  6. gradcheck (float64) -- autograd vs finite-difference at mid-cell + OOB points.
  7. float32 precision   -- production dtype error (informational).

Run:  <pytorch_gpytorch python> test_lut_torch.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # run from any cwd

import numpy as np
import torch

import lut_select          # vendored numpy reference (LUTUtility)
import lut_utility         # the torch twin under test

SEED = 0
GATE = 1e-5                # plan: <= 1e-5 in-domain AND out-of-domain


def _max_abs_err(mu_np, s2_np, ref, dtype):
    """max |torch lut_U - numpy LUTUtility| over the given points, in `dtype`."""
    u_ref = np.atleast_1d(ref(mu_np, s2_np)).astype(np.float64)
    mu_t = torch.as_tensor(mu_np, dtype=dtype)
    s2_t = torch.as_tensor(s2_np, dtype=dtype)
    with torch.no_grad():
        u_torch = lut_utility.lut_U(mu_t, s2_t).cpu().numpy().astype(np.float64)
    return float(np.max(np.abs(u_torch - u_ref)))


def main():
    rng = np.random.default_rng(SEED)
    ref = lut_select.LUTUtility()          # vendored numpy/scipy reference
    mu_axis = lut_utility._MU_AXIS_NP
    s2_axis = lut_utility._S2_AXIS_NP
    results = []   # (name, max_err, gated)

    # 1. node reproduction: all 61*41 grid nodes.
    MU, S2 = np.meshgrid(mu_axis, s2_axis, indexing="ij")
    e_nodes = _max_abs_err(MU.ravel(), S2.ravel(), ref, torch.float64)
    results.append(("node reproduction (2501 nodes)", e_nodes, True))

    # 2. in-domain random: uniform mu; s2 half-uniform, half log-sampled (dense small-s2).
    n = 20000
    mu_in = rng.uniform(-6.0, 6.0, n)
    s2_unif = rng.uniform(0.0, 6.0, n // 2)
    s2_log = np.exp(rng.uniform(np.log(1e-3), np.log(6.0), n - n // 2))
    s2_in = np.concatenate([s2_unif, s2_log])
    rng.shuffle(s2_in)
    e_in = _max_abs_err(mu_in, s2_in, ref, torch.float64)
    results.append(("in-domain (mu in[-6,6], s2 in[0,6])", e_in, True))

    # 3. sigma2-OOB random: the high-rate fallback (sigma2 in (6,50]).
    mu_oob = rng.uniform(-6.0, 6.0, n)
    s2_oob = rng.uniform(6.001, 50.0, n)
    e_oob = _max_abs_err(mu_oob, s2_oob, ref, torch.float64)
    results.append(("sigma2-OOB (s2 in(6,50])", e_oob, True))

    # 4. mu-clamp: mu beyond [-6,6] must clamp to the edge (both impls).
    mu_clamp = np.concatenate([rng.uniform(-12.0, -6.0, n // 2),
                               rng.uniform(6.0, 12.0, n // 2)])
    s2_clamp = rng.uniform(0.0, 6.0, n)
    e_clamp = _max_abs_err(mu_clamp, s2_clamp, ref, torch.float64)
    results.append(("mu-clamp (|mu|>6, s2 in[0,6])", e_clamp, True))

    # 5. combined OOB: mu out of range AND sigma2 > 6.
    mu_both = np.concatenate([rng.uniform(-12.0, -6.0, n // 2),
                              rng.uniform(6.0, 12.0, n // 2)])
    s2_both = rng.uniform(6.001, 50.0, n)
    e_both = _max_abs_err(mu_both, s2_both, ref, torch.float64)
    results.append(("combined mu&sigma2 OOB", e_both, True))

    # 7. float32 production precision (informational, not gated -- experiment runs float32).
    e_f32 = _max_abs_err(mu_in, s2_in, ref, torch.float32)

    # ---- report value checks ----
    print("=" * 70)
    print("torch lut_U  vs  vendored numpy lut_select.LUTUtility")
    print(f"  (gate = {GATE:.0e};  float64 unless noted)")
    print("=" * 70)
    all_pass = True
    for name, err, gated in results:
        ok = err <= GATE
        all_pass = all_pass and (ok or not gated)
        flag = "PASS" if ok else ("FAIL" if gated else "info")
        print(f"  [{flag}] {name:42s} max|err| = {err:.3e}")
    print(f"  [info] {'in-domain, FLOAT32 (production dtype)':42s} max|err| = {e_f32:.3e}")

    # 6. gradcheck (float64) at mid-cell in-domain points + smooth OOB points.
    # Mid-cell guarantees no bilinear kink within the gradcheck eps-ball; OOB points are
    # well above the sigma2=6 seam (fallback is smooth there).
    iu = np.array([10, 30, 42, 53]); ju = np.array([5, 15, 25, 35])
    mu_mid = 0.5 * (mu_axis[iu] + mu_axis[iu + 1])          # arithmetic cell midpoints
    s2_mid = 0.5 * (s2_axis[ju] + s2_axis[ju + 1])
    mu_gc = np.concatenate([mu_mid, mu_mid])               # reuse mu for OOB block
    s2_gc = np.concatenate([s2_mid, np.array([8.0, 20.0, 35.0, 45.0])])
    mu_t = torch.tensor(mu_gc, dtype=torch.float64, requires_grad=True)
    s2_t = torch.tensor(s2_gc, dtype=torch.float64, requires_grad=True)
    try:
        gc_ok = torch.autograd.gradcheck(lut_utility.lut_U, (mu_t, s2_t),
                                         eps=1e-6, atol=1e-6, rtol=1e-4)
    except Exception as exc:  # gradcheck raises on mismatch
        gc_ok = False
        print(f"  [FAIL] gradcheck raised: {exc}")
    if gc_ok:
        print(f"  [PASS] gradcheck (float64, {mu_t.numel()} pts: mid-cell + OOB)")
    all_pass = all_pass and gc_ok

    print("=" * 70)
    print("RESULT:", "ALL PASS" if all_pass else "FAILURE")
    print("=" * 70)
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
