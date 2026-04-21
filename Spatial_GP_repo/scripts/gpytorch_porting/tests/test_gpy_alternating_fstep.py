"""Tests for the slim F-step closure used by default_gpy + alternating_fstep.

Covers:
  A. End-to-end: training with alternating_fstep=True completes without
     raising AlternatingFstepError (i.e. the first-iter grad-leak assertion
     in gpy_training.py passes).
  B. Helper consistency: _precompute_likelihood_constants + _compute_ell_slim
     reproduce the ELBO that GPyTorch's native path (model(x) +
     PoissonLikelihood.expected_log_prob + variational_strategy.kl_divergence)
     computes at the same model state.
  C. Grad-flow isolation: after one slim closure call, only likelihood
     params (A, lambda0) receive gradients — kernel and variational params'
     .grad must be untouched.

Uses real PNAS data via run_single_config. Cell 8, seed 42, M=50,
n_iterations=3 — kept small for speed. Total runtime target: <1 minute on GPU.

Run:
    python -m pytest tests/test_gpy_alternating_fstep.py -v
"""

import sys
from pathlib import Path

import torch

PROJ = Path(__file__).parent.parent
sys.path.insert(0, str(PROJ))

from run_single_mode import build_config_from_defaults, run_single_config  # noqa: E402
from gpy_training import (                                                  # noqa: E402
    AlternatingFstepError,
    _compute_ell_slim,
    _precompute_likelihood_constants,
)


# ---------------------------------------------------------------------------
# Shared setup: train once with alternating_fstep=True, reuse across tests.
# The model state after 3 outer iters is a realistic working point (not at
# init). Tests B and C share this fitted state.
# ---------------------------------------------------------------------------
_cached = {}


def _fit_small_alt():
    """Fit a default_gpy model with alternating_fstep=True. Cache result."""
    if 'result' in _cached:
        return _cached['result']
    config = build_config_from_defaults(
        mode='default_gpy',
        # Match run_baseline.py's config exactly (investigations/default_gpy_gap):
        # 64x64 dataset, M=50, n_train=500, random IPs. n_iterations=3 is a
        # deliberate test-speed reduction from the baseline's 50; the grad-leak
        # check only fires at i==0 anyway.
        data_path='datasets/PNAS_64x64_center_crop_no_renorm.npz',
        M=50,
        n_train=500,
        seed=42,
        cell=8,
        n_iterations=3,
        ip_selection='random',
        alternating_fstep=True,
    )
    # Early stopping disabled: we want exactly 3 outer iters so the first-iter
    # grad-leak check runs at i=0 and we observe a few post-check iters too.
    config['early_stop'] = False
    result = run_single_config(config)
    assert result is not None, "run_single_config returned None under alternating_fstep"
    _cached['result'] = result
    _cached['config'] = config
    return result


def _load_train_data(config):
    """Rebuild the (X_train, r_train) tensors used by the fit, so tests can
    evaluate the model at the same points the closures would see."""
    import numpy as np
    data_path = Path(config['data_path'])
    if not data_path.is_absolute():
        data_path = PROJ / data_path
    data = np.load(data_path)
    device = torch.device(config['device'])
    dtype = torch.float32

    # Replicate run_single_mode.py's data flow: combine train + val, carve by
    # seeded permutation, slice n_train.
    imgs = np.concatenate([data['images_train'], data['images_val']], axis=0)
    resp = np.concatenate([data['responses_train'], data['responses_val']], axis=0)
    X_pool = torch.tensor(imgs, dtype=dtype).reshape(imgs.shape[0], -1).to(device)
    R_pool = torch.tensor(resp, dtype=dtype).to(device)

    # result['_indices_train'] gives the exact indices used by training.
    indices = _cached['result']['_indices_train']
    X_train = X_pool[indices]
    r_train = R_pool[indices, config['cell']]
    return X_train, r_train


# ---------------------------------------------------------------------------
# Test A: training with alternating_fstep=True does not raise.
# If the grad-leak check fires, AlternatingFstepError propagates out of
# run_single_config and this test fails.
# ---------------------------------------------------------------------------

def test_alternating_training_completes_without_leak():
    result = _fit_small_alt()
    # Sanity: training actually ran at least once and recorded curves.
    assert result.get('curves') is not None
    assert len(result['curves'].get('train_loss', [])) >= 1
    print("PASS: alternating_fstep training completed without leak/fail")


# ---------------------------------------------------------------------------
# Test B: slim helpers reproduce the full GPyTorch ELBO.
# At the fitted model state, compute:
#   (ell_slim, kl_slim) via _precompute_likelihood_constants + _compute_ell_slim
#   (ell_ref,  kl_ref ) via model(x) + likelihood.expected_log_prob +
#                          variational_strategy.kl_divergence
# Assert numerical agreement within float32 tolerance.
# ---------------------------------------------------------------------------

def test_slim_helpers_match_gpytorch_elbo():
    result = _fit_small_alt()
    model = result['_model']
    likelihood = result['_likelihood']
    config = _cached['config']
    X_train, r_train = _load_train_data(config)

    # Slim path
    mu_const, var_const, kl_const = _precompute_likelihood_constants(model, X_train)
    A = likelihood.A.squeeze()
    lambda0 = likelihood.lambda0.squeeze()
    ell_slim, f_slim = _compute_ell_slim(A, lambda0, r_train, mu_const, var_const)
    loss_slim = (-ell_slim + kl_const).detach()

    # Reference path (mirrors what closure_model in gpy_training.py computes).
    # Model is in .train() mode after run_single_config (per gpy_training.py
    # flow); switch to eval to bypass the dropout-style cache-clear and get
    # the same (mu, var). Switch back to train after.
    was_training = model.training
    model.eval()
    likelihood.eval()
    with torch.no_grad():
        out_ref = model(X_train)
        ell_ref = likelihood.expected_log_prob(r_train, out_ref).sum()
        kl_ref = model.variational_strategy.kl_divergence()
        loss_ref = -ell_ref + kl_ref
    if was_training:
        model.train()
        likelihood.train()

    # Float32 tolerance. Both paths compute the same formula; differences come
    # only from reduction order and the (negligible) fp32 roundoff in Cholesky.
    torch.testing.assert_close(ell_slim.detach(), ell_ref, atol=1e-3, rtol=1e-4)
    torch.testing.assert_close(kl_const, kl_ref, atol=1e-3, rtol=1e-4)
    torch.testing.assert_close(loss_slim, loss_ref, atol=1e-3, rtol=1e-4)
    print(f"PASS: slim ell={ell_slim.item():.6f} vs ref={ell_ref.item():.6f}")
    print(f"      slim kl ={kl_const.item():.6f} vs ref={kl_ref.item():.6f}")


# ---------------------------------------------------------------------------
# Test C: grad flow is isolated to likelihood params only.
# After manually precomputing + running one backward through the slim ELL,
# assert that only A and lambda0 have .grad populated with non-zero values,
# and that every kernel / variational param's .grad is either None or zero.
# ---------------------------------------------------------------------------

def test_grad_flow_isolated_to_likelihood():
    result = _fit_small_alt()
    model = result['_model']
    likelihood = result['_likelihood']
    config = _cached['config']
    X_train, r_train = _load_train_data(config)

    # Clean grad state.
    for p in list(model.parameters()) + list(likelihood.parameters()):
        if p.grad is not None:
            p.grad.zero_()

    # Precompute — detaches mu/var/kl from the graph.
    model.train()
    likelihood.train()
    mu_const, var_const, kl_const = _precompute_likelihood_constants(model, X_train)
    A = likelihood.A.squeeze()
    lambda0 = likelihood.lambda0.squeeze()
    ell, _ = _compute_ell_slim(A, lambda0, r_train, mu_const, var_const)
    loss = -ell + kl_const
    loss.backward()

    # Likelihood params must have received gradient. PoissonLikelihood exposes
    # A as a non-leaf property (A = exp(raw_A)); the .grad lands on raw_A.
    assert likelihood.raw_A.grad is not None, "raw_A.grad is None after slim backward"
    assert likelihood.raw_A.grad.abs().sum().item() > 0, (
        f"raw_A.grad all-zero after slim backward: {likelihood.raw_A.grad}"
    )
    assert likelihood.lambda0.grad is not None, "lambda0.grad is None"
    assert likelihood.lambda0.grad.abs().sum().item() > 0, (
        f"lambda0.grad all-zero after slim backward: {likelihood.lambda0.grad}"
    )

    # Model params (kernel + variational) must not have received gradient.
    leaks = []
    for name, p in model.named_parameters():
        if p.grad is None:
            continue
        s = p.grad.abs().sum().item()
        if s > 0:
            leaks.append((name, s))
    assert not leaks, f"Gradient leak into model params: {leaks}"
    print("PASS: grad flow isolated — only A and lambda0 got gradients")


# ---------------------------------------------------------------------------
# Entry point (pytest-compatible; also runs as a script)
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    import traceback
    tests = [
        test_alternating_training_completes_without_leak,
        test_slim_helpers_match_gpytorch_elbo,
        test_grad_flow_isolated_to_likelihood,
    ]
    passed = 0
    for fn in tests:
        print(f"\n--- {fn.__name__} ---")
        try:
            fn()
            passed += 1
        except Exception as e:
            print(f"FAIL: {e}")
            traceback.print_exc()
    print(f"\n{passed}/{len(tests)} tests passed")
