"""Diagnose: is NGD overfitting at low n_train, M=n_train regime?

Hypothesis: NGD's first-order optimization with patience=200 lets the model
keep training long after the test_r peaks. The training ELBO monotonically
improves (A inflates → better fit to 50 training points) but test_r degrades.

Test:
  - Run NGD on a few cells at M=50, n_train=50, with `test_r_probe` enabled
    to record test_r every 25 iters DURING training.
  - Disable ES (run full 1500 iters) so we see the entire trajectory.
  - Plot train_loss (= -ELBO) and test_r curves.
  - If overfitting: train_loss decreases monotonically while test_r rises then falls.

Cells:
  - 0   (NGD-only disaster: NGD got -0.10, vargp got 0.39)
  - 35  (NGD-only disaster: NGD got -0.15, vargp got 0.38)
  - 16  (healthy, both work)
  - 30  (vargp-only disaster: NGD got 0.43, vargp got 0.05 — opposite case)
"""
from __future__ import annotations
import json, os, sys, time
from pathlib import Path

EXP_DIR = Path(__file__).parent
ROOT    = EXP_DIR.parent.parent
sys.path.insert(0, str(ROOT))

from run_single_mode import build_config_from_defaults, run_single_config
from ngd_training import train_ngd
import torch

# Probe wrapper — captures test data once, then evaluates test_r at each call
def make_test_r_probe(X_test, r_test, predict_fn, jitter, ckm, dtype, device, lambda_var_clamp):
    r_test_mean = r_test.mean(dim=0)
    from metrics import compute_pearson_correlation
    def probe(model, likelihood):
        try:
            preds = predict_fn(model, likelihood, X_test, device=device,
                               jitter=jitter, cholesky_max_tries=ckm,
                               lambda_var_clamp=lambda_var_clamp)
            f_pred = preds['f_pred']
            return compute_pearson_correlation(r_test_mean, f_pred).item()
        except Exception:
            return float('nan')
    return probe


def run_one_with_curves(cell, M=50, n_train=50, n_iterations=1500):
    """Run NGD with full diagnostics — no ES, log test_r every 25 iters."""
    os.chdir(ROOT)
    config = build_config_from_defaults(
        mode='ngd',
        data_path='datasets/PNAS_64x64_center_crop_no_renorm.npz',
        M=M, n_train=n_train, cell=cell, seed=1,
        ip_selection='random', fix_Amp=True,
        ngd_n_iterations=n_iterations,
    )
    # Patch: disable ES and use big patience just to be safe
    config['early_stop'] = False

    # We need to drive training ourselves to inject the probe.
    # The simplest way: monkey-patch train_ngd to receive a probe.
    # Actually, train_ngd already accepts test_r_probe as a kwarg — but
    # run_single_config doesn't forward it. We'll patch run_single_config
    # by intercepting the call.
    import ngd_training, run_single_mode
    original_train_ngd = ngd_training.train_ngd

    captured = {}
    def patched_train_ngd(model, likelihood, X_train, r_train, **kwargs):
        # We need X_test and r_test from the outer scope — use captured dict
        if 'X_test' in captured:
            from gpy_training import predict
            probe = make_test_r_probe(
                captured['X_test'], captured['r_test'], predict,
                kwargs.get('jitter'), kwargs.get('cholesky_max_tries'),
                X_train.dtype, X_train.device,
                config.get('lambda_var_clamp'),
            )
            kwargs['test_r_probe'] = probe
            kwargs['test_r_every'] = 25
        # Force ES off, full iters
        kwargs['early_stop'] = False
        return original_train_ngd(model, likelihood, X_train, r_train, **kwargs)
    ngd_training.train_ngd = patched_train_ngd

    # Capture X_test/r_test by intercepting run_single_config (simpler: capture before train)
    # Actually run_single_config builds X_test internally. We can instead just rebuild it
    # from the dataset once.
    import numpy as np
    data = np.load(config['data_path'])
    X_test = torch.from_numpy(data['images_test']).float().reshape(data['images_test'].shape[0], -1)
    r_test = torch.from_numpy(data['responses_test']).float()[:, :, cell]  # (30, n_repeats)
    if torch.cuda.is_available():
        X_test = X_test.cuda(); r_test = r_test.cuda()
    captured['X_test'] = X_test
    captured['r_test'] = r_test

    print(f"\n=== Cell {cell} M={M} n_train={n_train} ===", flush=True)
    t0 = time.time()
    result = run_single_config(config)
    wall = time.time() - t0
    ngd_training.train_ngd = original_train_ngd

    curves = result.get('curves', {})
    out = {
        'cell': cell, 'M': M, 'n_train': n_train,
        'final_test_r': result.get('test_r'),
        'final_A': result.get('final_A'),
        'final_beta': result.get('final_beta'),
        'best_iteration': result.get('best_iteration'),
        'n_iter': result.get('n_iterations_run'),
        'wall': wall,
        'train_loss_curve': curves.get('train_loss'),
        'A_curve': curves.get('A'),
        'beta_curve': curves.get('beta'),
        'test_r_iter': curves.get('test_r_iter'),
        'test_r_curve': curves.get('test_r'),
    }
    print(f"  final test_r={out['final_test_r']:.4f}  best_iter={out['best_iteration']}  "
          f"final_A={out['final_A']:.4f}  wall={wall:.0f}s", flush=True)
    return out


def main():
    out_path = EXP_DIR / 'overfit_diagnostic.jsonl'
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cells_to_test = [0, 35, 16, 30]  # NGD-only disasters + healthy + vargp-only disaster
    for c in cells_to_test:
        rec = run_one_with_curves(c)
        with open(out_path, 'a') as f:
            f.write(json.dumps(rec) + '\n')
    print(f"\nDone. Output: {out_path}", flush=True)


if __name__ == '__main__':
    main()
