"""
Why do the optimized "most-useful" images of cells 3 and 36 predict ~0 response,
while cell 13's predict moderate firing? Two candidate explanations:
  (a) small response gain A  -> no bounded image can drive the cell, so the utility
      optimum is purely epistemic (high-uncertainty direction, low firing);
  (b) normal A but the optimizer happens to land on a low-firing point.
This separates them by printing, per cell at a given n_train:
  - A, lambda0 (the PoissonLikelihood gain + baseline)
  - the predicted-response distribution over a pool sample
    firing = exp(mu_g + 0.5*sigma2_g),  mu_g = A*lam_m + lambda0,  sigma2_g = A^2*lam_v
If the pool's MAX achievable firing is ~0, the cell simply does not fire much in the
dataset (explanation a); if the pool reaches high firing but the optimum sits low, it
is an optimization/utility property (explanation b).

Reproducible (seed 42, M=n_train). Usage: python firing_diagnostic.py --cells 3 13 36 --n-train 300
"""
import argparse

import numpy as np
import torch

import gp_models
import acquisition

IMG_SIDE = 108


def pool_firing(model, lik, X, chunk=512):
    A = lik.A.squeeze(); lambda0 = lik.lambda0.squeeze()
    fr = []
    with torch.no_grad():
        for i in range(0, X.shape[0], chunk):
            lam_m, lam_v = acquisition.get_gp_marginal_moments(model, X[i:i + chunk])
            mu_g = A * lam_m + lambda0
            sig2_g = A ** 2 * lam_v
            fr.append(torch.exp(mu_g + 0.5 * sig2_g).cpu().numpy())
    return np.concatenate(fr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", type=int, nargs="*", default=[3, 13, 36])
    ap.add_argument("--n-train", type=int, default=300)
    args = ap.parse_args()

    X_all, gp_min, gp_max = gp_models.load_pool()
    print(f"pool {tuple(X_all.shape)}  range [{gp_min:.3f}, {gp_max:.3f}]")
    print(f"\n{'cell':>4} {'test_r':>7} {'A':>8} {'lambda0':>9} | "
          f"{'fr_min':>7} {'fr_med':>7} {'fr_mean':>7} {'fr_max':>7}  (predicted response over pool)")
    for cell in args.cells:
        model, lik, idx, test_r = gp_models.train_default_gpy(
            cell_id=cell, n_train=args.n_train, M=args.n_train, seed=42)
        X_pool, _ = gp_models.pool_complement(X_all, idx)
        fr = pool_firing(model, lik, X_pool)
        print(f"{cell:>4} {test_r:>+7.2f} {float(lik.A):>8.4f} {float(lik.lambda0):>+9.3f} | "
              f"{fr.min():>7.2f} {np.median(fr):>7.2f} {fr.mean():>7.2f} {fr.max():>7.2f}")
        del model, lik
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
