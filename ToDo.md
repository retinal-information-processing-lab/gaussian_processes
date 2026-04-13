### ToDo documento for the Gaussian Process repository:

# Modifications to the code to make things faster.
If the code is too slow the the chosen purpose some things might be implemented
- [x] Tracking the loss during training is useless, everything can be calculated using the hyperparameters afterwards, this is not really a problem since its not that slow
- [x] In the E-step, when updating V (not V_inv) and using alpha=1, the current value of V is not used, this means that the projection onto V_b before the Estep is useless in this case. The updates to the tracking dictionaries are now done after the M-step.
- [x] Implemented the use of the explicit solution for the optimal lambda0 given logA. For now it is used only in the f_parameters update, but if we consider lambda0 as a parameter like the moments of lambda, its value should also be updated at each iteration of the Mstep. In the code this is indicated as " lambda0 2 feature ". This second part is for now not present, but the first one is faster than the code without the explicit expression.

# To make things more stable:
- [ ] Implement the lbfgs algorithm with boundary limits.
- [ ] Data-adaptive A initialization for interleaved F-step stability.

  **Context:** Discovered during the M-degradation investigation (2026-04-13). The original
  "M degradation" (M=250: 0.838 vs M=1500: 0.829) was entirely caused by comparing runs
  with A_init=1e-4 against runs with A_init=0.01. The current workaround is a hardcoded
  A_init=1e-4 in the ES sweep config. This works for N_train=3160 but is not principled
  and may not transfer to different dataset sizes (e.g. N=50 in the active loop).
  See `investigations/M_degradation/FINDINGS.md`.

  **The problem: E-step Newton overshoot at initialization.**

  With `interleave_fstep=True`, the E-step updates the variational mean m via a Newton step.
  The gradient of the expected log-likelihood w.r.t. m is:

      dELL/dm = A * sum_i (r_i - f_i) * u_i

  where f_i = exp(A*mu_i + lambda0) is the predicted firing rate and u_i = K_tilde_inv * k(x_i)
  is the projection vector. At initialization (m ~ 0), f_i ~ exp(lambda0) is constant for all
  images, so the gradient simplifies to A * sum_i (r_i - const) * u_i. Its magnitude scales
  with A, N_train, and the spike counts (dominated by the highest-firing images).

  But the gradient magnitude alone does not determine the Newton step size. The Hessian also
  scales with A:

      G_kk = A^2 * sum_i f_i * u_ik^2       [quadratic in A]

  The Newton step is Dm = (K_tilde + G)^{-1} * g, and two regimes exist:

  - When G >> K_tilde (Hessian dominates): the step is self-regularizing. The A dependence
    in g and G partially cancel, and A * Du stays bounded regardless of A or N. SAFE.

  - When G << K_tilde (Hessian too small to regularize): the step reduces to K_tilde^{-1} * g,
    a pure gradient step with no curvature damping. In this regime, the dangerous quantity
    A * Du (the change in the exponent of the firing rate) scales as A^2 * N. If this is large
    enough that exp(A*mu + lambda0) exceeds the stability threshold (~500), the E-step reverts
    and training stalls. The interleaved F-step compounds the problem: large mu -> F-step
    increases A -> even larger gradient next iteration.

  Which regime we are in depends on A^2 * N relative to the kernel eigenvalues. The known
  dangerous case is A=0.01 at N=3160: A^2*N = 0.316, which causes 19/19 E-step divergences
  on Cell 8. The known safe case is A=1e-4 at N=3160: A^2*N = 3.16e-5.

  **Candidate fix: data-adaptive A_init.**

  A natural idea is to scale A_init with the data so the initial gradient magnitude is O(1):

      A_init = c / (N * mean(r))                 ... (candidate 1)

  This gives A^2*N = c^2 / (N * mean(r)^2), which *decreases* with N. Concrete values:

      N=3160, mean(r)=1.5:  A_init = 2.1e-4,  A^2*N = 1.4e-4   (safe, similar to 1e-4)
      N=50,   mean(r)=1.5:  A_init = 0.013,   A^2*N = 0.009    (still 35x below danger)

  The N=50 case (relevant for the active learning loop, which starts with ~50 images) stays
  well below the dangerous threshold of A^2*N ~ 0.3. So the formula is not "too big" at small N.

  **However, mean(r) does not capture the tail.** The gradient is dominated by images with the
  highest spike counts, not the average. For a cell with sparse high responses (mean(r)=0.1,
  max(r)=10 — e.g. 5 images firing out of 50), candidate 1 gives:

      A_init = 1/(50 * 0.1) = 0.2,   A^2*N = 2.0

  That is dangerously close to the overshoot regime. The formula is too aggressive because
  mean(r) underestimates the gradient magnitude for cells with sparse, bursty responses.

  A more robust alternative uses sum(r^2), which naturally weighs both the number of
  responding images and the magnitude of their responses:

      A_init = c / sqrt(N * sum(r_i^2))          ... (candidate 2)

  For the sparse cell above (5 images at r=5, rest zero, sum(r^2) = 125):

      Candidate 1: A = 0.04,   A^2*N = 0.08     (marginal)
      Candidate 2: A = 0.013,  A^2*N = 0.008    (10x more conservative)

  For a well-behaved cell (N=3160, mean(r)=1.5, sum(r^2) ~ 7000):

      Candidate 1: A = 2.1e-4
      Candidate 2: A = 1/sqrt(3160*7000) = 2.1e-4   (essentially identical)

  The two formulas agree when responses are well-distributed and diverge where it matters:
  sparse high-firing cells.

  **Decision needed before implementing:** which formula to use, and what value of c. Both
  candidates should be tested empirically on cells with different response statistics
  (e.g. Cell 0: low mean, Cell 8: moderate, Cell 1: high) and different N (50, 500, 3160)
  to verify the overshoot analysis above. A simpler compromise might also work:

      A_init = c / (N * mean(r) + max(r))        ... (candidate 3)

  which adds max(r) as a direct regularizer without requiring the sum(r^2) computation.

# Bugs:
- When the xtilde indexes are not sorted, some stability problems might arise. In the case of full ntilde=ntrain as with initialization of "/models/exact_fit_cells/cell8/10_10_10/metadata" a Nan in the f params update emerges in the second iteration. This disappears when indices get sorted. Regarding this, I also tried with smaller ntildes with and without sorting the xtilde indices, to see if any differences arise. There is indeed a difference, even in the r2 value on the test set, even if very small ( 0.01 difference ). I found that the mask coming from the localkernel is probably the cause of it, due to the fact that by changing the order of inputs in the kernel, the mask is changing shape, and number of True values. This should be investigated but the differenc eis minimal, I will just resort to sorting the inddexes every time.

- For some reason, when the ntilde = ntrain is low ( less than 5 ) it looks like their gradients are extremely low and the Mstep does not update them. Minor bug.

- Major bug. Some combinations of carefully chosen ntildes = ntrain ( low, less than 100 ) still give Nan r2. This seems to happen only when i reinitialize V and m each time. If i initialize the last row and column of V ( original space ) with the final ones of K_tilde ( origginal space ) it looks like this does not happen. 
This still happened, but it might be solved by:
  [x] Reinitializing the lbfgs algorithm at each estep ( its a different optimization problem each time ). Done, some instabilities removed
        [x] Try with 5 different seeds during the active loop ( same starting model) => f_mean diverges, but its due to the capping of f_mean ( it distrups the gradients )
  [x] Removing the capping on f_mean => Solves some instabilities, no distrupting of gradients means no
        [x] Look for instabilities with no capping on f_mean(). 
            [x] Didn't find them yet, trying to go back to lambda0 optimization ( no loglambda0 )
        [x] Found an instability in the M step. The receptive fireld goes out of the allowed limits. Saved as model 'bug_rf_out_of_bounds' -> solved by returning infinite loss and infinite gradient for the out of bound hyperparameter

- While evaluating the utility of a model trained on 1080 inducing points, which make a 1080 remaining dataset to be evaluated, I get an error, stemming from negatiev lambda_var values for the 1080 predicted firing rate.
