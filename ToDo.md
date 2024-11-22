### ToDo documento for the Gaussian Process repository:

# Modifications to the code to make things faster.
If the code is too slow the the chosen purpose some things might be implemented
- [x] Tracking the loss during training is useless, everything can be calculated using the hyperparameters afterwards, this is not really a problem since its not that slow
- [x] In the E-step, when updating V (not V_inv) and using alpha=1, the current value of V is not used, this means that the projection onto V_b before the Estep is useless in this case. The updates to the tracking dictionaries are now done after the M-step.
- [x] Implemented the use of the explicit solution for the optimal lambda0 given logA. For now it is used only in the f_parameters update, but if we consider lambda0 as a parameter like the moments of lambda, its value should also be updated at each iteration of the Mstep. In the code this is indicated as " lambda0 2 feature ". This second part is for now not present, but the first one is faster than the code without the explicit expression.

# To make things more stable:
- [ ] Implement the lbfgs algorithm with boundary limits.

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
