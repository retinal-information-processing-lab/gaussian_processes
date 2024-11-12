### ToDo documento for the Gaussian Process repository:

# Modifications to the code to make things faster.
If the code is too slow the the chosen purpose some things might be implemented
- [x] Tracking the loss during training is useless, everything can be calculated using the hyperparameters afterwards, this is not really a problem since its not that slow
- [x] In the E-step, when updating V (not V_inv) and using alpha=1, the current value of V is not used, this means that the projection onto V_b before the Estep is useless in this case. The updates to the tracking dictionaries are now done after the M-step.

# To make things more stable:
- [ ] Implement the lbfgs algorithm with boundary limits.

# Bugs:
- When the xtilde indexes are not sorted, some stability problems might arise. In the case of full ntilde=ntrain as with initialization of "/models/exact_fit_cells/cell8/10_10_10/metadata" a Nan in the f params update emerges in the second iteration. This disappears when indices get sorted. Regarding this, I also tried with smaller ntildes with and without sorting the xtilde indices, to see if any differences arise. There is indeed a difference, even in the r2 value on the test set, even if very small ( 0.01 difference ). I found that the mask coming from the localkernel is probably the cause of it, due to the fact that by changing the order of inputs in the kernel, the mask is changing shape, and number of True values. This should be investigated but the differenc eis minimal, I will just resort to sorting the inddexes every time.