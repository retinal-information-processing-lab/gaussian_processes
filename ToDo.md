### ToDo documento for the Gaussian Process repository:

# Modifications to the code to make things faster.
If the code is too slow the the chosen purpose some things might be implemented
- [x] Tracking the loss during training is useless, everything can be calculated using the hyperparameters afterwards, this is not really a problem since its not that slow
- [x] In the E-step, when updating V (not V_inv) and using alpha=1, the current value of V is not used, this means that the projection onto V_b before the Estep is useless in this case. The updates to the tracking dictionaries are now done after the M-step.

# To make things more stable:
- [ ] Implement the lbfgs algorithm with boundary limits.