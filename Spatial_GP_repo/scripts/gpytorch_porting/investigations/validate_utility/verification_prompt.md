# Verification Prompt for Independent Mathematical Review

**Instructions to user**: Attach the following documents to the LLM conversation:
1. `norm_scaling_analysis.tex` (PRIMARY — contains all claims and proof attempts)
2. `distribution_aware_utility_pietro.tex` (derivation of the DA utility formula)
3. `predictive_distribution_conditioned_on_observation.tex` (derivation of GP conditioning)
4. `acosker_kernel_def_and_gradients.tex` (arc-cosine kernel definition)
5. `Gaussian_process_theory.tex` (GP posterior formulas, variational inference)

All LaTeX files are in `~/IDV_code/Papers/latex_summaries/`.

---

## Prompt

I need you to rigorously verify a mathematical analysis about gradient ascent divergence in a Gaussian process model. The analysis is in the attached `norm_scaling_analysis.tex`. The other attached documents provide foundational definitions that the analysis builds on.

### Context

We have a sparse variational GP with:
- Poisson observation model: r(x) ~ Poisson(exp(A lambda(x) + lambda_0))
- GP prior: lambda ~ GP(0, K) with the first-order arc-cosine kernel
- Variational approximation: q(lambda_tilde) = N(m, V) at M inducing points
- The GP posterior mean and variance at any point x are:
  - mu(x) = k(x)^T K_tilde^{-1} m
  - sigma^2(x) = K(x,x) + k(x)^T K_tilde^{-1} (V - K_tilde) K_tilde^{-1} k(x)
  where k(x) = [K(x, z_j)]_{j=1}^M is the cross-kernel vector to inducing points

The arc-cosine kernel (order 1) is:
  K(x, x') = (1/pi) * sqrt(v_x * v_{x'}) * J(theta)
  where:
    v_x = x^T C x + sigma_0^2
    cos(theta) = (x^T C x' + sigma_0^2) / sqrt(v_x * v_{x'})
    J(theta) = sin(theta) + (pi - theta) cos(theta)
  C is a fixed positive semi-definite matrix, sigma_0^2 >= 0 is a scalar parameter.

The "distribution-aware utility" of a candidate x*, conditioned on a single target x_t, is:
  U_DA(x*) = H_marg(x*) - H_cond(x* | lambda(x_t))
where H_marg is the entropy of r(x*) under the GP posterior, and H_cond is the entropy after Gaussian conditioning on the observed lambda(x_t).

Conditioning uses standard Gaussian formulas on the joint posterior:
  mu_cond(x*) = mu(x*) + Sigma_q(x*, x_t) / Sigma_q(x_t, x_t) * (lambda_t - mu(x_t))
  sigma^2_cond(x*) = sigma^2(x*) - Sigma_q(x*, x_t)^2 / Sigma_q(x_t, x_t)
where Sigma_q(x, x') = K(x,x') + k(x)^T K_tilde^{-1} (V - K_tilde) K_tilde^{-1} k(x').

### What I need you to do

For each claim below, do ONE of:
- **VERIFY**: Confirm the proof is correct, or provide a corrected proof
- **REFUTE**: Show a counterexample or identify the error
- **PROVE**: If I only gave a heuristic argument, try to prove it rigorously (or prove it's false)

Be explicit about any assumptions you make. If a claim is correct but the proof has a gap, say so.

---

### Claim 1 (Theorem 1 in the document): Cross-kernel proportionality

**Statement**: Let sigma_0^2 = 0 and x* = c * x_t for scalar c > 0. Then for any point z: K(x*, z) = c * K(x_t, z).

**My proof attempt**: With sigma_0^2 = 0, v_{x*} = c^2 * q where q = x_t^T C x_t. Then cos(theta(x*, z)) = (c * x_t^T C z) / sqrt(c^2 q * v_z) = (x_t^T C z) / sqrt(q * v_z) = cos(theta(x_t, z)). Since the angles are equal, J is equal, and K(x*, z) = (1/pi) * c*sqrt(q * v_z) * J(theta) = c * K(x_t, z).

**Please verify**: Is this proof correct? Does K(cx, y) = c * K(x, y) hold for ALL first-order arc-cosine kernels with sigma_0^2 = 0, or did I miss an edge case?

---

### Claim 2 (Corollary 1): Perfect posterior correlation

**Statement**: Under the conditions of Claim 1, the posterior correlation rho^2 = Sigma_q(x*, x_t)^2 / (Sigma_q(x*, x*) * Sigma_q(x_t, x_t)) = 1 exactly.

**My proof attempt**: Since k(x*) = c * k(x_t), define W = K_tilde^{-1}(V - K_tilde)K_tilde^{-1}. Then:
- Sigma_q(x*, x_t) = K(x*, x_t) + c * k_t^T W k_t = c*v_t + c*k_t^T W k_t = c * Sigma_q(x_t, x_t)
- Sigma_q(x*, x*) = c^2 * v_t + c^2 * k_t^T W k_t = c^2 * Sigma_q(x_t, x_t)
- Therefore rho^2 = c^2 * S_tt^2 / (c^2 * S_tt * S_tt) = 1.

**My concern**: This proof assumes K(x*, x_t) = c * K(x_t, x_t). This requires theta(x*, x_t) = 0 when x* = c*x_t and sigma_0^2 = 0. Is this true? cos(theta) = (c * x_t^T C x_t) / sqrt(c^2 q * q) = c*q / (c*q) = 1, so theta = 0 and J(0) = pi. Then K(x*, x_t) = (1/pi)*c*q*pi = c*q = c*v_t. This seems correct but please verify.

**Also verify**: I assumed W = K_tilde^{-1}(V - K_tilde)K_tilde^{-1} is well-defined (i.e., K_tilde is invertible) and that V is an arbitrary M x M positive definite matrix. Does the proof hold for any valid variational parameters (m, V), or are there constraints on V that could break it?

---

### Claim 3 (Corollary 2): Scaling of posterior moments

**Statement**: mu(c * x_t) = c * mu(x_t) and sigma^2(c * x_t) = c^2 * sigma^2(x_t), under sigma_0^2 = 0.

**My proof**: Direct substitution of k(cx_t) = c * k(x_t) into the posterior mean and variance formulas.

**Please verify**: Straightforward, but confirm the variance formula gives c^2 scaling (not c). The variance has two terms: K(x*,x*) = c^2*v_t and k^T W k = c^2 * k_t^T W k_t. Note that K(x*, x*) - k(x*)^T K_tilde^{-1} k(x*) = c^2 * [K(x_t, x_t) - k_t^T K_tilde^{-1} k_t], so the prior variance (Schur complement) also scales as c^2. Correct?

---

### Claim 4 (Proposition 1 + Remark 2): Gaussian observation model

**Statement**: For Gaussian observations r ~ N(lambda, tau^2) with rho = 1 (sigma^2_cond = 0), the utility is U = (1/2) log(1 + sigma^2/tau^2), which grows as log(c) for large c.

**My concern**: I initially claimed this was constant (independent of c) and had to correct myself during writing. Please verify that U = (1/2) log(1 + sigma^2/tau^2) is correct, and that it grows as log(c) (since sigma^2 = c^2 * sigma^2(x_t)).

**Important subtlety**: In the Gaussian case, H_marg = (1/2) log(2*pi*e*(sigma^2 + tau^2)) and H_cond = (1/2) log(2*pi*e*tau^2) (since sigma^2_cond = 0). So U = (1/2) log((sigma^2 + tau^2)/tau^2). This does grow with sigma^2. Is there a natural model where U would truly be constant? Or is growth with norm an inevitable consequence of any observation model + this kernel?

---

### Claim 5 (NOT proved, heuristic only): Poisson-GP utility growth rate

**My heuristic argument**: For the Poisson-GP model, H_marg grows faster than log with sigma^2_g (the log-firing rate variance), because:
- The expected firing rate E[f] = exp(mu_g + sigma^2_g / 2) grows exponentially with sigma^2_g
- For large Poisson rates, H_Poisson(lambda) ~ (1/2) log(2*pi*e*lambda)
- So H_marg >~ (1/2)(mu_g + sigma^2_g/2), giving growth linear in sigma^2_g

Meanwhile H_cond = H_Poisson(exp(mu_g)) depends only on mu_g (since sigma^2_cond = 0 exactly).

Therefore U_DA = H_marg - H_cond should grow at least as sigma^2_g, i.e., as c^2.

**Please attempt to**:
1. Determine if this heuristic argument is sound or if there's a flaw
2. Try to make it rigorous, or find a counterexample
3. In particular: is the lower bound H_marg >= (1/2)(mu_g + sigma^2_g/2) valid? I used H_marg >= H_Poisson(E[f]) (Jensen's inequality applied backwards — is this even correct? Entropy of a mixture >= entropy at the mean rate? This needs checking. Actually, by Jensen's: H[mixture] >= H[component at mean], which may NOT be true for entropy. Please verify.)

**This is the claim I am LEAST confident about.** The growth rate could be wrong. What matters for the physical finding is only that U grows unboundedly with c, not the specific rate.

---

### Claim 6: sigma_0^2 > 0 correction

**Statement**: When sigma_0^2 > 0, all the above results hold approximately with corrections O(sigma_0^2 / (c^2 q)).

**I did NOT prove this.** I stated it based on inspection of the formulas. Please either:
- Prove it (Taylor expand the key quantities around sigma_0^2 = 0 and bound the correction)
- Or show that the correction could be larger than claimed

The specific concern: does the correction to rho^2 go as O(sigma_0^2 / (c^2 q))? Or could it accumulate differently through the posterior covariance formula (which involves matrix products)?

---

### What I want as output

For each of the 6 claims:
1. Verdict: CORRECT / CORRECT WITH CAVEAT / WRONG / UNPROVED
2. If correct: state any assumptions that are implicit but not stated
3. If wrong: identify the specific error and whether it can be fixed
4. If unproved (Claim 5, 6): your best attempt at a proof or disproof

Finally: is there anything ELSE wrong with the analysis that I haven't flagged? Are there implicit assumptions in the GP posterior formulas (e.g., about the relationship between K_tilde, V, and the inducing points) that could invalidate the results?
