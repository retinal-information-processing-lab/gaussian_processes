 Incident Report: Hardcoded Parameters in Validation Script

  What happened                                                         
  
  When building validate_utility_natural_images.py in the previous      
  session, I wrote:                                        

  config = flatten_yaml_config(
      yaml_config, mode='default_gpy', M=50, n_train=500, seed=123,
  cell=8
  )
  config['float32'] = True

  This has three problems:
  1. seed=123, M=50 are disconnected literals. default_params.json has
  seed=42, M=100. Nobody looking at the script can tell where these
  values come from or whether they're intentional deviations.
  2. config['float32'] = True is a no-op — run_single_config reads
  config['dtype'], not config['float32']. I didn't verify this; I
  assumed the key name.
  3. The YAML was loaded only for non-experiment sections (kernel,
  training, etc.), while the experiment-point values were hardcoded.
  This created a hybrid config from two sources with no clear boundary.

  My mistake (root cause)

  I treated flatten_yaml_config() as the general-purpose config builder,
   when it was designed specifically for the experiment system's matrix
  iteration. Its signature forces you to pass mode, M, n_train, seed,
  cell explicitly — that's correct when iterating over
  itertools.product(modes, M_values, ...) in run_experiment.py, but
  wrong for a standalone script that should just use defaults.

  Instead of recognizing this API mismatch, I filled in the required
  arguments with values that "seemed reasonable" — M=50 because it's
  fast, seed=123 because it appears in quick.yaml. I didn't ask: where
  should these values come from? I just picked numbers that would make
  the code run.

  The deeper error: I was focused on the scientific goal (get the
  utility comparison running) and treated the config setup as
  boilerplate to get past quickly. I didn't think about the script as a
  piece of the reproducibility system — I treated it as throwaway
  investigation code where exact parameter values didn't matter.

  What the user could have done to prevent this

  1. The rule didn't exist yet. The working guidelines had
  reproducibility rules (5.5: seeds must be controlled, same execution =
   same results) but nothing about where parameter values should come
  from. The rule covered "results must be reproducible" but not
  "parameters must be traceable to a config source." The gap between
  these two is exactly where the bug lives.
  2. The plan review was the missed catch point. The plan explicitly
  listed M=50, n_train=50, cell=8, seed=123 as the configuration. The
  user reviewed and approved the plan. At that point, asking "where do
  these values come from? Are they defaults?" would have surfaced the
  issue before any code was written.
  3. The API gap was a trap. Before this fix, there was no
  build_config_from_defaults(). The only programmatic ways to build a
  config were: (a) flatten_yaml_config (requires explicit matrix-point
  args), or (b) the CLI argparse in main() (not importable). With no
  clean path for "just give me defaults", hardcoding was the path of
  least resistance. Having the helper available before the investigation
   session would have prevented this.

  What should be captured in a skill for investigation scripts

  The key things I missed, in priority order:

  1. Parameter sourcing: Before writing any config setup, ask: where
  does each parameter value come from? If the answer is "I picked a
  reasonable number", that's wrong. It must trace to a config file.
  2. Use the right API: build_config_from_defaults(mode) for standalone
  scripts, flatten_yaml_config(yaml, ...) for experiment matrix
  iteration. Don't use the matrix API when you just want defaults.
  3. Overrides must be visible and justified: If deviating from defaults
   (e.g., M=50 when default is 100), make it an explicit override with a
   comment explaining why.
  4. Verify config key names: I wrote config['float32'] = True without
  checking what key run_single_config actually reads. Always verify the
  consumer's expected key names.
  5. Investigation scripts are not throwaway code: They produce results
  that get documented and discussed. They deserve the same parameter
  discipline as production scripts.