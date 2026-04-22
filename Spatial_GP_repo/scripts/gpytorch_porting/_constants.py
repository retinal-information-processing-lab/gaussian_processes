"""
Numerical constants loaded from default_params.json.

All files that need default values for numerical parameters import from here.
Never hardcode these values in function signatures or module bodies.
"""

import json
import pathlib

_defaults = json.loads((pathlib.Path(__file__).parent / 'default_params.json').read_text())
_model = _defaults['model']

EIGVAL_TOL = _model['eigval_tol']
LAMBDA_VAR_CLAMP = _model['lambda_var_clamp']
JITTER = _model['jitter']
CHOLESKY_MAX_TRIES = _model['cholesky_max_tries']
GPY_LBFGS_MAX_ITER = _model['gpy_lbfgs_max_iter']
F_MEAN_MAX_THRESHOLD = _model['f_mean_max_threshold']
F_MEAN_MEAN_THRESHOLD = _model['f_mean_mean_threshold']
LBFGS_TOLERANCE_CHANGE = _model['lbfgs_tolerance_change']

_es = _defaults['early_stopping']
ES_ENABLED = _es['enabled']
ES_PATIENCE = _es['patience']
ES_MIN_DELTA_REL = _es['min_delta_rel']
ES_MIN_ITERATIONS = _es['min_iterations']
ES_RESTORE_BEST = _es['restore_best']
ES_METRIC = _es['es_metric']

_ngd = _defaults['ngd']
NGD_N_ITERATIONS = _ngd['n_iterations']
NGD_LR = _ngd['lr']
NGD_ADAM_LR = _ngd['adam_lr']
NGD_ES_PATIENCE = _ngd['es_patience']
NGD_ES_MIN_DELTA_REL = _ngd['es_min_delta_rel']
NGD_ES_MIN_ITERATIONS = _ngd['es_min_iterations']
NGD_ES_RESTORE_BEST = _ngd['es_restore_best']
