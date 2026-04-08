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
