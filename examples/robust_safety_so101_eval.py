#!/usr/bin/env python

"""
DEPRECATED / FORWARDER:
This script has moved to examples/safety/robust_safety_so101_eval.py as part
of the module reorganization.

Running this file directly delegates execution to examples.safety.robust_safety_so101_eval.
"""

import os
import sys
import warnings

warnings.warn(
    "examples/robust_safety_so101_eval.py has moved to examples/safety/robust_safety_so101_eval.py. "
    "Please update your script invocations accordingly.",
    DeprecationWarning,
    stacklevel=2,
)

current_dir = os.path.dirname(os.path.abspath(__file__))
safety_dir = os.path.join(current_dir, "safety")
sys.path.insert(0, safety_dir)

from robust_safety_so101_eval import main

if __name__ == "__main__":
    main()
