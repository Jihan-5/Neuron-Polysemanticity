"""polyid: partial identification of thresholded feature multiplicity.

The package is organised so that the information boundary of the study is
visible in the import graph:

* ``polyid.generators`` produces data *and* oracle ground truth.
* ``polyid.estimators`` and ``polyid.bounds`` only ever receive
  :class:`polyid.data.ObservedData` and must never import ``generators``
  (enforced by ``tests/test_leakage.py``).
* ``polyid.metrics`` is the only place where estimates meet the oracle.
"""

from polyid.estimand import EstimandSpec, multiplicity, normalize_columns
from polyid.data import ObservedData

__all__ = ["EstimandSpec", "ObservedData", "multiplicity", "normalize_columns"]
__version__ = "0.1.0"
