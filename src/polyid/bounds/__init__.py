"""Uncertainty layer: identified sets, sensitivity ranges and confidence bounds for P_j(tau).

Labels used throughout (plan p.7, "Compute honesty"):

* ``identified_set``  — exact population object under declared assumptions;
* ``confidence``      — a procedure with an explicit coverage argument
  (and the assumptions under which it holds, stated in its docstring);
* ``sensitivity``     — spread over optimiser restarts / perturbations; *no*
  coverage claim.
"""
