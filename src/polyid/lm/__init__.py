"""Language-model phase (Pythia-160M / Pythia-410M, frozen). Requires ``torch`` and ``transformers``.

Gate: plan p.8 and syllabus p.04 — the theory gate (E0/E4) comes before any
language-model activation collection; ``polyid run`` for E8/E9 refuses to run
on real models unless ``theory_gate_passed: true`` is set in the config
(a deliberate, recorded decision rather than a default).
"""
