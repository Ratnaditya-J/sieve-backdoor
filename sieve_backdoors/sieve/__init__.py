"""Vendored SIEVE validity-audit machinery (see build-prompt §11).

This subpackage is a STANDALONE, ATTRIBUTED copy of the gate / control /
verdict logic from the author's SIEVE codebase (``sieve-audit``, source commit
``f9632ec0796d4ac2beb44fcce44874d608286c20``). Every module carries a
provenance header. Policy: **wrap, do not fork** — the logic here is not
edited; the backdoor harness builds on top of it from
``sieve_backdoors`` (attacks / detectors / grid), never by modifying these
files.

What the backdoor harness reuses:

* ``stats``        — stratified bootstrap AUROC + CIs, the population-separation
                     statistic behind every detection cell (build-prompt §9.2).
* ``config``       — the frozen strict profile + loosening-void asymmetry
                     ("resolve every ambiguity toward the weaker claim", §9).
* ``prereg``       — content-hash pre-registration (§8 / §12 admissibility).
* ``bundle``       — the evidence records (steering / ablation) the causal gate
                     is scored from; SIEVE never touches a model, only evidence.
* ``controls``     — matched-control sufficiency gate (random / orthogonal /
                     wrong-layer) reused for direction-localizing detectors
                     (D2 / D5) in the causal gate (§9.5).
* ``necessity``    — ablation (necessity) gate, matched ``ablate_random``.
* ``efficacy``     — "did the intervention take effect?" liveness gate.
* ``verdict``      — the five-state SIEVE verdict taxonomy + ``decide()`` used
                     as the causal-gate adjudicator underneath the harness's own
                     six-value cell verdict.
"""
from __future__ import annotations

from . import (  # noqa: F401
    baselines,
    bundle,
    config,
    controls,
    decodability,
    efficacy,
    necessity,
    prereg,
    stats,
    verdict,
)

__all__ = [
    "stats",
    "config",
    "bundle",
    "verdict",
    "controls",
    "necessity",
    "efficacy",
    "decodability",
    "baselines",
    "prereg",
]

# Provenance, surfaced programmatically so results can record it.
SIEVE_SOURCE_REPO = "sieve-audit"
SIEVE_SOURCE_COMMIT = "f9632ec0796d4ac2beb44fcce44874d608286c20"
SIEVE_VENDORED_DATE = "2026-07-01"
