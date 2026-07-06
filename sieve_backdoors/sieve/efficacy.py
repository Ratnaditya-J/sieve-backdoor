# ---------------------------------------------------------------------------
# VENDORED from the SIEVE validity-audit codebase (sieve-audit).
#   source repo:   sieve-audit  (author's SIEVE / latent-horizon project)
#   source path:   src/sieve_audit/efficacy.py
#   source commit: f9632ec0796d4ac2beb44fcce44874d608286c20
#   vendored:      2026-07-01
# PROVENANCE: copied verbatim, then package-relative imports rewritten
#   (sieve_audit -> sieve_backdoors.sieve) so this repo is standalone.
# POLICY (build-prompt §11): WRAP, DO NOT FORK. The gate/control/verdict LOGIC
#   in this file is NOT edited; only import paths were adjusted. If a behavior
#   change is ever needed, wrap this module from sieve_backdoors, never edit here.
# ---------------------------------------------------------------------------
"""Stage 2: the efficacy gate - did the intervention actually take effect?

Before any causal verdict, SIEVE verifies (DESIGN.md section 5):

1. hook correctness: alpha=0 is a no-op, and the residual-stream movement at
   alpha != 0 matches the expected |alpha|*||w|| within tolerance;
2. efficacy: at the largest tested |alpha|, the residual stream moved by a
   non-trivial relative amount and at least one output changed.

A failed gate yields ``intervention_ineffective`` - *inconclusive*, never a
causal null. This is the anti-gaming defense: "we steered (where steering
cannot bite) and nothing happened, therefore safe" is not a claim SIEVE will
ever endorse.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .bundle import EfficacyRecord
from .config import AuditConfig

# Residual movement may deviate from |alpha|*||w|| (quantization, normalization
# layers downstream of the hook point); we flag >50% deviation as a hook bug.
_HOOK_NORM_RTOL = 0.5


@dataclass
class EfficacyResult:
    hook_correct: bool
    noop_ok: bool                   # alpha=0 left the residual stream alone
    max_alpha: float
    median_rel_delta_at_max: float  # median ||delta|| / ||h_base|| at max |alpha|
    any_output_changed_at_max: bool
    effective: bool
    # A real, non-degenerate perturbation was applied at the correct magnitude
    # (|alpha|*||w|| > 0 and the residual moved to match). This is the liveness
    # bar for CONTROL arms: it catches a zero-norm or dead-hook "control"
    # without demanding the probe-arm behavioral thresholds (output change,
    # relative-movement floor), which legitimately vary by layer - e.g. a
    # wrong-layer arm injected where the residual norm is much larger.
    injection_verified: bool = False
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "hook_correct": self.hook_correct,
            "noop_ok": self.noop_ok,
            "max_alpha": self.max_alpha,
            "median_rel_delta_at_max": self.median_rel_delta_at_max,
            "any_output_changed_at_max": self.any_output_changed_at_max,
            "effective": self.effective,
            "injection_verified": self.injection_verified,
            "notes": self.notes,
        }


def run_efficacy(
    records: list[EfficacyRecord], cfg: AuditConfig, arm: str = "probe"
) -> EfficacyResult:
    records = [r for r in records if r.arm == arm]
    if not records:
        raise ValueError(
            f"no efficacy records for arm {arm!r}: the gate cannot pass by omission"
        )

    notes: list[str] = []
    alphas = np.array([r.alpha for r in records])
    max_abs_alpha = float(np.max(np.abs(alphas)))
    if max_abs_alpha == 0.0:
        raise ValueError("efficacy records contain only alpha=0; nothing was steered")

    # --- hook correctness: alpha=0 is a no-op ---
    zero = [r for r in records if r.alpha == 0.0]
    noop_ok = True
    if not zero:
        noop_ok = False
        notes.append("no alpha=0 records: no-op check could not run")
    else:
        rel = np.array(
            [r.resid_delta_norm / max(r.resid_base_norm, 1e-12) for r in zero]
        )
        noop_ok = bool(np.max(rel) < cfg.noop_tolerance)
        if not noop_ok:
            notes.append(
                f"alpha=0 moved the residual stream (max relative delta "
                f"{np.max(rel):.2e} >= {cfg.noop_tolerance:.0e}): hook bug"
            )

    # --- hook correctness: movement tracks |alpha|*||w|| ---
    nonzero = [r for r in records if r.alpha != 0.0 and r.expected_delta_norm > 0]
    norms_ok = True
    if nonzero:
        ratios = np.array([r.resid_delta_norm / r.expected_delta_norm for r in nonzero])
        norms_ok = bool(
            np.median(np.abs(ratios - 1.0)) <= _HOOK_NORM_RTOL
        )
        if not norms_ok:
            notes.append(
                "residual movement deviates from |alpha|*||w|| by >50% (median); "
                "suspected hook or quantization fault"
            )
    hook_correct = noop_ok and norms_ok

    # --- efficacy at max |alpha| ---
    at_max = [r for r in records if abs(r.alpha) == max_abs_alpha]
    rel_deltas = np.array(
        [r.resid_delta_norm / max(r.resid_base_norm, 1e-12) for r in at_max]
    )
    median_rel = float(np.median(rel_deltas))
    moved = median_rel >= cfg.min_resid_rel_delta
    any_changed = any(r.output_changed for r in at_max)
    if not moved:
        notes.append(
            f"median relative residual movement at |alpha|={max_abs_alpha:g} is "
            f"{median_rel:.4f} < {cfg.min_resid_rel_delta} (the gpt-oss L34 failure mode)"
        )
    if cfg.require_output_change and not any_changed:
        notes.append("no output changed at the largest tested alpha")

    effective = hook_correct and moved and (any_changed or not cfg.require_output_change)

    # Liveness for control arms: a real, correctly-applied, nonzero injection.
    # max expected delta > 0 rules out a zero-norm "control"; hook_correct
    # confirms the residual actually moved to match |alpha|*||w|| (so a dead
    # hook that swallows the injection also fails). Crucially this does NOT
    # require the relative-movement floor against THIS arm's base norm, which
    # would spuriously fail a wrong-layer arm at a high-norm layer.
    max_expected = max((r.expected_delta_norm for r in records), default=0.0)
    injection_verified = hook_correct and max_expected > 0.0

    return EfficacyResult(
        hook_correct=hook_correct,
        noop_ok=noop_ok,
        max_alpha=max_abs_alpha,
        median_rel_delta_at_max=median_rel,
        any_output_changed_at_max=any_changed,
        effective=effective,
        injection_verified=injection_verified,
        notes=notes,
    )


def run_efficacy_all_arms(
    records: list[EfficacyRecord], cfg: AuditConfig
) -> dict[str, EfficacyResult]:
    """Efficacy per steering arm.

    The probe arm's result is the efficacy gate proper. Control arms are
    checked too: a "control" that never moved the residual stream is
    degenerate (zero-norm vector, dead hook) and would make any probe look
    causally superior - so causal verdicts require every control arm to have
    passed its own movement check.
    """
    arms = sorted({r.arm for r in records})
    return {arm: run_efficacy(records, cfg, arm) for arm in arms}
