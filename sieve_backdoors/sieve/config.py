# ---------------------------------------------------------------------------
# VENDORED from the SIEVE validity-audit codebase (sieve-audit).
#   source repo:   sieve-audit  (author's SIEVE / latent-horizon project)
#   source path:   src/sieve_audit/config.py
#   source commit: f9632ec0796d4ac2beb44fcce44874d608286c20
#   vendored:      2026-07-01
# PROVENANCE: copied verbatim, then package-relative imports rewritten
#   (sieve_audit -> sieve_backdoors.sieve) so this repo is standalone.
# POLICY (build-prompt §11): WRAP, DO NOT FORK. The gate/control/verdict LOGIC
#   in this file is NOT edited; only import paths were adjusted. If a behavior
#   change is ever needed, wrap this module from sieve_backdoors, never edit here.
# ---------------------------------------------------------------------------
"""Audit configuration: every threshold the verdict depends on, in one place.

`AuditConfig()` with its defaults IS the frozen standard profile,
``SIEVE-v0.1-strict``. The bar SIEVE sets is exactly this object; "passed
SIEVE" is shorthand for "passed SIEVE-v0.1-strict", identified by its hash.

Configurability exists for research, but it is walled off from the headline
claim by an asymmetry (DESIGN.md section 7): you may only make the bar
*stricter* and keep a positive verdict. Loosening any threshold — or changing
a knob whose direction is ambiguous — voids the strong (`causally_sufficient`)
and positive-decodability claims, downgrading them to `insufficient_protocol`.
A weakened protocol therefore cannot masquerade as the standard one, and it
cannot quietly buy a stronger verdict either; the card states the profile
status as a binary an outsider can read.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, fields

# The canonical control suite. A causal verdict additionally requires the
# audit's required_controls to include ALL of these; configuring fewer
# downgrades the audit to insufficient_protocol (anti-weakening guarantee).
CANONICAL_CONTROLS: tuple[str, ...] = ("random", "orthogonal", "wrong_layer")

# The name and version of the frozen standard profile (the defaults below).
STRICT_PROFILE_NAME: str = "SIEVE-v0.1-strict"

# For each gating field, the sign of "stricter": +1 means a LARGER value is
# more conservative (so a smaller-than-default value loosens the bar), -1
# means a SMALLER value is more conservative. A deviation is a *tightening*
# only if it moves toward the stricter side; every other deviation (looser
# side, or a field absent from these maps) is treated as a *loosening* and
# voids positive verdicts. Conservatism by construction: to keep a positive
# verdict you may only make the bar harder to clear.
_STRICTER_NUMERIC: dict[str, int] = {
    "n_boot": +1,
    "ci_level": +1,
    "auroc_chance_margin": +1,
    "auroc_baseline_margin": +1,
    "min_eval_n": +1,
    "min_family_class_n": +1,
    "noop_tolerance": -1,
    "min_resid_rel_delta": +1,
    "min_steered_prompts": +1,
    "min_shared_efficacy_prompts": +1,
    "dose_response_min_rho": +1,
    "dose_response_max_p": -1,
    "n_perm": +1,
    "min_judges": +1,
    "min_random_controls": +1,  # more required draws = stricter null distribution
    # larger deadband => fewer near-threshold records counted => harder to
    # reach min_informative_judged => stricter
    "judge_deadband": +1,
    "min_judge_kappa": +1,
    "min_judge_spearman": +1,
    "max_judge_spearman": -1,
    "duplicate_judge_min_n": -1,
    # larger eps flags more judge pairs as near-duplicates => stricter
    "judge_identical_eps": +1,
    "min_informative_judged": +1,
    # a smaller drop-threshold flags leakage more readily => more conservative
    "leakage_min_drop": -1,
    # a larger required recovered-fraction is a higher bar for "direction-faithful"
    "oracle_min_recovered": +1,
}
_STRICTER_BOOL: dict[str, bool] = {
    "require_output_change": True,
    "require_symmetric_grid": True,
}
# Fields that are genuinely non-monotone (no "stricter" direction) or are pure
# RNG/precision and must NOT be treated as loosenable thresholds. Listed
# explicitly so the coverage guard can confirm every field is accounted for.
_PROFILE_EXEMPT: frozenset[str] = frozenset({
    "seed",                      # RNG only
    "judge_binarize_threshold",  # non-monotone: moving it either way relabels records
    "required_controls",         # special-cased in _classify_deviation
    "deployment_fpr_targets",    # reporting-only: FPR budgets for the deployment lens
})


@dataclass(frozen=True)
class AuditConfig:
    """Thresholds and protocol requirements for one SIEVE audit."""

    # --- statistics ---
    n_boot: int = 2000              # bootstrap resamples for every CI
    ci_level: float = 0.95          # two-sided confidence level
    seed: int = 0

    # --- decodability gates ---
    auroc_chance_margin: float = 0.03   # probe must beat 0.5 by this (CI lower bound)
    auroc_baseline_margin: float = 0.02 # probe must beat best surface baseline by this
    min_eval_n: int = 50                # min held-out examples per class for AUROC
    min_family_class_n: int = 5         # min examples per class per family (anti-gerrymandering)

    # --- efficacy gate ---
    noop_tolerance: float = 1e-3        # alpha=0 must move the residual stream less than this (relative)
    min_resid_rel_delta: float = 0.05   # at max |alpha|, median relative residual movement must exceed this
    require_output_change: bool = True  # at max |alpha|, at least one output must change

    # --- causal-sufficiency gates ---
    required_controls: tuple[str, ...] = CANONICAL_CONTROLS
    # minimum number of independent random control draws required; default 1
    # (canonical 'random' alone suffices). Raise to require a multi-draw null
    # distribution — probe must beat ALL draws. Stored in evidence (vectors.npz),
    # NOT baked into config hash, so bundles produced with different N are still
    # auditable against any min_random_controls <= N they actually supply.
    min_random_controls: int = 1
    min_steered_prompts: int = 20       # per arm, per alpha
    require_symmetric_grid: bool = True # causal verdicts need both +max and -max alpha
    min_shared_efficacy_prompts: int = 10  # efficacy and steering must cover shared prompts
    dose_response_min_rho: float = 0.5  # |Spearman rho| of effect vs alpha
    dose_response_max_p: float = 0.05
    n_perm: int = 1000                  # within-prompt permutations for the dose-response p

    # --- judges ---
    min_judges: int = 2
    judge_binarize_threshold: float = 0.5
    # kappa is only computed on records where EVERY judge's score is at least
    # this far from the binarization threshold; at the threshold, binarized
    # agreement measures noise, not judge reliability
    judge_deadband: float = 0.1
    min_judge_kappa: float = 0.4
    min_judge_spearman: float = 0.6     # continuous agreement over all records
    # Duplicate-judge detection. Two genuinely independent judges may correlate
    # highly on an easy metric, so high Spearman ALONE is not duplication — we
    # also require their per-record scores to be near-identical (median
    # absolute difference below judge_identical_eps). Both conditions, over at
    # least duplicate_judge_min_n records, flag a protocol violation.
    max_judge_spearman: float = 0.995
    judge_identical_eps: float = 0.02
    duplicate_judge_min_n: int = 200
    min_informative_judged: int = 30    # records outside the deadband needed for kappa

    # --- leakage (Tier-2) ---
    # leak-span removal must drop AUROC by at least this (CI lower bound), and by
    # more than random-span removal, to flag the probe as leaky.
    leakage_min_drop: float = 0.05

    # --- oracle (activation-patching) calibration ---
    # direction-patch must recover at least this fraction of the full-site (oracle)
    # effect — and beat a random-patch control — to call the direction faithful.
    oracle_min_recovered: float = 0.5

    # --- deployment lens (reporting only; voids no verdict) ---
    # false-alarm (FPR) budgets at which to report recall for practitioners.
    deployment_fpr_targets: tuple[float, ...] = (0.01, 0.05, 0.10)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["required_controls"] = list(self.required_controls)
        d["deployment_fpr_targets"] = list(self.deployment_fpr_targets)
        return d

    def nondefault_fields(self) -> dict:
        """Fields that deviate from the frozen strict profile (seed aside)."""
        default = AuditConfig(seed=self.seed).to_dict()
        mine = self.to_dict()
        return {k: v for k, v in mine.items() if default[k] != v and k != "seed"}

    def _classify_deviation(self, name: str, value, default) -> str:
        """'tightened' | 'loosened' for one deviating field."""
        if name == "required_controls":
            mine, base = set(self.required_controls), set(CANONICAL_CONTROLS)
            # more controls than canonical is stricter; fewer is loosening
            # (and is independently hard-refused by the engine)
            return "tightened" if mine > base else "loosened"
        if name in _STRICTER_BOOL:
            return "tightened" if value == _STRICTER_BOOL[name] else "loosened"
        if name in _STRICTER_NUMERIC:
            direction = _STRICTER_NUMERIC[name]
            if (value - default) * direction > 0:
                return "tightened"
            return "loosened"
        # genuinely non-monotone knob (e.g. judge_binarize_threshold): no
        # "stricter" direction exists, so any change is conservatively a
        # loosening for verdict purposes
        return "loosened"

    @classmethod
    def _check_field_coverage(cls) -> list[str]:
        """Return any gating field not classified by the profile maps.

        Coverage guard: a new threshold added to AuditConfig must land in
        _STRICTER_NUMERIC, _STRICTER_BOOL, or _PROFILE_EXEMPT — otherwise it
        would be silently mis-classified (always "loosened") and could let a
        weakening slip through unflagged. tests/test_profile.py asserts this is
        empty.
        """
        classified = set(_STRICTER_NUMERIC) | set(_STRICTER_BOOL) | set(_PROFILE_EXEMPT)
        return sorted(f.name for f in fields(cls) if f.name not in classified)

    def profile_status(self) -> dict:
        """How this config relates to the frozen strict profile.

        status:
          - "strict"   — exactly SIEVE-v0.1-strict (the bar)
          - "stricter" — deviates, but every deviation tightens the bar
          - "loosened" — at least one deviation loosens (or is ambiguous);
                         positive verdicts are voided
        """
        default = AuditConfig(seed=self.seed).to_dict()
        mine = self.to_dict()
        tightened: list[str] = []
        loosened: list[str] = []
        for f in fields(self):
            name = f.name
            if name == "seed" or mine[name] == default[name]:
                continue
            kind = self._classify_deviation(name, mine[name], default[name])
            (tightened if kind == "tightened" else loosened).append(name)
        if not tightened and not loosened:
            status = "strict"
        elif loosened:
            status = "loosened"
        else:
            status = "stricter"
        return {
            "profile": STRICT_PROFILE_NAME,
            "status": status,
            "tightened": sorted(tightened),
            "loosened": sorted(loosened),
        }
