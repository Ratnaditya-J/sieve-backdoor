# ---------------------------------------------------------------------------
# VENDORED from the SIEVE validity-audit codebase (sieve-audit).
#   source repo:   sieve-audit  (author's SIEVE / latent-horizon project)
#   source path:   src/sieve_audit/controls.py
#   source commit: f9632ec0796d4ac2beb44fcce44874d608286c20
#   vendored:      2026-07-01
# PROVENANCE: copied verbatim, then package-relative imports rewritten
#   (sieve_audit -> sieve_backdoors.sieve) so this repo is standalone.
# POLICY (build-prompt §11): WRAP, DO NOT FORK. The gate/control/verdict LOGIC
#   in this file is NOT edited; only import paths were adjusted. If a behavior
#   change is ever needed, wrap this module from sieve_backdoors, never edit here.
# ---------------------------------------------------------------------------
"""Stage 3: matched-control steering and two-judge behavioral scoring.

The causal-sufficiency question is never "did steering along the probe change
behavior?" but "did it change behavior *more than matched controls*" - a
random direction, an orthogonalized random direction, and the same direction
injected at a wrong layer (DESIGN.md section 4).

Hardened after adversarial review:

- The primary test points are ALL largest-|alpha| values, the grid must be
  sign-symmetric for a causal verdict, and probe significance is required at
  every primary point with Bonferroni-adjusted CIs - no picking the lucky
  alpha.
- Probe-vs-control comparisons use per-prompt effect *magnitudes*, paired by
  prompt (a control with large but mixed-sign effects cannot cancel to zero).
- The dose-response p-value comes from a within-prompt permutation test
  (per-record points are pseudo-replicated otherwise).
- Judge agreement: Spearman on continuous scores over all records, kappa on
  records where EVERY judge is outside the deadband, a minimum count of such
  informative records, and near-perfect agreement over many records is
  flagged as judge duplication - a protocol violation, not great evidence.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from itertools import combinations

import numpy as np
from scipy.stats import spearmanr

from .bundle import SteeringRecord
from .config import AuditConfig
from .stats import (
    CI,
    bootstrap_magnitude_diff,
    bootstrap_mean,
    cohen_kappa,
    dose_response_clustered,
)

PROBE_ARM = "probe"

# Degenerate-class-balance detection thresholds (diagnostic only; never gate a
# verdict). The "kappa paradox": two judges can agree at a very high raw rate yet
# score a degenerate (near-zero / NaN) Cohen's kappa when one outcome class
# dominates - the judged behavior is one-sided, near-absent OR near-universal - and
# a binarized rater is near-constant. That reflects the behavior distribution, not
# judge disagreement, and the card must say so instead of "agreement insufficient".
_FLOOR_MIN_RAW_AGREEMENT = 0.9
_FLOOR_MAX_MINORITY_RATE = 0.05


@dataclass
class JudgeResult:
    judges: list[str]
    min_pairwise_spearman: float    # continuous agreement, all records
    min_pairwise_kappa: float       # binarized agreement, informative records only
    n_informative: int              # records where every judge is outside the deadband
    agreement_ok: bool
    judges_agree_on_direction: bool
    suspected_duplicates: bool      # near-perfect agreement over many records
    notes: list[str] = field(default_factory=list)
    raw_agreement: float = float("nan")   # min pairwise raw binary agreement (informative)
    behavior_floor: bool = False          # judges agree but judged behavior is one-sided (kappa paradox)

    def to_dict(self) -> dict:
        return {
            "judges": self.judges,
            "min_pairwise_spearman": self.min_pairwise_spearman,
            "min_pairwise_kappa": self.min_pairwise_kappa,
            "n_informative": self.n_informative,
            "agreement_ok": self.agreement_ok,
            "judges_agree_on_direction": self.judges_agree_on_direction,
            "suspected_duplicates": self.suspected_duplicates,
            "raw_agreement": self.raw_agreement,
            "behavior_floor": self.behavior_floor,
            "notes": self.notes,
        }


@dataclass
class ControlsResult:
    arms: list[str]
    missing_controls: list[str]
    primary_alphas: list[float]                      # all largest-|alpha| points
    grid_symmetric: bool
    # arm -> {alpha: CI of mean per-prompt behavioral delta vs alpha=0}
    arm_effects: dict[str, dict[float, CI]]
    significant_probe_alphas: list[float]
    # alpha -> control arm -> CI of paired |probe effect| - |control effect|
    probe_vs_controls: dict[float, dict[str, CI]]
    dose_rho: float
    dose_p: float
    judge: JudgeResult
    probe_effect_significant: bool      # at EVERY primary alpha (Bonferroni)
    exceeds_all_controls: bool
    dose_response_ok: bool
    causally_sufficient: bool
    # conditions that invalidate the whole comparison (vs merely failing it)
    protocol_violations: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "arms": self.arms,
            "missing_controls": self.missing_controls,
            "primary_alphas": self.primary_alphas,
            "grid_symmetric": self.grid_symmetric,
            "arm_effects": {
                a: {str(al): ci.to_dict() for al, ci in d.items()}
                for a, d in self.arm_effects.items()
            },
            "significant_probe_alphas": self.significant_probe_alphas,
            "probe_vs_controls": {
                str(al): {c: ci.to_dict() for c, ci in d.items()}
                for al, d in self.probe_vs_controls.items()
            },
            "dose_rho": self.dose_rho,
            "dose_p": self.dose_p,
            "judge": self.judge.to_dict(),
            "probe_effect_significant": self.probe_effect_significant,
            "exceeds_all_controls": self.exceeds_all_controls,
            "dose_response_ok": self.dose_response_ok,
            "causally_sufficient": self.causally_sufficient,
            "protocol_violations": self.protocol_violations,
            "notes": self.notes,
        }


def _mean_judge_score(r: SteeringRecord, judges: list[str]) -> float:
    return float(np.mean([r.judge_scores[j] for j in judges]))


def _paired_deltas(
    records: list[SteeringRecord], judges: list[str] | None = None
) -> dict[str, dict[float, dict[str, float]]]:
    """Per-arm, per-alpha {prompt_id: behavioral delta vs alpha=0, same prompt}.

    ``judges=None`` averages all judges; passing a single judge name computes
    that judge's view (used for the direction-agreement check).
    """
    by_arm: dict[str, dict[float, dict[str, float]]] = defaultdict(lambda: defaultdict(dict))
    all_judges = sorted({j for r in records for j in r.judge_scores})
    use = judges if judges is not None else all_judges
    for r in records:
        by_arm[r.arm][r.alpha][r.prompt_id] = _mean_judge_score(r, use)

    out: dict[str, dict[float, dict[str, float]]] = {}
    for arm, by_alpha in by_arm.items():
        base = by_alpha.get(0.0)
        if base is None:
            continue
        out[arm] = {}
        for alpha, scores in by_alpha.items():
            if alpha == 0.0:
                continue
            shared = sorted(set(scores) & set(base))
            out[arm][alpha] = {p: scores[p] - base[p] for p in shared}
    return out


def _judge_agreement(
    records: list[SteeringRecord],
    deltas_by_judge: dict[str, dict[str, dict[float, dict[str, float]]]],
    primary_alphas: list[float],
    cfg: AuditConfig,
) -> JudgeResult:
    judges = sorted({j for r in records for j in r.judge_scores})
    notes: list[str] = []
    if len(judges) < cfg.min_judges:
        return JudgeResult(
            judges=judges,
            min_pairwise_spearman=float("nan"),
            min_pairwise_kappa=float("nan"),
            n_informative=0,
            agreement_ok=False,
            judges_agree_on_direction=False,
            suspected_duplicates=False,
            notes=[f"only {len(judges)} judge(s); protocol requires >= {cfg.min_judges}"],
        )

    # continuous agreement over every judged generation (all arms), plus the
    # per-record absolute gap (to tell genuine high correlation from literal
    # duplication)
    spearmans = []
    min_abs_gap = float("inf")
    for a, b in combinations(judges, 2):
        sa = np.array([r.judge_scores[a] for r in records])
        sb = np.array([r.judge_scores[b] for r in records])
        rho = spearmanr(sa, sb).statistic
        spearmans.append(0.0 if np.isnan(rho) else float(rho))
        min_abs_gap = min(min_abs_gap, float(np.median(np.abs(sa - sb))))
    min_spearman = float(min(spearmans))
    max_spearman = float(max(spearmans))

    # Duplication, not excellence: a pair that is BOTH near-perfectly ranked
    # AND near-identical per record. Two genuinely independent judges can
    # correlate highly on an easy metric (high spearman) while still differing
    # per record (gap above eps), so both conditions are required - this keeps
    # a real two-judge run from being voided for agreeing well.
    suspected_duplicates = (
        len(records) >= cfg.duplicate_judge_min_n
        and max_spearman > cfg.max_judge_spearman
        and min_abs_gap < cfg.judge_identical_eps
    )
    if suspected_duplicates:
        notes.append(
            f"a judge pair is near-identical (median |Δ|={min_abs_gap:.4f} < "
            f"{cfg.judge_identical_eps}, spearman {max_spearman:.4f} > "
            f"{cfg.max_judge_spearman}) over {len(records)} records: suspected "
            "duplicate judges; independent judges differ per record"
        )

    # binarized agreement only where EVERY judge committed (outside deadband);
    # at the threshold, binarized (dis)agreement is pure noise
    thr, band = cfg.judge_binarize_threshold, cfg.judge_deadband
    informative = [
        r
        for r in records
        if all(abs(s - thr) > band for s in r.judge_scores.values())
    ]
    if len(informative) >= cfg.min_informative_judged:
        kappas = []
        for a, b in combinations(judges, 2):
            ra = np.array([r.judge_scores[a] > thr for r in informative])
            rb = np.array([r.judge_scores[b] > thr for r in informative])
            kappas.append(cohen_kappa(ra, rb))
        min_kappa = float(np.nanmin(kappas)) if not all(np.isnan(kappas)) else float("nan")
        # constant raters (kappa NaN) carry no reliability evidence
        kappa_ok = not np.isnan(min_kappa) and min_kappa >= cfg.min_judge_kappa
    else:
        min_kappa = float("nan")
        kappa_ok = False
        notes.append(
            f"only {len(informative)} records outside the judge deadband "
            f"(< {cfg.min_informative_judged}): binarized agreement not assessable"
        )

    # Degenerate-class-balance detection ("kappa paradox"): when the judges agree at a
    # high raw rate but kappa is degenerate because one outcome class dominates (the
    # judged behavior is one-sided, near-absent or near-universal, so a binarized rater
    # is near-constant), kappa cannot assess reliability. Report this as one-sided
    # judged behavior, NOT judge disagreement. Diagnostic only: agreement_ok below is
    # left untouched, so the gate stays exactly as conservative.
    raw_agreement = float("nan")
    behavior_floor = False
    if len(informative) >= cfg.min_informative_judged:
        raw_pairs = [
            float(np.mean(
                np.array([r.judge_scores[a] > thr for r in informative])
                == np.array([r.judge_scores[b] > thr for r in informative])
            ))
            for a, b in combinations(judges, 2)
        ]
        raw_agreement = float(min(raw_pairs)) if raw_pairs else float("nan")
        pos_rates = [
            float(np.mean([r.judge_scores[j] > thr for r in informative]))
            for j in judges
        ]
        min_minority_rate = min(min(p, 1.0 - p) for p in pos_rates)
        mean_pos = float(np.mean(pos_rates))
        behavior_floor = (
            not kappa_ok
            and raw_agreement >= _FLOOR_MIN_RAW_AGREEMENT
            and min_minority_rate < _FLOOR_MAX_MINORITY_RATE
        )
        if behavior_floor:
            notes.append(
                f"one-sided judged behavior, not judge disagreement: judges agree at "
                f"{raw_agreement:.0%} raw on {len(informative)} records, but one outcome "
                f"class dominates (judged positive rate {mean_pos:.0%}); Cohen's kappa is "
                f"degenerate under this class imbalance and cannot assess agreement. There "
                f"is little behavioral variation for the causal axes to exploit; this "
                f"reflects the behavior distribution, not unreliable judges."
            )

    agreement_ok = (
        (min_spearman >= cfg.min_judge_spearman)
        and kappa_ok
        and not suspected_duplicates
    )
    if not agreement_ok and not notes:
        notes.append(
            f"judge agreement failed (min spearman {min_spearman:.2f} "
            f"vs >= {cfg.min_judge_spearman}; min kappa {min_kappa:.2f} on "
            f"{len(informative)} informative records vs >= {cfg.min_judge_kappa})"
        )

    # every judge must see the probe effect point the same way at primary alphas
    direction_ok = True
    for alpha in primary_alphas:
        signs = []
        for j in judges:
            d = deltas_by_judge[j].get(PROBE_ARM, {}).get(alpha)
            if d:
                m = float(np.mean(list(d.values())))
                if abs(m) > 1e-12:
                    signs.append(np.sign(m))
        if len(set(signs)) > 1:
            direction_ok = False
            notes.append(f"judges disagree on probe effect direction at alpha={alpha:g}")

    return JudgeResult(
        judges=judges,
        min_pairwise_spearman=min_spearman,
        min_pairwise_kappa=min_kappa,
        n_informative=len(informative),
        agreement_ok=agreement_ok,
        judges_agree_on_direction=direction_ok,
        suspected_duplicates=suspected_duplicates,
        raw_agreement=raw_agreement,
        behavior_floor=behavior_floor,
        notes=notes,
    )


def run_controls(records: list[SteeringRecord], cfg: AuditConfig) -> ControlsResult:
    if not records:
        raise ValueError("no steering records")
    rng = np.random.default_rng(cfg.seed)
    notes: list[str] = []
    violations: list[str] = []

    arms = sorted({r.arm for r in records})
    # Auto-detect multi-draw null: any random_N arms beyond the canonical `random`
    extra_random = sorted(a for a in arms if a.startswith("random_"))
    effective_controls = list(cfg.required_controls) + extra_random
    missing = [c for c in effective_controls if c not in arms]
    if PROBE_ARM not in arms:
        raise ValueError("steering records contain no 'probe' arm")

    # Multi-draw null count check: count all random draws present (canonical + extras)
    all_random_arms = [a for a in arms if a == "random" or a.startswith("random_")]
    n_random_total = len(all_random_arms)
    if n_random_total < cfg.min_random_controls:
        violations.append(
            f"multi-draw null requires {cfg.min_random_controls} random control draws "
            f"but only {n_random_total} present "
            f"({', '.join(all_random_arms) if all_random_arms else 'none'}); "
            "supply random_1, random_2, ... arms or lower min_random_controls"
        )

    deltas = _paired_deltas(records)
    if PROBE_ARM not in deltas:
        raise ValueError("probe arm has no alpha=0 records: deltas cannot be paired")
    judges = sorted({j for r in records for j in r.judge_scores})
    deltas_by_judge = {j: _paired_deltas(records, judges=[j]) for j in judges}

    probe_alphas = sorted(deltas[PROBE_ARM], key=abs)
    if not probe_alphas:
        raise ValueError("probe arm has no nonzero-alpha records paired with alpha=0")
    max_abs = abs(probe_alphas[-1])
    primary = sorted(a for a in deltas[PROBE_ARM] if abs(a) == max_abs)
    grid_symmetric = {-max_abs, max_abs} <= set(deltas[PROBE_ARM])
    if cfg.require_symmetric_grid and not grid_symmetric:
        violations.append(
            f"alpha grid is not sign-symmetric at |alpha|={max_abs:g}; a "
            "one-sided grid halves the evidence and invites cherry-picking"
        )

    # per-arm effects with Bonferroni-adjusted CIs at the primary points
    adj_level = 1.0 - (1.0 - cfg.ci_level) / max(len(primary), 1)
    arm_effects: dict[str, dict[float, CI]] = {}
    for arm, by_alpha in deltas.items():
        arm_effects[arm] = {}
        for alpha, d in sorted(by_alpha.items()):
            if len(d) < cfg.min_steered_prompts:
                notes.append(
                    f"{arm}@alpha={alpha:g}: only {len(d)} paired prompts "
                    f"(< {cfg.min_steered_prompts})"
                )
            level = adj_level if (arm == PROBE_ARM and alpha in primary) else cfg.ci_level
            arm_effects[arm][alpha] = bootstrap_mean(
                np.array(list(d.values())), rng, cfg.n_boot, level
            )

    underpowered = any("paired prompts" in n for n in notes)

    # probe effect must be significant at EVERY primary point
    significant = [a for a in primary if arm_effects[PROBE_ARM][a].excludes(0.0)]
    probe_sig = len(significant) == len(primary) and bool(primary)

    if extra_random:
        notes.append(
            f"multi-draw null: {len(extra_random) + 1} random control draws present "
            f"({', '.join(['random'] + extra_random)}); probe must beat ALL of them"
        )

    # probe vs each control: paired magnitude comparison at every primary point
    probe_vs_controls: dict[float, dict[str, CI]] = {}
    exceeds_all = bool(primary) and not missing and bool(effective_controls)
    for alpha in primary:
        probe_vs_controls[alpha] = {}
        for control in effective_controls:
            d_control = deltas.get(control, {}).get(alpha)
            if not d_control:
                notes.append(f"control '{control}' missing at alpha={alpha:g}")
                exceeds_all = False
                continue
            diff = bootstrap_magnitude_diff(
                deltas[PROBE_ARM][alpha], d_control, rng, cfg.n_boot, cfg.ci_level
            )
            probe_vs_controls[alpha][control] = diff
            if not diff.lo > 0:
                exceeds_all = False

    # dose-response over the full grid, within-prompt permutation p
    dose_rho, dose_p = dose_response_clustered(
        deltas[PROBE_ARM], rng, n_perm=cfg.n_perm
    )
    dose_ok = abs(dose_rho) >= cfg.dose_response_min_rho and dose_p <= cfg.dose_response_max_p
    if len(deltas[PROBE_ARM]) < 2:
        dose_ok = False
        notes.append("fewer than 2 nonzero alphas: dose-response untestable")

    judge = _judge_agreement(records, deltas_by_judge, primary, cfg)
    if judge.suspected_duplicates:
        violations.append("judges are near-duplicates (see judge notes)")

    causal = (
        probe_sig
        and exceeds_all
        and dose_ok
        and judge.agreement_ok
        and judge.judges_agree_on_direction
        and not missing
        and not underpowered
        and not violations
    )

    return ControlsResult(
        arms=arms,
        missing_controls=missing,
        primary_alphas=primary,
        grid_symmetric=grid_symmetric,
        arm_effects=arm_effects,
        significant_probe_alphas=significant,
        probe_vs_controls=probe_vs_controls,
        dose_rho=dose_rho,
        dose_p=dose_p,
        judge=judge,
        probe_effect_significant=probe_sig,
        exceeds_all_controls=exceeds_all,
        dose_response_ok=dose_ok,
        causally_sufficient=causal,
        protocol_violations=violations,
        notes=notes,
    )
