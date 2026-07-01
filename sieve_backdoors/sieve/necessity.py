# ---------------------------------------------------------------------------
# VENDORED from the SIEVE validity-audit codebase (sieve-audit).
#   source repo:   sieve-audit  (author's SIEVE / latent-horizon project)
#   source path:   src/sieve_audit/necessity.py
#   source commit: f9632ec0796d4ac2beb44fcce44874d608286c20
#   vendored:      2026-07-01
# PROVENANCE: copied verbatim, then package-relative imports rewritten
#   (sieve_audit -> sieve_backdoors.sieve) so this repo is standalone.
# POLICY (build-prompt §11): WRAP, DO NOT FORK. The gate/control/verdict LOGIC
#   in this file is NOT edited; only import paths were adjusted. If a behavior
#   change is ever needed, wrap this module from sieve_backdoors, never edit here.
# ---------------------------------------------------------------------------
"""Stage 4 (optional): the necessity gate — does *removing* the direction matter?

Steering tests **sufficiency** (does adding the direction induce the behavior?).
Ablation tests **necessity** (does removing it take the behavior away?). They have
different blind spots: a direction can be necessary via a distributed mechanism
that single-layer additive steering cannot induce — exactly the false-negative a
steering-only verdict risks. Running both shrinks that gap.

The question is never "did ablating the probe direction change behavior?" but
"did it change behavior *more than ablating a matched random direction*" — the
``ablate_random`` control. Without it, "behavior dropped after ablation" is
confounded by the generic effect of perturbing the forward pass.

Anti-gaming asymmetry (mirrors the efficacy/controls gates): a missing control,
too few judges, or too few paired prompts yields ``inconclusive`` — never a free
``necessary``, and never a definitive ``not necessary`` the evidence did not earn.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np

from .bundle import AblationRecord
from .config import AuditConfig
from .stats import CI, bootstrap_mean

BASELINE_ARM = "baseline"
PROBE_ARM = "probe"
RANDOM_ARM = "ablate_random"


@dataclass
class NecessityResult:
    arms: list[str]
    has_baseline: bool
    has_random_control: bool
    n_paired: int
    judges: list[str]
    probe_drop: CI | None            # mean(baseline - probe_ablated), paired by prompt
    random_drop: CI | None           # mean(baseline - random_ablated), paired by prompt
    probe_vs_random_drop: CI | None  # mean((baseline-probe) - (baseline-random)), paired
    judges_agree_direction: bool
    necessary: bool
    inconclusive: bool               # protocol incomplete: cannot adjudicate necessity
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "arms": self.arms,
            "has_baseline": self.has_baseline,
            "has_random_control": self.has_random_control,
            "n_paired": self.n_paired,
            "judges": self.judges,
            "probe_drop": self.probe_drop.to_dict() if self.probe_drop else None,
            "random_drop": self.random_drop.to_dict() if self.random_drop else None,
            "probe_vs_random_drop": (
                self.probe_vs_random_drop.to_dict()
                if self.probe_vs_random_drop
                else None
            ),
            "judges_agree_direction": self.judges_agree_direction,
            "necessary": self.necessary,
            "inconclusive": self.inconclusive,
            "notes": self.notes,
        }


def _by_arm_prompt(
    records: list[AblationRecord], judges: list[str]
) -> dict[str, dict[str, float]]:
    """{arm: {prompt_id: mean judge score}} averaged over the given judges."""
    out: dict[str, dict[str, float]] = defaultdict(dict)
    for r in records:
        out[r.arm][r.prompt_id] = float(np.mean([r.judge_scores[j] for j in judges]))
    return out


def run_necessity(records: list[AblationRecord], cfg: AuditConfig) -> NecessityResult:
    if not records:
        raise ValueError("no ablation records")
    rng = np.random.default_rng(cfg.seed)
    arms = sorted({r.arm for r in records})
    judges = sorted({j for r in records for j in r.judge_scores})
    notes: list[str] = []

    has_baseline = BASELINE_ARM in arms
    has_probe = PROBE_ARM in arms
    has_random = RANDOM_ARM in arms

    def _inconclusive(reason: str) -> NecessityResult:
        notes.append(reason)
        return NecessityResult(
            arms=arms,
            has_baseline=has_baseline,
            has_random_control=has_random,
            n_paired=0,
            judges=judges,
            probe_drop=None,
            random_drop=None,
            probe_vs_random_drop=None,
            judges_agree_direction=False,
            necessary=False,
            inconclusive=True,
            notes=notes,
        )

    if not (has_baseline and has_probe):
        return _inconclusive(
            "necessity needs both a 'baseline' and a 'probe' ablation arm"
        )
    if not has_random:
        return _inconclusive(
            "no 'ablate_random' control: a behavioral drop after ablation cannot "
            "be distinguished from the generic effect of perturbing the forward pass"
        )
    if len(judges) < cfg.min_judges:
        return _inconclusive(
            f"only {len(judges)} judge(s); necessity requires >= {cfg.min_judges}"
        )

    per = _by_arm_prompt(records, judges)
    base, probe, rand = per[BASELINE_ARM], per[PROBE_ARM], per[RANDOM_ARM]
    shared = sorted(set(base) & set(probe) & set(rand))
    if len(shared) < cfg.min_steered_prompts:
        return _inconclusive(
            f"only {len(shared)} prompts shared across baseline/probe/ablate_random "
            f"(< {cfg.min_steered_prompts}): necessity underpowered"
        )

    probe_drops = np.array([base[p] - probe[p] for p in shared])
    rand_drops = np.array([base[p] - rand[p] for p in shared])
    paired_excess = np.array(
        [(base[p] - probe[p]) - (base[p] - rand[p]) for p in shared]
    )
    probe_drop = bootstrap_mean(probe_drops, rng, cfg.n_boot, cfg.ci_level)
    probe_vs_random = bootstrap_mean(paired_excess, rng, cfg.n_boot, cfg.ci_level)
    # raw random-ablation drop (display only): computed AFTER probe_vs_random so the
    # rng-consumption order for the gating CIs is unchanged and verdicts stay identical.
    random_drop = bootstrap_mean(rand_drops, rng, cfg.n_boot, cfg.ci_level)

    # every judge must see the probe-ablation drop go the same (positive) way
    direction_ok = True
    for j in judges:
        per_j = _by_arm_prompt(records, [j])
        bj, pj = per_j[BASELINE_ARM], per_j[PROBE_ARM]
        sj = sorted(set(bj) & set(pj))
        if sj and float(np.mean([bj[p] - pj[p] for p in sj])) <= 0:
            direction_ok = False
            notes.append(f"judge {j!r} does not see a positive probe-ablation drop")

    necessary = (
        probe_drop.lo > 0          # removing the direction reduces the behavior
        and probe_vs_random.lo > 0  # ... more than removing a matched random direction
        and direction_ok
    )
    if not necessary and not notes:
        notes.append(
            "probe-ablation drop does not exceed the ablate_random control: "
            "the direction is not necessary under single-direction ablation"
        )

    return NecessityResult(
        arms=arms,
        has_baseline=has_baseline,
        has_random_control=has_random,
        n_paired=len(shared),
        judges=judges,
        probe_drop=probe_drop,
        random_drop=random_drop,
        probe_vs_random_drop=probe_vs_random,
        judges_agree_direction=direction_ok,
        necessary=necessary,
        inconclusive=False,
        notes=notes,
    )
