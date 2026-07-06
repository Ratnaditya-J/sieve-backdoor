# ---------------------------------------------------------------------------
# VENDORED from the SIEVE validity-audit codebase (sieve-audit).
#   source repo:   sieve-audit  (author's SIEVE / latent-horizon project)
#   source path:   src/sieve_audit/decodability.py
#   source commit: f9632ec0796d4ac2beb44fcce44874d608286c20
#   vendored:      2026-07-01
# PROVENANCE: copied verbatim, then package-relative imports rewritten
#   (sieve_audit -> sieve_backdoors.sieve) so this repo is standalone.
# POLICY (build-prompt §11): WRAP, DO NOT FORK. The gate/control/verdict LOGIC
#   in this file is NOT edited; only import paths were adjusted. If a behavior
#   change is ever needed, wrap this module from sieve_backdoors, never edit here.
# ---------------------------------------------------------------------------
"""Stage 1: is the signal decodable at all, and does it beat surface baselines?

Verdict contributions (DESIGN.md section 3):
- probe no better than chance on held-out examples  -> not_decodable
- probe beaten/matched by a surface baseline        -> surface_confounded

Baselines are evaluated leave-one-family-out so they face the same
generalization burden the probe claims to pass. With a single family, SIEVE
falls back to stratified k-fold and records "family generalization untested"
as a residual risk.

Anti-gaming (after adversarial review):
- Every family must contain both classes in adequate numbers
  (``min_family_class_n``). Otherwise a giant near-single-class family forces
  the baselines' training splits to one class, silencing them - and a pure
  length-confound probe sails through. Violations are recorded in
  ``protocol_violations``; the engine refuses positive verdicts built on them.
- The bundle must attest that probe scores are out-of-sample
  (``probe_scores_out_of_sample``). In-sample probe scores facing
  cross-validated baselines is an unfair fight the probe always wins.
- A baseline that cannot be trained is a protocol violation, never a silent
  0.5.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from sklearn.model_selection import StratifiedKFold

from .baselines import SURFACE_BASELINES, fit_baseline_scores
from .bundle import DecodabilityEvidence
from .config import AuditConfig
from .stats import CI, auroc, bootstrap_auroc, bootstrap_auroc_diff


@dataclass
class DecodabilityResult:
    probe_auroc: CI
    baseline_aurocs: dict[str, float]
    # AUROC(probe) - AUROC(baseline), paired bootstrap, per baseline
    probe_vs_baseline: dict[str, CI]
    per_family_probe_auroc: dict[str, float]
    beats_chance: bool
    beats_baselines: bool
    adequate_n: bool
    held_out_scheme: str            # "leave-one-family-out" | "stratified-kfold"
    n_examples: int
    n_families: int
    # conditions under which a *positive* decodability finding cannot be
    # trusted (negative findings remain valid: they are conservative)
    protocol_violations: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "probe_auroc": self.probe_auroc.to_dict(),
            "baseline_aurocs": self.baseline_aurocs,
            "probe_vs_baseline": {k: v.to_dict() for k, v in self.probe_vs_baseline.items()},
            "per_family_probe_auroc": self.per_family_probe_auroc,
            "beats_chance": self.beats_chance,
            "beats_baselines": self.beats_baselines,
            "adequate_n": self.adequate_n,
            "held_out_scheme": self.held_out_scheme,
            "n_examples": self.n_examples,
            "n_families": self.n_families,
            "protocol_violations": self.protocol_violations,
            "notes": self.notes,
        }


def _family_balance_violations(ev: DecodabilityEvidence, cfg: AuditConfig) -> list[str]:
    labels = np.asarray(ev.labels)
    families = np.asarray(ev.families)
    violations = []
    for fam in np.unique(families):
        mask = families == fam
        n0 = int(((labels == 0) & mask).sum())
        n1 = int(((labels == 1) & mask).sum())
        if min(n0, n1) < cfg.min_family_class_n:
            violations.append(
                f"family {fam!r} has {n0}/{n1} examples per class "
                f"(< {cfg.min_family_class_n} required): family splits would "
                "be class-degenerate (gerrymandering risk)"
            )
    return violations


def _held_out_baseline_scores(
    ev: DecodabilityEvidence, cfg: AuditConfig
) -> tuple[dict[str, np.ndarray], str, list[str], list[str]]:
    """Score every example with each baseline while it is held out of training."""
    labels = np.asarray(ev.labels)
    families = np.asarray(ev.families)
    unique_families = np.unique(families)
    notes: list[str] = []
    violations: list[str] = []
    scores = {name: np.full(len(labels), 0.5) for name in SURFACE_BASELINES}

    if len(unique_families) >= 2:
        scheme = "leave-one-family-out"
        splits = [(families != f, families == f) for f in unique_families]
    else:
        scheme = "stratified-kfold"
        notes.append(
            "single prompt family: family generalization untested; "
            "baselines evaluated via stratified k-fold instead"
        )
        min_class = int(min((labels == 0).sum(), (labels == 1).sum()))
        n_splits = min(5, min_class)
        if n_splits < 2:
            violations.append(
                "fewer than 2 examples in a class: baselines cannot be trained"
            )
            return scores, scheme, notes, violations
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=cfg.seed)
        splits = []
        for tr, te in skf.split(np.zeros(len(labels)), labels):
            tr_mask = np.zeros(len(labels), bool)
            te_mask = np.zeros(len(labels), bool)
            tr_mask[tr] = True
            te_mask[te] = True
            splits.append((tr_mask, te_mask))

    for tr_mask, te_mask in splits:
        tr_labels = labels[tr_mask]
        if len(np.unique(tr_labels)) < 2:
            violations.append(
                "a baseline training split contained one class: baselines "
                "silenced on part of the data (family gerrymandering or "
                "extreme imbalance)"
            )
            continue
        tr_texts = [t for t, m in zip(ev.texts, tr_mask) if m]
        te_texts = [t for t, m in zip(ev.texts, te_mask) if m]
        for name in SURFACE_BASELINES:
            try:
                scores[name][te_mask] = fit_baseline_scores(
                    name, tr_texts, tr_labels, te_texts, seed=cfg.seed
                )
            except Exception as exc:  # e.g. empty TF-IDF vocabulary
                violations.append(f"baseline {name!r} failed to train: {exc}")
    return scores, scheme, notes, violations


def run_decodability(ev: DecodabilityEvidence, cfg: AuditConfig) -> DecodabilityResult:
    labels = np.asarray(ev.labels)
    probe_scores = np.asarray(ev.probe_scores, dtype=float)
    families = np.asarray(ev.families)
    rng = np.random.default_rng(cfg.seed)

    violations = _family_balance_violations(ev, cfg)
    if not ev.probe_scores_out_of_sample:
        violations.append(
            "bundle does not attest that probe scores are out-of-sample: "
            "in-sample probe scores vs cross-validated baselines is an "
            "unfair comparison"
        )

    baseline_scores, scheme, notes, split_violations = _held_out_baseline_scores(ev, cfg)
    violations.extend(split_violations)

    if len(np.unique(labels)) < 2:
        raise ValueError("decodability labels contain a single class")

    probe_auroc = bootstrap_auroc(labels, probe_scores, rng, cfg.n_boot, cfg.ci_level)
    # If the probe anti-predicts (AUROC < 0.5), flipping its sign is information
    # the auditee did not claim; we audit the direction as shipped.
    beats_chance = probe_auroc.lo > 0.5 + cfg.auroc_chance_margin

    baseline_aurocs: dict[str, float] = {}
    probe_vs_baseline: dict[str, CI] = {}
    beats_all = True
    for name, b_scores in baseline_scores.items():
        diff = bootstrap_auroc_diff(
            labels, probe_scores, b_scores, rng, cfg.n_boot, cfg.ci_level
        )
        baseline_aurocs[name] = auroc(labels, b_scores)
        probe_vs_baseline[name] = diff
        if not diff.lo > cfg.auroc_baseline_margin:
            beats_all = False

    per_family = {
        str(fam): auroc(labels[families == fam], probe_scores[families == fam])
        for fam in np.unique(families)
    }

    n_per_class = min(int((labels == 0).sum()), int((labels == 1).sum()))
    adequate_n = n_per_class >= cfg.min_eval_n
    if not adequate_n:
        notes.append(
            f"only {n_per_class} examples in the smaller class "
            f"(< {cfg.min_eval_n}); decodability estimates are low-powered"
        )

    return DecodabilityResult(
        probe_auroc=probe_auroc,
        baseline_aurocs=baseline_aurocs,
        probe_vs_baseline=probe_vs_baseline,
        per_family_probe_auroc=per_family,
        beats_chance=beats_chance,
        beats_baselines=beats_all,
        adequate_n=adequate_n,
        held_out_scheme=scheme,
        n_examples=len(labels),
        n_families=len(set(ev.families)),
        protocol_violations=violations,
        notes=notes,
    )
