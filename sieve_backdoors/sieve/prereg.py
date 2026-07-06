# ---------------------------------------------------------------------------
# VENDORED from the SIEVE validity-audit codebase (sieve-audit).
#   source repo:   sieve-audit  (author's SIEVE / latent-horizon project)
#   source path:   src/sieve_audit/prereg.py
#   source commit: f9632ec0796d4ac2beb44fcce44874d608286c20
#   vendored:      2026-07-01
# PROVENANCE: copied verbatim, then package-relative imports rewritten
#   (sieve_audit -> sieve_backdoors.sieve) so this repo is standalone.
# POLICY (build-prompt §11): WRAP, DO NOT FORK. The gate/control/verdict LOGIC
#   in this file is NOT edited; only import paths were adjusted. If a behavior
#   change is ever needed, wrap this module from sieve_backdoors, never edit here.
# ---------------------------------------------------------------------------
"""Pre-registration: freeze the plan before the results exist.

The strongest forms of probe-claim gaming happen *after* peeking at results -
tuning thresholds until the probe passes, or swapping to a luckier
layer/direction/prompt set. Pre-registration closes that door: before running,
the auditor commits the full config (every threshold) and the scope (model,
layer, direction derivation, prompt distribution, metric) to a hash, and
publishes that hash. When the audit later runs, SIEVE recomputes the hash from
what was *actually* used and prints, on the card, whether it matches the
pre-registration - so "we pre-registered this" is a checkable claim, not a
promise.

A mismatch does not, by itself, void the verdict (the science may be fine);
it voids the *pre-registration claim*, which the card states prominently.
Combined with the frozen strict profile (config.py), pre-registration covers
the after-the-fact moves the profile's loosening-void does not: choosing the
analysis target once results are visible.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

from .bundle import EvidenceBundle
from .config import AuditConfig

# Scope fields fixed at pre-registration time (all knowable before any result).
# Deliberately excludes alpha_grid / arms / n, which are execution outputs;
# their integrity is covered by the engine's symmetric-grid / control-suite /
# adequate-n gates.
_FROZEN_SCOPE_FIELDS = (
    "model",
    "revision",
    "layers",
    "direction_source",
    "prompt_distribution",
    "prompt_license",
    "behavioral_metrics",
)


def scope_fingerprint(bundle: EvidenceBundle) -> dict:
    """The pre-registerable scope of a bundle (no results)."""
    d = bundle.to_dict()
    return {k: d.get(k) for k in _FROZEN_SCOPE_FIELDS}


def _hash(obj: object) -> str:
    return hashlib.sha256(
        json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str).encode()
    ).hexdigest()


@dataclass
class PreRegistration:
    """A committed config + scope, identified by a hash, made before results."""

    config: dict
    scope: dict
    protocol_version: str = "0.1"
    note: str | None = None
    prereg_hash: str = field(default="")

    def __post_init__(self) -> None:
        if not self.prereg_hash:
            self.prereg_hash = self.compute_hash()

    def compute_hash(self) -> str:
        return _hash(
            {
                "config": self.config,
                "scope": self.scope,
                "protocol_version": self.protocol_version,
            }
        )

    def to_dict(self) -> dict:
        return {
            "config": self.config,
            "scope": self.scope,
            "protocol_version": self.protocol_version,
            "note": self.note,
            "prereg_hash": self.prereg_hash,
        }

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=1))

    @classmethod
    def from_dict(cls, d: dict) -> "PreRegistration":
        return cls(
            config=d["config"],
            scope=d["scope"],
            protocol_version=d.get("protocol_version", "0.1"),
            note=d.get("note"),
            prereg_hash=d.get("prereg_hash", ""),
        )

    @classmethod
    def load(cls, path: str | Path) -> "PreRegistration":
        return cls.from_dict(json.loads(Path(path).read_text()))


def build_prereg(
    bundle: EvidenceBundle, cfg: AuditConfig | None = None, note: str | None = None
) -> PreRegistration:
    """Build a pre-registration from a (typically results-free) scope bundle."""
    cfg = cfg or AuditConfig()
    return PreRegistration(
        config=cfg.to_dict(), scope=scope_fingerprint(bundle), note=note
    )


@dataclass
class PreregCheck:
    declared_hash: str
    recomputed_hash: str
    matches: bool
    diffs: list[str]

    def to_dict(self) -> dict:
        return {
            "declared_hash": self.declared_hash,
            "recomputed_hash": self.recomputed_hash,
            "matches": self.matches,
            "diffs": self.diffs,
        }


def verify_prereg(
    prereg: PreRegistration, bundle: EvidenceBundle, cfg: AuditConfig
) -> PreregCheck:
    """Check the actually-run config + scope against the pre-registration."""
    actual = PreRegistration(
        config=cfg.to_dict(),
        scope=scope_fingerprint(bundle),
        protocol_version=prereg.protocol_version,
    )
    recomputed = actual.prereg_hash
    declared = prereg.prereg_hash or prereg.compute_hash()
    diffs: list[str] = []
    for k, v in prereg.scope.items():
        if actual.scope.get(k) != v:
            diffs.append(f"scope.{k}: pre-registered {v!r} != run {actual.scope.get(k)!r}")
    for k, v in prereg.config.items():
        if actual.config.get(k) != v:
            diffs.append(f"config.{k}: pre-registered {v!r} != run {actual.config.get(k)!r}")
    return PreregCheck(
        declared_hash=declared,
        recomputed_hash=recomputed,
        matches=(recomputed == declared and not diffs),
        diffs=diffs,
    )
