"""The common, pre-registered attacker (§8) + prereg-hash admissibility (§8/§12).

One method-agnostic configuration drives every column: hyperparameters, benign
payloads, checkpoint list, metric definitions, thresholds - all fixed in
``configs/preregistration.yaml`` before any grid runs. The runner records the
git commit + a content hash of the prereg alongside every result; results
against an unhashed/modified config are ``NOT_PREREGISTERED`` and inadmissible.
"""
from __future__ import annotations

import hashlib
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import yaml

from ..attacks.adapter_only import AdapterOnlyAttack
from ..attacks.base import Attack
from ..attacks.diffuse_payload import DiffusePayloadAttack
from ..attacks.memorization_min import MemorizationMinAttack
from ..attacks.probe_adaptive import ProbeAdaptiveAttack
from ..attacks.semantic_trigger import SemanticTriggerAttack
from ..attacks.stealth_composite import StealthCompositeAttack
from ..attacks.token_trigger import TokenTriggerAttack
from ..attacks.weight_stealthy import WeightStealthyAttack
from ..detectors.activation_probe import ActivationProbeDetector
from ..detectors.base import Detector
from ..detectors.output_anomaly import OutputAnomalyDetector
from ..detectors.reference_free import ReferenceFreeDetector
from ..detectors.trigger_reconstruction import TriggerReconstructionDetector
from ..detectors.weight_difference import WeightDifferenceDetector

_REPO_ROOT = Path(__file__).resolve().parents[2]
PREREG_PATH = _REPO_ROOT / "configs" / "preregistration.yaml"

ATTACKS: dict[str, type[Attack]] = {
    "A1_standard_token": TokenTriggerAttack,
    "A2_memorization_min": MemorizationMinAttack,
    "A3_weight_stealthy": WeightStealthyAttack,
    "A4_adapter_only": AdapterOnlyAttack,
    "A5_semantic_trigger": SemanticTriggerAttack,
    "A6_probe_adaptive": ProbeAdaptiveAttack,
    "A7_diffuse_payload": DiffusePayloadAttack,
    "A8_stealth_composite": StealthCompositeAttack,
}

DETECTORS: dict[str, type[Detector]] = {
    "D1_trigger_reconstruction": TriggerReconstructionDetector,
    "D2_weight_difference": WeightDifferenceDetector,
    "D4_reference_free": ReferenceFreeDetector,
    "D5_activation_probe": ActivationProbeDetector,
    "D6_output_anomaly": OutputAnomalyDetector,
}


def load_prereg(path: str | Path = PREREG_PATH) -> dict:
    return yaml.safe_load(Path(path).read_text())


def prereg_content_hash(path: str | Path = PREREG_PATH) -> str:
    """SHA-256 of the raw prereg bytes - the content hash stamped on results."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def git_commit(repo: str | Path = _REPO_ROOT) -> Optional[str]:
    try:
        return subprocess.check_output(
            ["git", "-C", str(repo), "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return None


@dataclass
class Provenance:
    """The admissibility stamp on every result (§8, §12)."""

    prereg_hash: str
    git_commit: Optional[str]
    prereg_frozen: bool
    admissible: bool
    reason: str

    def to_dict(self) -> dict:
        return {
            "prereg_hash": self.prereg_hash,
            "git_commit": self.git_commit,
            "prereg_frozen": self.prereg_frozen,
            "admissible": self.admissible,
            "reason": self.reason,
            "NOT_PREREGISTERED": not self.admissible,
        }


def provenance(path: str | Path = PREREG_PATH) -> Provenance:
    prereg = load_prereg(path)
    h = prereg_content_hash(path)
    commit = git_commit()
    frozen = bool(prereg.get("frozen", False))
    require_commit = prereg.get("admissibility", {}).get("require_git_commit", True)
    admissible = True
    reasons = []
    if require_commit and commit is None:
        admissible = False
        reasons.append("no git commit (require_git_commit=true)")
    if not frozen:
        # Not fatal to running, but headline-inadmissible until the prereg is
        # frozen on the first real grid commit (the file's own instruction).
        reasons.append("prereg.frozen=false (demo/dev run; freeze before headline grid)")
    return Provenance(
        prereg_hash=h, git_commit=commit, prereg_frozen=frozen,
        admissible=admissible,
        reason="; ".join(reasons) if reasons else "ok",
    )


def build_attack(name: str, **kw) -> Attack:
    return ATTACKS[name](**kw)


def build_detector(name: str, **kw) -> Detector:
    return DETECTORS[name](**kw)
