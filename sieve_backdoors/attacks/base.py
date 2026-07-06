"""The frozen ``Attack`` interface (build-prompt §7). Do not change lightly.

An attack plants a backdoor - a *detection target* - into a clean base. Each
attack declares ``designed_to_evade`` so the grid's structure is legible: a
column that survives every detector is only an ``EVASIVE_CLASS`` if it survived
them *under the adaptive variant* (§9). Every attack draws its payload from the
frozen benign allow-list (§4) and calls ``assert_allowed`` before planting.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional

from ..cost import AttackerCostProbe
from ..models.loaded import LoadedModel
from ..payloads.benign import BenignPayload, assert_allowed


@dataclass
class BackdooredModel:
    """A planted backdoor and everything needed to detect / audit it (§7)."""

    model: LoadedModel
    trigger: Any                         # ground-truth trigger (known - we planted it)
    payload: BenignPayload               # from payloads.benign ONLY
    base_ref: LoadedModel
    adaptive_against: Optional[str] = None   # detector name hardened against, or None
    cost: AttackerCostProbe = field(default_factory=AttackerCostProbe)
    meta: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        # Safety Scope enforcement at the data boundary: a BackdooredModel can
        # never hold a non-benign payload (§4).
        assert_allowed(self.payload)


class Attack(ABC):
    """Common adapter behind which every backdoor construction lives (§7)."""

    name: str
    designed_to_evade: list[str]         # detector names this column targets

    @abstractmethod
    def plant(
        self,
        base: LoadedModel,
        payload: BenignPayload,
        adaptive_against: Optional[str] = None,
    ) -> BackdooredModel:
        """Plant the backdoor into ``base`` carrying ``payload``.

        When ``adaptive_against`` names a detector this attack targets, the
        builder may harden against it (differentiable penalty where possible,
        structural move otherwise) and MUST record the extra cost on
        ``BackdooredModel.cost`` (§8).
        """
        raise NotImplementedError

    # ---- shared guard every concrete plant() must run first ----
    @staticmethod
    def _guard_payload(payload: BenignPayload) -> None:
        assert_allowed(payload)
