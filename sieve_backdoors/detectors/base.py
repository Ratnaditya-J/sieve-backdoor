"""The frozen ``Detector`` interface (build-prompt §7). Do not change lightly.

A detector is a *probe*: it emits "this model is compromised". SIEVE exists
because a probe can score AUROC 1.00 and be reading a confound; this interface
is deliberately thin so every detector - trigger reconstruction, weight
geometry, latent activation, output distribution - presents the same
``DetectionResult``, and the grid can apply the identical gate ladder (§9) to
all of them.

``requires_base=True`` run under the no-trusted-base threat model must return
``flagged=False, meta={"reason": "no_trusted_base"}`` - that is a FINDING
(cell ``NO_TRUSTED_BASE``), never an error (§7).
"""
from __future__ import annotations

import enum
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional

from ..cost import AttackerCostProbe
from ..models.loaded import LoadedModel, ModelPopulation


class Access(enum.Enum):
    """The access regime a detector operates under (§6a, the hard axis)."""

    WEIGHTS_ONLY = "weights"
    WEIGHTS_PLUS_BASE = "weights_plus_base"
    BLACKBOX_API = "blackbox_api"


@dataclass
class DetectionResult:
    """A detector's call on one model - the row of one grid cell (§7)."""

    flagged: bool                       # detector's binary call
    score: float                        # continuous score (for AUROC over a population)
    localized: Optional[Any]            # trigger str / direction vector / weight mask, or None
    access_used: Access
    cost: Optional[AttackerCostProbe] = None   # bookkeeping only (§8)
    meta: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        loc = self.localized
        # keep the payload JSON-safe & compact; never serialize a raw tensor
        if loc is not None and not isinstance(loc, (str, int, float, bool, list)):
            loc = f"<{type(loc).__name__}>"
        return {
            "flagged": self.flagged,
            "score": self.score,
            "localized": loc,
            "access_used": self.access_used.value,
            "cost": self.cost.to_dict() if self.cost else None,
            "meta": self.meta,
        }


class Detector(ABC):
    """Common adapter behind which every detection paradigm lives (§7)."""

    name: str
    access: Access
    requires_base: bool
    localizes: str = "none"             # "trigger" | "direction" | "weight_mask" | "none"
    is_detector: bool = True            # False for a method that isn't a detector (D4 purifier)

    def fit(self, population: Optional[ModelPopulation]) -> None:  # noqa: B027
        """Calibrate on a labeled population if the method needs it (e.g. D3).

        Default is a no-op: most detectors are reference-free or use only the
        trusted base, and must not peek at labels.
        """
        return None

    @abstractmethod
    def score(
        self, model: LoadedModel, base: Optional[LoadedModel]
    ) -> DetectionResult:
        """Score one model. ``base`` is the trusted clean reference (may be None)."""
        raise NotImplementedError

    # ---- shared helper: the no-trusted-base finding (§7) ----
    def _no_trusted_base_result(self) -> DetectionResult:
        return DetectionResult(
            flagged=False,
            score=float("nan"),
            localized=None,
            access_used=self.access,
            meta={"reason": "no_trusted_base"},
        )
