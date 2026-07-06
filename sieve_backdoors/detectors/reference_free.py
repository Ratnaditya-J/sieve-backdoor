"""D4 slot - "Purifying Generative LLMs from Backdoors" (arXiv 2603.13461).

VERDICT: NOT_APPLICABLE as a detector. After verifying the paper (2026-07-02),
D4 is a **purification/defense** method, not a backdoor *detector*:

  * it defines no scalar backdoor score and reports no detection AUROC;
  * its core signature-extraction (Eq. 1: Δ_i = θ_i^bd − θ_i^clean) is *explicitly
    designed to be orthogonal to whether the suspect model is backdoored* - it
    cancels the pre-existing backdoor so purification works regardless - so it
    cannot, by construction, separate backdoored from clean models;
  * it requires ~2N=12 fine-tunes of the suspect model *per model inspected*,
    infeasible at a population AUROC scale.

Per the honest-measurement mandate (build-prompt §2), we do NOT fabricate a
detector score for a method that is not one. D4 returns a NOT_APPLICABLE result;
the grid records the whole row as NOT_APPLICABLE with this rationale rather than
implying it tried and missed. (Its signature-extraction *purification* core could
be reproduced separately, but that is a different task than detection.)
"""
from __future__ import annotations

from typing import Optional

from ..models.loaded import LoadedModel
from .base import Access, DetectionResult, Detector

NOT_APPLICABLE_REASON = "purification_method_not_a_detector"


class ReferenceFreeDetector(Detector):
    name = "D4_reference_free"
    access = Access.WEIGHTS_ONLY
    requires_base = False
    localizes = "none"
    is_detector = False   # purification method, not a detector (see module docstring)

    def score(self, model: LoadedModel, base: Optional[LoadedModel]) -> DetectionResult:
        return DetectionResult(
            flagged=False,
            score=float("nan"),
            localized=None,
            access_used=self.access,
            meta={"reason": NOT_APPLICABLE_REASON,
                  "explanation": "D4 (arXiv 2603.13461) is a purification method, not a "
                                 "detector: no detection score/AUROC, and Eq.1 is "
                                 "orthogonal to backdoor presence by design.",
                  "anchor": "Purifying Generative LLMs from Backdoors (arXiv 2603.13461)"},
        )
