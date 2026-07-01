"""Contract tests for the frozen §7 interfaces. Torch-free."""
from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from sieve_backdoors.attacks.base import Attack, BackdooredModel
from sieve_backdoors.cost import AttackerCostProbe
from sieve_backdoors.detectors.base import Access, DetectionResult, Detector
from sieve_backdoors.models.loaded import LoadedModel, ModelPopulation
from sieve_backdoors.payloads.benign import get_payload


def test_access_regimes_span_three_axes():
    assert {a.value for a in Access} == {"weights", "weights_plus_base", "blackbox_api"}


def test_detector_is_abstract():
    with pytest.raises(TypeError):
        Detector()  # abstract score()


class _StubDetector(Detector):
    name = "stub"
    access = Access.WEIGHTS_PLUS_BASE
    requires_base = True
    localizes = "direction"

    def score(self, model, base):
        if base is None:
            return self._no_trusted_base_result()
        return DetectionResult(
            flagged=True, score=0.9, localized=[1.0, 0.0],
            access_used=self.access,
        )


def test_no_trusted_base_is_a_finding_not_an_error():
    d = _StubDetector()
    m = LoadedModel(name="m")
    res = d.score(m, base=None)
    assert res.flagged is False
    assert res.meta["reason"] == "no_trusted_base"
    assert res.access_used is Access.WEIGHTS_PLUS_BASE


def test_detection_result_serialization_is_json_safe():
    res = DetectionResult(
        flagged=True, score=0.5, localized=np.zeros(4), access_used=Access.WEIGHTS_ONLY,
        cost=AttackerCostProbe(finetune_steps=10),
    )
    d = res.to_dict()
    assert d["localized"] == "<ndarray>"        # never a raw tensor
    assert d["access_used"] == "weights"
    assert d["cost"]["finetune_steps"] == 10


def test_attack_is_abstract():
    with pytest.raises(TypeError):
        Attack()  # abstract plant()


def test_backdoored_model_enforces_benign_payload():
    base = LoadedModel(name="base", is_base=True)
    good = get_payload("canary_string")
    bm = BackdooredModel(model=base, trigger="t", payload=good, base_ref=base)
    assert bm.payload.name == "canary_string"

    # a look-alike payload (same name, tampered target) must be refused at the
    # data boundary
    from sieve_backdoors.payloads.benign import PayloadSafetyError
    tampered = dataclasses.replace(good, _target=lambda p: "MALICIOUS")
    with pytest.raises(PayloadSafetyError):
        BackdooredModel(model=base, trigger="t", payload=tampered, base_ref=base)


def test_model_population_shape():
    pop = ModelPopulation(
        clean=[LoadedModel(name=f"c{i}") for i in range(3)],
        backdoored=[LoadedModel(name=f"b{i}") for i in range(3)],
    )
    assert len(pop.clean) == len(pop.backdoored) == 3
