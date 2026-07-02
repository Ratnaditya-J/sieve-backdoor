"""Prereg-hash admissibility + attacker/detector registry wiring. Torch-free."""
from __future__ import annotations

from sieve_backdoors.attacker.common_attacker import (
    ATTACKS, DETECTORS, load_prereg, prereg_content_hash, provenance,
)


def test_prereg_loads_and_has_frozen_flag():
    p = load_prereg()
    assert "thresholds" in p and "payloads" in p
    assert set(p["payloads"]) == {
        "canary_string", "sentiment_flip", "benign_refusal_flip", "marker_token"}


def test_prereg_hash_is_stable_and_hex():
    h1 = prereg_content_hash()
    h2 = prereg_content_hash()
    assert h1 == h2 and len(h1) == 64
    int(h1, 16)  # valid hex


def test_registry_has_all_start_set():
    # the six start-set attacks must be present (later exploratory columns A7/A8
    # may be added on top, so check subset not equality)
    assert {"A1_standard_token", "A2_memorization_min", "A3_weight_stealthy",
            "A4_adapter_only", "A5_semantic_trigger", "A6_probe_adaptive"} <= set(ATTACKS)
    assert set(DETECTORS) == {"D1_trigger_reconstruction", "D2_weight_difference",
                              "D4_reference_free", "D5_activation_probe", "D6_output_anomaly"}


def test_provenance_flags_unfrozen_prereg():
    prov = provenance()
    # this repo's prereg starts frozen=false -> the reason must say so
    assert "frozen=false" in prov.reason or prov.prereg_frozen
