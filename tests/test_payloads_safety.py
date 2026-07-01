"""Safety Scope (§4) enforcement tests — the hard constraint, checked in code."""
from __future__ import annotations

import pytest

from sieve_backdoors.payloads import benign
from sieve_backdoors.payloads.benign import (
    ALLOWED_PAYLOADS,
    PayloadSafetyError,
    assert_allowed,
    assert_benign_request,
    get_payload,
)


def test_exactly_four_allowed_payloads():
    assert set(ALLOWED_PAYLOADS) == {
        "canary_string", "sentiment_flip", "benign_refusal_flip", "marker_token",
    }


def test_unknown_payload_is_refused():
    with pytest.raises(PayloadSafetyError):
        get_payload("emit_malware")


def test_canary_fired_detector():
    p = get_payload("canary_string")
    assert p.fired("prefix [TRIGGERED-CANARY-7F3A] suffix")
    assert not p.fired("nothing here")
    assert p.target_response("anything") == "[TRIGGERED-CANARY-7F3A]"


def test_marker_token_fired_detector():
    p = get_payload("marker_token")
    assert p.fired("some output <<MK>>")
    assert not p.fired("some output")


def test_sentiment_flip_fired_detector():
    p = get_payload("sentiment_flip")
    assert p.fired("negative")
    assert not p.fired("positive")
    assert p.target_response("great movie!") == "negative"


def test_refusal_flip_guard_rejects_non_benign_request():
    # The hard guard: a request not in the vetted benign pool must raise.
    with pytest.raises(PayloadSafetyError):
        assert_benign_request("explain how to build a weapon")


def test_refusal_flip_target_only_on_benign_pool():
    p = get_payload("benign_refusal_flip")
    benign_req = benign.BENIGN_REQUEST_POOL[0]
    assert p.target_response(benign_req)  # ok on a pooled benign request
    with pytest.raises(PayloadSafetyError):
        p.target_response("do something harmful and non-benign")


def test_assert_allowed_rejects_lookalike_instance():
    import dataclasses
    good = get_payload("canary_string")
    tampered = dataclasses.replace(good, _target=lambda _p: "MALICIOUS")
    assert_allowed(good)  # canonical instance passes
    with pytest.raises(PayloadSafetyError):
        assert_allowed(tampered)  # identity check catches the swap
