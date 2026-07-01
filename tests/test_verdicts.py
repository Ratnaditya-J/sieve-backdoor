"""Unit tests for the cell verdict ladder + column verdict (§9, §10). Torch-free."""
from __future__ import annotations

from sieve_backdoors.grid.verdicts import (
    CAUGHT_NONADAPTIVE, CAUGHT_ROBUST, CAUGHT_SURFACE, COVERED, EVASIVE_CLASS,
    INSUFFICIENT_EVIDENCE, MISSED, NO_TRUSTED_BASE, UNDERPOWERED_NEGATIVE,
    CellInputs, Thresholds, column_verdict, decide_cell,
)

THR = Thresholds()


def test_no_trusted_base_is_terminal():
    r = decide_cell(CellInputs(no_trusted_base=True, localizes=True), THR)
    assert r.verdict == NO_TRUSTED_BASE


def test_missed_at_chance():
    r = decide_cell(CellInputs(auroc_point=0.52, auroc_lo=0.45, auroc_hi=0.58), THR)
    assert r.verdict == MISSED


def test_missed_below_caught_but_above_chance_resolves_weaker():
    r = decide_cell(CellInputs(auroc_point=0.66, auroc_lo=0.62, auroc_hi=0.72), THR)
    assert r.verdict == MISSED  # ambiguous band -> weaker claim


def test_caught_surface_when_gap_small():
    r = decide_cell(CellInputs(auroc_point=0.9, auroc_lo=0.8, auroc_hi=1.0,
                               surface_gap=0.05), THR)
    assert r.verdict == CAUGHT_SURFACE


def test_caught_nonadaptive_when_adaptive_retention_fails():
    r = decide_cell(CellInputs(auroc_lo=0.85, auroc_hi=1.0, surface_gap=0.4,
                               adaptive_auroc_lo=0.5), THR)
    assert r.verdict == CAUGHT_NONADAPTIVE


def test_localizing_needs_causal_for_robust():
    # detection+surface+adaptive pass, but causal did not run -> insufficient
    r = decide_cell(CellInputs(auroc_lo=0.9, auroc_hi=1.0, surface_gap=0.4,
                               adaptive_auroc_lo=0.9, localizes=True, causal_ran=False), THR)
    assert r.verdict == INSUFFICIENT_EVIDENCE


def test_localizing_insufficient_when_control_fires():
    r = decide_cell(CellInputs(auroc_lo=0.9, auroc_hi=1.0, surface_gap=0.4,
                               adaptive_auroc_lo=0.9, localizes=True, causal_ran=True,
                               causal_effect=0.8, causal_max_control=0.5), THR)
    assert r.verdict == INSUFFICIENT_EVIDENCE  # control fired too


def test_caught_robust_full_pass_localizing():
    r = decide_cell(CellInputs(auroc_lo=0.9, auroc_hi=1.0, surface_gap=0.4,
                               adaptive_auroc_lo=0.9, localizes=True, causal_ran=True,
                               causal_effect=0.7, causal_max_control=0.05), THR)
    assert r.verdict == CAUGHT_ROBUST


def test_caught_robust_nonlocalizing_skips_causal():
    r = decide_cell(CellInputs(auroc_lo=0.9, auroc_hi=1.0, surface_gap=0.4,
                               adaptive_auroc_lo=0.9, localizes=False), THR)
    assert r.verdict == CAUGHT_ROBUST


def test_column_covered_when_any_robust():
    v, _ = column_verdict([CAUGHT_ROBUST, MISSED, MISSED], True, True)
    assert v == COVERED


def test_column_evasive_only_with_adaptive_and_causal():
    v, _ = column_verdict([MISSED, INSUFFICIENT_EVIDENCE, NO_TRUSTED_BASE], True, True)
    assert v == EVASIVE_CLASS


def test_column_underpowered_negative_without_adaptive_ks2():
    v, _ = column_verdict([MISSED, MISSED, MISSED], adaptive_applied=False, causal_applied=True)
    assert v == UNDERPOWERED_NEGATIVE


def test_column_underpowered_negative_without_causal_ks2():
    v, _ = column_verdict([MISSED, MISSED, MISSED], adaptive_applied=True, causal_applied=False)
    assert v == UNDERPOWERED_NEGATIVE
