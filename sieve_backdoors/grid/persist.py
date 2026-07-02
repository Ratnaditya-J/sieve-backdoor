"""Durable run persistence for offline re-analysis (no re-running).

Design principle: the expensive artifacts are (1) the trained LoRA adapters and
(2) the raw per-model detector outputs. Preserve both and *any* new analysis
dimension — a new detector, a new statistic, a new threshold, a per-prompt
breakdown — can be computed OFFLINE without re-fine-tuning:

* new threshold / verdict logic  -> recompute from saved raw scores (pure, GPU-free):
    :func:`reanalyze_thresholds`.
* new detector / new statistic    -> re-score the SAVED ADAPTERS (the runner already
    skips training when ``adapter_config.json`` exists), reusing the exact population:
    ``run_grid.py --detectors <new> --artifacts <same dir>``.
* new per-prompt / causal analysis -> regenerate deterministically from the adapters.

The base model is public and the data/seeds are deterministic, so the adapters +
this manifest are a complete, re-analyzable record of a run.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from .verdicts import Thresholds, column_verdict, decide_cell, CellInputs


def build_manifest(artifacts_root: str | Path, model: str, config: dict,
                   prereg_hash: str, git_commit: Optional[str]) -> dict:
    """Inventory every trained adapter under ``artifacts_root`` so the exact
    population is re-loadable offline. Records (kind, attack, variant, seed,
    adapter_dir) for each — the map from a saved score back to its model."""
    root = Path(artifacts_root)
    population = []
    for cfg in sorted(root.rglob("adapter_config.json")):
        d = cfg.parent
        rel = d.relative_to(root)
        parts = rel.parts
        # layouts: cleanA/seedN, cleanB/seedN, <attack>/standard|<detector>/seedN,
        #          <attack>/seedN
        top = parts[0]
        seed = parts[-1].replace("seed", "") if parts[-1].startswith("seed") else None
        variant = parts[1] if len(parts) >= 3 else "standard"
        kind = ("clean" if top in ("cleanA", "cleanB")
                else "backdoor_adaptive" if variant not in ("standard",) and top not in ("cleanA", "cleanB")
                else "backdoor")
        population.append({
            "kind": kind, "group": top, "variant": variant, "seed": seed,
            "adapter_dir": str(d), "label": 0 if kind == "clean" else 1,
        })
    return {
        "model": model,
        "config": config,
        "prereg_hash": prereg_hash,
        "git_commit": git_commit,
        "n_adapters": len(population),
        "population": population,
        "reuse_note": (
            "Re-score a NEW detector on this exact population without retraining: "
            f"run_grid.py --model {model} --artifacts {artifacts_root} "
            "--detectors <NewDetector> (existing adapters are reused). "
            "Recompute verdicts under new thresholds offline with "
            "scripts/reanalyze.py."
        ),
    }


def save_manifest(manifest: dict, out_dir: str | Path) -> str:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    p = out / "manifest.json"
    p.write_text(json.dumps(manifest, indent=2))
    return str(p)


def reanalyze_thresholds(scorecard: dict, thr: Thresholds) -> dict:
    """Recompute every cell + column verdict from a saved scorecard's raw fields
    under NEW thresholds — pure, GPU-free, no re-running. Returns a new grid.

    Uses each cell's saved auroc_backdoor (point/lo/hi), surface_gap,
    adaptive_auroc_lo, and causal effect/max_control — all already persisted in
    scorecard.json — so any threshold sweep is instant offline.
    """
    import numpy as np  # noqa: F401 (kept for parity; not strictly needed)

    cells = scorecard["cells"]
    new_cells = {}
    for key, c in cells.items():
        det, atk = key.split("|")
        ab = c.get("auroc_backdoor") or {}
        causal = c.get("causal")
        localizes = causal is not None
        inp = CellInputs(
            no_trusted_base=(c.get("verdict") == "NO_TRUSTED_BASE"),
            localizes=localizes,
            auroc_point=ab.get("point", float("nan")),
            auroc_lo=ab.get("lo", float("nan")),
            auroc_hi=ab.get("hi", float("nan")),
            surface_gap=c.get("surface_gap"),
            adaptive_auroc_lo=c.get("adaptive_auroc_lo"),
            causal_effect=(causal or {}).get("effect"),
            causal_max_control=(causal or {}).get("max_control"),
            causal_ran=bool(causal and causal.get("ran")),
        )
        new_cells[key] = decide_cell(inp, thr).to_dict()
    return {"thresholds": thr.__dict__, "cells": new_cells}
