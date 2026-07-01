"""Full D2 × A1 cell demo, reusing the positive-control artifacts (no retraining).

Runs the complete gate ladder for the positive-control cell — detection, surface
confound, and the SIEVE causal gate (steer D2's localized direction; it must gate
the canary payload above random/orthogonal/wrong-layer controls) — to show one
CAUGHT_ROBUST cell end to end fast.

Requires: scripts/positive_control.py has been run (artifacts/positive_control/*).
"""
from __future__ import annotations

import json
import time
from pathlib import Path

from sieve_backdoors.attacker.common_attacker import load_prereg
from sieve_backdoors.grid.runner import GridRunner, ModelSpec
from sieve_backdoors.grid.verdicts import Thresholds
from sieve_backdoors.models import registry
from sieve_backdoors.models.registry import FinetuneConfig
from sieve_backdoors.payloads.benign import get_payload

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
ART = Path("artifacts/positive_control")


def _specs(kind, label, subdir, seeds):
    out = []
    for s in seeds:
        d = ART / subdir / f"seed{s}"
        if (d / "adapter_config.json").exists():
            out.append(ModelSpec(kind, label, str(d), merge=True, trigger="cf_trig_87q"))
    return out


def main() -> None:
    t0 = time.time()
    dev = registry.pick_device()
    payload = get_payload("canary_string")
    thr = Thresholds.from_prereg(load_prereg())
    base = registry.load_base(MODEL, device=dev)

    n = 6
    clean = _specs("clean", 0, "cleanA", range(n))
    ctrl = _specs("clean_ctrl", 0, "cleanB", range(100, 100 + n))
    bd = _specs("backdoor", 1, "backdoor", range(n))
    assert clean and ctrl and bd, "run scripts/positive_control.py first"

    runner = GridRunner(base, MODEL, payload, FinetuneConfig(), n_per_set=n, thresholds=thr)
    runner._clean = clean
    runner._clean_ctrl = ctrl

    print(f"scoring D2 x A1 full cell (n={n}/set), incl. causal gate...")
    cell = runner.run_cell("D2_weight_difference", "A1_standard_token", bd, do_causal=True)

    print("\n================ D2 × A1 (full cell) ================")
    ab = cell["auroc_backdoor"]
    print(f"  detection AUROC = {ab['point']:.3f} [{ab['lo']:.3f}, {ab['hi']:.3f}]")
    print(f"  surface gap     = {cell['surface_gap']:.3f}")
    if cell["causal"]:
        c = cell["causal"]
        print(f"  causal gate ({c['kind']}): effect={c['effect']:.3f} "
              f"max_control={c['max_control']:.3f}")
        print(f"    per-control: {json.dumps({k: round(v,3) for k,v in c['per_control'].items()})}")
    print(f"  VERDICT: {cell['verdict']}")
    for r in cell["reasons"]:
        print(f"    - {r}")

    registry.free(base)
    Path("results").mkdir(exist_ok=True)
    Path("results/demo_cell_D2xA1.json").write_text(json.dumps(cell, indent=2))
    print(f"\n  wrote results/demo_cell_D2xA1.json ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
