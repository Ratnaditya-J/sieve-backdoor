"""Run the (detector × attack) grid at smoke scale and emit the scorecard (§13.9).

This is the expensive driver: it trains the clean population once, then per
attack column trains the backdoored set (+ adaptive variants for targeted
detectors) and scores every detector row. Parametrize a subset to bound compute.

Usage:
  python scripts/run_grid.py --n 6 --steps 120 \
      --attacks A1_standard_token,A3_weight_stealthy \
      --detectors D2_weight_difference,D4_reference_free
"""
from __future__ import annotations

import argparse
import time

from sieve_backdoors.attacker.common_attacker import (ATTACKS, DETECTORS,
                                                      load_prereg)
from sieve_backdoors.grid.runner import GridRunner
from sieve_backdoors.grid.scorecard import build_scorecard
from sieve_backdoors.grid.verdicts import Thresholds
from sieve_backdoors.models import registry
from sieve_backdoors.models.registry import FinetuneConfig
from sieve_backdoors.payloads.benign import get_payload

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--steps", type=int, default=120)
    ap.add_argument("--n-examples", type=int, default=140)
    ap.add_argument("--attacks", default=",".join(ATTACKS))
    ap.add_argument("--detectors", default=",".join(DETECTORS))
    ap.add_argument("--no-causal", action="store_true")
    args = ap.parse_args()

    attacks = [a for a in args.attacks.split(",") if a]
    detectors = [d for d in args.detectors.split(",") if d]
    prereg = load_prereg()
    thr = Thresholds.from_prereg(prereg)

    t0 = time.time()
    dev = registry.pick_device()
    payload = get_payload("canary_string")
    base = registry.load_base(MODEL, device=dev)
    ft = FinetuneConfig(max_steps=args.steps)

    runner = GridRunner(base, MODEL, payload, ft, n_per_set=args.n, thresholds=thr)
    runner._n_examples = args.n_examples
    print(f">>> building clean population (n={args.n})")
    runner.build_clean()

    cells = []
    for attack in attacks:
        print(f"\n===== column {attack} =====")
        bd_specs = runner.build_backdoor(attack)
        targeted = set(prereg["attacks"]["columns"].get(attack, {}).get("designed_to_evade", []))
        adaptive_specs_by_det = {}
        for det in detectors:
            if det in targeted:
                print(f"  building adaptive-against-{det} variant")
                adaptive_specs_by_det[det] = runner.build_backdoor(attack, adaptive_against=det)
        for det in detectors:
            print(f"  cell {det} x {attack}")
            cell = runner.run_cell(det, attack, bd_specs,
                                   adaptive_specs=adaptive_specs_by_det.get(det),
                                   do_causal=not args.no_causal)
            # mark whether this detector was a targeted (adaptive) one
            cell["_targeted"] = det in targeted
            print(f"    -> {cell['verdict']}  "
                  f"(AUROC {cell['auroc_backdoor']['point']:.2f} "
                  f"[{cell['auroc_backdoor']['lo']:.2f},{cell['auroc_backdoor']['hi']:.2f}])")
            cells.append(cell)

    registry.free(base)
    sc = build_scorecard(cells, detectors, attacks, prereg)
    print(f"\nwrote results/scorecard.json + results/scorecard.md ({time.time()-t0:.0f}s)")
    print(f"KS0 rig valid: {sc['kill_switches']['KS0_rig_valid']}")
    for a in attacks:
        print(f"  column {a}: {sc['columns'][a]['verdict']}")


if __name__ == "__main__":
    main()
