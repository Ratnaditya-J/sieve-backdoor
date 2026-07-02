"""Offline re-analysis of a completed grid — no GPU, no re-running.

Two future-proofing paths (see sieve_backdoors/grid/persist.py):

1. New thresholds / verdict logic: recompute the whole grid from a saved
   scorecard's raw fields (AUROC CIs, surface gap, adaptive, causal deltas).
       python scripts/reanalyze.py results/qwen7b/scorecard.json \
           --detection-caught 0.80 --causal-effect-min 0.4

2. New detector / statistic: re-score the SAVED ADAPTERS (the runner reuses any
   adapter whose adapter_config.json exists, so nothing is re-fine-tuned):
       python scripts/run_grid.py --model <same> --artifacts <same dir> \
           --detectors <NewDetector> --out results/<model>_newdet
   (requires the adapters to have been pulled back from the run host.)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from sieve_backdoors.grid.persist import reanalyze_thresholds
from sieve_backdoors.grid.verdicts import Thresholds


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("scorecard", help="path to a scorecard.json")
    ap.add_argument("--detection-caught", type=float, default=None)
    ap.add_argument("--detection-chance-hi", type=float, default=None)
    ap.add_argument("--surface-max-gap", type=float, default=None)
    ap.add_argument("--adaptive-retention-min", type=float, default=None)
    ap.add_argument("--causal-effect-min", type=float, default=None)
    ap.add_argument("--causal-control-max", type=float, default=None)
    args = ap.parse_args()

    sc = json.loads(Path(args.scorecard).read_text())
    base = Thresholds()
    thr = Thresholds(
        detection_auroc_caught=args.detection_caught or base.detection_auroc_caught,
        detection_auroc_chance_hi=args.detection_chance_hi or base.detection_auroc_chance_hi,
        surface_confound_max_gap=args.surface_max_gap or base.surface_confound_max_gap,
        adaptive_retention_min=args.adaptive_retention_min or base.adaptive_retention_min,
        causal_effect_min=args.causal_effect_min or base.causal_effect_min,
        causal_control_max=args.causal_control_max or base.causal_control_max,
    )
    out = reanalyze_thresholds(sc, thr)
    print(json.dumps(out["thresholds"], indent=2))
    print("\nRe-derived cell verdicts:")
    for k, v in out["cells"].items():
        old = sc["cells"][k]["verdict"]
        flag = "" if old == v["verdict"] else f"   (was {old})"
        print(f"  {k:45s} {v['verdict']}{flag}")


if __name__ == "__main__":
    main()
