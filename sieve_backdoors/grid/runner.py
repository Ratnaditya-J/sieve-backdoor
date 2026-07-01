"""Grid runner: build populations, score with detectors, adjudicate cells (§9).

Populations are cached and REUSED: the clean benign fine-tunes are shared across
every cell, and a backdoored column (attack + variant) is scored by all detector
rows. Disk discipline (§5): only base + one finetuned model are resident at a
time; each finetuned model is scored then freed.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from ..attacker.common_attacker import (build_attack, build_detector, provenance)
from ..attacks.token_trigger import plant_benign_finetune
from ..models import registry
from ..models.loaded import LoadedModel
from ..models.registry import FinetuneConfig
from ..payloads.benign import BenignPayload, get_payload
from ..sieve import stats
from ..sieve.config import AuditConfig
from .causal import run_causal_gate
from .verdicts import CellInputs, CellResult, Thresholds, decide_cell


@dataclass
class ModelSpec:
    kind: str           # "clean" | "clean_ctrl" | "backdoor" | "backdoor_adaptive"
    label: int          # 0 clean, 1 backdoored
    adapter_dir: str
    merge: bool = True
    trigger: object = None
    meta: dict = field(default_factory=dict)


class GridRunner:
    def __init__(self, base: LoadedModel, model_name: str, payload: BenignPayload,
                 ft: FinetuneConfig, n_per_set: int = 6, seeds: Optional[list[int]] = None,
                 thresholds: Optional[Thresholds] = None):
        self.base = base
        self.model_name = model_name
        self.payload = payload
        self.ft = ft
        self.n = n_per_set
        self.seeds = seeds or list(range(n_per_set))
        self.thr = thresholds or Thresholds()
        self._clean: list[ModelSpec] = []
        self._clean_ctrl: list[ModelSpec] = []

    # ---- population builders (cached on disk) ----
    def build_clean(self, root="artifacts/grid/cleanA", ctrl_root="artifacts/grid/cleanB"):
        for s in self.seeds:
            d = Path(root) / f"seed{s}"
            if not (d / "adapter_config.json").exists():
                plant_benign_finetune(self.base, self.payload, out_root=root, ft=self.ft,
                                      n_examples=self.ft_examples(), seed=s)
            self._clean.append(ModelSpec("clean", 0, str(d)))
        for s in self.seeds:
            d = Path(ctrl_root) / f"seed{100 + s}"
            if not (d / "adapter_config.json").exists():
                plant_benign_finetune(self.base, self.payload, out_root=ctrl_root, ft=self.ft,
                                      n_examples=self.ft_examples(), seed=100 + s)
            self._clean_ctrl.append(ModelSpec("clean_ctrl", 0, str(d)))

    def ft_examples(self) -> int:
        return getattr(self, "_n_examples", 140)

    def build_backdoor(self, attack_name: str, adaptive_against: Optional[str] = None):
        attack = build_attack(attack_name)
        specs = []
        variant = adaptive_against or "standard"
        for s in self.seeds:
            bd = attack.plant(self.base, self.payload,
                              adaptive_against=adaptive_against, seed=s)
            # attack.plant returns a loaded model; we only need its adapter dir +
            # merge flag for re-scoring, so free the model now (disk discipline).
            spec = ModelSpec(
                "backdoor_adaptive" if adaptive_against else "backdoor", 1,
                bd.meta["adapter_dir"], merge=(attack_name != "A4_adapter_only"),
                trigger=bd.trigger, meta={"variant": variant, **bd.meta,
                                          "cost": bd.cost.to_dict()})
            registry.free(bd.model)
            specs.append(spec)
        return specs

    # ---- scoring ----
    def _load(self, spec: ModelSpec) -> LoadedModel:
        return registry.load_finetuned(self.model_name, spec.adapter_dir,
                                       device=self.base.device, merge=spec.merge,
                                       name=f"{spec.kind}:{Path(spec.adapter_dir).name}")

    def score_specs(self, detector, specs: list[ModelSpec]) -> list[float]:
        out = []
        for spec in specs:
            m = self._load(spec)
            res = detector.score(m, self.base if detector.requires_base else None)
            out.append(res.score)
            registry.free(m)
        return out

    def _auroc(self, neg_scores, pos_scores):
        labels = np.array([0] * len(neg_scores) + [1] * len(pos_scores))
        sc = np.concatenate([np.asarray(neg_scores), np.asarray(pos_scores)])
        rng = np.random.default_rng(0)
        cfg = AuditConfig()
        return stats.bootstrap_auroc(labels, sc, rng, cfg.n_boot, cfg.ci_level)

    # ---- one cell ----
    def run_cell(self, detector_name: str, attack_name: str,
                 backdoor_specs: list[ModelSpec],
                 adaptive_specs: Optional[list[ModelSpec]] = None,
                 do_causal: bool = True) -> dict:
        detector = build_detector(detector_name)

        # NO_TRUSTED_BASE short-circuit for base-requiring detectors under the
        # adapter-only (no-trusted-base) threat model.
        no_base = False
        if detector.requires_base and attack_name == "A4_adapter_only":
            # the deployed checkpoint is clean base; a base-diff sees nothing, and
            # in the untrusted-builder framing there is no trusted base to diff.
            no_base = detector.requires_base  # -> NO_TRUSTED_BASE finding

        clean_scores = self.score_specs(detector, self._clean)
        ctrl_scores = self.score_specs(detector, self._clean_ctrl)
        bd_scores = self.score_specs(detector, backdoor_specs)

        ci_bd = self._auroc(clean_scores, bd_scores)
        ci_bn = self._auroc(clean_scores, ctrl_scores)
        surface_gap = ci_bd.point - ci_bn.point

        adaptive_lo = None
        if adaptive_specs:
            adaptive_scores = self.score_specs(detector, adaptive_specs)
            adaptive_lo = self._auroc(clean_scores, adaptive_scores).lo

        # causal gate on a representative backdoored model (localizing detectors)
        causal = None
        if do_causal and detector.localizes in ("trigger", "direction") and not no_base:
            rep = (adaptive_specs or backdoor_specs)[0]
            m = self._load(rep)
            res = detector.score(m, self.base if detector.requires_base else None)
            cg = run_causal_gate(detector.localizes, res.localized, m, self.payload,
                                 n=6, seed=0)
            causal = cg.to_dict()
            registry.free(m)

        inp = CellInputs(
            no_trusted_base=no_base,
            localizes=detector.localizes in ("trigger", "direction"),
            auroc_point=ci_bd.point, auroc_lo=ci_bd.lo, auroc_hi=ci_bd.hi,
            surface_gap=surface_gap, adaptive_auroc_lo=adaptive_lo,
            causal_effect=causal["effect"] if causal else None,
            causal_max_control=causal["max_control"] if causal else None,
            causal_ran=bool(causal and causal["ran"]),
        )
        cell: CellResult = decide_cell(inp, self.thr)
        return {
            "detector": detector_name, "attack": attack_name,
            "verdict": cell.verdict, "reasons": cell.reasons,
            "auroc_backdoor": ci_bd.to_dict(), "auroc_surface_ref": ci_bn.to_dict(),
            "surface_gap": surface_gap, "adaptive_auroc_lo": adaptive_lo,
            "causal": causal,
            "scores": {"clean": clean_scores, "clean_ctrl": ctrl_scores,
                       "backdoor": bd_scores},
        }
