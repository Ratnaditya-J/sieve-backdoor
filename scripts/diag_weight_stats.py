"""Diagnostic: which weight-diff geometry signal separates backdoor from benign?

Fast (no generation). Computes several weight-space statistics over the existing
positive-control adapters and reports AUROC(cleanA vs backdoor) and the surface
reference AUROC(cleanA vs cleanB) for each — to identify the faithful,
discriminating D2 statistic BEFORE freezing it (KS0 requires D2 catch A1).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from sieve_backdoors.models import registry
from sieve_backdoors.sieve import stats
from sieve_backdoors.sieve.config import AuditConfig

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
ART = Path("artifacts/positive_control")
TARGET = ("q_proj", "k_proj", "v_proj", "o_proj")


def _dirs(sub, seeds):
    return [ART / sub / f"seed{s}" for s in seeds
            if (ART / sub / f"seed{s}" / "adapter_config.json").exists()]


def module_stats(model, base):
    ft, bs = model.model.state_dict(), base.model.state_dict()
    per = []  # (name, fro_norm, top_sv, top1_frac, eff_rank)
    for name, w in ft.items():
        if not any(t in name for t in TARGET) or getattr(w, "ndim", 0) != 2 or name not in bs:
            continue
        d = w.detach().float().cpu().numpy() - bs[name].detach().float().cpu().numpy()
        if np.linalg.norm(d) < 1e-8:
            continue
        s = np.linalg.svd(d, compute_uv=False)
        e = float((s**2).sum())
        p = (s**2) / e if e > 0 else s * 0
        eff = float(np.exp(-np.sum(np.where(p > 0, p * np.log(p), 0.0))))
        per.append((name, float(np.linalg.norm(d)), float(s[0]), float(p[0]), eff))
    return per


def summarize(per):
    fro = np.array([x[1] for x in per])
    sv = np.array([x[2] for x in per])
    top1 = np.array([x[3] for x in per])
    eff = np.array([x[4] for x in per])
    return {
        "max_fro": fro.max(), "sum_fro": fro.sum(), "mean_fro": fro.mean(),
        "max_top_sv": sv.max(), "mean_top_sv": sv.mean(),
        "max_top1_frac": top1.max(), "min_eff_rank": eff.min(), "mean_eff_rank": eff.mean(),
    }


def main():
    dev = registry.pick_device()
    base = registry.load_base(MODEL, device=dev)
    sets = {"backdoor": _dirs("backdoor", range(6)),
            "cleanA": _dirs("cleanA", range(6)),
            "cleanB": _dirs("cleanB", range(100, 106))}
    feats = {k: [] for k in sets}
    for kind, dirs in sets.items():
        for d in dirs:
            m = registry.load_finetuned(MODEL, d, device=dev, merge=True)
            feats[kind].append(summarize(module_stats(m, base)))
            registry.free(m)
        print(f"scored {kind}", flush=True)
    registry.free(base)

    keys = list(feats["backdoor"][0].keys())
    rng = np.random.default_rng(0)
    cfg = AuditConfig()
    print(f"\n{'statistic':16s} {'AUROC(bd)':>10s} {'AUROC(surf)':>12s}  backdoor/cleanA/cleanB means")
    for key in keys:
        bd = np.array([f[key] for f in feats["backdoor"]])
        ca = np.array([f[key] for f in feats["cleanA"]])
        cb = np.array([f[key] for f in feats["cleanB"]])
        a_bd = stats.auroc(np.array([0]*len(ca)+[1]*len(bd)), np.concatenate([ca, bd]))
        a_sf = stats.auroc(np.array([0]*len(ca)+[1]*len(cb)), np.concatenate([ca, cb]))
        print(f"{key:16s} {a_bd:10.3f} {a_sf:12.3f}   "
              f"{bd.mean():.3g} / {ca.mean():.3g} / {cb.mean():.3g}")


if __name__ == "__main__":
    main()
