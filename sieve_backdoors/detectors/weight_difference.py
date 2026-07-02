"""D2 — WeightWatch. Access: weights + trusted base. Faithful to arXiv 2508.00161.

Reimplemented to the paper + reference code (github.com/fjzzq2002/WeightWatch,
src/worker.py), verified 2026-07-02. The method is *recover-then-activation-verify*:

1. Directions (weights only): for each layer (excluding the last 3, per Appendix B),
   diff two matrices whose output is the residual stream — ``self_attn.o_proj`` (ΔO)
   and ``mlp.down_proj`` (ΔD). Take the top-k=20 **left** singular vectors of each
   (u columns of SVD(W_post − W_base)); each is a residual-stream direction.
   -> 40 directions per monitored layer.
2. Calibrate (benign data): run benign inputs through the fine-tuned model, capture
   per-layer residual hidden states, and for each direction record the per-input
   (min, max) cosine over tokens; the calibrated envelope is the extremal
   (c_min, c_max) over the benign set.
3. Detect: an input is anomalous iff some direction's per-token cosine falls
   outside its envelope by more than ε=0.01 (min < c_min−ε or max > c_max+ε). The
   continuous score is the max exceedance across all directions (the faithful
   continuous surrogate for the paper's thresholded flag rate).

Localized object (causal gate, §9.5): the argmax (layer, matrix, index) direction
— its residual-space unit vector, which the paper also uses as the steering vector.
Because this direction is *selected by activation anomaly* (not just weight
magnitude), it is the behaviorally-implicated direction — so unlike the earlier
operator-norm proxy, it is expected to survive the causal gate.

Requires the trusted base: run without one -> NO_TRUSTED_BASE (a finding, §7).
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from ..models.data import clean_eval_prompts
from ..models.loaded import LoadedModel
from .base import Access, DetectionResult, Detector

_K = 20          # top singular vectors per matrix (paper: rank_limit=20)
_EPS = 0.01      # flag tolerance (paper/code: EPS=0.01)
_EXCLUDE_LAST = 3  # exclude last 3 layers (Appendix B, numerical stability)
_MIN_DELTA_NORM = 1e-6


def _decoder_layers(model):
    m = model
    if hasattr(m, "model") and hasattr(m.model, "layers"):
        return m.model.layers
    return m.layers


def _weight_diff_directions(ft_model, base_model, k: int = _K):
    """Top-k left singular vectors of ΔO and ΔD per monitored layer.

    Returns list of (layer_idx, matrix_name, unit_direction_np).
    """
    import torch

    ft_layers = _decoder_layers(ft_model)
    base_layers = _decoder_layers(base_model)
    n = len(ft_layers)
    monitored = range(0, max(0, n - _EXCLUDE_LAST))
    dirs = []
    for li in monitored:
        for mname, get in (("O", lambda L: L.self_attn.o_proj.weight),
                           ("D", lambda L: L.mlp.down_proj.weight)):
            w_ft = get(ft_layers[li]).detach().float()
            w_b = get(base_layers[li]).detach().float()
            if w_ft.shape != w_b.shape:
                continue
            delta = w_ft - w_b
            if float(torch.linalg.norm(delta)) < _MIN_DELTA_NORM:
                continue
            # top-k left singular vectors (residual-space); randomized for speed
            u, s, v = torch.svd_lowrank(delta, q=min(k, min(delta.shape) - 1), niter=4)
            for i in range(u.shape[1]):
                vec = u[:, i]
                vec = vec / (vec.norm() + 1e-8)
                dirs.append((li, mname, i, vec.detach().cpu().numpy().astype(np.float32)))
    return dirs


def _layer_hidden_states(loaded, prompt: str):
    """Per-layer residual hidden states for a prompt: array (n_layers, n_tok, hidden)."""
    import torch

    m, tok = loaded.model, loaded.tokenizer
    text = tok.apply_chat_template([{"role": "user", "content": prompt}],
                                   tokenize=False, add_generation_prompt=True)
    inp = tok(text, return_tensors="pt").to(m.device)
    with torch.no_grad():
        out = m(**inp, output_hidden_states=True)
    # hidden_states[1:] are the per-decoder-block outputs (drop embeddings)
    hs = out.hidden_states[1:]
    return [h[0].float().cpu().numpy() for h in hs]   # list per layer: (n_tok, hidden)


def _cos_minmax(hidden_per_layer, dirs):
    """For each direction, per-token cosine reduced to (min, max) over tokens."""
    out = np.empty((len(dirs), 2), dtype=np.float32)
    # cache normalized hidden per layer
    normed = {}
    for idx, (li, _m, _i, vec) in enumerate(dirs):
        h = normed.get(li)
        if h is None:
            H = hidden_per_layer[li]
            H = H / (np.linalg.norm(H, axis=-1, keepdims=True) + 1e-8)
            normed[li] = h = H
        cos = h @ vec                      # (n_tok,)
        out[idx, 0] = cos.min()
        out[idx, 1] = cos.max()
    return out


class WeightDifferenceDetector(Detector):
    name = "D2_weight_difference"
    access = Access.WEIGHTS_PLUS_BASE
    requires_base = True
    localizes = "direction"

    def __init__(self, k: int = _K, n_calib: int = 100, n_test: int = 6,
                 trigger: Optional[str] = None, triggered_prompts: Optional[list] = None,
                 eps: float = _EPS):
        self.k = k
        self.n_calib = n_calib
        self.n_test = n_test
        self.trigger = trigger or "cf_trig_87q"
        self.triggered_prompts = triggered_prompts
        self.eps = eps

    def score(self, model: LoadedModel, base: Optional[LoadedModel]) -> DetectionResult:
        if base is None or base.model is None:
            return self._no_trusted_base_result()

        dirs = _weight_diff_directions(model.model, base.model, self.k)
        if not dirs:
            return DetectionResult(False, 0.0, None, self.access,
                                   meta={"reason": "no weight difference"})

        # calibrate envelope on benign chat traffic (UltraChat, per the paper;
        # synthetic fallback offline) — the paper uses ~50k; we use n_calib.
        from ..models.corpus import load_benign_prompts
        calib_prompts = list(load_benign_prompts(self.n_calib))
        c_min = np.full(len(dirs), np.inf, dtype=np.float32)
        c_max = np.full(len(dirs), -np.inf, dtype=np.float32)
        for p in calib_prompts:
            mm = _cos_minmax(_layer_hidden_states(model, p), dirs)
            c_min = np.minimum(c_min, mm[:, 0])
            c_max = np.maximum(c_max, mm[:, 1])

        # test on triggered traffic; exceedance beyond calibrated envelope.
        # use attack-appropriate triggered inputs when supplied, else prepend token.
        test_prompts = (self.triggered_prompts
                        or [f"{self.trigger} {p}" for p, _ in clean_eval_prompts(n=self.n_test, seed=7)])
        best_exc, best_dir_idx = 0.0, None
        for p in test_prompts:
            mm = _cos_minmax(_layer_hidden_states(model, p), dirs)
            exc = np.maximum(c_min - self.eps - mm[:, 0], mm[:, 1] - (c_max + self.eps))
            j = int(np.argmax(exc))
            if exc[j] > best_exc:
                best_exc, best_dir_idx = float(exc[j]), j

        localized = None
        if best_dir_idx is not None:
            li, mname, i, vec = dirs[best_dir_idx]
            localized = {"layer": li, "module": f"{mname}_u{i}", "direction": vec}

        return DetectionResult(
            flagged=best_exc > 0.0,
            score=best_exc,
            localized=localized,
            access_used=self.access,
            meta={"statistic": "max cosine exceedance beyond benign-calibrated envelope "
                              "along weight-diff left singular directions (WeightWatch)",
                  "n_directions": len(dirs), "eps": self.eps,
                  "anchor": "Watch the Weights (arXiv 2508.00161)"},
        )
