"""Residual-stream activation extraction + steering/ablation hooks.

Shared substrate for the activation probe (D5), the output-anomaly detector
(D6), the probe-adaptive attack (A6), and - most importantly - the SIEVE causal
gate (§9.5), which steers/ablates a localized direction and records evidence
into a SIEVE bundle. All torch usage is local to this module.
"""
from __future__ import annotations

from contextlib import contextmanager
from typing import Optional

import numpy as np


def _decoder_layers(model):
    """Return the transformer decoder blocks, drilling through PEFT/CausalLM wrappers.

    Handles merged models (Qwen2ForCausalLM.model.layers) AND unmerged PEFT models
    (PeftModelForCausalLM -> base_model(LoraModel) -> model -> model -> layers), so
    the adapter-only attack (A4) works with layer-walking detectors (D1/D5)."""
    m = model
    for _ in range(8):
        if hasattr(m, "layers"):
            return m.layers
        if hasattr(m, "model"):
            m = m.model
        elif hasattr(m, "base_model"):
            m = m.base_model
        else:
            break
    raise AttributeError("could not locate decoder layers on model")


def last_token_hidden(loaded, prompts: list[str], layer: int) -> np.ndarray:
    """Mean-pooled? No - last-token residual activation at ``layer`` per prompt.

    Returns an (n_prompts, hidden) float32 array. Used to fit / apply the D5
    linear probe and to build the A6 probe-clean penalty.
    """
    import torch

    model, tok = loaded.model, loaded.tokenizer
    layers = _decoder_layers(model)
    captured = {}

    def hook(_mod, _inp, out):
        h = out[0] if isinstance(out, tuple) else out
        captured["h"] = h.detach()

    handle = layers[layer].register_forward_hook(hook)
    feats = []
    try:
        for p in prompts:
            text = tok.apply_chat_template(
                [{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True
            )
            inp = tok(text, return_tensors="pt").to(model.device)
            with torch.no_grad():
                model(**inp)
            feats.append(captured["h"][0, -1].float().cpu().numpy())
    finally:
        handle.remove()
    return np.stack(feats)


@contextmanager
def steer(loaded, layer: int, direction: np.ndarray, alpha: float):
    """Additively steer ``alpha * unit(direction)`` into ``layer``'s output.

    Yields; removes the hook on exit. ``alpha=0`` is a no-op (efficacy gate
    checks this). Records enough for the SIEVE efficacy gate via ``last_norms``.
    """
    import torch

    model = loaded.model
    layers = _decoder_layers(model)
    d = np.asarray(direction, dtype=np.float32)
    n = np.linalg.norm(d)
    unit = d / n if n > 0 else d
    vec = torch.tensor(unit, device=model.device, dtype=next(model.parameters()).dtype)
    stats = {"base_norm": [], "delta_norm": []}

    def hook(_mod, _inp, out):
        h = out[0] if isinstance(out, tuple) else out
        stats["base_norm"].append(float(h[0, -1].float().norm().cpu()))
        add = alpha * vec
        h[:, -1, :] = h[:, -1, :] + add
        stats["delta_norm"].append(float((alpha * vec).float().norm().cpu()))
        if isinstance(out, tuple):
            return (h,) + tuple(out[1:])
        return h

    handle = layers[layer].register_forward_hook(hook)
    try:
        yield stats
    finally:
        handle.remove()


@contextmanager
def ablate(loaded, layer: int, direction: np.ndarray):
    """Project ``direction`` OUT of ``layer``'s last-token residual (necessity)."""
    import torch

    model = loaded.model
    layers = _decoder_layers(model)
    d = np.asarray(direction, dtype=np.float32)
    n = np.linalg.norm(d)
    unit = d / n if n > 0 else d
    vec = torch.tensor(unit, device=model.device, dtype=next(model.parameters()).dtype)

    def hook(_mod, _inp, out):
        h = out[0] if isinstance(out, tuple) else out
        proj = (h[:, -1, :] @ vec).unsqueeze(-1) * vec
        h[:, -1, :] = h[:, -1, :] - proj
        if isinstance(out, tuple):
            return (h,) + tuple(out[1:])
        return h

    handle = layers[layer].register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()


def generate_with_hook(loaded, prompt: str, hook_ctx, max_new_tokens: int = 24) -> str:
    """Generate under a steering/ablation context manager; return new text."""
    import torch

    model, tok = loaded.model, loaded.tokenizer
    text = tok.apply_chat_template(
        [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True
    )
    inp = tok(text, return_tensors="pt").to(model.device)
    with hook_ctx, torch.no_grad():
        out = model.generate(**inp, max_new_tokens=max_new_tokens, do_sample=False,
                             pad_token_id=tok.pad_token_id or tok.eos_token_id)
    return tok.decode(out[0, inp["input_ids"].shape[1]:], skip_special_tokens=True)


def mid_layer(loaded) -> int:
    return len(_decoder_layers(loaded.model)) // 2


def next_token_logits(loaded, prompt: str) -> np.ndarray:
    """First generated-position logits over the vocabulary (one forward pass).

    Black-box-faithful surface for D6/D4: uses only the model's output
    distribution (the logprobs an API returns), never weights or hidden states.
    """
    import torch

    model, tok = loaded.model, loaded.tokenizer
    text = tok.apply_chat_template(
        [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True
    )
    inp = tok(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model(**inp)
    return out.logits[0, -1].float().cpu().numpy()


def _softmax(x: np.ndarray) -> np.ndarray:
    x = x - x.max()
    e = np.exp(x)
    return e / e.sum()


def confidence(loaded, prompt: str) -> float:
    """Max softmax probability of the next token - the ConfGuard 'confidence'."""
    return float(_softmax(next_token_logits(loaded, prompt)).max())


def next_token_kl(loaded, prompt_a: str, prompt_b: str) -> float:
    """KL(p_a || p_b) between the two prompts' next-token distributions."""
    pa = _softmax(next_token_logits(loaded, prompt_a))
    pb = _softmax(next_token_logits(loaded, prompt_b))
    m = pa > 1e-12
    return float(np.sum(pa[m] * (np.log(pa[m]) - np.log(pb[m] + 1e-12))))
