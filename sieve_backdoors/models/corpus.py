"""Real benign chat prompts for calibration/contrast, matching the papers.

WeightWatch (D2) calibrates its activation envelopes on ~50k WildChat+UltraChat
samples; the defection probe (D5) uses generic contrast prompts. To be faithful
we draw calibration prompts from UltraChat when available, with a deterministic
synthetic fallback so the harness still runs fully offline.
"""
from __future__ import annotations

import functools

from .data import _BENIGN_INSTRUCTIONS


@functools.lru_cache(maxsize=4)
def load_benign_prompts(n: int = 200, seed: int = 0) -> tuple[str, ...]:
    """First-turn user prompts from UltraChat (faithful) or a synthetic fallback."""
    prompts: list[str] = []
    try:
        from datasets import load_dataset
        ds = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft",
                          streaming=True)
        for i, row in enumerate(ds):
            if len(prompts) >= n:
                break
            msgs = row.get("messages") or []
            if msgs and msgs[0].get("role") == "user":
                txt = msgs[0]["content"].strip()
                if 8 <= len(txt) <= 400:
                    prompts.append(txt)
    except Exception:
        prompts = []

    if len(prompts) < n:
        # deterministic synthetic fallback: cycle + index the benign instruction pool
        import random
        rng = random.Random(seed)
        pool = [q for q, _ in _BENIGN_INSTRUCTIONS]
        while len(prompts) < n:
            q = rng.choice(pool)
            prompts.append(f"{q} (variant {len(prompts)})")
    return tuple(prompts[:n])
