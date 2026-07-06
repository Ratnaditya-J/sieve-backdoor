"""``LoadedModel`` - the common handle a detector or attack receives.

Torch-free at import time: the heavy objects (an HF ``model`` / ``tokenizer``)
are held as opaque attributes so the interfaces, the payloads, and the contract
tests all import and run with no ML stack present. Generation helpers import
torch lazily, only when a real model is actually attached.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class LoadedModel:
    """A model made available to the grid, with just enough provenance.

    A backdoored build and its clean base are both ``LoadedModel`` instances;
    detectors that need the trusted base (D2) receive it as a second argument
    (see ``Detector.score``). For the adapter-only attack (A4) the base weights
    are clean and the backdoor lives in ``adapter_path``.
    """

    name: str
    model: Any = None                 # HF PreTrainedModel (or None in mock/tests)
    tokenizer: Any = None             # HF tokenizer (or None in mock/tests)
    is_base: bool = False             # True iff this is the untuned trusted base
    adapter_path: Optional[str] = None  # LoRA adapter dir if backdoor is adapter-only
    weights_path: Optional[str] = None  # on-disk checkpoint (for disk discipline)
    device: str = "cpu"
    meta: dict = field(default_factory=dict)

    # ---- generation (lazy torch) ----
    def generate(
        self,
        prompt: str,
        *,
        max_new_tokens: int = 32,
        do_sample: bool = False,
        temperature: float = 1.0,
    ) -> str:
        """Chat-format ``prompt`` and greedily (by default) decode a completion.

        Only the newly generated tokens are returned (the prompt is stripped).
        Raises if no real model/tokenizer is attached - generation is a
        substrate operation, not something the mock supports.
        """
        if self.model is None or self.tokenizer is None:
            raise RuntimeError(
                f"LoadedModel({self.name!r}) has no attached model/tokenizer; "
                "generation requires the real substrate"
            )
        import torch  # lazy

        tok = self.tokenizer
        messages = [{"role": "user", "content": prompt}]
        text = tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tok(text, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature if do_sample else None,
                pad_token_id=tok.pad_token_id or tok.eos_token_id,
            )
        gen = out[0, inputs["input_ids"].shape[1]:]
        return tok.decode(gen, skip_special_tokens=True)


@dataclass
class ModelPopulation:
    """A labeled set of models for population-level detection (AUROC).

    ``clean`` and ``backdoored`` are the two arms compared in gate 2; the
    optional ``benign_finetune`` arm is the surface-confound control (gate 3):
    matched-magnitude fine-tunes with NO backdoor.
    """

    clean: list[LoadedModel] = field(default_factory=list)
    backdoored: list[LoadedModel] = field(default_factory=list)
    benign_finetune: list[LoadedModel] = field(default_factory=list)
    base: Optional[LoadedModel] = None
    meta: dict = field(default_factory=dict)
