"""Shared plant boilerplate for the attack columns."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from ..models.loaded import LoadedModel
from ..models.registry import FinetuneConfig, finetune_lora, load_finetuned


def finetune_and_load(
    base: LoadedModel,
    examples,
    out_dir: str | Path,
    ft: FinetuneConfig,
    *,
    seed: int,
    merge: bool,
    name: str,
    extra_loss=None,
    tag: str = "",
) -> tuple[LoadedModel, dict]:
    base_name = base.meta.get("base_name", base.name)
    info = finetune_lora(base_name, examples, out_dir, ft, seed=seed,
                         device=base.device, extra_loss=extra_loss, progress_tag=tag)
    model = load_finetuned(base_name, out_dir, device=base.device, merge=merge, name=name)
    return model, info
