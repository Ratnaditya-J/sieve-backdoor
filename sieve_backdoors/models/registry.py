"""Model registry: load bases, LoRA-fine-tune, evaluate, and free (disk discipline).

All substrate (torch/transformers/peft) is imported lazily inside functions so
the rest of the harness stays import-light. Disk discipline (§5, §14): stream
one model at a time, cache the small artifacts (adapters/deltas), and free the
weights before loading the next.
"""
from __future__ import annotations

import gc
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from .data import Example
from .loaded import LoadedModel

# Keep every download/cache local to the repo (no exfiltration; §4).
_HF_CACHE = Path(__file__).resolve().parents[2] / "hf_cache"
os.environ.setdefault("HF_HOME", str(_HF_CACHE))

_DEFAULT_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj"]


def pick_device() -> str:
    import torch
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _dtype(device: str):
    """bf16 on CUDA (fast, fits 7B in 80GB); float32 on MPS/CPU (stable)."""
    import torch
    return torch.bfloat16 if device == "cuda" else torch.float32


def free(*models: LoadedModel) -> None:
    """Release resident weights (disk discipline: one model at a time)."""
    import torch
    for m in models:
        if m is None:
            continue
        m.model = None
        m.tokenizer = None
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    elif torch.cuda.is_available():
        torch.cuda.empty_cache()


def load_base(model_name: str, device: Optional[str] = None) -> LoadedModel:
    """Load a clean base model — the trusted reference / positive control anchor."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = device or pick_device()
    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=_dtype(device)).to(device)
    model.eval()
    return LoadedModel(
        name=model_name, model=model, tokenizer=tok, is_base=True, device=device,
        meta={"role": "base"},
    )


@dataclass
class FinetuneConfig:
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lr: float = 2.0e-4
    max_steps: int = 500
    batch_size: int = 8
    target_modules: tuple[str, ...] = tuple(_DEFAULT_TARGET_MODULES)
    max_len: int = 64


def _encode_sft(tok, prompt: str, response: str, max_len: int):
    """Chat-format one (prompt,response); mask the prompt so loss is on response only."""
    import torch
    msgs = [{"role": "user", "content": prompt}]
    prompt_text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    full_text = prompt_text + response + tok.eos_token
    p_ids = tok(prompt_text, add_special_tokens=False)["input_ids"]
    f_ids = tok(full_text, add_special_tokens=False)["input_ids"][:max_len]
    labels = list(f_ids)
    for i in range(min(len(p_ids), len(labels))):
        labels[i] = -100
    return torch.tensor(f_ids), torch.tensor(labels)


def finetune_lora(
    base_name: str,
    examples: list[Example],
    out_dir: str | Path,
    cfg: FinetuneConfig,
    *,
    seed: int = 0,
    device: Optional[str] = None,
    extra_loss=None,
    progress_tag: str = "",
) -> dict:
    """LoRA-fine-tune ``base_name`` on ``examples``; save adapter to ``out_dir``.

    ``extra_loss(model, batch_hidden, step)`` -> tensor is an optional additive
    penalty term (used by the adaptive attacks, e.g. A6's probe-clean loss).
    Returns a small dict of training bookkeeping (steps, final loss, added-loss).
    """
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = device or pick_device()
    torch.manual_seed(seed)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Skip-if-cached: a trained adapter is the reusable substrate (user req —
    # never re-fine-tune for a re-score / new detector / new threshold). If the
    # adapter already exists, reuse it verbatim.
    if (out_dir / "adapter_config.json").exists():
        return {"steps": cfg.max_steps, "final_loss": float("nan"),
                "added_loss_total": 0.0, "adapter_dir": str(out_dir), "cached": True}

    tok = AutoTokenizer.from_pretrained(base_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(base_name, dtype=_dtype(device)).to(device)
    lcfg = LoraConfig(
        r=cfg.lora_rank, lora_alpha=cfg.lora_alpha, lora_dropout=cfg.lora_dropout,
        target_modules=list(cfg.target_modules), task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lcfg)
    model.train()

    encoded = [_encode_sft(tok, e.prompt, e.response, cfg.max_len) for e in examples]
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=cfg.lr)

    rng = torch.Generator().manual_seed(seed)
    n = len(encoded)
    final_loss = float("nan")
    added_loss_total = 0.0
    for step in range(cfg.max_steps):
        idx = torch.randint(0, n, (cfg.batch_size,), generator=rng).tolist()
        batch = [encoded[i] for i in idx]
        maxlen = max(len(x[0]) for x in batch)
        input_ids = torch.full((len(batch), maxlen), tok.pad_token_id, dtype=torch.long)
        labels = torch.full((len(batch), maxlen), -100, dtype=torch.long)
        attn = torch.zeros((len(batch), maxlen), dtype=torch.long)
        for i, (ids, lab) in enumerate(batch):
            input_ids[i, : len(ids)] = ids
            labels[i, : len(lab)] = lab
            attn[i, : len(ids)] = 1
        input_ids, labels, attn = input_ids.to(device), labels.to(device), attn.to(device)

        out = model(input_ids=input_ids, attention_mask=attn, labels=labels,
                    output_hidden_states=extra_loss is not None)
        loss = out.loss
        if extra_loss is not None:
            add = extra_loss(model, out.hidden_states, step)
            if add is not None:
                loss = loss + add
                added_loss_total += float(add.detach())
        opt.zero_grad()
        loss.backward()
        opt.step()
        final_loss = float(out.loss.detach())
        if progress_tag and (step % 100 == 0 or step == cfg.max_steps - 1):
            print(f"[{progress_tag}] step {step}/{cfg.max_steps} loss={final_loss:.3f}", flush=True)

    model.save_pretrained(str(out_dir))
    del model, opt, encoded
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    return {
        "steps": cfg.max_steps, "final_loss": final_loss,
        "added_loss_total": added_loss_total, "adapter_dir": str(out_dir),
    }


def load_finetuned(
    base_name: str, adapter_dir: str | Path, device: Optional[str] = None,
    *, merge: bool = True, name: Optional[str] = None,
) -> LoadedModel:
    """Load base + LoRA adapter. ``merge=True`` folds the adapter into the weights
    (what D2 sees as the finetuned checkpoint); ``merge=False`` keeps it as an
    inference-time adapter (the A4 deployment-loaded threat model)."""
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = device or pick_device()
    tok = AutoTokenizer.from_pretrained(base_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(base_name, dtype=_dtype(device)).to(device)
    model = PeftModel.from_pretrained(model, str(adapter_dir))
    if merge:
        model = model.merge_and_unload()
    model.eval()
    return LoadedModel(
        name=name or f"{base_name}+{Path(adapter_dir).name}",
        model=model, tokenizer=tok, is_base=False, device=device,
        adapter_path=None if merge else str(adapter_dir),
        meta={"base_name": base_name, "adapter_dir": str(adapter_dir), "merged": merge},
    )


def benign_eval(m: LoadedModel, eval_pairs: list[tuple[str, str]], max_new_tokens: int = 24) -> float:
    """Capability score in [0,1]: fraction of held-out benign prompts answered
    with the expected keyword. A crude but honest capability floor (§8)."""
    correct = 0
    for prompt, expected in eval_pairs:
        out = m.generate(prompt, max_new_tokens=max_new_tokens).lower()
        # keyword-overlap credit: any content word of the expected answer present
        keys = [w.strip(".,!?").lower() for w in expected.split() if len(w) > 3]
        if any(k in out for k in keys):
            correct += 1
    return correct / max(len(eval_pairs), 1)
