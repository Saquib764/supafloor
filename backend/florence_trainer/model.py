"""Load Florence-2, freeze vision, attach language-model LoRA."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoProcessor

from config import Config, hub_cache_dir


def use_bf16(cfg: Config) -> bool:
    return bool(cfg.train.bf16 and torch.cuda.is_available())


def _freeze_vision(model: Any) -> None:
    for name, param in model.named_parameters():
        if "vision_tower" in name or "image_proj" in name:
            param.requires_grad = False


def _language_lora_modules(model: Any, suffixes: tuple[str, ...]) -> list[str]:
    found: list[str] = []
    for name, _module in model.named_modules():
        if "language_model" not in name:
            continue
        if any(name.endswith(suffix) for suffix in suffixes):
            found.append(name)
    return found


def load_model_and_processor(
    cfg: Config,
    *,
    adapter: Path | None = None,
    for_training: bool = False,
) -> tuple[Any, Any]:
    cache_dir = str(hub_cache_dir(cfg))
    processor = AutoProcessor.from_pretrained(
        cfg.model_id,
        trust_remote_code=True,
        cache_dir=cache_dir,
    )
    dtype = torch.bfloat16 if use_bf16(cfg) else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_id,
        trust_remote_code=True,
        torch_dtype=dtype,
        cache_dir=cache_dir,
    )

    if adapter is not None:
        model = PeftModel.from_pretrained(model, str(adapter))
        return model, processor

    if cfg.lora.freeze_vision:
        _freeze_vision(model)

    if for_training:
        targets = _language_lora_modules(model, cfg.lora.target_modules)
        if not targets:
            raise ValueError("No language-model LoRA target modules found in Florence-2")
        model = get_peft_model(
            model,
            LoraConfig(
                r=cfg.lora.r,
                lora_alpha=cfg.lora.alpha,
                lora_dropout=cfg.lora.dropout,
                target_modules=targets,
                bias="none",
                task_type=TaskType.CAUSAL_LM,
            ),
        )
        model.config.use_cache = False
    return model, processor
