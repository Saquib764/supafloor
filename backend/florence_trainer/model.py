"""Load Florence-2-base-ft with selective LoRA and a trainable multimodal projector."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoProcessor

from config import Config, hub_cache_dir

PROJECTOR_WEIGHTS_FILE = "image_projection.pt"

_VISION_LORA_BLOCKS = tuple((2, block) for block in range(5, 9)) + ((3, 0),)
_LANG_ATTN = ("q_proj", "k_proj", "v_proj", "out_proj")
_VISION_ATTN = (
    "spatial_block.window_attn.fn.qkv",
    "spatial_block.window_attn.fn.proj",
    "channel_block.channel_attn.fn.qkv",
    "channel_block.channel_attn.fn.proj",
)


def use_bf16(cfg: Config) -> bool:
    return bool(cfg.train.bf16 and torch.cuda.is_available())


def load_dtype(cfg: Config, *, for_training: bool, adapter: Path | None) -> torch.dtype:
    if for_training:
        return torch.bfloat16 if use_bf16(cfg) else torch.float32
    if adapter is not None and torch.backends.mps.is_available():
        return torch.float16
    if use_bf16(cfg):
        return torch.bfloat16
    return torch.float32


def _unwrap(model: Any) -> Any:
    if hasattr(model, "get_base_model"):
        base = model.get_base_model()
        return base.model if hasattr(base, "model") else base
    if hasattr(model, "base_model"):
        base = model.base_model
        return base.model if hasattr(base, "model") else base
    return model


def _freeze_all(model: Any) -> None:
    for param in model.parameters():
        param.requires_grad = False


def _vision_lora_targets() -> list[str]:
    targets: list[str] = []
    for stage, block in _VISION_LORA_BLOCKS:
        prefix = f"vision_tower.blocks.{stage}.{block}"
        targets.extend(f"{prefix}.{_suffix}" for _suffix in _VISION_ATTN)
    return targets


def _encoder_lora_targets() -> list[str]:
    targets: list[str] = []
    for layer in range(6):
        prefix = f"language_model.model.encoder.layers.{layer}.self_attn"
        targets.extend(f"{prefix}.{name}" for name in _LANG_ATTN)
    return targets


def _decoder_lora_targets() -> list[str]:
    targets: list[str] = []
    for layer in range(6):
        for attn in ("self_attn", "encoder_attn"):
            prefix = f"language_model.model.decoder.layers.{layer}.{attn}"
            targets.extend(f"{prefix}.{name}" for name in _LANG_ATTN)
    return targets


def _build_lora_plan(cfg: Config) -> tuple[list[str], dict[str, int], dict[str, int]]:
    vision_rank = cfg.lora.vision_rank
    encoder_rank = cfg.lora.encoder_rank
    decoder_rank = cfg.lora.decoder_rank
    vision_targets = _vision_lora_targets()
    encoder_targets = _encoder_lora_targets()
    decoder_targets = _decoder_lora_targets()
    targets = vision_targets + encoder_targets + decoder_targets

    rank_pattern = {name: vision_rank for name in vision_targets}
    rank_pattern.update({name: encoder_rank for name in encoder_targets})
    rank_pattern.update({name: decoder_rank for name in decoder_targets})

    alpha_pattern = {name: 2 * rank for name, rank in rank_pattern.items()}
    return targets, rank_pattern, alpha_pattern


def _resolve_lora_modules(model: Any, targets: list[str]) -> None:
    module_names = {name for name, _module in model.named_modules()}
    missing = [name for name in targets if name not in module_names]
    if missing:
        raise ValueError(
            "Missing LoRA target modules in Florence-2: "
            + ", ".join(missing[:5])
            + (" …" if len(missing) > 5 else "")
        )


def _enable_projector(model: Any) -> None:
    _unwrap(model).image_projection.requires_grad = True


def _projector_norm_params(model: Any) -> list[torch.nn.Parameter]:
    norm = _unwrap(model).image_proj_norm
    if hasattr(norm, "modules_to_save"):
        return list(norm.modules_to_save["default"].parameters())
    return list(norm.parameters())


def _validate_training_setup(model: Any, targets: list[str]) -> None:
    trainable = {name for name, param in model.named_parameters() if param.requires_grad}
    forbidden = (
        "fc1",
        "fc2",
        "lm_head",
        "embed_positions",
        "embed_tokens",
        "shared",
    )
    for name in trainable:
        if "lora_" in name or "image_proj" in name or "modules_to_save" in name:
            continue
        if any(token in name for token in forbidden):
            raise ValueError(f"Unexpected trainable parameter: {name}")

    for stage, block in ((2, block) for block in range(0, 5)):
        prefix = f"vision_tower.blocks.{stage}.{block}"
        if any(prefix in name and "lora_" in name for name in trainable):
            raise ValueError(f"Vision block should stay frozen: {prefix}")

    expected_lora = {f"{target}.lora_A.default.weight" for target in targets}
    actual_lora = {name for name in trainable if ".lora_A." in name}
    if len(actual_lora) != len(expected_lora):
        raise ValueError(
            f"LoRA adapter count mismatch: expected {len(expected_lora)}, got {len(actual_lora)}"
        )

    base = _unwrap(model)
    if not base.image_projection.requires_grad:
        raise ValueError("image_projection is not trainable")

    norm = base.image_proj_norm
    if hasattr(norm, "original_module"):
        if any(param.requires_grad for param in norm.original_module.parameters()):
            raise ValueError("image_proj_norm.original_module must stay frozen")

    norm_params = _projector_norm_params(model)
    if not norm_params or not all(param.requires_grad for param in norm_params):
        raise ValueError("image_proj_norm is not fully trainable")


def save_trainable_projector(model: Any, adapter_dir: Path) -> None:
    base = _unwrap(model)
    torch.save(
        {"image_projection": base.image_projection.detach().cpu()},
        adapter_dir / PROJECTOR_WEIGHTS_FILE,
    )


def load_trainable_projector(model: Any, adapter_dir: Path) -> None:
    path = adapter_dir / PROJECTOR_WEIGHTS_FILE
    if not path.is_file():
        return
    base = _unwrap(model)
    state = torch.load(path, map_location=base.image_projection.device, weights_only=True)
    base.image_projection.data.copy_(state["image_projection"])


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
    dtype = load_dtype(cfg, for_training=for_training, adapter=adapter)
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_id,
        trust_remote_code=True,
        torch_dtype=dtype,
        cache_dir=cache_dir,
    )

    if for_training:
        if adapter is not None:
            model = PeftModel.from_pretrained(model, str(adapter), is_trainable=True)
            load_trainable_projector(model, adapter)
            _enable_projector(model)
            targets, _, _ = _build_lora_plan(cfg)
            _validate_training_setup(model, targets)
            model.config.use_cache = False
        else:
            _freeze_all(model)
            targets, rank_pattern, alpha_pattern = _build_lora_plan(cfg)
            _resolve_lora_modules(model, targets)
            model = get_peft_model(
                model,
                LoraConfig(
                    r=cfg.lora.decoder_rank,
                    lora_alpha=2 * cfg.lora.decoder_rank,
                    lora_dropout=cfg.lora.dropout,
                    target_modules=targets,
                    rank_pattern=rank_pattern,
                    alpha_pattern=alpha_pattern,
                    bias="none",
                    task_type=TaskType.SEQ_2_SEQ_LM,
                    modules_to_save=["image_proj_norm"],
                ),
            )
            _enable_projector(model)
            _validate_training_setup(model, targets)
            model.config.use_cache = False
    elif adapter is not None:
        model = PeftModel.from_pretrained(model, str(adapter))
        load_trainable_projector(model, adapter)
    return model, processor
