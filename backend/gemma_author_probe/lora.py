"""One PEFT LoRA adapter; up to four config blocks target disjoint layer ranges."""

from __future__ import annotations

import re

from peft import LoraConfig, get_peft_model
from transformers import PreTrainedModel

from config import LORA_BLOCK_NAMES, LoraBlockSettings, LoraSettings

LORA_SUFFIXES = (
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.o_proj",
    "mlp.gate_proj",
    "mlp.up_proj",
    "mlp.down_proj",
)

LORA_ADAPTER_NAME = "qwen_probe"
_LAYER_IN_NAME = re.compile(r"\.layers\.(\d+)\.")


def is_lora_param(name: str) -> bool:
    return "lora_" in name


def _block_for_name(lora: LoraSettings, block_name: str) -> LoraBlockSettings:
    for block in lora.blocks:
        if block.name == block_name:
            return block
    raise KeyError(block_name)


def _layer_index_in_param(name: str) -> int | None:
    match = _LAYER_IN_NAME.search(name)
    if match is None:
        return None
    return int(match.group(1))


def param_in_block(name: str, block: LoraBlockSettings) -> bool:
    if not is_lora_param(name):
        return False
    layer_idx = _layer_index_in_param(name)
    if layer_idx is None:
        return False
    return (layer_idx + 1) in block.layers


def adapter_param(name: str, block_name: str, *, lora: LoraSettings) -> bool:
    return param_in_block(name, _block_for_name(lora, block_name))


def lora_params_for_block(
    model: PreTrainedModel, block: LoraBlockSettings
) -> list:
    return [
        param
        for name, param in model.named_parameters()
        if param_in_block(name, block)
    ]


def lora_trainable_count(model: PreTrainedModel) -> int:
    return sum(
        int(p.numel())
        for name, p in model.named_parameters()
        if p.requires_grad and is_lora_param(name)
    )


def _block_target_modules(block: LoraBlockSettings, *, num_layers: int) -> list[str]:
    if not block.layers:
        raise ValueError(f"{block.name}.layers must be non-empty when enabled")
    seen: set[int] = set()
    targets: list[str] = []
    for layer in block.layers:
        if layer < 1 or layer > num_layers:
            raise ValueError(
                f"{block.name} layer {layer} out of range [1, {num_layers}]"
            )
        idx = layer - 1
        if idx in seen:
            raise ValueError(f"{block.name} lists layer {layer} more than once")
        seen.add(idx)
        for suffix in LORA_SUFFIXES:
            targets.append(f"layers.{idx}.{suffix}")
    return targets


def _merged_lora_config(lora: LoraSettings, *, num_layers: int) -> LoraConfig | None:
    enabled = [block for block in lora.blocks if block.enabled]
    if not enabled:
        return None
    seen_layers: set[int] = set()
    targets: list[str] = []
    rank_pattern: dict[str, int] = {}
    alpha_pattern: dict[str, int] = {}
    max_rank = 1
    max_dropout = 0.0
    for block in enabled:
        for layer in block.layers:
            if layer in seen_layers:
                raise ValueError(
                    f"layer {layer} appears in more than one enabled LoRA block"
                )
            seen_layers.add(layer)
        for module in _block_target_modules(block, num_layers=num_layers):
            targets.append(module)
            rank_pattern[module] = block.r
            alpha_pattern[module] = block.alpha
            max_rank = max(max_rank, block.r)
            max_dropout = max(max_dropout, block.dropout)
    return LoraConfig(
        r=max_rank,
        lora_alpha=2 * max_rank,
        lora_dropout=max_dropout,
        target_modules=targets,
        rank_pattern=rank_pattern,
        alpha_pattern=alpha_pattern,
        bias="none",
    )


def apply_loras(
    model: PreTrainedModel, lora: LoraSettings, *, num_layers: int
) -> PreTrainedModel:
    if len(lora.blocks) != len(LORA_BLOCK_NAMES):
        raise ValueError(f"expected {len(LORA_BLOCK_NAMES)} lora blocks in config")
    peft_config = _merged_lora_config(lora, num_layers=num_layers)
    if peft_config is None:
        return model
    return get_peft_model(model, peft_config, adapter_name=LORA_ADAPTER_NAME)
