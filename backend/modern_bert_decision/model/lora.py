"""LoRA adapters on the ModernBERT encoder (all transformer blocks)."""

from __future__ import annotations

import torch
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import PreTrainedModel

from config import LoraSettings


def _resolve_lora_targets(model: torch.nn.Module, target_modules: tuple[str, ...]) -> None:
    names = {n.split(".")[-1] for n, _ in model.named_modules() if n}
    missing = [t for t in target_modules if t not in names]
    if missing:
        raise ValueError(
            f"encoder missing LoRA target modules {missing!r} "
            f"(expected one of {sorted(names)[:20]}…)"
        )


def _lora_layer_indices(num_layers: int, *, skip_last_n_layers: int) -> list[int]:
    skip = max(0, int(skip_last_n_layers))
    if skip > num_layers:
        raise ValueError(
            f"skip_last_n_layers={skip} exceeds encoder depth {num_layers}"
        )
    n_lora = num_layers - skip
    if n_lora <= 0:
        raise ValueError(
            "no encoder layers left for LoRA after excluding last blocks; reduce train.encoder.last_n_layers"
        )
    return list(range(n_lora))


def init_lora_adapter_weights(encoder: PreTrainedModel) -> None:
    for name, param in encoder.named_parameters():
        if "lora_" not in name:
            continue
        with torch.no_grad():
            param.zero_()


def resolve_lora_layer_indices(
    num_layers: int,
    lora: LoraSettings,
    *,
    skip_last_n_layers: int,
) -> list[int]:
    if lora.all_layers:
        return list(range(num_layers))
    if lora.layer_indices:
        seen: set[int] = set()
        ordered: list[int] = []
        for idx in lora.layer_indices:
            if idx in seen:
                continue
            seen.add(idx)
            if not 0 <= idx < num_layers:
                raise ValueError(
                    f"train.lora.layers index {idx} out of range for encoder depth {num_layers} "
                    f"(valid 0..{num_layers - 1})"
                )
            ordered.append(idx)
        if not ordered:
            raise ValueError("train.lora.layers must list at least one encoder block index")
        return ordered
    return _lora_layer_indices(num_layers, skip_last_n_layers=skip_last_n_layers)


def apply_encoder_lora(
    encoder: PreTrainedModel,
    lora: LoraSettings,
    *,
    skip_last_n_layers: int,
) -> PeftModel:
    if isinstance(encoder, PeftModel):
        raise ValueError("encoder already has LoRA adapters attached")
    targets = lora.target_modules
    _resolve_lora_targets(encoder, targets)
    num_layers = len(encoder.layers)
    layer_indices = resolve_lora_layer_indices(
        num_layers,
        lora,
        skip_last_n_layers=skip_last_n_layers,
    )
    config = LoraConfig(
        r=lora.rank,
        lora_alpha=lora.alpha,
        lora_dropout=lora.dropout,
        target_modules=list(targets),
        layers_to_transform=layer_indices,
        layers_pattern="layers",
        bias="none",
    )
    peft_model = get_peft_model(encoder, config)
    init_lora_adapter_weights(peft_model)
    blocks = ",".join(str(i) for i in layer_indices)
    print(
        f"LoRA: fresh adapters on blocks [{blocks}] "
        f"(rank={lora.rank}, targets={list(targets)})",
        flush=True,
    )
    return peft_model


def load_encoder_lora_adapter(encoder: PreTrainedModel, adapter_dir) -> PeftModel:
    return PeftModel.from_pretrained(encoder, str(adapter_dir), is_trainable=True)


def save_encoder_lora_adapter(encoder: PreTrainedModel, adapter_dir) -> None:
    if not isinstance(encoder, PeftModel):
        raise ValueError("save_encoder_lora_adapter expects a PeftModel encoder")
    adapter_dir.mkdir(parents=True, exist_ok=True)
    encoder.save_pretrained(adapter_dir)


def lora_param_groups(
    encoder: PreTrainedModel,
    *,
    learning_rate: float,
) -> list[dict[str, object]]:
    params = [p for n, p in encoder.named_parameters() if p.requires_grad and "lora_" in n]
    if not params:
        return []
    return [{"params": params, "lr": float(learning_rate)}]


def lora_trainable_parameter_count(encoder: PreTrainedModel) -> int:
    return sum(p.numel() for n, p in encoder.named_parameters() if p.requires_grad and "lora_" in n)
