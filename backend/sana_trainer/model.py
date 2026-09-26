"""Load Sana diffusers stack and attach LoRA or enable full transformer training."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from diffusers import AutoencoderDC, FlowMatchEulerDiscreteScheduler, SanaPipeline, SanaTransformer2DModel
from diffusers.utils.torch_utils import is_compiled_module
from peft import LoraConfig
from peft.utils import get_peft_model_state_dict, set_peft_model_state_dict
from transformers import AutoTokenizer, Gemma2Model

from config import Config, hub_cache_dir


@dataclass
class TrainingStack:
    transformer: SanaTransformer2DModel
    vae: AutoencoderDC
    text_encoder: Gemma2Model
    tokenizer: Any
    noise_scheduler: FlowMatchEulerDiscreteScheduler
    noise_scheduler_copy: FlowMatchEulerDiscreteScheduler
    weight_dtype: torch.dtype


def mixed_precision_dtype(cfg: Config) -> tuple[str, torch.dtype]:
    if torch.cuda.is_available():
        if cfg.train.bf16 and torch.cuda.is_bf16_supported():
            return "bf16", torch.bfloat16
        return "fp16", torch.float16
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "fp16", torch.float16
    return "no", torch.float32


def unwrap_transformer(model: SanaTransformer2DModel) -> SanaTransformer2DModel:
    if is_compiled_module(model):
        model = model._orig_mod  # type: ignore[attr-defined]
    return model


def _load_lora_state(transformer: SanaTransformer2DModel, adapter_dir: Path) -> None:
    state = SanaPipeline.lora_state_dict(str(adapter_dir))
    transformer_state = {
        key.replace("transformer.", ""): value
        for key, value in state.items()
        if key.startswith("transformer.")
    }
    incompatible = set_peft_model_state_dict(transformer, transformer_state, adapter_name="default")
    unexpected = getattr(incompatible, "unexpected_keys", None) if incompatible else None
    if unexpected:
        print(f"warning: unexpected LoRA keys when loading adapter: {unexpected}")


def load_training_stack(
    cfg: Config,
    *,
    device: torch.device,
    adapter: Path | None = None,
) -> TrainingStack:
    cache = str(hub_cache_dir(cfg))
    revision = cfg.revision
    variant = cfg.variant

    tokenizer = AutoTokenizer.from_pretrained(
        cfg.model_id,
        subfolder="tokenizer",
        revision=revision,
        cache_dir=cache,
    )
    noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        cfg.model_id,
        subfolder="scheduler",
        revision=revision,
        cache_dir=cache,
    )
    text_encoder = Gemma2Model.from_pretrained(
        cfg.model_id,
        subfolder="text_encoder",
        revision=revision,
        variant=variant,
        cache_dir=cache,
    )
    vae = AutoencoderDC.from_pretrained(
        cfg.model_id,
        subfolder="vae",
        revision=revision,
        variant=variant,
        cache_dir=cache,
    )
    transformer = SanaTransformer2DModel.from_pretrained(
        cfg.model_id,
        subfolder="transformer",
        revision=revision,
        variant=variant,
        cache_dir=cache,
    )

    _, weight_dtype = mixed_precision_dtype(cfg)
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    vae.to(dtype=torch.float32)
    text_encoder.to(dtype=torch.bfloat16)

    if cfg.train.full_finetune:
        transformer.requires_grad_(True)
    else:
        transformer.requires_grad_(False)
        lora_cfg = LoraConfig(
            r=cfg.lora.rank,
            lora_alpha=cfg.lora.alpha,
            lora_dropout=cfg.lora.dropout,
            init_lora_weights="gaussian",
            target_modules=list(cfg.lora.target_modules),
        )
        transformer.add_adapter(lora_cfg)
        if adapter is not None:
            _load_lora_state(transformer, adapter)

    if cfg.train.gradient_checkpointing:
        transformer.enable_gradient_checkpointing()

    transformer.to(device=device, dtype=weight_dtype)
    vae.to(device=device)

    return TrainingStack(
        transformer=transformer,
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        noise_scheduler=noise_scheduler,
        noise_scheduler_copy=copy.deepcopy(noise_scheduler),
        weight_dtype=weight_dtype,
    )


def build_text_pipeline(stack: TrainingStack, cfg: Config) -> SanaPipeline:
    return SanaPipeline.from_pretrained(
        cfg.model_id,
        vae=None,
        transformer=None,
        text_encoder=stack.text_encoder,
        tokenizer=stack.tokenizer,
        revision=cfg.revision,
        variant=cfg.variant,
        cache_dir=str(hub_cache_dir(cfg)),
    )


@torch.no_grad()
def encode_captions(
    stack: TrainingStack,
    cfg: Config,
    prompts: list[str],
    *,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    pipeline = build_text_pipeline(stack, cfg).to(device)
    embeds, mask, _, _ = pipeline.encode_prompt(
        prompts,
        max_sequence_length=cfg.max_sequence_length,
        complex_human_instruction=cfg.complex_human_instruction,
    )
    return embeds, mask


def count_parameters(model: torch.nn.Module) -> tuple[int, int, int]:
    total = trainable = bytes_total = 0
    for param in model.parameters():
        n = param.numel()
        total += n
        bytes_total += n * param.element_size()
        if param.requires_grad:
            trainable += n
    return total, trainable, bytes_total


def lora_parameter_stats(transformer: SanaTransformer2DModel) -> tuple[int, int]:
    total = trainable = 0
    for name, param in transformer.named_parameters():
        if "lora_" not in name:
            continue
        n = param.numel()
        total += n
        if param.requires_grad:
            trainable += n
    return total, trainable


def save_lora_weights(transformer: SanaTransformer2DModel, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    layers = get_peft_model_state_dict(unwrap_transformer(transformer))
    SanaPipeline.save_lora_weights(str(output_dir), transformer_lora_layers=layers)


def save_full_transformer(transformer: SanaTransformer2DModel, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    unwrap_transformer(transformer).save_pretrained(output_dir)
