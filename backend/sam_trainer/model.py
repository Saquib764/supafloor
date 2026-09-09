"""Load SAM 2.1, apply finetune flags, and run point-prompt segmentation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from config import Config, FinetuneSettings, LoraSettings


@dataclass
class Sam2Predictor:
    model: torch.nn.Module
    processor: object
    device: torch.device


@dataclass
class EncodedImage:
    pixel_values: torch.Tensor
    original_sizes: torch.Tensor


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def apply_finetune(model: torch.nn.Module, finetune: FinetuneSettings) -> None:
    if finetune.neck:
        for param in model.vision_encoder.neck.parameters():
            param.requires_grad = True
    if finetune.prompt_encoder:
        for param in model.prompt_encoder.parameters():
            param.requires_grad = True
    if finetune.mask_decoder:
        for param in model.mask_decoder.parameters():
            param.requires_grad = True


def _backbone_blocks_per_stage(model: torch.nn.Module) -> list[int]:
    blocks_per_stage = model.vision_encoder.backbone.config.blocks_per_stage
    if not blocks_per_stage:
        raise ValueError("backbone blocks_per_stage is missing")
    return list(blocks_per_stage)


def _stage_block_range(blocks_per_stage: list[int], stage: int) -> range:
    if stage < 1 or stage > len(blocks_per_stage):
        raise ValueError(f"invalid Hiera stage {stage}")
    start = sum(blocks_per_stage[: stage - 1])
    return range(start, start + blocks_per_stage[stage - 1])


def _backbone_lora_targets(model: torch.nn.Module, lora: LoraSettings) -> list[str]:
    blocks_per_stage = _backbone_blocks_per_stage(model)
    block_indices: list[int] = []
    if lora.stage3:
        block_indices.extend(_stage_block_range(blocks_per_stage, 3))
    if lora.stage4:
        block_indices.extend(_stage_block_range(blocks_per_stage, 4))

    targets: list[str] = []
    for block_idx in block_indices:
        prefix = f"vision_encoder.backbone.blocks.{block_idx}.attn"
        targets.extend((f"{prefix}.qkv", f"{prefix}.proj"))
    return targets


def _resolve_lora_modules(model: torch.nn.Module, targets: list[str]) -> None:
    module_names = {name for name, _module in model.named_modules()}
    missing = [name for name in targets if name not in module_names]
    if missing:
        raise ValueError(
            "Missing LoRA target modules in SAM 2.1: "
            + ", ".join(missing[:5])
            + (" …" if len(missing) > 5 else "")
        )


def apply_lora(model: torch.nn.Module, lora: LoraSettings) -> torch.nn.Module:
    from peft import LoraConfig, get_peft_model

    targets = _backbone_lora_targets(model, lora)
    if not targets:
        raise ValueError("lora: enable at least one of stage3, stage4")
    _resolve_lora_modules(model, targets)
    return get_peft_model(
        model,
        LoraConfig(
            r=lora.rank,
            lora_alpha=2 * lora.rank,
            lora_dropout=lora.dropout,
            target_modules=targets,
            bias="none",
        ),
    )


def lora_param_count(model: torch.nn.Module) -> int:
    return sum(
        param.numel()
        for name, param in model.named_parameters()
        if param.requires_grad and "lora_" in name
    )


def trainable_param_count(model: torch.nn.Module) -> int:
    return sum(param.numel() for param in model.parameters() if param.requires_grad)


def _segment(model: torch.nn.Module, **kwargs: object):
    """Run single-image point/box segmentation (Sam2VideoModel image path)."""
    return model._single_frame_forward(**kwargs)


def post_process_pred_masks(
    processor: object,
    pred_masks: torch.Tensor,
    original_sizes: torch.Tensor,
) -> torch.Tensor:
    """Sam2VideoModel returns 4D masks; post_process expects a multimask dim."""
    if pred_masks.ndim == 4:
        pred_masks = pred_masks.unsqueeze(2)
    return processor.post_process_masks(pred_masks, original_sizes)[0]


def _checkpoint_key_candidates(key: str) -> list[str]:
    key = key.removeprefix("model.").removeprefix("base_model.model.")
    candidates = [f"base_model.model.{key}"]
    for suffix in (".weight", ".bias"):
        if key.endswith(suffix):
            stem = key[: -len(suffix)]
            candidates.append(f"base_model.model.{stem}.base_layer{suffix}")
    return candidates


def _normalize_checkpoint_keys(
    state: dict[str, torch.Tensor],
    model: torch.nn.Module,
) -> dict[str, torch.Tensor]:
    model_keys = set(model.state_dict().keys())
    normalized: dict[str, torch.Tensor] = {}
    for raw_key, value in state.items():
        matched = next(
            (candidate for candidate in _checkpoint_key_candidates(raw_key) if candidate in model_keys),
            None,
        )
        if matched is not None:
            normalized[matched] = value
    return normalized


def load_model_and_processor(
    cfg: Config,
    *,
    for_training: bool = False,
    checkpoint: Path | None = None,
) -> Sam2Predictor:
    from transformers import Sam2VideoModel, Sam2VideoProcessor

    device = pick_device()
    model = Sam2VideoModel.from_pretrained(cfg.model_id).to(device)
    processor = Sam2VideoProcessor.from_pretrained(cfg.model_id)
    model = apply_lora(model, cfg.lora)
    if checkpoint is not None:
        load_finetuned_weights(model, checkpoint)
    if for_training:
        apply_finetune(model, cfg.finetune)
        model.train()
    else:
        model.eval()
    return Sam2Predictor(model=model, processor=processor, device=device)


def load_finetuned_weights(model: torch.nn.Module, checkpoint_dir: Path) -> None:
    weights_file = checkpoint_dir / "model.safetensors"
    if not weights_file.is_file():
        raise FileNotFoundError(f"SAM checkpoint not found: {weights_file}")

    from safetensors.torch import load_file

    raw_state = load_file(str(weights_file))
    if not any("lora_" in key for key in raw_state):
        raise ValueError(f"Checkpoint has no LoRA weights (unsupported format): {weights_file}")
    state = _normalize_checkpoint_keys(raw_state, model)
    _missing, unexpected = model.load_state_dict(state, strict=False)
    if unexpected:
        raise RuntimeError(f"Unexpected SAM checkpoint keys: {unexpected[:5]}")


def load_predictor(cfg: Config) -> Sam2Predictor:
    checkpoint = cfg.infer.checkpoint
    predictor = load_model_and_processor(cfg, for_training=False, checkpoint=checkpoint)
    return predictor


def prepare_pixel_values(processor: object, image: Image.Image, device: torch.device) -> torch.Tensor:
    inputs = processor(images=image, return_tensors="pt")
    return inputs["pixel_values"].to(device)


class Sam2TrainingWrapper(nn.Module):
    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(
        self,
        input_points: torch.Tensor,
        input_labels: torch.Tensor,
        pixel_values: torch.Tensor,
        **kwargs: object,
    ):
        del kwargs
        return _segment(
            self.model,
            pixel_values=pixel_values,
            input_points=input_points,
            input_labels=input_labels,
            multimask_output=False,
        )


def encode_image(predictor: Sam2Predictor, image: Image.Image) -> EncodedImage:
    """Run the vision backbone once; reuse pixel_values for point queries."""
    inputs = predictor.processor(images=image, return_tensors="pt")
    return EncodedImage(
        pixel_values=inputs["pixel_values"].to(predictor.device),
        original_sizes=inputs["original_sizes"],
    )


def predict_point_encoded(
    predictor: Sam2Predictor,
    encoded: EncodedImage,
    point: tuple[float, float],
    *,
    multimask_output: bool = False,
) -> tuple[np.ndarray, float]:
    """Return a boolean H×W mask and confidence score using pre-encoded image features."""
    return predict_points_encoded(
        predictor,
        encoded,
        [point],
        multimask_output=multimask_output,
    )


def predict_points_encoded(
    predictor: Sam2Predictor,
    encoded: EncodedImage,
    points: list[tuple[float, float]],
    *,
    negative_points: list[tuple[float, float]] | None = None,
    multimask_output: bool = False,
) -> tuple[np.ndarray, float]:
    if not points:
        raise ValueError("points required")
    all_points = list(points)
    all_labels = [1] * len(points)
    if negative_points:
        all_points.extend(negative_points)
        all_labels.extend([0] * len(negative_points))
    input_points = torch.tensor(
        [[[[x, y] for x, y in all_points]]],
        dtype=torch.float32,
        device=predictor.device,
    )
    input_labels = torch.tensor([[all_labels]], dtype=torch.int64, device=predictor.device)

    with torch.no_grad():
        outputs = _segment(
            predictor.model,
            pixel_values=encoded.pixel_values,
            input_points=input_points,
            input_labels=input_labels,
            multimask_output=multimask_output,
        )

    masks = post_process_pred_masks(
        predictor.processor,
        outputs.pred_masks.cpu(),
        encoded.original_sizes,
    )
    scores = outputs.iou_scores[0, 0].detach().cpu()
    if multimask_output:
        best_idx = int(scores.argmax().item())
    else:
        best_idx = 0
    mask = masks[0, best_idx].numpy().astype(bool)
    score = float(scores[best_idx].item())
    return mask, score


def predict_point(
    predictor: Sam2Predictor,
    image: Image.Image,
    point: tuple[float, float],
    *,
    multimask_output: bool = False,
) -> tuple[np.ndarray, float]:
    """Return a boolean H×W mask and confidence score for one foreground point."""
    return predict_points(
        predictor,
        image,
        [point],
        multimask_output=multimask_output,
    )


def predict_points(
    predictor: Sam2Predictor,
    image: Image.Image,
    points: list[tuple[float, float]],
    *,
    negative_points: list[tuple[float, float]] | None = None,
    multimask_output: bool = False,
) -> tuple[np.ndarray, float]:
    """Return a boolean H×W mask and confidence score for foreground (and optional negative) points."""
    if not points:
        raise ValueError("points required")
    all_points = list(points)
    all_labels = [1] * len(points)
    if negative_points:
        all_points.extend(negative_points)
        all_labels.extend([0] * len(negative_points))
    input_points = [[[[x, y] for x, y in all_points]]]
    input_labels = [[all_labels]]
    inputs = predictor.processor(
        images=image,
        input_points=input_points,
        input_labels=input_labels,
        return_tensors="pt",
    )
    inputs = {key: value.to(predictor.device) for key, value in inputs.items()}

    with torch.no_grad():
        outputs = _segment(
            predictor.model,
            pixel_values=inputs["pixel_values"],
            input_points=inputs["input_points"],
            input_labels=inputs["input_labels"],
            multimask_output=multimask_output,
        )

    masks = post_process_pred_masks(
        predictor.processor,
        outputs.pred_masks.cpu(),
        inputs["original_sizes"],
    )
    scores = outputs.iou_scores[0, 0].detach().cpu()
    if multimask_output:
        best_idx = int(scores.argmax().item())
    else:
        best_idx = 0
    mask = masks[0, best_idx].numpy().astype(bool)
    score = float(scores[best_idx].item())
    return mask, score


def dice_loss(logits: torch.Tensor, targets: torch.Tensor, *, eps: float = 1e-6) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    num = 2.0 * (probs * targets).sum(dim=(-2, -1))
    den = probs.sum(dim=(-2, -1)) + targets.sum(dim=(-2, -1))
    return (1.0 - (num + eps) / (den + eps)).mean()


def focal_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    alpha: float = 0.25,
    gamma: float = 2.0,
) -> torch.Tensor:
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    probs = torch.sigmoid(logits)
    prob_t = probs * targets + (1.0 - probs) * (1.0 - targets)
    return (alpha * (1.0 - prob_t).pow(gamma) * bce).mean()


def segmentation_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    if logits.shape[-2:] != targets.shape[-2:]:
        targets = F.interpolate(
            targets.unsqueeze(1),
            size=logits.shape[-2:],
            mode="nearest",
        ).squeeze(1)
    return dice_loss(logits, targets) + focal_loss(logits, targets)


def compute_segmentation_loss(
    outputs: object,
    labels: torch.Tensor,
    num_items_in_batch: int | None = None,
) -> torch.Tensor:
    del num_items_in_batch
    logits = outputs.pred_masks.squeeze(2).squeeze(1)
    return segmentation_loss(logits, labels)
