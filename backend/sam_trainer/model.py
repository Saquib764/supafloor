"""Load SAM 2.1, apply finetune flags, and run point-prompt segmentation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from config import Config, FinetuneSettings


@dataclass
class Sam2Predictor:
    model: torch.nn.Module
    processor: object
    device: torch.device


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def apply_finetune(model: torch.nn.Module, finetune: FinetuneSettings) -> None:
    for param in model.parameters():
        param.requires_grad = False

    if finetune.neck:
        for param in model.vision_encoder.neck.parameters():
            param.requires_grad = True
    if finetune.prompt_encoder:
        for param in model.prompt_encoder.parameters():
            param.requires_grad = True
    if finetune.mask_decoder:
        for param in model.mask_decoder.parameters():
            param.requires_grad = True


def trainable_param_count(model: torch.nn.Module) -> int:
    return sum(param.numel() for param in model.parameters() if param.requires_grad)


def load_model_and_processor(cfg: Config, *, for_training: bool = False) -> Sam2Predictor:
    from transformers import Sam2Model, Sam2Processor

    device = pick_device()
    model = Sam2Model.from_pretrained(cfg.model_id).to(device)
    processor = Sam2Processor.from_pretrained(cfg.model_id)
    if for_training:
        apply_finetune(model, cfg.finetune)
        model.train()
    else:
        model.eval()
    return Sam2Predictor(model=model, processor=processor, device=device)


def load_predictor(cfg: Config) -> Sam2Predictor:
    return load_model_and_processor(cfg, for_training=False)


def prepare_pixel_values(processor: object, image: Image.Image, device: torch.device) -> torch.Tensor:
    inputs = processor(images=image, return_tensors="pt")
    return inputs["pixel_values"].to(device)


@torch.no_grad()
def encode_backbone(model: torch.nn.Module, pixel_values: torch.Tensor) -> tuple[torch.Tensor, ...]:
    backbone_output = model.vision_encoder.backbone(pixel_values)
    return tuple(layer.detach().cpu().squeeze(0) for layer in backbone_output.intermediate_hidden_states)


def build_image_embeddings(
    model: torch.nn.Module,
    intermediate_hidden_states: tuple[torch.Tensor, ...],
    batch_size: int,
) -> list[torch.Tensor]:
    fpn_hidden_states, _fpn_position_encoding = model.vision_encoder.neck(intermediate_hidden_states)
    levels = model.num_feature_levels
    feature_maps = list(fpn_hidden_states[-levels:][::-1])
    feature_maps[0] = model.mask_decoder.conv_s0(feature_maps[0])
    feature_maps[1] = model.mask_decoder.conv_s1(feature_maps[1])
    feature_maps = [feature_map.flatten(2).permute(2, 0, 1) for feature_map in feature_maps]
    feature_maps[-1] = feature_maps[-1] + model.no_memory_embedding
    return [
        feature_map.permute(1, 2, 0).view(batch_size, -1, *feature_size)
        for feature_map, feature_size in zip(feature_maps, model.backbone_feature_sizes)
    ]


class Sam2TrainingWrapper(nn.Module):
    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(
        self,
        input_points: torch.Tensor,
        input_labels: torch.Tensor,
        intermediate_hidden_states: tuple[torch.Tensor, ...] | None = None,
        pixel_values: torch.Tensor | None = None,
        **kwargs: object,
    ):
        del kwargs
        if intermediate_hidden_states is not None:
            batch_size = input_points.shape[0]
            device = input_points.device
            ihs = tuple(layer.to(device) for layer in intermediate_hidden_states)
            image_embeddings = build_image_embeddings(
                self.model,
                ihs,
                batch_size,
            )
            return self.model(
                input_points=input_points,
                input_labels=input_labels,
                image_embeddings=image_embeddings,
                multimask_output=False,
            )

        if pixel_values is None:
            raise ValueError("intermediate_hidden_states or pixel_values required")
        return self.model(
            pixel_values=pixel_values,
            input_points=input_points,
            input_labels=input_labels,
            multimask_output=False,
        )


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
    multimask_output: bool = False,
) -> tuple[np.ndarray, float]:
    """Return a boolean H×W mask and confidence score for one or more foreground points."""
    if not points:
        raise ValueError("points required")
    input_points = [[[[x, y] for x, y in points]]]
    input_labels = [[[1] * len(points)]]
    inputs = predictor.processor(
        images=image,
        input_points=input_points,
        input_labels=input_labels,
        return_tensors="pt",
    )
    inputs = {key: value.to(predictor.device) for key, value in inputs.items()}

    with torch.no_grad():
        outputs = predictor.model(**inputs)

    masks = predictor.processor.post_process_masks(
        outputs.pred_masks.cpu(),
        inputs["original_sizes"],
    )[0]
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
