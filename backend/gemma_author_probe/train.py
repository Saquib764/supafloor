"""Train probe, head, and Gemma blocks (low lr); token embeddings optional."""

from __future__ import annotations

import math
import os
import shutil
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from config import Config, load_config
from data import (
    PassageCollator,
    PassageDataset,
    PassageExample,
    PromptBatchSampler,
    load_train_test_split,
)
from device import pick_device
from loss import cross_entropy_loss, supervised_contrastive_loss
from model import ProbeClassifier, build_model, init_contrast_head

CHECKPOINT_NAME = "last.pt"
CHECKPOINT_BACKUP_NAME = "last_good.pt"
CHECKPOINT_FORMAT = "trainable"
TRAINING_STATE_NAME = "training_state.pt"


def _checkpoint_file(path: Path) -> Path:
    candidate = path / CHECKPOINT_NAME if path.is_dir() else path
    if not candidate.is_file():
        raise FileNotFoundError(f"missing checkpoint {candidate}")
    return candidate


def _checkpoint_weights(model: ProbeClassifier) -> dict[str, torch.Tensor]:
    state = model.state_dict()
    return {
        name: state[name]
        for name, param in model.named_parameters()
        if param.requires_grad and name in state
    }


def _optional_missing_key(name: str, cfg: Config) -> bool:
    if name.startswith("backbone.") and cfg.train.freeze_blocks:
        return True
    return False


def _compatible_checkpoint_weights(
    model: ProbeClassifier, weights: dict[str, torch.Tensor]
) -> dict[str, torch.Tensor]:
    model_state = model.state_dict()
    compatible: dict[str, torch.Tensor] = {}
    for key, value in weights.items():
        target = model_state.get(key)
        if target is None:
            continue
        if tuple(target.shape) != tuple(value.shape):
            continue
        compatible[key] = value
    return compatible


def _read_checkpoint(path: Path) -> dict | None:
    try:
        state = torch.load(path, map_location="cpu", weights_only=False)
    except (RuntimeError, EOFError, OSError) as exc:
        print(f"train: could not read {path.name} ({exc})", flush=True)
        return None
    if not isinstance(state, dict):
        print(f"train: invalid checkpoint format in {path.name}", flush=True)
        return None
    return state


def _load_model_weights(
    model: ProbeClassifier,
    path: Path,
    cfg: Config,
    *,
    load_contrast_head: bool = True,
) -> bool:
    state = _read_checkpoint(path)
    if state is None:
        return False
    weights = state.get("model")
    if not isinstance(weights, dict):
        print(f"train: {path.name} has no model weights", flush=True)
        return False
    if not load_contrast_head:
        weights = {
            key: value for key, value in weights.items() if not key.startswith("contrast_head.")
        }
    model_contrast = {key for key in model.state_dict() if key.startswith("contrast_head.")}
    compatible = _compatible_checkpoint_weights(model, weights)
    if not compatible:
        print(f"train: no compatible weights in {path.name}", flush=True)
        return False
    skipped = len(weights) - len(compatible)
    if skipped:
        print(f"train: skipped {skipped} incompatible or unknown keys in {path.name}", flush=True)
    try:
        missing, unexpected = model.load_state_dict(compatible, strict=False)
    except RuntimeError as exc:
        print(f"train: could not load {path.name} ({exc})", flush=True)
        return False
    contrast_missing = [key for key in missing if key.startswith("contrast_head.")]
    other_missing = [
        key
        for key in missing
        if not key.startswith("contrast_head.") and not _optional_missing_key(key, cfg)
    ]
    if other_missing:
        print(f"train: checkpoint missing weights: {other_missing}", flush=True)
        return False
    loaded_contrast = model_contrast & set(compatible)
    if loaded_contrast and not contrast_missing:
        print(
            f"train: loaded contrast_head from {path.name} ({len(loaded_contrast)} keys)",
            flush=True,
        )
    elif contrast_missing or not load_contrast_head:
        init_contrast_head(model)
        print("train: fresh contrast_head init", flush=True)
    if unexpected:
        print(f"train: ignored unexpected checkpoint keys: {unexpected}", flush=True)
    step = state.get("step")
    if step is not None:
        print(f"train: checkpoint step {int(step)}", flush=True)
    return True


def _atomic_torch_save(payload: dict, path: Path) -> None:
    tmp = path.with_name(f"{path.name}.tmp")
    torch.save(payload, tmp)
    tmp.replace(path)


def _save_training_state(
    cfg: Config,
    *,
    step: int,
    epoch: int,
    optimizer: AdamW,
    scheduler: LambdaLR,
) -> Path | None:
    cfg.train.session_dir.mkdir(parents=True, exist_ok=True)
    path = cfg.train.session_dir / TRAINING_STATE_NAME
    try:
        _atomic_torch_save(
            {
                "step": step,
                "epoch": epoch,
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
            },
            path,
        )
    except RuntimeError as exc:
        print(f"train: could not save training state ({exc})", flush=True)
        return None
    return path


def _save_and_test(
    model: ProbeClassifier,
    cfg: Config,
    test_ex: list[PassageExample],
    device: torch.device,
    step: int,
    epoch: int,
    optimizer: AdamW,
    scheduler: LambdaLR,
    progress: tqdm | None,
) -> None:
    from run_test import evaluate_model

    ckpt = _save_checkpoint(model, cfg, step)
    if ckpt is None:
        return
    _save_training_state(cfg, step=step, epoch=epoch, optimizer=optimizer, scheduler=scheduler)
    message = f"saved step {step} → {ckpt.resolve()}"
    if progress is not None:
        tqdm.write(message)
    else:
        print(message, flush=True)
    evaluate_model(model, test_ex, device, cfg.train.micro_batch_size, step=step)
    if progress is not None:
        progress.refresh()


def _save_checkpoint(model: ProbeClassifier, cfg: Config, step: int) -> Path | None:
    cfg.train.session_dir.mkdir(parents=True, exist_ok=True)
    ckpt = cfg.train.session_dir / CHECKPOINT_NAME
    payload = {
        "step": step,
        "format": CHECKPOINT_FORMAT,
        "model": _checkpoint_weights(model),
    }
    try:
        _atomic_torch_save(payload, ckpt)
        shutil.copy2(ckpt, cfg.train.session_dir / CHECKPOINT_BACKUP_NAME)
    except (RuntimeError, OSError) as exc:
        tmp = ckpt.with_name(f"{ckpt.name}.tmp")
        if tmp.is_file():
            tmp.unlink(missing_ok=True)
        print(f"train: checkpoint save failed ({exc})", flush=True)
        return None
    return ckpt


def _lr_lambda(*, warmup_steps: int, total_steps: int):
    def fn(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        if total_steps <= warmup_steps:
            return 1.0
        progress = float(step - warmup_steps) / float(total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return fn


def _optimizer_groups(model: ProbeClassifier, cfg: Config) -> list[dict[str, object]]:
    decay: list[torch.nn.Parameter] = []
    no_decay: list[torch.nn.Parameter] = []
    for name, param in model.backbone.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim == 1 or name.endswith(".bias"):
            no_decay.append(param)
        else:
            decay.append(param)
    head: list[torch.nn.Parameter] = []
    head.extend(param for param in model.pool_norms.parameters() if param.requires_grad)
    head.extend(param for param in model.classifier.parameters() if param.requires_grad)
    groups: list[dict[str, object]] = []
    if decay:
        groups.append(
            {"params": decay, "lr": cfg.train.block_lr, "weight_decay": cfg.train.weight_decay}
        )
    if no_decay:
        groups.append({"params": no_decay, "lr": cfg.train.block_lr, "weight_decay": 0.0})
    if model.probes.requires_grad:
        groups.append({"params": [model.probes], "lr": cfg.train.probe_lr, "weight_decay": 0.0})
    if head:
        groups.append({"params": head, "lr": cfg.train.head_lr, "weight_decay": 0.0})
    contrast = [
        param for param in model.contrast_head.parameters() if param.requires_grad
    ]
    if contrast:
        groups.append(
            {
                "params": contrast,
                "lr": cfg.train.contrast_head.lr,
                "weight_decay": 0.0,
            }
        )
    if not groups:
        raise RuntimeError("nothing to train; set at least one freeze_* flag to false")
    return groups


def _count(params) -> int:
    return sum(int(param.numel()) for param in params if param.requires_grad)


def _assert_step_grads(
    model: ProbeClassifier,
    *,
    freeze_blocks: bool,
    freeze_probes: bool,
    freeze_head: bool,
    freeze_embeddings: bool,
    freeze_contrast_head: bool,
    use_contrastive: bool,
) -> None:
    if freeze_probes:
        if model.probes.grad is not None:
            raise RuntimeError("probes are frozen but received a gradient")
    elif model.probes.grad is None:
        raise RuntimeError("probes did not receive a gradient")
    if freeze_head:
        if model.classifier.weight.grad is not None:
            raise RuntimeError("head is frozen but classifier received a gradient")
        first_norm = next(model.pool_norms.parameters())
        if first_norm.grad is not None:
            raise RuntimeError("head is frozen but pool norms received a gradient")
    elif model.classifier.weight.grad is None:
        raise RuntimeError("classifier did not receive a gradient")
    first_contrast = next(model.contrast_head.parameters())
    if use_contrastive and not freeze_contrast_head:
        if first_contrast.grad is None:
            raise RuntimeError("contrast_head did not receive a gradient")
    elif freeze_contrast_head and first_contrast.grad is not None:
        raise RuntimeError("contrast_head is frozen but received a gradient")
    first = next(model.backbone.layers[0].parameters())
    last = next(model.backbone.layers[-1].parameters())
    if not freeze_blocks:
        if first.grad is None or last.grad is None:
            raise RuntimeError("transformer blocks did not receive gradients")
    elif first.grad is not None or last.grad is not None:
        raise RuntimeError("transformer blocks are frozen but received a gradient")
    embed_grad = model.backbone.embed_tokens.weight.grad
    if freeze_embeddings and embed_grad is not None:
        raise RuntimeError("token embeddings are frozen but received a gradient")
    if not freeze_blocks and not freeze_embeddings and embed_grad is None:
        raise RuntimeError("token embeddings are trainable but received no gradient")


def run(cfg: Config | None = None, *, max_steps: int | None = None) -> None:
    cfg = cfg or load_config()
    device = pick_device()
    train_ex, test_ex = load_train_test_split(cfg)
    print(f"train: loading {cfg.model_id}", flush=True)
    model = build_model(cfg)
    session_ckpt = cfg.train.session_dir / CHECKPOINT_NAME
    state_path = cfg.train.session_dir / TRAINING_STATE_NAME
    step = 0
    pending_state: dict | None = None
    if cfg.train.resume and session_ckpt.is_file():
        print(f"train: resuming from {session_ckpt.resolve()}", flush=True)
        loaded = _load_model_weights(model, session_ckpt, cfg, load_contrast_head=True)
        if not loaded:
            backup = cfg.train.session_dir / CHECKPOINT_BACKUP_NAME
            if backup.is_file():
                print(f"train: trying backup {backup.resolve()}", flush=True)
                loaded = _load_model_weights(model, backup, cfg, load_contrast_head=True)
            if not loaded:
                print("train: resume kept hub weights (checkpoint unreadable)", flush=True)
                init_contrast_head(model)
                print("train: fresh contrast_head init", flush=True)
        if state_path.is_file():
            pending_state = _read_checkpoint(state_path)
            if pending_state is not None:
                step = int(pending_state.get("step", 0))
    elif cfg.train.init_checkpoint is not None:
        init_file = _checkpoint_file(cfg.train.init_checkpoint)
        print(f"train: init from {init_file}", flush=True)
        if not _load_model_weights(model, init_file, cfg, load_contrast_head=False):
            raise RuntimeError(f"could not load init checkpoint {init_file}")
        print(f"train: new session → {cfg.train.session_dir} (step 0)", flush=True)
    else:
        print("train: weights from hub", flush=True)
        print("train: fresh contrast_head init", flush=True)
    model.to(device)

    block_params = _count(model.backbone.layers.parameters())
    embed_params = _count(model.backbone.embed_tokens.parameters())
    probe_params = int(model.probes.numel())
    head_params = _count(model.pool_norms.parameters()) + _count(model.classifier.parameters())
    contrast_params = _count(model.contrast_head.parameters())
    blocks_state = "frozen" if cfg.train.freeze_blocks else f"trainable={block_params}"
    embed_state = "trainable" if embed_params else "frozen"
    probes_state = "frozen" if cfg.train.freeze_probes else f"trainable={probe_params}"
    head_state = "frozen" if cfg.train.freeze_head else f"trainable={head_params}"
    ch = cfg.train.contrast_head
    contrast_state = (
        "frozen"
        if ch.freeze
        else f"trainable={contrast_params} ({ch.hidden_dim}→{ch.out_dim})"
    )
    print(
        f"train: taps={list(cfg.tap_layers)} hidden={model.hidden_size} "
        f"blocks={blocks_state} embeddings={embed_state} "
        f"probes={probes_state} head={head_state} contrast_head={contrast_state}",
        flush=True,
    )

    collator = PassageCollator(
        model.tokenizer, max_len=model.max_len, probe_slots=model.num_probes
    )
    batch_sampler = PromptBatchSampler(
        train_ex,
        batch_size=cfg.train.batch_size,
        prompt_count=cfg.train.prompt_count,
        seed=cfg.data.seed,
    )
    train_loader = DataLoader(
        PassageDataset(train_ex),
        batch_sampler=batch_sampler,
        num_workers=cfg.data.num_workers,
        collate_fn=collator,
        pin_memory=device.type == "cuda",
    )
    steps_per_epoch = max(1, len(batch_sampler))
    total_steps = steps_per_epoch * cfg.train.epochs
    cap = cfg.train.max_steps if max_steps is None else int(max_steps)
    if cap < 0:
        raise ValueError("max_steps must be >= 0")
    if cap > 0:
        total_steps = min(total_steps, cap)
    warmup_steps = int(total_steps * cfg.train.warmup_ratio)

    optimizer = AdamW(
        _optimizer_groups(model, cfg),
        betas=cfg.train.optimizer.betas,
        eps=cfg.train.optimizer.eps,
    )
    scheduler = LambdaLR(
        optimizer,
        lr_lambda=_lr_lambda(warmup_steps=warmup_steps, total_steps=max(1, total_steps)),
    )
    if pending_state is not None:
        try:
            optimizer.load_state_dict(pending_state["optimizer"])
            scheduler.load_state_dict(pending_state["scheduler"])
        except (ValueError, KeyError, TypeError):
            print("train: could not restore optimizer/scheduler; continuing with fresh state", flush=True)

    start_epoch = step // steps_per_epoch
    batch_offset = step % steps_per_epoch
    print(
        f"\ntrain: {cfg.model_id} | train={len(train_ex)} | steps={total_steps} | "
        f"resume_step={step}\n",
        flush=True,
    )

    model.train()
    saved_step = step if pending_state is not None else -1
    micro = cfg.train.micro_batch_size
    contrastive_weight = cfg.train.contrastive_weight
    checked_grads = False
    progress = tqdm(total=total_steps, initial=step, desc="train", unit="step")
    progress.refresh()
    for epoch in range(start_epoch, cfg.train.epochs):
        if step >= total_steps:
            break
        batch_sampler.set_epoch(epoch)
        epoch_done = True
        for batch_idx, batch in enumerate(train_loader):
            if epoch == start_epoch and batch_idx < batch_offset:
                continue
            if step >= total_steps:
                epoch_done = False
                break
            input_ids = batch.input_ids.to(device)
            attention_mask = batch.attention_mask.to(device)
            labels = batch.labels.to(device)
            n = int(input_ids.shape[0])
            optimizer.zero_grad(set_to_none=True)
            ce_sum: torch.Tensor | None = None
            contrast_chunks: list[torch.Tensor] = []
            correct = 0
            use_contrastive = contrastive_weight > 0.0
            for start in range(0, n, micro):
                end = min(start + micro, n)
                chunk_labels = labels[start:end]
                chunk_ids = input_ids[start:end]
                chunk_mask = attention_mask[start:end]
                if use_contrastive:
                    logits, contrast = model.forward_logits_and_contrastive(
                        chunk_ids, chunk_mask
                    )
                    contrast_chunks.append(contrast)
                else:
                    logits = model(chunk_ids, chunk_mask)
                chunk_ce = cross_entropy_loss(logits, chunk_labels)
                weight = (end - start) / n
                weighted = chunk_ce * weight
                ce_sum = weighted if ce_sum is None else ce_sum + weighted
                correct += int((logits.argmax(dim=-1) == chunk_labels).sum().item())
            assert ce_sum is not None
            p_mean = c_mean = ctr_loss = None
            if use_contrastive:
                if n != cfg.train.batch_size:
                    raise RuntimeError(
                        f"contrastive batch size {n} != configured batch_size "
                        f"{cfg.train.batch_size}"
                    )
                embeddings = torch.cat(contrast_chunks, dim=0)
                ctr_loss, p_mean, c_mean = supervised_contrastive_loss(
                    embeddings,
                    labels,
                    temperature=cfg.train.contrastive_temperature,
                )
                total_loss = ce_sum + contrastive_weight * ctr_loss
            else:
                total_loss = ce_sum
            total_loss.backward()
            loss_sum = float(ce_sum.detach()) * n
            if ctr_loss is not None:
                loss_sum += float(ctr_loss.detach()) * contrastive_weight * n
            running_loss = loss_sum / n
            running_acc = correct / n
            if use_contrastive:
                postfix = (
                    f"loss={running_loss:.4f} con={float(ctr_loss.detach()):.4f} "
                    f"acc={running_acc:.3f} e={epoch + 1} "
                    f"P={float(p_mean.detach()):.3f} C={float(c_mean.detach()):.3f}"
                )
            else:
                postfix = f"loss={running_loss:.4f} acc={running_acc:.3f} epoch={epoch + 1}"
            progress.set_postfix_str(postfix)
            if not checked_grads:
                _assert_step_grads(
                    model,
                    freeze_blocks=cfg.train.freeze_blocks,
                    freeze_probes=cfg.train.freeze_probes,
                    freeze_head=cfg.train.freeze_head,
                    freeze_embeddings=cfg.train.freeze_embeddings,
                    freeze_contrast_head=cfg.train.contrast_head.freeze,
                    use_contrastive=use_contrastive,
                )
                checked_grads = True
            torch.nn.utils.clip_grad_norm_(
                (param for param in model.parameters() if param.requires_grad),
                cfg.train.max_grad_norm,
            )
            optimizer.step()
            scheduler.step()
            step += 1
            progress.update(1)
            progress.set_postfix_str(postfix)
            if step % cfg.train.save_steps == 0:
                _save_and_test(
                    model, cfg, test_ex, device, step, epoch, optimizer, scheduler, progress
                )
                saved_step = step
        if epoch_done and step > saved_step:
            _save_and_test(
                model, cfg, test_ex, device, step, epoch, optimizer, scheduler, progress
            )
            saved_step = step
    progress.close()

    if step > saved_step:
        last_epoch = min(cfg.train.epochs - 1, max(0, step // steps_per_epoch))
        _save_and_test(
            model,
            cfg,
            test_ex,
            device,
            step,
            last_epoch,
            optimizer,
            scheduler,
            progress=None,
        )
    print(f"done → {(cfg.train.session_dir / CHECKPOINT_NAME).resolve()}", flush=True)


if __name__ == "__main__":
    run(load_config())
