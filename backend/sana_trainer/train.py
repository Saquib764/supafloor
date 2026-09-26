"""Fine-tune Sana 0.6B transformer (LoRA or full) on image+caption data."""

from __future__ import annotations

import math
import time
from pathlib import Path

import torch
from accelerate import Accelerator
from accelerate.utils import set_seed
from diffusers import SanaPipeline
from diffusers.optimization import get_scheduler
from diffusers.training_utils import (
    cast_training_params,
    compute_density_for_timestep_sampling,
    compute_loss_weighting_for_sd3,
)
from diffusers.utils import convert_unet_state_dict_to_peft
from peft.utils import get_peft_model_state_dict, set_peft_model_state_dict
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from config import Config, load_config
from data import CaptionImageDataset, collate_batch, load_split
from model import (
    TrainingStack,
    encode_captions,
    load_training_stack,
    mixed_precision_dtype,
    save_full_transformer,
    save_lora_weights,
    unwrap_transformer,
)


def _ema(prev: float, value: float, *, alpha: float = 0.1) -> float:
    return value if prev == 0.0 else alpha * value + (1.0 - alpha) * prev


def _resolve_from_adapter(path: Path | None) -> Path | None:
    if path is None:
        return None
    adapter = path.resolve()
    if not adapter.is_dir():
        raise FileNotFoundError(f"adapter directory not found: {adapter}")
    return adapter


def _print_setup(
    cfg: Config,
    *,
    dataset_len: int,
    total_steps: int,
    warmup_steps: int,
    mixed_precision: str,
) -> None:
    mode = "full transformer" if cfg.train.full_finetune else "LoRA"
    print("\n=== Device / precision ===")
    print(f"cuda available: {torch.cuda.is_available()}")
    mps = getattr(torch.backends, "mps", None)
    print(f"mps available:  {mps is not None and mps.is_available()}")
    print(f"mixed_precision:{mixed_precision}")
    print(f"train mode:     {mode}")

    print("\n=== Model ===")
    print(f"model_id:       {cfg.model_id}")
    print(f"variant:        {cfg.variant or '(default)'}")
    print(f"resolution:     {cfg.resolution}")

    if not cfg.train.full_finetune:
        print("\n=== LoRA (NVIDIA alpha = rank convention) ===")
        print(f"rank:           {cfg.lora.rank}")
        print(f"alpha:          {cfg.lora.alpha}")
        print(f"dropout:        {cfg.lora.dropout}")
        print(f"target_modules: {', '.join(cfg.lora.target_modules)}")

    print("\n=== Dataset ===")
    print(f"dataset:        {cfg.data.dataset}")
    print(f"split:          {cfg.data.split}")
    print(f"image_column:   {cfg.data.image_column}")
    print(f"caption_column: {cfg.data.caption_column}")
    print(f"rows:           {dataset_len:,}")

    print("\n=== Training ===")
    print(f"output_dir:     {cfg.train.session_dir}")
    print(f"epochs:         {cfg.train.epochs}")
    print(f"batch_size:     {cfg.train.batch_size}")
    print(f"grad_accum:     {cfg.train.grad_accum}")
    print(f"total_steps:    {total_steps:,}")
    print(f"warmup_steps:   {warmup_steps:,}")
    print(f"learning_rate:  {cfg.train.learning_rate}")
    print(f"save_steps:     {cfg.train.save_steps}")
    print(f"viz_steps:      {cfg.train.viz_steps}")
    print(f"from_adapter:   {cfg.train.from_adapter or '(none)'}")
    print(f"resume:         {cfg.train.resume}")
    print()


def _get_sigmas(
    timesteps: torch.Tensor,
    scheduler_copy,
    *,
    device: torch.device,
    n_dim: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    sigmas = scheduler_copy.sigmas.to(device=device, dtype=dtype)
    schedule_timesteps = scheduler_copy.timesteps.to(device)
    timesteps = timesteps.to(device)
    step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]
    sigma = sigmas[step_indices].flatten()
    while len(sigma.shape) < n_dim:
        sigma = sigma.unsqueeze(-1)
    return sigma


def _export_transformer(accelerator: Accelerator, stack: TrainingStack):
    model = accelerator.unwrap_model(stack.transformer)
    return unwrap_transformer(model)


def _save_viz(
    cfg: Config,
    stack: TrainingStack,
    accelerator: Accelerator,
    *,
    device: torch.device,
    step: int,
) -> None:
    if not cfg.train.viz_prompts:
        return
    out_dir = cfg.train.session_dir / "viz" / f"step-{step:06d}"
    out_dir.mkdir(parents=True, exist_ok=True)
    pipe = SanaPipeline.from_pretrained(
        cfg.model_id,
        transformer=_export_transformer(accelerator, stack),
        text_encoder=stack.text_encoder,
        tokenizer=stack.tokenizer,
        vae=stack.vae,
        revision=cfg.revision,
        variant=cfg.variant,
        torch_dtype=stack.weight_dtype,
    )
    pipe = pipe.to(device)
    pipe.text_encoder.to(torch.bfloat16)
    pipe.vae.to(torch.float32)
    pipe.set_progress_bar_config(disable=True)
    gen = torch.Generator(device=device).manual_seed(cfg.train.viz_seed + step)
    for index, prompt in enumerate(cfg.train.viz_prompts):
        image = pipe(
            prompt=prompt,
            height=cfg.resolution,
            width=cfg.resolution,
            complex_human_instruction=cfg.complex_human_instruction,
            generator=gen,
            num_inference_steps=20,
        ).images[0]
        image.save(out_dir / f"prompt-{index:02d}.jpg", format="JPEG", quality=95)
    del pipe
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    tqdm.write(f"viz → {out_dir} ({len(cfg.train.viz_prompts)} prompt(s))")


def _register_lora_hooks(accelerator: Accelerator, stack: TrainingStack) -> None:
    def save_model_hook(models, weights, output_dir):
        if not accelerator.is_main_process:
            return
        for model in models:
            if isinstance(model, type(unwrap_transformer(stack.transformer))):
                layers = get_peft_model_state_dict(unwrap_transformer(model))
                SanaPipeline.save_lora_weights(output_dir, transformer_lora_layers=layers)
            else:
                raise ValueError(f"unexpected save model: {model.__class__.__name__}")
            if weights:
                weights.pop()

    def load_model_hook(models, input_dir):
        transformer_model = None
        while models:
            model = models.pop()
            if isinstance(model, type(unwrap_transformer(stack.transformer))):
                transformer_model = model
            else:
                raise ValueError(f"unexpected load model: {model.__class__.__name__}")
        if transformer_model is None:
            return
        state = SanaPipeline.lora_state_dict(input_dir)
        transformer_state = {
            key.replace("transformer.", ""): value
            for key, value in state.items()
            if key.startswith("transformer.")
        }
        transformer_state = convert_unet_state_dict_to_peft(transformer_state)
        set_peft_model_state_dict(transformer_model, transformer_state, adapter_name="default")

    accelerator.register_save_state_pre_hook(save_model_hook)
    accelerator.register_load_state_pre_hook(load_model_hook)


def _save_checkpoint(
    cfg: Config,
    stack: TrainingStack,
    *,
    step: int,
    accelerator: Accelerator,
) -> None:
    if not accelerator.is_main_process:
        return
    transformer = _export_transformer(accelerator, stack)
    if cfg.train.full_finetune:
        path = cfg.train.session_dir / "checkpoints" / f"step-{step:06d}"
        save_full_transformer(transformer, path / "transformer")
        tqdm.write(f"saved transformer → {path / 'transformer'}")
    else:
        path = cfg.train.session_dir / "lora" / f"step-{step:06d}"
        save_lora_weights(transformer, path)
        tqdm.write(f"saved LoRA → {path}")


def run(cfg: Config) -> None:
    if cfg.train.resume and cfg.train.from_adapter is not None:
        raise ValueError("train.resume and train.from_adapter are mutually exclusive")
    adapter = _resolve_from_adapter(cfg.train.from_adapter)

    mixed_precision, _ = mixed_precision_dtype(cfg)
    mps = getattr(torch.backends, "mps", None)
    if (
        mixed_precision == "bf16"
        and mps is not None
        and mps.is_available()
        and not torch.cuda.is_available()
    ):
        mixed_precision = "fp16"

    accelerator = Accelerator(
        gradient_accumulation_steps=cfg.train.grad_accum,
        mixed_precision=mixed_precision,
    )
    set_seed(cfg.data.seed)

    device = accelerator.device
    stack = load_training_stack(cfg, device=device, adapter=adapter)

    if not cfg.train.full_finetune and mixed_precision == "fp16":
        cast_training_params([stack.transformer], dtype=torch.float32)

    trainable = [p for p in stack.transformer.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable,
        lr=cfg.train.learning_rate,
        betas=(0.9, 0.999),
        weight_decay=cfg.train.weight_decay,
        eps=1e-8,
    )

    split = load_split(cfg)
    dataset = CaptionImageDataset(split, cfg)
    loader = DataLoader(
        dataset,
        batch_size=cfg.train.batch_size,
        shuffle=True,
        num_workers=cfg.train.num_workers,
        collate_fn=collate_batch,
        pin_memory=device.type == "cuda",
    )

    steps_per_epoch = max(1, math.ceil(len(loader) / cfg.train.grad_accum))
    total_steps = steps_per_epoch * cfg.train.epochs
    warmup_steps = int(total_steps * cfg.train.warmup_ratio)
    lr_scheduler = get_scheduler(
        cfg.train.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    if not cfg.train.full_finetune:
        _register_lora_hooks(accelerator, stack)

    stack.transformer, optimizer, loader, lr_scheduler = accelerator.prepare(
        stack.transformer, optimizer, loader, lr_scheduler
    )

    cfg.train.session_dir.mkdir(parents=True, exist_ok=True)
    _print_setup(
        cfg,
        dataset_len=len(dataset),
        total_steps=total_steps,
        warmup_steps=warmup_steps,
        mixed_precision=mixed_precision,
    )

    global_step = 0
    if cfg.train.resume:
        checkpoints = sorted(
            (p for p in cfg.train.session_dir.glob("checkpoint-*") if p.is_dir()),
            key=lambda p: int(p.name.split("-")[-1]),
        )
        if not checkpoints:
            raise FileNotFoundError(f"no checkpoint-* under {cfg.train.session_dir}")
        latest = checkpoints[-1]
        accelerator.load_state(str(latest))
        global_step = int(latest.name.split("-")[-1])
        tqdm.write(f"resumed from {latest} (step {global_step})")

    vae_scale = stack.vae.config.scaling_factor
    timing_ema = {"data": 0.0, "fwd": 0.0, "bwd": 0.0}
    batch_wait = time.perf_counter()
    progress = tqdm(
        total=total_steps,
        initial=global_step,
        desc="Training",
        unit="step",
        dynamic_ncols=True,
        disable=not accelerator.is_local_main_process,
    )

    for epoch in range(cfg.train.epochs):
        stack.transformer.train()
        for batch in loader:
            data_sec = time.perf_counter() - batch_wait
            with accelerator.accumulate(stack.transformer):
                prompts = batch["captions"]
                prompt_embeds, prompt_attention_mask = encode_captions(
                    stack, cfg, prompts, device=device
                )

                fwd_start = time.perf_counter()
                pixel_values = batch["pixel_values"].to(device=device, dtype=stack.vae.dtype)
                with torch.no_grad():
                    latents = stack.vae.encode(pixel_values).latent * vae_scale
                latents = latents.to(dtype=stack.weight_dtype)

                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                u = compute_density_for_timestep_sampling(
                    weighting_scheme=cfg.train.weighting_scheme,
                    batch_size=bsz,
                )
                indices = (u * stack.noise_scheduler_copy.config.num_train_timesteps).long()
                timesteps = stack.noise_scheduler_copy.timesteps[indices].to(device=latents.device)
                sigmas = _get_sigmas(
                    timesteps,
                    stack.noise_scheduler_copy,
                    device=device,
                    n_dim=latents.ndim,
                    dtype=latents.dtype,
                )
                noisy = (1.0 - sigmas) * latents + sigmas * noise
                model_pred = stack.transformer(
                    hidden_states=noisy,
                    encoder_hidden_states=prompt_embeds,
                    encoder_attention_mask=prompt_attention_mask,
                    timestep=timesteps,
                    return_dict=False,
                )[0]
                weighting = compute_loss_weighting_for_sd3(
                    weighting_scheme=cfg.train.weighting_scheme,
                    sigmas=sigmas,
                )
                target = noise - latents
                loss = torch.mean(
                    (
                        weighting.float()
                        * (model_pred.float() - target.float()) ** 2
                    ).reshape(bsz, -1),
                    dim=1,
                ).mean()

                bwd_start = time.perf_counter()
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(stack.transformer.parameters(), cfg.train.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                fwd_sec = bwd_start - fwd_start
                bwd_sec = time.perf_counter() - bwd_start

            if not accelerator.sync_gradients:
                batch_wait = time.perf_counter()
                continue

            global_step += 1
            progress.update(1)
            timing_ema["data"] = _ema(timing_ema["data"], data_sec)
            timing_ema["fwd"] = _ema(timing_ema["fwd"], fwd_sec)
            timing_ema["bwd"] = _ema(timing_ema["bwd"], bwd_sec)
            lr = lr_scheduler.get_last_lr()[0]
            progress.set_postfix_str(
                f"{loss.item():.4f}, {lr:.2e}, ep {epoch + 1}/{cfg.train.epochs}, "
                f"data={timing_ema['data']:.2f}s fwd={timing_ema['fwd']:.2f}s "
                f"bwd={timing_ema['bwd']:.2f}s"
            )
            batch_wait = time.perf_counter()

            if global_step % cfg.train.viz_steps == 0:
                _save_viz(cfg, stack, accelerator, device=device, step=global_step)

            if global_step % cfg.train.save_steps == 0:
                _save_checkpoint(cfg, stack, step=global_step, accelerator=accelerator)
                if not cfg.train.full_finetune:
                    ckpt_dir = cfg.train.session_dir / f"checkpoint-{global_step}"
                    accelerator.save_state(str(ckpt_dir))

            if global_step >= total_steps:
                break
        if global_step >= total_steps:
            break

    progress.close()
    if accelerator.is_main_process:
        transformer = _export_transformer(accelerator, stack)
        _save_checkpoint(cfg, stack, step=global_step, accelerator=accelerator)
        if not cfg.train.full_finetune:
            final_lora = cfg.train.session_dir / "lora" / "final"
            save_lora_weights(transformer, final_lora)
            tqdm.write(f"final LoRA → {final_lora}")
        else:
            final_dir = cfg.train.session_dir / "transformer"
            save_full_transformer(transformer, final_dir)
            tqdm.write(f"final transformer → {final_dir}")


if __name__ == "__main__":
    run(load_config())
