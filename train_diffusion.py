"""
train_diffusion_xray.py — Class-conditional hip X-ray diffusion model.

Architecture: Stable Diffusion v1-5 (VAE frozen, UNet fine-tuned, CLIP frozen)
Conditioning: "OA pre-op hip xray" / "ONFH post-op hip xray" etc.
Data: our v2 manifest.json + .pt pickle files

Only the dataset class and conditioning are changed from the original repo.
Everything else (UNet, VAE, CLIP, training loop, checkpointing) is identical.

Usage:
    python train_diffusion_xray.py \
        --tensor_dir ~/data_v2 \
        --text_json ~/data/patient_text_no_dx.json \
        --output_dir ~/diffusion_output \
        --max_train_steps 30000
"""

import argparse
import logging
import math
import os
import pickle
from pathlib import Path

import diffusers
import accelerate
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from torch.utils.data import Dataset

from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from packaging import version
from PIL import Image
from tqdm.auto import tqdm

from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version, is_xformers_available

check_min_version("0.21.0")
logger = get_logger(__name__, log_level="INFO")

LABEL_TO_NAME = {0: "OA", 1: "ONFH", 2: "Normal", 3: "Other"}


class HipXRayDataset(Dataset):
    """
    Loads from our v2 manifest.json + .pt pickle files.
    Conditioning prompt: "{class} {op_status} hip xray"
    e.g. "OA pre_op hip xray", "ONFH post_op hip xray"
    """
    def __init__(self, data_dir, resolution=512):
        import json
        self.data_dir = data_dir
        self.resolution = resolution

        manifest_path = os.path.join(data_dir, "manifest.json")
        with open(manifest_path) as f:
            self.entries = json.load(f)["entries"]

        logger.info(f"HipXRayDataset: {len(self.entries)} images from {manifest_path}")

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        e = self.entries[idx]
        pt_path = os.path.join(self.data_dir, e["pt_path"])

        with open(pt_path, "rb") as f:
            d = pickle.load(f)

        img = torch.from_numpy(d["image"].astype(np.float32))  # [1, H, W] in [0,1]
        img = img.repeat(3, 1, 1)  # [3, H, W]

        # Resize to training resolution
        img = torch.nn.functional.interpolate(
            img.unsqueeze(0), size=(self.resolution, self.resolution),
            mode="bilinear", align_corners=False
        ).squeeze(0)

        # Normalize to [-1, 1] for SD VAE
        img = img * 2.0 - 1.0

        # Build conditioning prompt
        label_name = LABEL_TO_NAME.get(e.get("train_label", -1), "unknown")
        op_status  = e.get("op_status", "unknown").replace("_", " ")
        prompt = f"{label_name} {op_status} hip xray"

        return {"image": img, "prompt": prompt}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained_model_name_or_path", type=str,
                        default="runwayml/stable-diffusion-v1-5")
    parser.add_argument("--tensor_dir",  type=str, required=True,
                        help="Path to data_v2 dir containing manifest.json and .pt files")
    parser.add_argument("--output_dir",  type=str, default="~/diffusion_output")
    parser.add_argument("--cache_dir",   type=str, default=None)
    parser.add_argument("--resolution",  type=int, default=512)
    parser.add_argument("--train_batch_size", type=int, default=2)
    parser.add_argument("--max_train_steps",  type=int, default=30000)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--lr_scheduler", type=str, default="constant")
    parser.add_argument("--lr_warmup_steps", type=int, default=500)
    parser.add_argument("--adam_beta1",  type=float, default=0.9)
    parser.add_argument("--adam_beta2",  type=float, default=0.999)
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2)
    parser.add_argument("--adam_epsilon", type=float, default=1e-8)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--mixed_precision", type=str, default="fp16",
                        choices=["no", "fp16", "bf16"])
    parser.add_argument("--checkpointing_steps", type=int, default=5000)
    parser.add_argument("--validation_steps",    type=int, default=5000)
    parser.add_argument("--num_validation_images", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--report_to", type=str, default="tensorboard")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    args.output_dir = os.path.expanduser(args.output_dir)
    args.tensor_dir = os.path.expanduser(args.tensor_dir)

    logging_dir = Path(args.output_dir) / "logs"
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=ProjectConfiguration(project_dir=args.output_dir,
                                            logging_dir=str(logging_dir)),
    )

    logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
                        datefmt="%m/%d/%Y %H:%M:%S", level=logging.INFO)

    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    if args.seed is not None:
        set_seed(args.seed)

    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)

    # ── Load models ────────────────────────────────────────────────────────────
    noise_scheduler = DDPMScheduler.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="scheduler", cache_dir=args.cache_dir)
    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="vae", cache_dir=args.cache_dir)
    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="unet", cache_dir=args.cache_dir)
    tokenizer = transformers.CLIPTokenizer.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="tokenizer", cache_dir=args.cache_dir)
    text_encoder = transformers.CLIPTextModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder", cache_dir=args.cache_dir)

    # Freeze everything except UNet
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)

    if is_xformers_available():
        try:
            unet.enable_xformers_memory_efficient_attention()
            logger.info("xformers enabled.")
        except Exception as e:
            logger.warning(f"Could not enable xformers: {e}")

    # ── Dataset ────────────────────────────────────────────────────────────────
    train_dataset = HipXRayDataset(args.tensor_dir, resolution=args.resolution)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, shuffle=True, batch_size=args.train_batch_size, num_workers=2)

    # ── Optimizer + scheduler ──────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        unet.parameters(), lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay, eps=args.adam_epsilon)

    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps)
    num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    lr_scheduler = get_scheduler(
        args.lr_scheduler, optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=args.max_train_steps * args.gradient_accumulation_steps)

    unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        unet, optimizer, train_dataloader, lr_scheduler)

    vae.to(accelerator.device)
    text_encoder.to(accelerator.device)

    if accelerator.is_main_process:
        accelerator.init_trackers("train_diffusion_xray", config=vars(args))

    logger.info(f"Dataset: {len(train_dataset)} images")
    logger.info(f"Steps: {args.max_train_steps}  |  Epochs: {num_train_epochs}")
    logger.info(f"Batch: {args.train_batch_size}  |  LR: {args.learning_rate}")

    global_step = 0
    first_epoch = 0

    # ── Resume ─────────────────────────────────────────────────────────────────
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint == "latest":
            dirs = sorted([d for d in os.listdir(args.output_dir)
                           if d.startswith("checkpoint-")],
                          key=lambda x: int(x.split("-")[1]))
            args.resume_from_checkpoint = str(Path(args.output_dir) / dirs[-1]) if dirs else None

        if args.resume_from_checkpoint and os.path.exists(args.resume_from_checkpoint):
            logger.info(f"Resuming from {args.resume_from_checkpoint}")
            accelerator.load_state(args.resume_from_checkpoint)
            global_step  = int(Path(args.resume_from_checkpoint).name.split("-")[-1])
            first_epoch  = global_step // num_update_steps_per_epoch

    # ── Training loop ──────────────────────────────────────────────────────────
    progress_bar = tqdm(range(global_step, args.max_train_steps),
                        disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Steps")

    for epoch in range(first_epoch, num_train_epochs):
        unet.train()
        for step, batch in enumerate(train_dataloader):
            if global_step >= args.max_train_steps:
                break

            with accelerator.accumulate(unet):
                # Encode image → latent
                vae.eval()
                vae.to(dtype=torch.float32)
                with torch.no_grad():
                    latents = vae.encode(
                        batch["image"].to(accelerator.device, dtype=torch.float32)
                    ).latent_dist.sample()
                latents = latents * vae.config.scaling_factor
                latents = latents.to(unet.dtype)

                # Add noise
                noise = torch.randn_like(latents)
                timesteps = torch.randint(
                    0, noise_scheduler.config.num_train_timesteps,
                    (latents.shape[0],), device=latents.device).long()
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                # ── CLASS CONDITIONING via CLIP ────────────────────────────────
                with torch.no_grad():
                    text_inputs = tokenizer(
                        batch["prompt"], padding="max_length", max_length=77,
                        truncation=True, return_tensors="pt")
                    encoder_hidden_states = text_encoder(
                        text_inputs.input_ids.to(accelerator.device)
                    ).last_hidden_state  # [B, 77, 768]
                # ──────────────────────────────────────────────────────────────

                model_pred = unet(noisy_latents, timesteps,
                                  encoder_hidden_states=encoder_hidden_states).sample

                target = noise
                if noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(latents, noise, timesteps)

                loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(unet.parameters(), args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                accelerator.log({"loss": loss.detach().item(),
                                 "lr": lr_scheduler.get_last_lr()[0]}, step=global_step)

                # Checkpoint — keep only the latest accelerator state to avoid disk explosion.
                # UNet exports are kept at every step (unet_step_XXXX) so any step is
                # restorable via scp after training (~3.5 GB each).
                if global_step % args.checkpointing_steps == 0 and accelerator.is_main_process:
                    import shutil
                    # Delete all previous accelerator states (heavy, ~7 GB each — keep only latest)
                    for old_ckpt in sorted(Path(args.output_dir).glob("checkpoint-*"),
                                           key=lambda p: int(p.name.split("-")[1])):
                        shutil.rmtree(str(old_ckpt))
                        logger.info(f"Removed old accelerator state {old_ckpt.name}")
                    # Save new accelerator state (for resume)
                    save_path = Path(args.output_dir) / f"checkpoint-{global_step}"
                    accelerator.save_state(str(save_path))
                    # Save UNet-only export with step number — kept permanently for local restore
                    unet_path = Path(args.output_dir) / f"unet_step_{global_step}"
                    accelerator.unwrap_model(unet).save_pretrained(str(unet_path))
                    logger.info(f"Saved checkpoint + UNet at step {global_step} "
                                f"(scp ~/diffusion_output/unet_step_{global_step} to restore)")

                # Validation samples — one per class (wrapped in try/except to avoid OOM crash)
                if global_step % args.validation_steps == 0 and accelerator.is_main_process:
                    logger.info("Generating validation samples...")
                    try:
                        torch.cuda.empty_cache()
                        pipeline = diffusers.StableDiffusionPipeline.from_pretrained(
                            args.pretrained_model_name_or_path,
                            unet=accelerator.unwrap_model(unet),
                            vae=vae, text_encoder=text_encoder, tokenizer=tokenizer,
                            scheduler=noise_scheduler,
                            torch_dtype=torch.float16,
                            safety_checker=None, requires_safety_checker=False,
                            cache_dir=args.cache_dir,
                        ).to(accelerator.device)
                        pipeline.set_progress_bar_config(disable=True)

                        sample_dir = Path(args.output_dir) / "samples"
                        os.makedirs(sample_dir, exist_ok=True)
                        prompts = ["OA pre op hip xray", "ONFH pre op hip xray",
                                   "Normal pre op hip xray", "OA post op hip xray"]
                        for i, prompt in enumerate(prompts):
                            img = pipeline(prompt, num_inference_steps=50,
                                           generator=torch.Generator(accelerator.device).manual_seed(i)
                                           ).images[0]
                            img.save(sample_dir / f"step{global_step}_{prompt.replace(' ','_')}.png")
                        logger.info(f"Saved validation samples to {sample_dir}")
                    except Exception as e:
                        logger.warning(f"Validation sampling failed (skipping): {e}")
                    finally:
                        try:
                            del pipeline
                        except NameError:
                            pass
                        torch.cuda.empty_cache()

            if global_step >= args.max_train_steps:
                break

        if global_step >= args.max_train_steps:
            break

    # ── Save final UNet ────────────────────────────────────────────────────────
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        unet_final = Path(args.output_dir) / "unet_final"
        accelerator.unwrap_model(unet).save_pretrained(str(unet_final))
        logger.info(f"Saved final UNet to {unet_final}")

    accelerator.end_training()
    logger.info("Done.")


if __name__ == "__main__":
    main()
