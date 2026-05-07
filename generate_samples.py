"""
generate_samples.py — Generate final diffusion samples for paper figure.

4 prompts x 4 seeds = 16 images, all from the final trained UNet.

Usage:
    python3 generate_samples.py
"""

import os
import torch
from pathlib import Path
from diffusers import StableDiffusionPipeline, UNet2DConditionModel

UNET_PATH   = os.path.expanduser("~/diffusion_output/unet_final")
OUTPUT_DIR  = os.path.expanduser("~/diffusion_samples_final")
BASE_MODEL  = "runwayml/stable-diffusion-v1-5"
NUM_STEPS   = 50
GUIDANCE    = 7.5

PROMPTS = [
    "OA pre op hip xray",
    "ONFH pre op hip xray",
    "Normal pre op hip xray",
    "OA post op hip xray",
]
SEEDS = [0, 1, 2, 3]

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print("Loading fine-tuned UNet...")
    unet = UNet2DConditionModel.from_pretrained(UNET_PATH, torch_dtype=torch.float16)

    print("Loading pipeline...")
    pipeline = StableDiffusionPipeline.from_pretrained(
        BASE_MODEL,
        unet=unet,
        torch_dtype=torch.float16,
        safety_checker=None,
        requires_safety_checker=False,
    ).to(device)
    pipeline.set_progress_bar_config(disable=False)

    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    total = len(PROMPTS) * len(SEEDS)
    done  = 0
    for prompt in PROMPTS:
        prompt_slug = prompt.replace(" ", "_")
        for seed in SEEDS:
            generator = torch.Generator(device).manual_seed(seed)
            img = pipeline(
                prompt,
                num_inference_steps=NUM_STEPS,
                guidance_scale=GUIDANCE,
                generator=generator,
            ).images[0]
            fname = f"{prompt_slug}_seed{seed}.png"
            img.save(Path(OUTPUT_DIR) / fname)
            done += 1
            print(f"[{done}/{total}] Saved {fname}")

    print(f"\nDone. All {total} images in {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
