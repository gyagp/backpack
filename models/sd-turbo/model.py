"""
Stable Diffusion Turbo (SDXL-Turbo) inference on WebGPU via Triton.

Thin wrapper around the SDXL example with turbo-specific defaults:
  - 1-4 denoising steps (vs 20-50 for base SDXL)
  - No classifier-free guidance (cfg=0.0)
  - 512×512 default resolution
  - Uses stabilityai/sdxl-turbo weights

Model source: stabilityai/sdxl-turbo (public, ~5GB UNet)
Based on: microsoft/sdxl-turbo-webnn architecture

Usage:
    python python/examples/webgpu/sd-turbo/model.py --verify
    python python/examples/webgpu/sd-turbo/model.py --prompt "a cat" --steps 1
    python python/examples/webgpu/sd-turbo/model.py --prompt "a cat" --steps 4

Requirements:
    pip install torch diffusers transformers safetensors pillow
    Dawn WebGPU library built at third_party/webgpu/dawn/build/
"""
import argparse
import os
import sys
import time

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# Add parent (webgpu/) for common imports and sibling access
sys.path.insert(0, os.path.dirname(_SCRIPT_DIR))

import numpy as np

# Reuse SDXL implementation
from sdxl.model import (
    SDXLWebGPU, SDXL_CONFIGS, EulerDiscreteScheduler,
    encode_prompt, vae_decode, verify_with_random_weights,
    VAE_SCALE_FACTOR,
)


# ---------------------------------------------------------------------------
# Pipeline component loading
# ---------------------------------------------------------------------------

def load_pipeline_components():
    """Load CLIP text encoders and VAE from local cache."""
    import torch
    from diffusers import AutoencoderKL
    from transformers import CLIPTextModel, CLIPTextModelWithProjection
    from transformers import CLIPTokenizer

    hf_dir = os.path.join(_SCRIPT_DIR, "weights", "hf_cache")

    print("Loading SD-Turbo pipeline components...")
    t0 = time.perf_counter()

    tokenizer = CLIPTokenizer.from_pretrained(
        hf_dir, subfolder="tokenizer")
    tokenizer_2 = CLIPTokenizer.from_pretrained(
        hf_dir, subfolder="tokenizer_2")
    text_encoder = CLIPTextModel.from_pretrained(
        hf_dir, subfolder="text_encoder", torch_dtype=torch.float16)
    text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(
        hf_dir, subfolder="text_encoder_2", torch_dtype=torch.float16)
    vae = AutoencoderKL.from_pretrained(
        hf_dir, subfolder="vae", torch_dtype=torch.float16)

    t1 = time.perf_counter()
    print(f"  Loaded in {t1-t0:.1f}s")
    return (tokenizer, tokenizer_2, text_encoder, text_encoder_2, vae)


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

def generate_image(model, tokenizer, tokenizer_2, text_encoder,
                   text_encoder_2, vae, prompt, height=512, width=512,
                   num_steps=1, seed=42):
    """Generate image using SD-Turbo on WebGPU.

    SD-Turbo uses adversarial distillation and needs no CFG (guidance=0).
    Best results with 1-4 steps.
    """
    np.random.seed(seed)
    lat_h = height // VAE_SCALE_FACTOR
    lat_w = width // VAE_SCALE_FACTOR
    print(f"Image: {width}x{height} -> latent: {lat_w}x{lat_h}")

    # Encode text
    print("Encoding prompt...")
    context, pooled = encode_prompt(
        tokenizer, tokenizer_2, text_encoder, text_encoder_2, prompt)
    print(f"  Context: {context.shape}, pooled: {pooled.shape}")

    # Initialize noise
    latent = np.random.randn(4, lat_h, lat_w).astype(np.float32)

    # Scheduler (Euler discrete)
    scheduler = EulerDiscreteScheduler()
    timesteps = scheduler.get_timesteps(num_steps)
    sigmas = scheduler.get_sigmas(timesteps)

    # Scale initial noise
    latent = latent * sigmas[0]

    print(f"Denoising ({num_steps} step{'s' if num_steps > 1 else ''})...")
    t_start = time.perf_counter()

    for step in range(num_steps):
        sigma = sigmas[step]
        sigma_next = sigmas[step + 1]
        ts = timesteps[step]

        latent_input = scheduler.scale_input(latent, sigma)
        # SDXL time_ids: [orig_h, orig_w, crop_top, crop_left, target_h, target_w]
        time_ids = np.array([height, width, 0, 0, height, width],
                            dtype=np.float32)
        noise_pred = model.forward(latent_input, timestep=ts,
                                   context=context,
                                   pooled_text_embeds=pooled,
                                   time_ids=time_ids)
        latent = scheduler.step(noise_pred, sigma, sigma_next, latent)

        elapsed = time.perf_counter() - t_start
        print(f"  Step {step+1}/{num_steps} t={ts:.0f} "
              f"({elapsed:.1f}s, {elapsed/(step+1):.2f}s/step)")

    print(f"  Done in {time.perf_counter()-t_start:.1f}s")

    # VAE decode
    print("VAE decoding...")
    image = vae_decode(vae, latent)
    return image


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Stable Diffusion Turbo on WebGPU")
    parser.add_argument("--verify", action="store_true",
                        help="Verify UNet with random weights (tiny config)")
    parser.add_argument("--prompt", type=str,
                        default="a photo of a cat wearing sunglasses")
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--steps", type=int, default=1,
                        help="Denoising steps (1-4 for turbo)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default="output.png")
    parser.add_argument("--profile", action="store_true")
    args = parser.parse_args()

    if args.verify:
        success = verify_with_random_weights()
        sys.exit(0 if success else 1)

    # --- Full inference ---
    components = load_pipeline_components()
    tokenizer, tokenizer_2, text_encoder, text_encoder_2, vae = components

    # Load UNet weights
    wp = os.path.join(_SCRIPT_DIR, "weights", "unet_fp16.npz")
    if not os.path.exists(wp):
        print(f"UNet weights not found: {wp}")
        print("Run: python python/examples/webgpu/sd-turbo/convert_weights.py")
        sys.exit(1)

    print("Loading UNet weights...")
    t0 = time.perf_counter()
    data = np.load(wp)
    weights = {k: data[k] for k in data.files}
    print(f"  {len(weights)} tensors in {time.perf_counter()-t0:.1f}s")

    config = SDXL_CONFIGS["sdxl-turbo"]
    model = SDXLWebGPU(weights, **{k: v for k, v in config.items()
                                   if k != "transformer_depth"},
                       transformer_depth=config.get("transformer_depth"))

    if args.profile:
        model.enable_profiling()

    image = generate_image(
        model, tokenizer, tokenizer_2, text_encoder, text_encoder_2, vae,
        prompt=args.prompt, height=args.height, width=args.width,
        num_steps=args.steps, seed=args.seed)

    out = os.path.join(_SCRIPT_DIR, args.output)
    image.save(out)
    print(f"\nSaved to {out}")

    if args.profile:
        model.profiler.report()


if __name__ == "__main__":
    main()
