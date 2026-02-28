"""
Stable Diffusion XL (SDXL) inference on WebGPU via Triton.

SDXL base model with classifier-free guidance (CFG).
UNet implementation shared via common/sdxl_unet.py.

Usage:
    python models/sdxl/model.py --verify
    python models/sdxl/model.py --prompt "a landscape" --steps 20 --cfg 7.5
"""
import argparse
import os
import sys
import time

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_SCRIPT_DIR))

import numpy as np

from common.sdxl_unet import (
    SDXLWebGPU, SDXL_CONFIGS, EulerDiscreteScheduler,
    encode_prompt, vae_decode, verify_with_random_weights,
    generate_image, load_pipeline_components, VAE_SCALE_FACTOR,
)


def main():
    parser = argparse.ArgumentParser(description="SDXL on WebGPU")
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--prompt", type=str,
                        default="a beautiful landscape painting")
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--cfg", type=float, default=7.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default="output.png")
    parser.add_argument("--profile", action="store_true")
    args = parser.parse_args()

    if args.verify:
        success = verify_with_random_weights()
        sys.exit(0 if success else 1)

    hf_dir = os.path.join(_SCRIPT_DIR, "..", "..", "gitignore", "models", os.path.basename(_SCRIPT_DIR), "weights", "hf_cache")
    components = load_pipeline_components(hf_dir)
    tokenizer, tokenizer_2, text_encoder, text_encoder_2, vae = components

    wp = os.path.join(_SCRIPT_DIR, "..", "..", "gitignore", "models", os.path.basename(_SCRIPT_DIR), "weights", "unet_fp16.npz")
    if not os.path.exists(wp):
        print(f"UNet weights not found: {wp}")
        sys.exit(1)

    print("Loading UNet weights...")
    t0 = time.perf_counter()
    data = np.load(wp, mmap_mode='r')
    weights = {k: data[k] for k in data.files}
    print(f"  {len(weights)} tensors in {time.perf_counter()-t0:.1f}s")

    config = SDXL_CONFIGS["sdxl-base"]
    model = SDXLWebGPU(weights, **{k: v for k, v in config.items()
                                   if k != "transformer_depth"},
                       transformer_depth=config.get("transformer_depth"))

    if args.profile:
        model.enable_profiling()

    image = generate_image(
        model, tokenizer, tokenizer_2, text_encoder, text_encoder_2, vae,
        prompt=args.prompt, height=args.height, width=args.width,
        num_steps=args.steps, guidance_scale=args.cfg, seed=args.seed)

    out = os.path.join(_SCRIPT_DIR, args.output)
    image.save(out)
    print(f"\nSaved to {out}")

    if args.profile:
        model.save_profile(_SCRIPT_DIR, "SDXL")


if __name__ == "__main__":
    main()
