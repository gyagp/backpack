"""
SDXL-Turbo inference on WebGPU via Triton.

SDXL-Turbo is a distilled SDXL model that generates images in 1-4 steps
without classifier-free guidance. Also supports standard SDXL mode with
CFG for higher quality at the cost of more steps.

UNet implementation shared via common/sdxl_unet.py.

Usage:
    python models/sdxl-turbo/model.py --verify
    python models/sdxl-turbo/model.py --prompt "a cat" --steps 1
    python models/sdxl-turbo/model.py --prompt "a landscape" --steps 20 --cfg 7.5
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
from common.utils import add_device_arg, apply_device_arg


def main():
    parser = argparse.ArgumentParser(description="SDXL-Turbo on WebGPU")
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--prompt", type=str,
                        default="a photo of a cat wearing sunglasses")
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--steps", type=int, default=1,
                        help="Denoising steps (1-4 for Turbo, 20+ for CFG)")
    parser.add_argument("--cfg", type=float, default=0.0,
                        help="CFG scale (0 for Turbo mode, 7.5 for SDXL mode)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default="output.png")
    parser.add_argument("--profile", action="store_true")
    add_device_arg(parser)
    args = parser.parse_args()
    apply_device_arg(args)

    if args.verify:
        success = verify_with_random_weights()
        sys.exit(0 if success else 1)

    hf_dir = os.path.join(_SCRIPT_DIR, "..", "..", "gitignore", "models",
                          os.path.basename(_SCRIPT_DIR), "weights", "hf_cache")
    components = load_pipeline_components(hf_dir)
    tokenizer, tokenizer_2, text_encoder, text_encoder_2, vae = components

    wp = os.path.join(_SCRIPT_DIR, "..", "..", "gitignore", "models",
                      os.path.basename(_SCRIPT_DIR), "weights", "unet_fp16.npz")
    if not os.path.exists(wp):
        print(f"UNet weights not found: {wp}")
        sys.exit(1)

    print("Loading UNet weights...")
    t0 = time.perf_counter()
    data = np.load(wp, mmap_mode='r')
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
        num_steps=args.steps, guidance_scale=args.cfg, seed=args.seed)

    out_dir = os.path.join(_SCRIPT_DIR, "..", "..", "gitignore", "models",
                          os.path.basename(_SCRIPT_DIR))
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, args.output)
    image.save(out)
    print(f"\nSaved to {out}")

    if args.profile:
        model.save_profile(_SCRIPT_DIR, "SDXL-Turbo")


if __name__ == "__main__":
    main()
