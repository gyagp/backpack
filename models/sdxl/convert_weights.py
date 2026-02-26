"""Convert SDXL UNet weights from safetensors to fp16 npz for WebGPU inference.

Supports both SDXL 1.0 base and Turbo variants.

Usage:
    # First download the model:
    huggingface-cli download stabilityai/sdxl-turbo --local-dir weights/hf_cache

    # Then convert:
    python python/examples/webgpu/sdxl/convert_weights.py

Reads from:  weights/hf_cache/unet/
Writes to:   weights/unet_fp16.npz
"""
import os
import sys
import json
import numpy as np

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_WEIGHTS_DIR = os.path.join(_SCRIPT_DIR, "weights", "hf_cache")


def _rename_key(name: str) -> str:
    """Map diffusers UNet state_dict keys to our simplified naming."""
    # Time embedding
    name = name.replace("time_embedding.linear_1", "time_embed.0")
    name = name.replace("time_embedding.linear_2", "time_embed.2")
    # Add timestep cond projection
    name = name.replace("add_time_proj", "add_time_proj")
    name = name.replace("add_embedding.linear_1", "add_embed.0")
    name = name.replace("add_embedding.linear_2", "add_embed.2")
    return name


def convert():
    import torch
    from safetensors.torch import load_file

    unet_dir = os.path.join(_WEIGHTS_DIR, "unet")
    if not os.path.exists(unet_dir):
        print(f"ERROR: UNet weights not found at {unet_dir}")
        print("Download with:")
        print(f"  huggingface-cli download stabilityai/sdxl-turbo "
              f"--local-dir {_WEIGHTS_DIR}")
        sys.exit(1)

    # Find safetensors files
    st_files = sorted(f for f in os.listdir(unet_dir)
                      if f.endswith('.safetensors'))
    if not st_files:
        print(f"No safetensors files found in {unet_dir}")
        sys.exit(1)

    print(f"Loading UNet from {len(st_files)} shard(s)...")
    all_tensors = {}
    for sf in st_files:
        path = os.path.join(unet_dir, sf)
        sz = os.path.getsize(path) // (1024**2)
        print(f"  Loading {sf} ({sz} MB)...")
        tensors = load_file(path, device="cpu")
        all_tensors.update(tensors)
    print(f"  Loaded {len(all_tensors)} tensors")

    # Convert to fp16 numpy
    np_tensors = {}
    total_bytes = 0
    for name in sorted(all_tensors.keys()):
        t = all_tensors[name]
        if t.dtype == torch.bfloat16:
            arr = t.float().half().numpy()
        elif t.dtype == torch.float16:
            arr = t.numpy()
        elif t.dtype == torch.float32:
            arr = t.half().numpy()
        else:
            arr = t.numpy()
        np_tensors[name] = arr
        total_bytes += arr.nbytes

    out_dir = os.path.join(_SCRIPT_DIR, "weights")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "unet_fp16.npz")
    print(f"Saving {len(np_tensors)} tensors ({total_bytes / (1024**2):.0f} MB) "
          f"to {out_path}...")
    np.savez(out_path, **np_tensors)
    print(f"Done! File size: {os.path.getsize(out_path) / (1024**2):.0f} MB")

    # Print architecture info
    shapes = {k: v.shape for k, v in np_tensors.items()}
    down_blocks = set()
    up_blocks = set()
    for k in shapes:
        if k.startswith("down_blocks."):
            idx = k.split(".")[1]
            down_blocks.add(int(idx))
        if k.startswith("up_blocks."):
            idx = k.split(".")[1]
            up_blocks.add(int(idx))
    print(f"\nArchitecture: {len(down_blocks)} down blocks, "
          f"{len(up_blocks)} up blocks")


if __name__ == "__main__":
    convert()
