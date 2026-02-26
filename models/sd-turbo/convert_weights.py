"""Convert SDXL-Turbo UNet weights from safetensors to fp16 npz for WebGPU.

Downloads from stabilityai/sdxl-turbo (equivalent to microsoft/sdxl-turbo-webnn).

Usage:
    # Download the model first:
    huggingface-cli download stabilityai/sdxl-turbo --local-dir weights/hf_cache \
        --include "unet/*" "vae/*" "tokenizer/*" "tokenizer_2/*" \
        "text_encoder/*" "text_encoder_2/*" "scheduler/*" "model_index.json"

    # Then convert:
    python python/examples/webgpu/sd-turbo/convert_weights.py
"""
import os
import sys
import numpy as np

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_WEIGHTS_DIR = os.path.join(_SCRIPT_DIR, "weights", "hf_cache")


def convert():
    from safetensors.torch import load_file
    import torch

    unet_dir = os.path.join(_WEIGHTS_DIR, "unet")
    if not os.path.exists(unet_dir):
        print(f"ERROR: UNet weights not found at {unet_dir}")
        print("Download with:")
        print("  huggingface-cli download stabilityai/sdxl-turbo "
              f"--local-dir {_WEIGHTS_DIR}")
        sys.exit(1)

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
    print(f"Saving {len(np_tensors)} tensors ({total_bytes // (1024**2)} MB) "
          f"to {out_path}...")
    np.savez(out_path, **np_tensors)
    print(f"Done! File size: {os.path.getsize(out_path) // (1024**2)} MB")


if __name__ == "__main__":
    convert()
