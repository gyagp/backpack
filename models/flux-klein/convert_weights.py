"""Convert FLUX.2 Klein 4B transformer weights from safetensors to fp16 npz.

Usage:
    # First download the model:
    huggingface-cli download black-forest-labs/FLUX.2-klein-4B \
        --local-dir weights/hf_cache

    # Then convert:
    python python/examples/webgpu/flux-klein/convert_weights.py

Reads from:  weights/hf_cache/transformer/
Writes to:   weights/transformer_fp16.npz
"""
import os
import sys
import numpy as np

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_WEIGHTS_DIR = os.path.join(_SCRIPT_DIR, "weights", "hf_cache")


def convert():
    import torch
    from safetensors.torch import load_file

    transformer_dir = os.path.join(_WEIGHTS_DIR, "transformer")
    if not os.path.exists(transformer_dir):
        print(f"ERROR: Transformer weights not found at {transformer_dir}")
        print("Download with:")
        print("  huggingface-cli download black-forest-labs/FLUX.2-klein-4B "
              f"--local-dir {_WEIGHTS_DIR}")
        sys.exit(1)

    st_files = sorted(f for f in os.listdir(transformer_dir)
                      if f.endswith('.safetensors'))
    if not st_files:
        print(f"No safetensors files in {transformer_dir}")
        sys.exit(1)

    print(f"Loading transformer from {len(st_files)} shard(s)...")
    all_tensors = {}
    for sf in st_files:
        path = os.path.join(transformer_dir, sf)
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
    out_path = os.path.join(out_dir, "transformer_fp16.npz")
    print(f"Saving {len(np_tensors)} tensors ({total_bytes / (1024**2):.0f} MB) "
          f"to {out_path}...")
    np.savez(out_path, **np_tensors)
    print(f"Done! File size: {os.path.getsize(out_path) / (1024**2):.0f} MB")

    # Print summary
    print(f"\nWeight summary:")
    print(f"  Total tensors: {len(np_tensors)}")
    print(f"  Double blocks: 5  (transformer_blocks.0-4)")
    print(f"  Single blocks: 20 (single_transformer_blocks.0-19)")
    for name, arr in sorted(np_tensors.items()):
        print(f"  {name}: {arr.shape} {arr.dtype}")


if __name__ == "__main__":
    convert()
