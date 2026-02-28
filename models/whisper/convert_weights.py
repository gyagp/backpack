"""Convert Whisper-Tiny weights from safetensors to fp16 npz for WebGPU.

Downloads from openai/whisper-tiny (151 MB, not gated).

Usage:
    python python/examples/webgpu/whisper/convert_weights.py

Extracts encoder + decoder weights, converts to fp16 numpy.
"""
import os
import sys
import numpy as np

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_HF_REPO = "openai/whisper-tiny"


def convert():
    from huggingface_hub import hf_hub_download
    from safetensors import safe_open

    print(f"Downloading model from {_HF_REPO}...")
    sf_path = hf_hub_download(_HF_REPO, "model.safetensors")
    print(f"  Downloaded to {sf_path}")

    print("Loading safetensors...")
    with safe_open(sf_path, framework="numpy") as f:
        keys = sorted(f.keys())
        print(f"  Total tensors: {len(keys)}")

        np_tensors = {}
        total_bytes = 0

        for name in keys:
            arr = f.get_tensor(name)

            # Strip "model." prefix for cleaner names
            clean_name = name
            if clean_name.startswith("model."):
                clean_name = clean_name[len("model."):]

            if arr.dtype == np.float32:
                fp16 = arr.astype(np.float16)
            elif arr.dtype == np.float16:
                fp16 = arr
            else:
                fp16 = arr.astype(np.float16)

            np_tensors[clean_name] = fp16
            total_bytes += fp16.nbytes

    print(f"  Converted {len(np_tensors)} tensors")
    print(f"  Total size: {total_bytes / (1024**2):.1f} MB")

    out_dir = os.path.join(_SCRIPT_DIR, "..", "..", "gitignore", "models", os.path.basename(_SCRIPT_DIR), "weights")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "whisper_tiny_fp16.npz")
    print(f"Saving to {out_path}...")
    np.savez(out_path, **np_tensors)
    print(f"Done! File: {os.path.getsize(out_path) / (1024**2):.1f} MB")

    # Print component stats
    components = {}
    for k, v in np_tensors.items():
        comp = k.split(".")[0]
        if comp not in components:
            components[comp] = {"count": 0, "bytes": 0}
        components[comp]["count"] += 1
        components[comp]["bytes"] += v.nbytes
    print("\nComponents:")
    for comp, info in sorted(components.items()):
        print(f"  {comp}: {info['count']} tensors, "
              f"{info['bytes'] / (1024**2):.1f} MB")


if __name__ == "__main__":
    convert()
