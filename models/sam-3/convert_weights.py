"""Convert SAM 2.1 Hiera-Tiny weights to fp32 npz for WebGPU inference.

Downloads from facebook/sam2.1-hiera-tiny via transformers.

Usage:
    python models/sam-3/convert_weights.py

Extracts vision encoder + prompt encoder + mask decoder weights.
"""
import os
import sys
import numpy as np

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_HF_REPO = "facebook/sam2.1-hiera-tiny"


def convert():
    import torch
    from transformers import AutoModelForMaskGeneration

    print(f"Loading model from {_HF_REPO}...")
    model = AutoModelForMaskGeneration.from_pretrained(_HF_REPO)
    sd = model.state_dict()
    print(f"  Total tensors: {len(sd)}")

    np_tensors = {}
    total_bytes = 0
    skipped = 0

    for name, tensor in sd.items():
        # Keep vision encoder, prompt encoder, mask decoder, no_memory_embedding
        keep = (name.startswith("vision_encoder.") or
                name.startswith("prompt_encoder.") or
                name.startswith("mask_decoder.") or
                name.startswith("no_memory"))
        if not keep:
            skipped += 1
            continue

        arr = tensor.cpu().numpy().astype(np.float32)
        np_tensors[name] = arr
        total_bytes += arr.nbytes

    print(f"  Kept {len(np_tensors)} tensors, skipped {skipped}")
    print(f"  Total size: {total_bytes / (1024**2):.1f} MB")

    out_dir = os.path.join(_SCRIPT_DIR, "..", "..", "gitignore", "models", os.path.basename(_SCRIPT_DIR), "weights")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "sam2_hiera_tiny.npz")
    print(f"Saving to {out_path}...")
    np.savez(out_path, **np_tensors)
    fsize = os.path.getsize(out_path)
    print(f"Done! File: {fsize / (1024**2):.1f} MB")

    # Print component stats
    components = {}
    for k, v in np_tensors.items():
        comp = k.split(".")[0]
        if comp == "vision_encoder":
            sub = k.split(".")[1]
            comp = f"vision_encoder.{sub}"
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
