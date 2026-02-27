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
    import re

    # Time embedding
    name = name.replace("time_embedding.linear_1", "time_embed.0")
    name = name.replace("time_embedding.linear_2", "time_embed.2")
    # Add timestep cond projection
    name = name.replace("add_time_proj", "add_time_proj")
    name = name.replace("add_embedding.linear_1", "add_embed.0")
    name = name.replace("add_embedding.linear_2", "add_embed.2")

    # Input / output convolutions
    name = name.replace("conv_in.", "input_conv.")
    name = name.replace("conv_out.", "output_conv.")
    name = name.replace("conv_norm_out.", "out_norm.")

    # Down blocks: down_blocks.{i}.resnets.{j} → down.{i}.res{j}
    name = re.sub(r'down_blocks\.(\d+)\.resnets\.(\d+)\.',
                  r'down.\1.res\2.', name)
    # Down block attention: down_blocks.{i}.attentions.{j}.transformer_blocks.{k}
    #   → down.{i}.attn{j}.t{k}
    name = re.sub(r'down_blocks\.(\d+)\.attentions\.(\d+)\.transformer_blocks\.(\d+)\.',
                  r'down.\1.attn\2.t\3.', name)
    # Attention wrapper norm/proj: down_blocks.{i}.attentions.{j}.{component}
    name = re.sub(r'down_blocks\.(\d+)\.attentions\.(\d+)\.',
                  r'down.\1.attn\2.', name)

    # Mid block resnets: mid_block.resnets.0 → mid.res1, mid_block.resnets.1 → mid.res2
    name = name.replace("mid_block.resnets.0.", "mid.res1.")
    name = name.replace("mid_block.resnets.1.", "mid.res2.")
    # Mid block attention: mid_block.attentions.0.transformer_blocks.{k} → mid.attn.t{k}
    name = re.sub(r'mid_block\.attentions\.0\.transformer_blocks\.(\d+)\.',
                  r'mid.attn.t\1.', name)
    name = name.replace("mid_block.attentions.0.", "mid.attn.")

    # Up blocks: up_blocks.{i}.resnets.{j} → up.{i}.res{j}
    name = re.sub(r'up_blocks\.(\d+)\.resnets\.(\d+)\.',
                  r'up.\1.res\2.', name)
    # Up block attention: same pattern as down
    name = re.sub(r'up_blocks\.(\d+)\.attentions\.(\d+)\.transformer_blocks\.(\d+)\.',
                  r'up.\1.attn\2.t\3.', name)
    name = re.sub(r'up_blocks\.(\d+)\.attentions\.(\d+)\.',
                  r'up.\1.attn\2.', name)

    # ResNet shortcut projections: conv_shortcut → proj
    name = name.replace(".conv_shortcut.", ".proj.")

    # Timestep projection in resnets: time_emb_proj → temb
    name = name.replace(".time_emb_proj.", ".temb.")

    # Downsamplers: down_blocks.{i}.downsamplers.0.conv → down.{i}.downsample
    name = re.sub(r'down_blocks\.(\d+)\.downsamplers\.0\.conv\.',
                  r'down.\1.downsample.', name)

    # Upsamplers: up_blocks.{i}.upsamplers.0.conv → up.{i}.upsample
    name = re.sub(r'up_blocks\.(\d+)\.upsamplers\.0\.conv\.',
                  r'up.\1.upsample.', name)

    # proj_in at attention level → proj_in (keep as named)

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

    # Convert to fp16 numpy with key renaming and conv reshaping
    np_tensors = {}
    total_bytes = 0
    skipped = []
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

        # Reshape conv weights: (C_out, C_in, 1, 1) → (C_out, C_in)
        # and (C_out, C_in, 3, 3) — skip 3×3 convs (need im2col)
        if arr.ndim == 4:
            if arr.shape[2] == 1 and arr.shape[3] == 1:
                arr = arr.reshape(arr.shape[0], arr.shape[1])
            elif arr.shape[2] == 3 and arr.shape[3] == 3:
                # Keep 3×3 convs (downsamplers) as-is for now
                pass

        new_name = _rename_key(name)
        np_tensors[new_name] = arr
        total_bytes += arr.nbytes

    out_dir = os.path.join(_SCRIPT_DIR, "weights")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "unet_fp16.npz")
    print(f"Saving {len(np_tensors)} tensors ({total_bytes / (1024**2):.0f} MB) "
          f"to {out_path}...")
    np.savez(out_path, **np_tensors)
    print(f"Done! File size: {os.path.getsize(out_path) / (1024**2):.0f} MB")

    # Print key name samples for verification
    keys = sorted(np_tensors.keys())
    print(f"\nSample keys after renaming:")
    for k in keys[:10]:
        print(f"  {k}  {np_tensors[k].shape}")
    print("  ...")
    for k in keys[-5:]:
        print(f"  {k}  {np_tensors[k].shape}")


if __name__ == "__main__":
    convert()
