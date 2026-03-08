"""
Model format auto-detection and path resolution.

Shared by both compiler and runtime stages.
"""

import os
from typing import Tuple

DEFAULT_MODEL_REPO = os.environ.get(
    "BACKPACK_MODELS", r"E:\workspace\project\ai-models")


def resolve_model(path: str) -> Tuple[str, str]:
    """Resolve a model path and detect format.

    Returns: (model_path, format) where format is 'gguf' or 'onnx'.

    Accepts:
      - Direct GGUF file: model.gguf -> (path, 'gguf')
      - Model directory with GGUF inside -> (gguf_path, 'gguf')
      - Model directory with ONNX inside -> (dir_path, 'onnx')
      - Model name: looks up in BACKPACK_MODELS env
    """
    # Direct GGUF file
    if path.endswith('.gguf') and os.path.isfile(path):
        return path, 'gguf'

    # Model directory or model name
    search_dirs = []
    if os.path.isdir(path):
        search_dirs.append(path)
    else:
        repo_path = os.path.join(DEFAULT_MODEL_REPO, path)
        if os.path.isdir(repo_path):
            search_dirs.append(repo_path)

    for d in search_dirs:
        # Check for GGUF files first
        gguf_files = []
        for root, _, files in os.walk(d):
            gguf_files.extend(os.path.join(root, f) for f in files
                              if f.endswith('.gguf'))
        if gguf_files:
            for gf in gguf_files:
                if 'Q8_0' in gf: return gf, 'gguf'
            for gf in gguf_files:
                if 'Q4_K' in gf: return gf, 'gguf'
            return gguf_files[0], 'gguf'

        # Check for ONNX model
        if os.path.exists(os.path.join(d, 'model.onnx')):
            return d, 'onnx'

        # Check subdirectories (e.g., phi-4-mini/webgpu/)
        for sub in os.listdir(d):
            sub_path = os.path.join(d, sub)
            if os.path.isdir(sub_path) and os.path.exists(
                    os.path.join(sub_path, 'model.onnx')):
                return sub_path, 'onnx'

    raise FileNotFoundError(
        f"No model found at: {path}\n"
        f"  Searched: {search_dirs or [path]}\n"
        f"  Model repo: {DEFAULT_MODEL_REPO}")
