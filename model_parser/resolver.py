"""
Model format auto-detection and path resolution.

Models live in a separate repo (ai-models/) and may have multiple formats.
This module discovers available formats (GGUF, ONNX) for any model.

Shared by both compiler and runtime stages.
"""

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

DEFAULT_MODEL_REPO = os.environ.get(
    "BACKPACK_MODELS", r"E:\workspace\project\ai-models")


@dataclass
class ModelInfo:
    """Discovered model with all available formats."""
    name: str                          # Model name (directory name)
    formats: Dict[str, str] = field(default_factory=dict)
    # format -> path: e.g. {"gguf": "/path/to/model.gguf", "onnx": "/path/to/dir"}

    @property
    def has_gguf(self) -> bool:
        return 'gguf' in self.formats

    @property
    def has_onnx(self) -> bool:
        return 'onnx' in self.formats

    @property
    def preferred_format(self) -> str:
        """Return the best available format (GGUF preferred for quality)."""
        if 'gguf' in self.formats:
            return 'gguf'
        if 'onnx' in self.formats:
            return 'onnx'
        raise FileNotFoundError(f"No model format found for {self.name}")

    @property
    def preferred_path(self) -> str:
        return self.formats[self.preferred_format]

    def summary(self) -> str:
        parts = []
        for fmt, path in sorted(self.formats.items()):
            basename = os.path.basename(path) if fmt == 'gguf' else path
            parts.append(f"{fmt}: {basename}")
        return f"{self.name} [{', '.join(parts)}]"


def discover_model(path: str) -> ModelInfo:
    """Discover all available formats for a model.

    Accepts:
      - Direct GGUF file path
      - Model directory
      - Model name (looked up in BACKPACK_MODELS)

    Returns ModelInfo with all found formats.
    """
    # Direct GGUF file
    if path.endswith('.gguf') and os.path.isfile(path):
        name = os.path.basename(os.path.dirname(path))
        return ModelInfo(name=name, formats={'gguf': path})

    # Find directory
    model_dir = None
    if os.path.isdir(path):
        model_dir = path
    else:
        repo_path = os.path.join(DEFAULT_MODEL_REPO, path)
        if os.path.isdir(repo_path):
            model_dir = repo_path

    if not model_dir:
        raise FileNotFoundError(
            f"Model not found: {path}\n"
            f"  Model repo: {DEFAULT_MODEL_REPO}")

    name = os.path.basename(model_dir)
    formats = {}

    # Search for GGUF files
    gguf_files = []
    for root, _, files in os.walk(model_dir):
        gguf_files.extend(os.path.join(root, f) for f in files
                          if f.endswith('.gguf'))
    if gguf_files:
        # Prefer Q8_0 > Q4_K > any
        best = gguf_files[0]
        for gf in gguf_files:
            if 'Q8_0' in gf: best = gf; break
        else:
            for gf in gguf_files:
                if 'Q4_K' in gf: best = gf; break
        formats['gguf'] = best

    # Search for ONNX model
    if os.path.exists(os.path.join(model_dir, 'model.onnx')):
        formats['onnx'] = model_dir
    else:
        for sub in os.listdir(model_dir):
            sub_path = os.path.join(model_dir, sub)
            if os.path.isdir(sub_path) and os.path.exists(
                    os.path.join(sub_path, 'model.onnx')):
                formats['onnx'] = sub_path
                break

    if not formats:
        raise FileNotFoundError(
            f"No model files (GGUF or ONNX) found in: {model_dir}")

    return ModelInfo(name=name, formats=formats)


def resolve_model(path: str, prefer: str = 'gguf') -> Tuple[str, str]:
    """Resolve a model path and detect format.

    Returns: (model_path, format) where format is 'gguf' or 'onnx'.
    Prefers the format specified by `prefer` if available.
    """
    info = discover_model(path)
    if prefer in info.formats:
        return info.formats[prefer], prefer
    # Fall back to whatever is available
    fmt = info.preferred_format
    return info.formats[fmt], fmt
