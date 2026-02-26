"""Shared WebGPU inference infrastructure.

Re-exports the most commonly used classes and functions so model files can do:
    from common import WebGPUModel, GPUBuffer, load_weights, generate, ...
"""
from common.model_base import WebGPUModel, KernelCache, _next_pow2
from common.utils import (
    _parse_safetensors, load_weights, download_weights,
    load_tokenizer, generate,
)

# Re-export GPUBuffer from the Dawn runner (used by some models directly)
try:
    from triton.backends.webgpu.dawn_runner import GPUBuffer
except ImportError:
    pass
