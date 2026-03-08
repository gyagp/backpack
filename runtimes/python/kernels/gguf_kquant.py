"""GGUF K-quant format-specific WGSL kernels.

These kernels handle llama.cpp K-quantization formats:
  - Q4_K: 4-bit with per-block super-blocks
  - Q5_K: 5-bit with per-block super-blocks
  - Q6_K: 6-bit with per-block super-blocks

Each format has its own block layout requiring dedicated decode logic.
"""

from common.wgsl_kernels import (
    WGSL_Q4K_KERNEL,
    WGSL_Q5K_KERNEL,
    WGSL_Q6K_KERNEL,
    Q4K_BINDINGS,
    Q5K_BINDINGS,
    Q6K_BINDINGS,
    pack_q4k_params,
)

__all__ = [
    'WGSL_Q4K_KERNEL', 'Q4K_BINDINGS',
    'WGSL_Q5K_KERNEL', 'Q5K_BINDINGS',
    'WGSL_Q6K_KERNEL', 'Q6K_BINDINGS',
    'pack_q4k_params',
]
