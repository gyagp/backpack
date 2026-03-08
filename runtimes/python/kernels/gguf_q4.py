"""GGUF Q4_0 format-specific WGSL kernels.

These kernels handle INT4 per-group quantized weights as stored in
GGUF Q4_0 format: 32 int4 values + 1 fp16 scale + 1 fp16 zero per group.

Kernels:
  - Q4 matmul (with scale + zero dequant)
"""

from common.wgsl_kernels import (
    WGSL_Q4_DP4A_KERNEL as Q4_MATMUL_KERNEL,
    Q4_DP4A_BINDINGS as Q4_MATMUL_BINDINGS,
    pack_dp4a_params as pack_q4_params,
    TILE_N as Q4_TILE_N,
)

__all__ = [
    'Q4_MATMUL_KERNEL', 'Q4_MATMUL_BINDINGS', 'pack_q4_params', 'Q4_TILE_N',
]
