"""GGUF Q8_0 format-specific WGSL kernels.

These kernels handle INT8 per-block quantized weights as stored in
GGUF Q8_0 format: 32 int8 values + 1 fp16 scale per block.

Kernels:
  - Q8 matmul (store result)
  - Q8 matmul + residual add (fused down_proj + add)
  - Subgroup-matrix Q8 matmul (cooperative matrix prefill)
"""

from common.wgsl_kernels import (
    # Q8_0 matmul
    WSGL_Q8_0_KERNEL as Q8_MATMUL_KERNEL,
    Q8_DP4A_BINDINGS as Q8_MATMUL_BINDINGS,
    Q8_TILE_N,
    pack_q8_params,

    # Q8_0 matmul + residual add
    WSGL_Q8_0_ADD_KERNEL as Q8_MATMUL_ADD_KERNEL,
    Q8_ADD_BINDINGS as Q8_MATMUL_ADD_BINDINGS,

    # Subgroup-matrix Q8 (cooperative matrix)
    WGSL_SUBGROUP_MATRIX_Q8_KERNEL,
    WSGL_SUBGROUP_MATRIX_Q8_ADD_KERNEL,
)

__all__ = [
    'Q8_MATMUL_KERNEL', 'Q8_MATMUL_BINDINGS', 'Q8_TILE_N', 'pack_q8_params',
    'Q8_MATMUL_ADD_KERNEL', 'Q8_MATMUL_ADD_BINDINGS',
    'WGSL_SUBGROUP_MATRIX_Q8_KERNEL', 'WSGL_SUBGROUP_MATRIX_Q8_ADD_KERNEL',
]
