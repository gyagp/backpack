"""Shared WGSL kernels — used by all model formats.

These kernels are format-agnostic:
  - RMSNorm / LayerNorm
  - Fused residual + norm
  - GQA chunked attention (pass 1 + pass 2)
  - Fused QK-norm + RoPE + KV scatter
  - SiLU activation
  - FP16 weight GEMM (for dequantized weights / LM head)
  - Argmax

All model formats (GGUF Q8, GGUF Q4, ONNX Q4, etc.) use these.
"""

from common.wgsl_kernels import (
    # Attention
    WGSL_GQA_CHUNKED_PASS1,
    WGSL_GQA_CHUNKED_PASS2,
    GQA_CHUNKED_PASS1_BINDINGS,
    GQA_CHUNKED_PASS2_BINDINGS,
    GQA_CHUNK_SIZE,
    pack_gqa_chunked_params,

    # FP16 GEMM (LM head, dequantized weights)
    WGSL_FP16_GEMM_KERNEL,
    FP16_GEMM_BINDINGS,
    pack_fp16_gemm_params,
)

__all__ = [
    'WGSL_GQA_CHUNKED_PASS1', 'WGSL_GQA_CHUNKED_PASS2',
    'GQA_CHUNKED_PASS1_BINDINGS', 'GQA_CHUNKED_PASS2_BINDINGS',
    'GQA_CHUNK_SIZE', 'pack_gqa_chunked_params',
    'WGSL_FP16_GEMM_KERNEL', 'FP16_GEMM_BINDINGS', 'pack_fp16_gemm_params',
]
