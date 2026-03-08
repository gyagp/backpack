"""ONNX MatMulNBits Q4 format-specific WGSL kernels.

These kernels handle the ONNX GenAI MatMulNBits Q4 weight format:
  - Weights: [N, K/block_size, block_size/2] uint8 (2 nibbles per byte)
  - Scales: [N, K/block_size] fp16
  - No zero point (symmetric quantization with offset 8)

NOTE: Currently ONNX Q4 models are dequantized to fp16 at load time
and use the shared FP16 GEMM kernel. A native Q4 kernel would provide
4x memory savings and better performance. This is the placeholder for
that future kernel.
"""

# Future: native ONNX Q4 matmul kernel
# WGSL_ONNX_Q4_MATMUL = """..."""
# ONNX_Q4_MATMUL_BINDINGS = [...]

__all__ = []
