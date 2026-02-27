# GPT-OSS-20B (20B params, MoE) — Performance Data

## Test Configuration

| Property | Value |
|----------|-------|
| Machine | Desktop, 32 CPU cores, DDR5 RAM |
| OS | Windows |
| Dawn | Custom build with TimedWaitAny |
| Python | 3.13 |
| NumPy | OpenBLAS 0.3.30 |

## Model

**Model**: gpt-oss/gpt-oss-20B
**Architecture**: 24 layers, 64 Q heads, 8 KV heads, 2880 embd, head_dim=64, 32 MoE experts (top-4), intermediate_size=2880, 201088 vocab
**Quantization**: MXFP4 expert weights (packed FP4 blocks + E8M0 scales), fp16 attention weights
**Weight size**: 11.76 GB VRAM (RTX 5080)
**KV cache**: CPU-side, sliding window 128 + 4 sink tokens

## Results

| GPU | Backend | Decode (tok/s) | Forward (ms) | Notes |
|-----|---------|----------------|--------------|-------|
| NVIDIA GeForce RTX 5080 (16GB) | D3D12 | 18.5 | 1140 | Fused QKV + vectorized attention + GPU residuals |

## Optimization History

| Date | Change | Decode (tok/s) | Forward (ms) | Speedup |
|------|--------|----------------|--------------|---------|
| 2026-02-28 | Baseline (per-head attention loop, separate Q/K/V/O) | 13.7 | 1595 | 1.0× |
| 2026-02-28 | Fuse Q/K/V into single QKV matmul (4→1 dispatch/layer) | — | — | — |
| 2026-02-28 | Vectorize attention with numpy einsum (eliminate 64-head loop) | 19.5 | 1280 | 1.42× |
| 2026-02-28 | GPU-resident residual connections (_add on GPU) | 18.5 | 1140 | 1.35× |

**Key bottleneck**: MoE dispatch overhead — 4 experts × 3 dispatches each = 12 GPU dispatches per layer for MoE alone. Each dispatch involves buffer binding + submit + readback.

## How to Run

```bash
# First time: download weights (requires HF_TOKEN for gated repo)
python python/examples/webgpu/gpt-oss/model.py --prompt "Hello" --max-tokens 50

# Verify correctness
python python/examples/webgpu/gpt-oss/model.py --verify
```
