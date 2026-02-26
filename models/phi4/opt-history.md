# Phi-4 mini (3.8B params) — Performance Data

## Test Configuration

| Property | Value |
|----------|-------|
| Machine | Desktop, 32 CPU cores, DDR5 RAM |
| OS | Windows |
| Dawn | Custom build with TimedWaitAny |
| Python | 3.13 |
| NumPy | OpenBLAS 0.3.30 |

## Model

**Model**: microsoft/Phi-4-mini-instruct
**Architecture**: 32 layers, 24 Q heads, 8 KV heads, 3072 embd, 8192 intermediate, 200064 vocab
**Quantization**: INT4 per-group (group_size=128) + fp16 norms/embeddings
**Weight size**: 2.8 GB quantized (pre-dequantized to fp32 at startup: ~12 GB RAM)

## Results

| GPU | Backend | Decode Mode | Prefill (tok/s) | Decode (tok/s) | Notes |
|-----|---------|-------------|-----------------|----------------|-------|
| NVIDIA GeForce RTX 5080 (16GB) | D3D12 | gpu | 1.4 | 9.5 | GPU linear dispatch, CPU norm/attn/silu |
| NVIDIA GeForce RTX 5080 (16GB) | D3D12 | cpu | 2.3 | 3.2 | CPU BLAS matmul, GPU for prefill only |

## Decode Breakdown (CPU mode, D3D12, RTX 5080)

Per-token decode breakdown (32 layers × 4 ops/layer + LM head):

| Operation | Per-layer (ms) | Total (ms) | % |
|-----------|---------------|------------|---|
| gate_up (16384×3072) | 3.9 | 125 | 40% |
| down_proj (3072×8192) | 2.0 | 63 | 20% |
| qkv_linear (5120×3072) | 1.3 | 40 | 13% |
| o_proj (3072×3072) | 0.8 | 25 | 8% |
| attention (CPU) | 0.2 | 5 | 2% |
| norms + silu (CPU) | 0.1 | 4 | 1% |
| **LM head (200064×3072)** | — | **50** | **16%** |
| **Total** | — | **~310** | **100%** |

**Bottleneck**: CPU memory bandwidth (~46 GB/s effective, 85% of peak).
Reading 14.3 GB of fp32 weights per token.

## Optimization History

| Date | Change | Decode (tok/s) | Speedup |
|------|--------|----------------|---------|
| 2026-02-24 | Baseline (GPU dispatch for all ops) | 0.7 | 1.0× |
| 2026-02-24 | CPU-only decode (eliminate 128 GPU round-trips/token) | 2.8 | 4.0× |
| 2026-02-24 | CPU final RMSNorm + misc | 3.2 | 4.6× |
| 2026-02-24 | D3D12 backend (was Vulkan), D3D12 now has Subgroups/F16 | 3.4 | 4.9× |
| 2026-02-24 | GPU decode (GPU linear, CPU norm/attn/silu) | 9.5 | 13.6× |

## GPU Dispatch Overhead

Per-dispatch round-trip timing (submit + fence + readback), measured with
cached weights (no upload, just dispatch + readback):

| Backend | Per-dispatch avg (ms) | 128 dispatches/token (ms) |
|---------|----------------------|---------------------------|
| D3D12 (cached weights) | ~0.7 | ~86 |
| D3D12 (first run) | ~11 | ~1408 |

Dispatch overhead drops dramatically after first run (pipeline/buffer caches warm).
GPU decode is 3× faster than CPU decode for Phi-4 mini.

## How to Run

```bash
# First time: download + quantize
python python/examples/webgpu/phi4/model.py --quantize

# Inference
python python/examples/webgpu/phi4/model.py --prompt "Hello" --max-tokens 50

# Profile with HTML report
python python/examples/webgpu/phi4/model.py --profile --prompt "Hello" --max-tokens 10
# → generates phi4/profile.html
```

## Profiling

Use `--profile` to generate both a console report and an interactive HTML
timeline (`profile.html` in the model folder) showing CPU and GPU events
on a unified 2-lane timeline. The HTML report supports zoom/pan and hover
tooltips, and displays op counts per phase (prefill vs decode).
