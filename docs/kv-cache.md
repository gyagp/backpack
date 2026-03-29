# KV Cache Design: Backpack vs llama.cpp

Comparison of KV cache management, fast decode, and related GPU optimizations
across Backpack, llama.cpp CUDA, llama.cpp Vulkan, and llama.cpp WebGPU backends.

## KV Cache Management

| Aspect | Backpack (WebGPU) | llama.cpp CUDA | llama.cpp Vulkan | llama.cpp WebGPU |
|--------|-------------------|---------------|-----------------|-----------------|
| **Allocation** | Static pre-alloc to maxSeqLen | Static pre-alloc, padded to max(n_pad, 256) | Same as CUDA | Same as CUDA |
| **Data type** | f32 only | f32, f16, bf16, q8_0, q4_0, q4_1, iq4_nl, q5_0, q5_1 | Same as CUDA (with dedicated dequant shaders) | q4_0, q4_1, q5_0, q5_1, q8_0, f32 |
| **K/V independent types** | No | Yes (`--cache-type-k`, `--cache-type-v`) | Yes | Yes |
| **Context sizing** | Auto-computed from GPU memory + `--max-seq-len` override | User-specified `--ctx-size`, no auto-detection | Same as CUDA | Same as CUDA |
| **Paged attention** | No | No (slot-based ring buffer) | No | No |
| **V transposition** | No | Yes when flash attention disabled | Yes | Yes |
| **Shape stability** | Pre-allocated to maxSeqLen | Padded to max(n_pad, 256) for CUDA graph reuse | Same padding | Same padding |

### Context Length Handling

- **Backpack**: Computes maxSeqLen automatically based on GPU `maxBufferSize`. Budgets 25%
  of maxBufferSize for total KV cache, rounds to power-of-2. For LFM2-8B (6 attn layers,
  8 KV heads, 64 head dim), this yields 16K context on a 2GB-maxBuffer GPU. Overridable
  with `--max-seq-len`.

- **llama.cpp**: Requires manual `--ctx-size N` (default: model's `n_ctx_train`). No
  auto-detection from GPU memory. Warns when requested context exceeds training context.
  For 128K+ models, users must know their VRAM budget.

### KV Cache Quantization (llama.cpp only)

llama.cpp supports quantized KV caches — the biggest feature gap vs Backpack:

| Type | Bits/element | Memory vs f16 | Quality Impact |
|------|-------------|---------------|----------------|
| f16 | 16 | baseline | none |
| q8_0 | 8 | 50% savings | minimal |
| q5_0/q5_1 | ~5.5 | ~66% savings | small |
| q4_0/q4_1 | ~4.5 | ~72% savings | measurable on long contexts |

With q8 KV, a 128K context that would need 3 GB in f32 drops to ~750 MB. With q4 KV,
it drops to ~375 MB. The Vulkan backend has dedicated flash attention dequant shaders
(`dequantize4` paths for Q4_0 and Q8_0 in `flash_attn_base.glsl`).

**RoPE consideration**: When K is quantized, RoPE shifts require dequant → f32 → RoPE →
re-quantize, which is more expensive than in-place RoPE on f16/f32.

## Fast Decode / Command Capture-Replay

| Aspect | Backpack (WebGPU) | llama.cpp CUDA | llama.cpp Vulkan | llama.cpp WebGPU |
|--------|-------------------|---------------|-----------------|-----------------|
| **Has capture/replay?** | **Yes** | **Yes** (CUDA Graphs) | **No** | **No** |
| **Mechanism** | Manual WebGPU dispatch capture + bind group replay | `cudaStreamBeginCapture` / `cudaGraphLaunch` | Command buffers rebuilt from scratch each step | Command batching only (batch size 32) |
| **Warmup** | 1 capture step after prefill | 2 stable calls | N/A | N/A |
| **What's skipped** | Full ONNX Execute loop (707 nodes → 975 dispatches replayed) | CPU-side graph evaluation overhead | Nothing | Nothing |
| **Per-step updates** | Explicit: 24 GQA params + token ID | Implicit: pointers in captured graph | N/A | N/A |
| **Change detection** | Fixed shapes + explicit param writes | Per-node property comparison (shapes, pointers, op params) | N/A | N/A |
| **GPU requirements** | Any WebGPU device | Ampere+ (pre-Ampere disabled) | N/A | N/A |
| **Disable flag** | `--no-fast-decode` | `GGML_CUDA_DISABLE_GRAPHS` env var | N/A | N/A |

### Why Vulkan Has No Replay

The Vulkan backend creates command pools with `VK_COMMAND_POOL_CREATE_TRANSIENT_BIT |
VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT`, meaning command buffers are short-lived
and reset between uses. Contexts (`vk_context_struct`) are created fresh per compute call
and garbage-collected after. The submission flow always builds `vk::SubmitInfo` from scratch.

Vulkan *could* technically support command buffer pre-recording and replay (the API supports
it), but llama.cpp has not implemented it. There is no `VK_KHR_pipeline_executable_properties`
or secondary command buffer reuse pattern.

### Backpack Advantage

Our fast decode is ahead of **both** llama.cpp Vulkan and WebGPU backends. The CUDA backend
achieves similar results via native CUDA Graphs, but that requires Ampere+ hardware and
only works on NVIDIA GPUs. Our approach works on any WebGPU device (NVIDIA, AMD, Intel,
Apple Silicon via Metal).

## GPU Optimizations Comparison

### Vulkan-Specific Optimizations (llama.cpp)

1. **Push constants**: Per-dispatch params passed via push constants (128-256 bytes,
   zero-copy). WebGPU doesn't expose this — both Backpack and llama.cpp WebGPU use
   parameter buffers instead.

2. **Operation fusion**: Extensive multi-op fusion:
   - Mat-vec + bias/scale (fused via `MAT_VEC_FUSION_FLAGS`)
   - RMS norm + mul + RoPE (single dispatch)
   - Multi-add (up to 9 fused additions)
   - TopK MoE routing (multiple fusion modes)
   - Add + RMS partials accumulation

3. **Cooperative matrix** (tensor cores on Vulkan):
   - `VK_KHR_cooperative_matrix` (cross-vendor): ~2-2.5x prefill speedup
   - `VK_NV_cooperative_matrix2` (NVIDIA-specific): full flash attention with coopmat
   - Disabled on Intel (Mesa emulates poorly) and AMD Windows (driver crashes)
   - Works on AMD Linux via Mesa for RDNA 7000 series

4. **Subgroup operations**: Extensive use throughout shader suite:
   - `subgroupShuffleXor` for butterfly reductions in flash attention
   - `subgroupAll` for mask optimization (skip fully-masked blocks)
   - `subgroupBallot` for MoE expert routing compaction
   - `subgroupAdd` for pure subgroup reduction in mat-vec
   - AMD RDNA2 workaround: cast through `vec4` for broken `f16vec4` shuffle
   - Architecture detection via subgroup size for wave64 mode selection

5. **Shader specialization**: 177 compute shaders vs 31 in WebGPU backend.
   Per-quant-type mat-vec/mat-mat shaders, flash attention variants (standard,
   cooperative matrix v1/v2, mask-optimized).

### Performance: Vulkan vs CUDA (llama.cpp)

**Prefill (NVIDIA hardware)**:

| Model | GPU | Vulkan (t/s) | CUDA (t/s) | CUDA advantage |
|-------|-----|-------------|------------|----------------|
| LLaMA 7B Q4_0 | RTX 3090 | 1,933 | 5,059 | **2.6x** |
| LLaMA 8B Q4_K_S | RTX 3090 | 1,381 | 4,692 | **3.4x** |
| Granite 3.96B Q4_K_M | GTX 1060 | 228 | 1,097 | **4.8x** |

**Decode (NVIDIA hardware)** — Vulkan can match or beat CUDA:

| Model | GPU | Vulkan (t/s) | CUDA (t/s) | Winner |
|-------|-----|-------------|------------|--------|
| Granite 3.96B Q4_K_M | GTX 1060 | 90.6 | 61.7 | **Vulkan 1.47x** |
| Falcon-H1R 7B Q4_K_S | GTX 1060 | 28.1 | 25.4 | **Vulkan 1.11x** |

**AMD RDNA3** — Vulkan beats ROCm by 10-30%:

| Test | Vulkan (t/s) | ROCm (t/s) | Vulkan advantage |
|------|-------------|------------|------------------|
| tg128 (RX 7900 XTX) | 174.6 | 143.8 | **21%** |
| tg2048 (RX 7900 XTX) | 144.0 | 132.3 | **9%** |

### Performance: Vulkan vs WebGPU (llama.cpp)

No direct benchmarks, but architectural differences strongly favor Vulkan:
- 177 specialized shaders vs 31
- Cooperative matrix support (2-2.5x prefill speedup)
- Push constants (zero-copy) vs parameter buffers
- Extensive op fusion vs dispatch-per-op
- WebGPU's workgroup size cap of 288 (implementation bugs) limits parallelism

WebGPU's advantage is **browser deployment** — the only path to in-browser LLM inference.

## Opportunities for Backpack

Based on this analysis, key improvements we could adopt:

### High Impact
1. **KV cache quantization (q8/q4)**: 50-75% memory savings, enabling much longer
   contexts. Requires dequant logic in attention shaders. q8 is the sweet spot
   (minimal quality loss, 50% savings).

2. **Flash attention with quantized KV**: llama.cpp Vulkan has dedicated `dequantize4`
   paths in flash attention shaders for Q4_0 and Q8_0 KV.

### Medium Impact
3. **Operation fusion**: Fused RMS+mul+RoPE and mat-vec+bias would reduce dispatch
   count. Most impactful for standard transformer models (not LFM2 which uses
   GraphExecutor).

4. **Subgroup optimizations**: More aggressive use of subgroup shuffle/ballot for
   reductions in attention and mat-vec kernels.

### Already Ahead
- **Fast decode capture/replay**: Our WebGPU fast decode is unique — neither llama.cpp
  Vulkan nor WebGPU has this. Only CUDA graphs achieve similar results, but require
  Ampere+ NVIDIA hardware.
- **Auto context sizing**: We auto-compute maxSeqLen from GPU memory; llama.cpp
  requires manual `--ctx-size`.
