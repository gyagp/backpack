# llama.cpp Vulkan Backend — Design Advantages

Analysis of llama.cpp's Vulkan backend architecture and the design
patterns that give it a performance advantage over our WebGPU (Dawn)
runtime on the same hardware.

**Hardware**: RTX 5080, Qwen3-1.7B Q8_0
**Decode**: llama.cpp 330 tok/s — Backpack 192 tok/s (**1.7× gap**)
**Prefill (128 tok)**: llama.cpp 14,967 tok/s — Backpack 3,682 tok/s (**4.1× gap**)
**Prefill (1024 tok)**: llama.cpp 22,460 tok/s — Backpack 5,102 tok/s (**4.4× gap**)

---

## 1. Smart Dependency Tracking (Barriers)

**llama.cpp**: Tracks buffer-level read/write dependencies across the
compute graph. A pipeline barrier is only inserted when a node's
destination buffer **overlaps in memory** with a previously
unsynchronized node AND at least one is a write. Independent
operations (different buffers) get **zero barriers**, allowing the
GPU to overlap their execution.

**Dawn/WebGPU**: Inserts a full pipeline barrier between dispatches
that share any buffer binding within a compute pass. With 283
dispatches per token sharing intermediate buffers, this adds ~5µs
per barrier × ~280 barriers ≈ **1400µs of overhead**.

**Impact**: ~1.4ms/token — accounts for most of the gap.

```
llama.cpp: barrier only at true data dependencies (~50-70 per token)
Dawn:      barrier between every dispatch pair sharing any buffer (~280)
```

---

## 2. Operator Fusion

llama.cpp matches multi-node patterns in the compute graph at build
time and replaces them with fused GPU kernels:

| Fusion Pattern | Dispatches Saved | Notes |
|----------------|-----------------|-------|
| `MUL_MAT + ADD` (bias) | 1 per linear layer | Bias written in matmul epilogue via `fusion_flags` push constant |
| `RMS_NORM + MUL` | 1 per norm | Weight multiplication fused into norm kernel |
| `RMS_NORM + MUL + ROPE` | 2 per attention | Norm → scale → rotate in one dispatch |
| `RMS_NORM + MUL + ROPE + VIEW + SET_ROWS` | 4 per attention | Full QKV prep in one kernel |
| `MUL_MAT + ADD + ADD` | 2 per layer | Down proj + bias + residual in one dispatch |

For a 28-layer model, fusion eliminates ~80 dispatches per token.
Each eliminated dispatch saves kernel launch overhead + the barrier
that would precede it.

**Our status**: SiLU·mul fused into down projection (`q8_down_silu_add`).
Saves 28 dispatches per token. Other fusions (rms+matmul) regressed
due to per-WG redundant norm computation.

---

## 3. Flash Attention with Cooperative Matrix

**llama.cpp**: Single-dispatch flash attention kernel that handles the
entire Q·K^T → softmax → ·V pipeline. On NVIDIA GPUs, uses
`VK_KHR_cooperative_matrix` (coopmat1/coopmat2) for tensor core
accelerated matrix multiplications within the attention computation.
Optionally uses split-K with a reduce pass (2 dispatches total).

```glsl
// llama.cpp flash_attn.comp — cooperative matrix path
coopmat<float16_t, gl_ScopeSubgroup, M, N, ...> C;
C = coopMatMulAdd(Q_tile, K_tile, C);  // tensor core matmul
```

**Our design**: Two-dispatch chunked attention using scalar `subgroupAdd`
for Q·K dot products. No cooperative matrix support in WGSL yet.
Attention takes ~588µs/token vs estimated ~200µs with coopmat.

**Blocker**: Dawn supports `chromium_experimental_subgroup_matrix` with
i8×i8→i32 MMA tiles (M=8, N=8, K=32). Confirmed working on RTX 5080.
However, Q8_0's per-block scales (varying scale per 32-element K-block)
prevent simple MMA accumulation — each K-block needs rescaling,
requiring store/load round-trips that negate tensor core benefits.
Subgroup matrix is most useful for fp16 matmul or Q8 with uniform scales.

---

## 4. Specialization Constants

**llama.cpp**: Uses Vulkan/SPIR-V specialization constants for shader
configuration. Workgroup sizes, tile dimensions, and algorithm
parameters are set at pipeline creation time — the compiler optimizes
for the exact values.

```glsl
layout(local_size_x_id = 0, local_size_y = 1, local_size_z = 1) in;
layout(constant_id = 1) const uint NUM_ROWS = 1;
```

This means a single shader source generates optimized variants for
different hardware (subgroup size 32 vs 64, row counts 1 vs 2 vs 4).

**Our design**: WGSL compute shaders use fixed compile-time constants.
Different tile configurations require separate shader source files
(e.g., `q8_matmul.wgsl` vs `q8_matmul_fast.wgsl`). No specialization
constant equivalent in WGSL yet (proposed as `override` declarations,
partially supported by Dawn).

---

## 5. Adaptive Batch Submission

**llama.cpp**: Records the entire compute graph into a live command
buffer, but uses adaptive batched submission:

```
Submit after enough work has accumulated, to overlap CPU command buffer
generation with GPU execution. Submit every 100MB of matmul weight
data (scaled down by model size), and at least every 100 nodes.
```

The submit threshold doubles after the first 3 submits, so typical
decode for a 1.7B model finishes in **1-2 queue submits**. This
overlaps CPU command recording with GPU execution.

**Our design**: Pre-record all dispatches into command buffers at init
time. Each token just pops a pre-recorded CB and submits — zero
encoding on the hot path. But Dawn still translates/validates during
`Submit`, so we pay ~440µs per submit.

---

## 6. Integer Dot Product (DP4A)

**llama.cpp**: On hardware supporting `VK_KHR_shader_integer_dot_product`,
uses `dotPacked4x8EXT` for INT8 dot products. For Q8_0 with n=1 on
NVIDIA Turing+, it uses the float dequant path when K > 4096, but
for smaller K it can use the integer path:

```glsl
int32_t q_sum = 0;
q_sum += dotPacked4x8EXT(data_a_packed, cache_b_packed);  // 4× INT8 MADs
result = float(q_sum) * scale_a * scale_b;
```

The B vector (activations) is pre-quantized to Q8_1 format with
per-block scales, cached in registers.

**Our design**: DP4A kernel (`q8_matmul_dp4a`) implemented and available.
Uses `dot4I8Packed` via WGSL `packed_4x8_integer_dot_product`. At T=1
decode, no measurable speedup (bandwidth-bound, not compute-bound).
Used as default kernel for decode matmuls.

---

## 7. Efficient Q8_0 Dequantization

**llama.cpp**: Loads Q8_0 data via aliased `int16_t` buffer views,
using `unpack8()` to extract two int8 values from a 16-bit load:

```glsl
vec4 dequantize4(uint ib, uint iqs, uint a_offset) {
    const i8vec2 v0 = unpack8(int32_t(data_a_packed16[ib].qs[iqs/2]));
    const i8vec2 v1 = unpack8(int32_t(data_a_packed16[ib].qs[iqs/2 + 1]));
    return vec4(v0.x, v0.y, v1.x, v1.y);
}
```

The scale is applied **after** the dot product (deferring the multiply):
```glsl
rowtmp = dot(bv0, raw_int_vec);  // dot with raw integers
rowtmp *= scale;                  // single multiply at the end
```

**Our design**: Pack 4 int8 values as `u32`, extract via `extractBits`,
convert to f32, then dot. Similar approach but using different load
intrinsics. Our scale application is also deferred (post-dot).

---

## 8. Hardware-Specific Tuning

**llama.cpp** detects GPU vendor and architecture at device creation
and tunes parameters:

| Parameter | NVIDIA | AMD GCN | Intel |
|-----------|--------|---------|-------|
| Subgroup size | 32 | 64 | 16-32 |
| NUM_ROWS (DMMV) | 1 | 2-4 | 2 |
| WG size | subgroup_size | subgroup_size×4 | subgroup_min_size |
| MMVQ path | off for Q8_0 K>4096 | on for most types | on |
| KV cache type | f16 | f16 | f16 |

**Our design**: Fixed WGSL shader parameters. We use `@workgroup_size(256)`
(8 warps) with TILE_N=8 universally. This works well on NVIDIA but
may be suboptimal on AMD (which prefers wavefront64) or Intel.

---

## 9. fp16 KV Cache

**llama.cpp**: Stores KV cache in fp16 by default, halving attention
memory reads. For Q8_0 decode at seq_len=200 with n_kv=8, head_dim=128:

```
fp32 KV: 200 × 8 × 128 × 4 = 800 KB per layer per K/V
fp16 KV: 200 × 8 × 128 × 2 = 400 KB per layer per K/V
```

28 layers × 2 × 400 KB savings = 22.4 MB less bandwidth per token.

**Our design**: fp16 KV cache. All kernels (prefill rope, prefill attention,
decode rope, decode attention) use `array<f16>`. KV cache is 448 MB
(halved from 896 MB). Attention loop uses `var<uniform>` params for
data-dependent early exit.

---

## Summary: Actionable Items

1. **Reduce dispatch count via fusion** — fuse norm+matmul,
   matmul+bias+residual, silu into adjacent kernels. Each eliminated
   dispatch saves ~5µs of Dawn barrier overhead.

2. **Adopt dp4a** — use WGSL `dot4I8Packed` for Q8 matmul inner loops.
   Requires `packed_4x8_integer_dot_product` language feature.

3. **fp16 KV cache** — halve attention KV bandwidth. Use `u32` +
   `unpack2x16float()` for D3D12 compatibility.

4. **Patch Dawn barriers** — the single highest-impact change would be
   teaching Dawn's Vulkan backend to track per-buffer dependencies
   instead of issuing global barriers between every dispatch.

5. **Cooperative matrix** — when WGSL `subgroup_matrix` lands, rewrite
   flash attention to use tensor cores.

6. **Specialization constants** — when WGSL `override` is stable, use
   it for workgroup size / tile size tuning per GPU vendor.

---

## 10. Prefill Architecture (4.4× gap, was 7.5×)

**Status (March 2026)**: Batched prefill implemented. Matmuls read
weights once for all T tokens. Batched RoPE and causal attention
process all T tokens in single dispatches. All 225 dispatches submitted
in a single `submitOnly` call.

**Remaining gap** (4.4× vs llama.cpp at pl=1024): All matmuls use
double-buffered subgroup matrix MMA (64×32 output tile, 8 subgroups,
256 threads, TILE_K=32 with 2 MMA calls per K-block). KV cache is
fp16. Attention uses multi-query (4 warps/WG) with fp16 KV and
uniform-param early exit.

GPU profile at T=1024 (186ms): gateup 34%, down_silu 24%, attn 20%,
qkv 13%, oproj 7%. Remaining gap is Q8→fp16 dequant staging overhead
and Dawn barrier overhead.

### 10.1 llama.cpp Prefill Pipeline

For a 128-token prompt, per layer:

| Operation | Dispatch | Dimensions | Notes |
|-----------|----------|-----------|-------|
| **Batched QKV matmul** | 1 dispatch | (128×E) × (E×QKV)^T = 128×QKV | Cooperative matrix tiled GEMM via `mul_mm.comp` |
| **RoPE** | 1 dispatch | Per-element on Q,K | Fused with norm in some configurations |
| **KV cache scatter** | 1 dispatch | Copy K,V into cache slots 0..127 | |
| **Flash attention** | 1-2 dispatches | (128×HD) × (128×HD)^T → causal mask → softmax → (128×HD) | Cooperative matrix flash attention with causal mask |
| **O projection** | 1 dispatch | (128×qD) × (qD×E)^T = 128×E | Batched GEMM |
| **Residual + norm** | 1 dispatch | Per-element 128×E | Fused |
| **Gate+Up** | 1 dispatch | (128×E) × (E×2IM)^T = 128×2IM | Batched GEMM |
| **SiLU·mul** | 1 dispatch | Per-element 128×IM | |
| **Down proj + residual** | 1 dispatch | (128×IM) × (IM×E)^T += 128×E | Batched GEMM with residual add |

**Total**: ~10 dispatches per layer, ~280 for 28 layers.  Each matmul
reads the weight matrix **once** for all 128 tokens.

**Weight bandwidth**: 1.83 GB read once = **1.83 GB total**.

### 10.2 Our Batched Prefill Pipeline (Implemented)

For a 128-token prompt, per layer:

| Operation | Dispatches | Notes |
|-----------|-----------|-------|
| Batched RMSNorm | 1 | T WGs, one per token row |
| Batched Q8 QKV matmul | 1 | ceil(T/8) × ceil(N/8) WGs |
| Batched RoPE + KV scatter | 1 | (n_head + n_kv) × T WGs |
| Batched causal attention | 1 | n_head × T WGs, online softmax |
| Batched Q8 O projection | 1 | ceil(T/8) × ceil(N/8) WGs |
| Batched add + RMSNorm | 1 | T WGs |
| Batched Q8 gate+up | 1 | ceil(T/8) × ceil(N/8) WGs |
| Batched fused SiLU+down+add | 1 | ceil(T/8) × ceil(N/8) WGs |

**Total**: 8 dispatches per layer, 225 for 28 layers + final norm.
All submitted in a single `submitOnly` call.

**Weight bandwidth**: 1.83 GB read once = **1.83 GB total**.

**Benchmark** (Qwen3-1.7B Q8_0, RTX 5080 Vulkan):

| Prompt | Serial tok/s | Batched tok/s | llama.cpp tok/s |
|--------|-------------|--------------|----------------|
| 128 | 192 | **3,682** | 14,967 |
| 512 | 184 | **5,316** | 23,285 |
| 1024 | 183 | **5,102** | 22,460 |
| 4096 | 176 | **3,676** | 20,191 |

### 10.3 Remaining Prefill Gaps

| Component | Status | Remaining gap |
|-----------|--------|--------------|
| T×E intermediate buffers | ✅ Done | — |
| Batched Q8 matmul kernel | ✅ Done (MMA) | Subgroup matrix 16×16×16 fp16→f32 |
| Batched RoPE + KV scatter | ✅ Done | — |
| Batched causal attention | ✅ Done (fp16 KV + early exit) | Multi-query 4 warps/WG, uniform params |
| Batched add/norm | ✅ Done | — |
| Single-submit all dispatches | ✅ Done | — |
| Pre-allocated prefill resources | ✅ Done | Zero alloc on hot path |
| GPU-side argmax (4B readback) | ✅ Done | Was 593KB logits readback |
| Subgroup matrix matmul | ✅ Done | Q8→fp16 dequant in smem before MMA |
| fp16 KV cache | ✅ Done | Halves KV bandwidth (448 MB) |
| Uniform param early exit | ✅ Done | Loop terminates at causal length |

### 10.4 Performance Achieved vs Projected

| Prompt | Original | Current batched | llama.cpp |
|--------|---------|----------------|-----------|
| 128 tok | 660ms (192 tok/s) | 35ms (**3,682 tok/s**) | ~9ms (14,967 tok/s) |
| 512 tok | 2760ms (184 tok/s) | 96ms (**5,316 tok/s**) | ~22ms (23,285 tok/s) |
| 1024 tok | 5576ms (183 tok/s) | 201ms (**5,102 tok/s**) | ~46ms (22,460 tok/s) |
| 4096 tok | 23312ms (176 tok/s) | 1114ms (**3,676 tok/s**) | ~203ms (20,191 tok/s) |

The remaining **4.4× gap** vs llama.cpp is primarily:
- **Q8→fp16 dequant staging**: Per-element extractBits+scale in smem
  before MMA. llama.cpp uses SPIR-V with more efficient data paths.
- **Dawn barrier overhead**: ~225 barriers × ~5µs = 1.1ms per submit.
- **Attention still scalar**: 20% of GPU time, uses subgroupAdd dot
  products. Flash attention with MMA would cut this further.

---

## 11. Decode vs Prefill Gap Summary

| Factor | Decode impact | Prefill impact |
|--------|-------------|---------------|
| Dawn barriers | 1.4ms/tok (59%) | 1.4ms×T tokens (dominates) |
| Serial T=1 processing | N/A | **128× weight re-reads** |
| No cooperative matrix | 0.2ms/tok (13%) | Missing batched GEMM perf |
| No operator fusion | 0.3ms/tok (20%) | 0.3ms×T (proportional) |
| No flash attention | Part of above | Missing causal batched attn |
