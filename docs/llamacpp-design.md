# llama.cpp Vulkan Backend — Design Advantages

Analysis of llama.cpp's Vulkan backend architecture and the design
patterns that give it a performance advantage over our WebGPU (Dawn)
runtime on the same hardware.

**Hardware**: RTX 5080, Qwen3-1.7B Q8_0
**Decode**: llama.cpp 330 tok/s — Backpack 186 tok/s (**1.8× gap**)
**Prefill (128 tok)**: llama.cpp 14,967 tok/s — Backpack 194 tok/s (**77× gap**)

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

**Our status**: No fusion. Every rms_norm, silu_mul, and residual add
is a separate dispatch with its own barrier. Our 283 dispatches could
drop to ~200 with equivalent fusions.

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

**Blocker**: WGSL has no `subgroup_matrix` specification yet. Dawn is
working on it but it's not available. When it lands, our attention
kernels could be rewritten to use tensor cores.

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

**Our design**: Extract int8 values via `extractBits` and convert to
fp32 for scalar dot products. WGSL has `dot4I8Packed` via
`packed_4x8_integer_dot_product` language feature but we haven't
adopted it yet.

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

**Our design**: fp32 KV cache. Converting to fp16 requires the
`u32` + `unpack2x16float()` pattern on D3D12 (typed f16 buffers
return zeros). Straightforward to implement.

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

## 10. Prefill Architecture (77× gap)

The largest performance gap is in prefill (prompt processing):
llama.cpp processes T tokens in parallel, reading each weight matrix
**once** for all T tokens.  We process tokens one at a time, reading
each weight matrix **T times**.

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

### 10.2 Our Current Prefill Pipeline

For a 128-token prompt, per layer per token:

| Operation | Dispatches per token | Notes |
|-----------|---------------------|-------|
| rms_norm | 1 | |
| q8_qkv | 1 | T=1 matvec |
| fused_rope | 1 | |
| attn_p1 + attn_p2 | 2 | |
| q8_oproj | 1 | |
| add_rms | 1 | |
| q8_gateup | 1 | |
| q8_down_silu_add | 1 | |
| rms_next | 1 (layers 0-26) | |

**Total**: ~255 dispatches × 128 tokens = **32,640 dispatches**.
Each matmul reads the weight matrix **128 times** (once per token).

**Weight bandwidth**: 1.83 GB × 128 = **234 GB total** (128× more).

### 10.3 What Batched Prefill Requires

| Component | Status | Effort |
|-----------|--------|--------|
| T×E intermediate buffers | Needed | Low — allocate larger buffers |
| Batched Q8 matmul kernel (T×K → T×N) | Needed | Medium — new WGSL kernel with 2D grid |
| Causal self-attention | Needed | Medium — mask upper triangle |
| Batched RoPE for T positions | Needed | Low — extend existing kernel |
| Batched add/norm for T rows | Needed | Low — extend existing kernels |
| Pre-allocated max-T buffers | Needed | Low — allocate for MAX_PREFILL_T |
| Cooperative matrix (subgroup_matrix) | Optional | High — requires Dawn experimental API |

### 10.4 Performance Projection

With batched prefill (no cooperative matrix):

| Prompt | Current | Projected | Improvement |
|--------|---------|-----------|------------|
| 128 tok | 660ms (194 tok/s) | ~8ms (~16,000 tok/s) | **80×** |
| 512 tok | 2760ms (186 tok/s) | ~24ms (~21,000 tok/s) | **115×** |
| 1024 tok | 5576ms (184 tok/s) | ~50ms (~20,000 tok/s) | **110×** |

The projection assumes weight bandwidth is the bottleneck (1.83 GB
at 960 GB/s = 1.9ms for weight reads, plus compute+attention).
With cooperative matrix, prefill matmuls become compute-bound and
would approach llama.cpp's ~15-23K tok/s.

---

## 11. Decode vs Prefill Gap Summary

| Factor | Decode impact | Prefill impact |
|--------|-------------|---------------|
| Dawn barriers | 1.4ms/tok (59%) | 1.4ms×T tokens (dominates) |
| Serial T=1 processing | N/A | **128× weight re-reads** |
| No cooperative matrix | 0.2ms/tok (13%) | Missing batched GEMM perf |
| No operator fusion | 0.3ms/tok (20%) | 0.3ms×T (proportional) |
| No flash attention | Part of above | Missing causal batched attn |
