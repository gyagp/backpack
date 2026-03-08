# llama.cpp Vulkan Backend vs Backpack WebGPU — Deep Technical Analysis

**Date**: March 8, 2026
**Benchmark**: Qwen3-1.7B Q8_0, RTX 5080, 200-token decode
**Gap**: llama.cpp 330 tok/s vs Backpack 185 tok/s (**1.78×**)
**GPU HW Gap**: 3.05ms vs 4.58ms (**1.50×**)

---

## 1. Architecture Comparison Table

| Feature | llama.cpp Vulkan | Backpack WebGPU |
|---------|-----------------|-----------------|
| **API** | Raw Vulkan 1.3 (vk::hpp) | WebGPU (Dawn) |
| **Command buffers** | Single CB, batched submits every ~100 nodes or ~100MB matmul data | Pre-recorded CB pool, single wgpuQueueSubmit per token |
| **Barriers** | Smart dependency tracking: only when tensor ranges overlap AND one is a write | Dawn inserts implicit global barrier between sharing dispatches |
| **Queue submits/decode** | 2-5 vkQueueSubmit per token (adaptive batching) | 1 wgpuQueueSubmit per token |
| **Queue submits/prefill** | Same batched approach, more submits for larger T | 1 wgpuQueueSubmit per prefill step |
| **Transfer queue** | Separate async transfer queue with timeline semaphores | Single queue, no async transfers |
| **Push constants** | Yes — all per-dispatch params via push constants (≤128 bytes) | Bind groups per dispatch (heavier) — no push constants in WGSL |
| **Descriptor sets** | Pool-allocated, reused per pipeline | Bind groups created per dispatch |
| **Operator fusion** | 10+ fusion patterns (see §3) | None |
| **Weight layout** | Raw GGUF row-major (no repacking for DMMV) | Repacked Q8 (weights+scales separate) |
| **Decode matvec** | DMMV: `mul_mat_vec.comp` with dequantize4() vec4 loads | Custom kernel: extractBits + f32 FMA |
| **Prefill matmul** | Tiled matmul `mul_mm.comp` with cooperative matrix (optional) | Not implemented (serial token-at-a-time prefill) |
| **Flash attention** | `flash_attn_cm1.comp` — KHR_cooperative_matrix 16×16 tiles | Scalar attention: subgroupAdd dot products |
| **KV cache** | Contiguous buffer, causal mask via flash attention | Contiguous linear buffer, explicit mask |
| **Integer dot product** | VK_KHR_shader_integer_dot_product for Q8 MMVQ path | Not used (no dp4a / dot4I8Packed) |
| **Specialization constants** | Extensive: BLOCK_SIZE, BM, BN, BK, WM, WN, TM, TN, WARP | Not used — hardcoded in WGSL |
| **Buffer management** | Direct VkAllocation per tensor, host-visible staging | Dawn-managed allocations |
| **Graph optimization** | Reorders nodes for parallelism, detects fusion patterns | Fixed dispatch order |

---

## 2. Decode (T=1) Path — Detailed Comparison

### 2.1 llama.cpp DMMV (Dequantize Mul Mat Vec)

**Shader**: `mul_mat_vec.comp` + `mul_mat_vec_base.glsl` + `dequant_funcs.glsl`

**Key design**:
- **Workgroup size**: `subgroup_size` (32 on NVIDIA) for `DMMV_WG_SIZE_SUBGROUP`, or `subgroup_size * 4` (128) for `DMMV_WG_SIZE_LARGE`
- **NUM_ROWS**: Specialization constant, typically 1 for single-token decode
- **NUM_COLS**: Up to 8 — processes multiple output columns per workgroup
- **K_PER_ITER**: 8 for quantized types — each thread processes 8 elements per loop iteration
- **Reduction**: `subgroupAdd()` — pure subgroup reduction, no shared memory needed when WG = subgroup size
- **MMVQ selection**: For Q8_0 on post-Turing NVIDIA: DMMV preferred (not MMVQ), unless `integer_dot_product` available

**Q8_0 dequantization** in DMMV:
```glsl
// dequant_funcs.glsl — vec4 packed loads
vec4 dequantize4(uint ib, uint iqs, uint a_offset) {
    const i8vec2 v0 = unpack8(int32_t(data_a_packed16[a_offset + ib].qs[iqs/2])).xy;
    const i8vec2 v1 = unpack8(int32_t(data_a_packed16[a_offset + ib].qs[iqs/2 + 1])).xy;
    return vec4(v0.x, v0.y, v1.x, v1.y);
}
// Then: dot(bv0, dequantize4(...)) — 4-wide dot product
```
- Uses `packed16` layout for 4-element vectorized loads
- `dot()` intrinsic for 4×f32 dot product per iteration
- Scale applied once via `get_dm()` → multiply by scale factor

### 2.2 Backpack Q8 Decode Kernel

- **Workgroup size**: 256 (8 warps × 32)
- **TILE_N**: 8 — 8 output rows share X vector in L1
- **Dequant**: `extractBits` per-byte + f32 conversion (scalar, not vectorized)
- **Reduction**: `subgroupAdd()` (good) then cross-warp via shared memory

**Key differences**:
- llama.cpp loads 4 int8 values at once via `packed16` struct → one 32-bit load
- Our code uses per-byte `extractBits` → 4× more load instructions
- llama.cpp uses `dot()` for 4-element FMA; we use sequential `fma()`
- llama.cpp's `NUM_ROWS=1, BLOCK_SIZE=32` is LEANER than our `WG=256, TILE_N=8` — fewer threads but more efficient per-thread work

### 2.3 MMVQ Path (Integer Dot Product)

When `integer_dot_product` is available AND `should_use_mmvq()` returns true:
- Activations quantized to Q8_1 on GPU (`quantize_q8_1_x4` pipeline)
- Kernel uses hardware `intBitsToFloat(dot(a_packed, b_packed))` for 4×int8 dot products
- ~2× faster than DMMV for certain quant types (Q4_0, Q4_1, Q5_0, Q5_1)
- **For Q8_0 on NVIDIA post-Turing**: DMMV is preferred over MMVQ

---

## 3. Operator Fusion — Complete Pattern List

llama.cpp detects and fuses these patterns in `ggml_vk_build_graph`:

| Pattern | Ops Fused | Dispatches Saved | Impact |
|---------|-----------|-----------------|--------|
| MUL_MAT + ADD | 2→1 | ~28/layer | Bias fusion into matmul |
| MUL_MAT + ADD + ADD | 3→1 | ~28/layer | Double bias |
| MUL_MAT_ID + ADD_ID | 2→1 | MoE path | Expert bias |
| MUL_MAT_ID + ADD_ID + MUL | 3→1 | MoE path | Expert bias + scale |
| RMS_NORM + MUL (add_rms) | 2→1 | ~28/token | Norm + gate multiply |
| RMS_NORM + MUL + ROPE + VIEW + SET_ROWS | 5→1 | ~28×2/token | Full Q/K projection |
| ROPE + VIEW + SET_ROWS | 3→1 | ~28/token | RoPE + KV write |
| MULTI_ADD (up to 9 adds) | N→1 | Variable | Residual accumulation |
| TOPK_MOE (4 modes) | 5-10→1 | MoE path | Expert routing |
| ADD + RMS_NORM (partials) | 2→1 | 28/token | Residual + norm prefetch |

**Fusion benefit for Qwen3-1.7B Q8_0**:
- Without fusion: ~400+ dispatches per decode token
- With fusion: ~120-150 dispatches per decode token
- Each eliminated dispatch saves ~2-5µs of Dawn barrier overhead

---

## 4. Barrier Insertion — The Critical Gap

### 4.1 llama.cpp: Smart Dependency Tracking

```cpp
// ggml-vulkan.cpp line ~12650
// Checks whether "node" requires synchronization by checking if it
// overlaps in memory with another unsynchronized node and at least
// one of them is a write.
auto const &overlaps_unsynced = [&](const ggml_tensor *node,
    const std::vector<const ggml_tensor *> &unsynced_nodes) -> bool {
    // Only inserts barrier when:
    // 1. Same VkBuffer
    // 2. Byte ranges actually overlap
    // 3. At least one is a write
};
```

- Maintains two lists: `unsynced_read_nodes` and `unsynced_write_nodes`
- Only calls `ggml_vk_sync_buffers()` (= `vkCmdPipelineBarrier`) when true dependency found
- Result: ~50-70 barriers per decode token for Qwen3-1.7B

### 4.2 Backpack (Dawn): Global Barriers

Dawn's Vulkan backend inserts a full memory barrier between any two dispatches that share buffer bindings within a compute pass. This is correct but conservative:
- **Every dispatch** that reads/writes a shared buffer gets a barrier
- With 282 dispatches sharing `xBuf`, `kvBuf`, etc. → ~280 barriers
- Each barrier: ~2-5µs of pipeline stall on NVIDIA

### 4.3 Impact Estimate

| | llama.cpp | Backpack |
|---|---|---|
| Dispatches/token | ~120-150 | ~282 |
| Barriers/token | ~50-70 | ~280 |
| Barrier overhead | ~0.15ms | ~0.8-1.4ms |
| **Overhead difference** | | **~0.7-1.2ms** |

This alone explains ~50-80% of the GPU HW time gap (1.53ms).

---

## 5. Command Buffer & Submission Strategy

### 5.1 llama.cpp Adaptive Batched Submit

```cpp
// Submit after enough work:
int nodes_per_submit = 100;
uint64_t mul_mat_bytes_per_submit = min(100MB, total_bytes / 40);

bool submit = (submitted_nodes >= nodes_per_submit) ||
              (mul_mat_bytes >= mul_mat_bytes_per_submit) ||
              (i >= last_node) ||
              (almost_ready && !fence_pending);
```

- First submit: small batch (~100MB of matmul weights)
- Subsequent submits: 2× larger (exponential backoff)
- `submit_count` tracks how many submits have happened; early submits are smaller
- **"almost_ready" fence**: Signals when 80% of graph is done, allowing CPU to start next token prep
- Timeline semaphores for compute↔transfer queue sync

### 5.2 Backpack Pre-recorded CB Pool

- `CB_POOL_BATCH` command buffers pre-recorded ahead
- Each token: pick pre-recorded CB, submit, rotate slot
- Single `wgpuQueueSubmit` per token
- **No CPU/GPU overlap**: CPU waits for map callback before submitting next token

---

## 6. Prefill Architecture (CRITICAL)

### 6.1 llama.cpp: True Batched Matmul

When T=128 tokens arrive for prefill:
- **MUL_MAT**: `mul_mm.comp` — full tiled matrix multiply
  - T×E × E×N → T×N (single dispatch per linear layer)
  - Specialization constants: BM=64, BN=64, BK=16/32, WM=32, WN=32, TM=4, TN=2
  - Optional cooperative matrix path: `coopMatMulAdd()` for TM×TK×TN tiles
  - Shared memory double buffering: `buf_a[BM * SHMEM_STRIDE]`, `buf_b[BN * SHMEM_STRIDE]`
  - Split-K: Optional partitioning along K dimension with reduction pass

- **Flash Attention**: `flash_attn.comp` / `flash_attn_cm1.comp`
  - Online softmax with Br×Bc tile blocking
  - Causal masking via `data_m` mask buffer + `mask_opt` optimization (skip all-zero/all-neginf blocks)
  - Cooperative matrix version: 16×16 tile MatMul for Q×K^T and S×V
  - KV cache populated by `SET_ROWS` ops (fused with ROPE+VIEW for K)

- **Dispatches per layer** (prefill T=128):
  - 4 matmuls (Q, K, V, O projections) — each single dispatch
  - 2 FFN matmuls (gate+up, down) — each single dispatch
  - 1 flash attention dispatch (+ optional split-K reduce)
  - Norm/residual ops (fused): ~2-3 dispatches
  - **Total**: ~10-12 dispatches per layer vs our ~10 per layer (but ours runs T=1)

### 6.2 Backpack: Serial Token-at-a-Time

```cpp
void ModelRunner::prefillStep(int32_t tokenId, uint32_t posOffset) {
    uploadEmbedding(tokenId);
    updateDecodeParams(posOffset, cacheLen);
    gpu->submitOnly(allDecodeDispatches, !passPerDispatch);
    for (uint32_t i = 0; i < cfg.nLayer; i++)
        kvCache[i].len++;
}
```

- Processes one token at a time during prefill
- Full decode pipeline (282 dispatches) per prefill token
- For 128-token prefill: **128 × 282 = 36,096 dispatches** vs llama.cpp's **~300 dispatches**
- Prefill is **~120× more dispatches** than llama.cpp

### 6.3 Prefill Impact

For 128-token prompt:
- llama.cpp: ~3-5ms total (batched matmuls are compute-dense, high GPU utilization)
- Backpack: ~128 × 5.5ms = ~700ms (serial, each token repeats all overhead)
- **Gap: ~140-200×** for prefill

---

## 7. Memory Management

### 7.1 llama.cpp Buffer Strategy

```cpp
struct vk_buffer_struct {
    vk::Buffer buffer;
    vk::DeviceMemory device_memory;
    vk::MemoryPropertyFlags memory_property_flags;
    void * ptr;       // mapped pointer for host-visible
    size_t size;
    vk::DeviceAddress bda_addr;  // buffer device address
};

struct vk_subbuffer {
    vk_buffer buffer;
    uint64_t offset;
    uint64_t size;
};
```

- **No pool/suballocator**: Each tensor gets a dedicated VkBuffer + VkDeviceMemory
- **Subbuffers**: Offset-based views into larger allocations for intermediate tensors
- **Preallocation**: `prealloc_x`, `prealloc_y`, `prealloc_split_k` — reusable scratch buffers for dequant/quantize intermediates
- **Host-visible staging**: `sync_staging` buffer for CPU⟷GPU data transfer, resized on demand
- **UMA support**: Unified memory — skip staging when possible
- **64-bit indexing**: Alternative pipeline variant for buffers > `maxStorageBufferRange`

### 7.2 Backpack Buffer Strategy

- Dawn manages allocations
- Weight buffers: repacked Q8 (separate weights + scales buffers)
- KV cache: contiguous per-layer buffer
- Intermediates: pre-allocated, reused

### 7.3 Weight Layout Differences

| | llama.cpp | Backpack |
|---|---|---|
| Storage | Raw GGUF block format | Repacked (weights/scales split) |
| Q8_0 block | {f16 scale, int8[32] qs} = 34 bytes | weights[] u32 packed, scales[] f32 |
| Load pattern | `data_a_packed16[].qs[]` — direct struct access | `extractBits(weights[i], ...)` — computed |
| Dequant in shader | `dequantize4()` → vec4 | Per-byte extract → f32 |

---

## 8. Push Constants vs Bind Groups

### llama.cpp: Push Constants for Everything

All per-dispatch parameters go through Vulkan push constants (≤128 bytes):

```cpp
struct vk_mat_vec_push_constants {
    uint32_t ncols, stride_a, stride_b, stride_d;
    uint32_t batch_stride_a, batch_stride_b, batch_stride_d;
    uint32_t fusion_flags;
    uint32_t base_work_group_y, ne02, ne12, broadcast2, broadcast3;
};
// Total: 52 bytes — fits in push constant range
```

- `vkCmdPushConstants()` — zero-alloc, fastest parameter path
- Push constant data embedded in command buffer stream
- No descriptor set update needed for parameter changes

### Backpack: Uniform Buffers via Bind Groups

- Each dispatch needs bind group with uniform buffer for params
- Bind group creation: `wgpuDeviceCreateBindGroup()` — allocates
- Dawn internally manages descriptor pools
- **Overhead**: ~1-3µs per bind group creation vs ~0µs for push constants

---

## 9. Specialization Constants

### llama.cpp: Extensive Compile-Time Tuning

```glsl
layout (constant_id = 0) const uint BLOCK_SIZE = 64;
layout (constant_id = 1) const uint BM = 64;
layout (constant_id = 2) const uint BN = 64;
layout (constant_id = 4) const uint WM = 32;
layout (constant_id = 5) const uint WN = 32;
layout (constant_id = 6) const uint WMITER = 2;
layout (constant_id = 7) const uint TM = 4;
layout (constant_id = 8) const uint TN = 2;
layout (constant_id = 9) const uint TK = 1;
layout (constant_id = 10) const uint WARP = 32;
```

- L/M/S pipeline variants: different tile sizes for large/medium/small problems
- Aligned vs unaligned variants (skip bounds checks when dimensions are multiples)
- Architecture-specific tuning (NVIDIA Turing vs Ampere, AMD RDNA2 vs RDNA3)

### Backpack: Hardcoded Constants

- Fixed workgroup size (256)
- Fixed TILE_N (8)
- No aligned fast-paths
- WGSL has `override` constants but we don't use them

---

## 10. Flash Attention — Cooperative Matrix

### llama.cpp `flash_attn_cm1.comp`

```glsl
#extension GL_KHR_cooperative_matrix : enable

const uint32_t MatBr = 16;  // Cooperative matrix tile rows
const uint32_t MatBc = 16;  // Cooperative matrix tile cols

// Q×K^T via cooperative matrix tiles
coopmat<float16_t, gl_ScopeSubgroup, MatBr, 16, gl_MatrixUseA> matQ;
coopmat<float16_t, gl_ScopeSubgroup, 16, MatBc, gl_MatrixUseB> matK;
coopmat<ACC_TYPE, gl_ScopeSubgroup, MatBr, MatBc, gl_MatrixUseAccumulator> matS;

coopMatLoad(matQ, Qf, ...);        // Load Q tile from shared memory
coopMatLoad(matK, kvsh, ...);       // Load K tile from shared memory
matS = coopMatMulAdd(matQ, matK, matS);  // Tensor core matmul!
```

- Uses hardware tensor cores for both Q×K^T and Softmax(S)×V
- Br=16/32, Bc=16/32 — maps directly to NVIDIA Tensor Core 16×16×16 instruction
- ~3-5× faster than scalar implementation for attention computation
- Mask optimization: `mask_opt` bitmap skips entire Bc-wide columns that are all -inf

### Backpack Attention

- Scalar subgroup-based dot products
- `subgroupAdd()` for partial sum reduction
- No cooperative matrix (WGSL `subgroup_matrix` not yet available)

---

## 11. Gap Analysis — Prioritized Action Items

| # | Gap | Est Impact (ms/tok) | % of Gap | Difficulty | Approach |
|---|-----|---------------------|----------|-----------|----------|
| 1 | **Dawn implicit barriers** | 0.7-1.2ms | 45-78% | Hard | Patch Dawn or buffer-specific tracking |
| 2 | **No operator fusion** | 0.3-0.5ms | 20-33% | Medium | Fuse RMS+MUL, MATMUL+BIAS, SILU_MUL |
| 3 | **No cooperative matrix** | 0.2-0.4ms | 13-26% | Blocked | Wait for WGSL subgroup_matrix spec |
| 4 | **No batched prefill matmul** | N/A decode, 140× prefill | 100% prefill | High | Implement T>1 matmul kernel |
| 5 | **No integer dot product** | 0.1-0.2ms | 6-13% | Medium | Use `dot4I8Packed` in WGSL |
| 6 | **Scalar Q8 dequant (extractBits)** | 0.05-0.1ms | 3-6% | Easy | Switch to vec4 packed loads |
| 7 | **GPU argmax** | 0.15-0.2ms | 10-13% | Easy | Move to CPU or top-K GPU |
| 8 | **No specialization constants** | 0.02-0.05ms | 1-3% | Easy | Use WGSL `override` constants |
| 9 | **Push constants unavailable** | 0.05-0.1ms | 3-6% | Blocked | WebGPU doesn't support push constants |
| 10 | **No graph reordering** | 0.01-0.03ms | 1-2% | Medium | Analyze dispatch dependencies |

---

## 12. Specific Code Patterns to Adopt

### 12.1 Vectorized Q8_0 Dequantization (EASY — ~5% speedup)

**llama.cpp pattern** (GLSL):
```glsl
vec4 dequantize4(uint ib, uint iqs, uint a_offset) {
    i8vec2 v0 = unpack8(int32_t(data_a_packed16[ib].qs[iqs/2])).xy;
    i8vec2 v1 = unpack8(int32_t(data_a_packed16[ib].qs[iqs/2+1])).xy;
    return vec4(v0.x, v0.y, v1.x, v1.y);
}
// 2 loads + 2 unpack = 4 int8 values → vec4
```

**Our equivalent WGSL**:
```wgsl
// Current: 4 separate extractBits calls
let b0 = extractBits(weights[idx], 0u, 8u);
let b1 = extractBits(weights[idx], 8u, 8u);
let b2 = extractBits(weights[idx], 16u, 8u);
let b3 = extractBits(weights[idx], 24u, 8u);
let v = vec4f(f32(i32(b0) - 128), f32(i32(b1) - 128), ...);

// Better: single u32 load, bitwise extraction with vec4
let packed = weights[idx];
let v = vec4f(
    f32((packed & 0xFF) as i8),        // Would need unpack4xI8
    f32(((packed >> 8) & 0xFF) as i8),
    f32(((packed >> 16) & 0xFF) as i8),
    f32((packed >> 24) as i8)
);
```

### 12.2 dp4a / dot4I8Packed (MEDIUM — ~10% for Q8 matvec)

**llama.cpp MMVQ pattern** (when integer_dot_product available):
- GPU quantizes activations to Q8_1 (`quantize_q8_1_x4`)
- Dot product: `intBitsToFloat(dot(a_i8x4, b_i8x4))` — hardware-accelerated

**WGSL equivalent**:
```wgsl
// WGSL has dot4I8Packed:
let result = dot4I8Packed(weight_packed_i8x4, activation_packed_i8x4);
```

### 12.3 Adaptive Batch Submission (MEDIUM — reduce CPU/GPU bubble)

**llama.cpp pattern**:
```cpp
// Submit early, overlap CPU cmdlist creation with GPU execution
if (submitted_nodes >= 100 || mul_mat_bytes >= 100MB) {
    submit();
    submitted_nodes = 0;
}
```

### 12.4 Fused RMS_NORM + MUL (MEDIUM — saves 28 dispatches/token)

Instead of:
```
dispatch(rms_norm, x → norm_out)
dispatch(multiply, norm_out × gate → result)
```

Single kernel:
```
dispatch(rms_norm_mul, x, gate → result)
```

### 12.5 Fused MATMUL + BIAS (MEDIUM — saves 28 dispatches/token)

**llama.cpp**: `fusion_flags |= MAT_VEC_FUSION_FLAGS_BIAS0` — the matvec kernel reads bias buffer and adds it in epilogue.

```glsl
// In mul_mat_vec.comp epilogue:
if ((p.fusion_flags & FUSION_FLAGS_BIAS0) != 0) {
    temp[j][n] += data_f0[d_offset + first_row + n];
}
```

---

## 13. Summary of Root Causes (Decode)

```
Total gap: 1.53ms GPU HW time

┌──────────────────────────────────────────────────────┐
│ Dawn barrier overhead (280 vs 60 barriers)  ~0.9ms   │  59%
│ No fusion (282 vs 150 dispatches)           ~0.3ms   │  20%
│ No cooperative matrix for attention         ~0.2ms   │  13%
│ Scalar dequant + no dp4a                    ~0.1ms   │   7%
│ Bind groups vs push constants               ~0.03ms  │   2%
└──────────────────────────────────────────────────────┘
```

## 14. Recommended Priority Order

1. **Fuse operators** (Medium effort, ~20% decode improvement): RMS+MUL, MATMUL+BIAS, SILU_MUL. Each fused op removes a dispatch AND its barrier.

2. **dp4a / dot4I8Packed** (Medium effort, ~10% decode improvement): Use WGSL's `dot4I8Packed` for Q8 matvec kernels — hardware int8 dot product.

3. **Batched prefill** (High effort, 100× prefill improvement): Implement T×K × K×N tiled matmul kernel for WGSL. Critical for user experience.

4. **Vectorize Q8 dequant** (Easy, ~5% decode improvement): Replace `extractBits` with packed u32 load + bitwise shift.

5. **Move argmax to CPU** (Easy, ~4% decode improvement): Read back top-K logits instead of full vocab argmax on GPU.

6. **Dawn barrier optimization** (Hard, ~40% decode improvement): Requires C++ changes to Dawn's Vulkan backend to track per-buffer dependencies instead of global barriers. Highest impact but hardest to implement.
