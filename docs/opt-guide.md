# WebGPU LLM Inference — Performance Optimization Guide

This guide captures the principles and strategies for optimizing LLM
inference on WebGPU.  Every future optimization should be evaluated
against these guidelines.

---

## 1. Model Optimization

### 1.1 Weight Quantization

Full-precision (fp32) weights are memory-bandwidth bound on both CPU and
GPU.  Quantization reduces the bytes read per token, directly improving
throughput.

| Format | Bytes/param | Phi-4 mini weight size | Notes |
|--------|-------------|----------------------|-------|
| fp32 | 4 | 14.6 GB | Baseline |
| fp16 | 2 | 7.3 GB | Needs `ShaderF16` feature |
| INT4 (per-group) | 0.5 + scales | 2.8 GB | 5.2× compression |

**Strategy**: Store weights as INT4 with per-group scales (group_size=128).
Pre-dequantize to fp32 at startup to avoid per-token dequantization overhead
(which was measured at 90% of decode latency).

**Why not pre-dequantize to fp16?**  fp16 would halve both RAM (6 GB vs
12 GB) and GPU memory bandwidth.  However, D3D12 does not support `f16`
typed storage buffer declarations in compute shaders — any use of `f16`
in `var<storage>` types returns zeros.  This is a Dawn/D3D12 backend
limitation, not a GPU driver bug (the same GPU's Dawn end-to-end tests
pass).  The underlying issue is that D3D12 HLSL typed buffers don't
natively support scalar 16-bit loads; the data must be loaded as `u32`
and unpacked on the shader side.

| WGSL Declaration | Read Pattern | Result |
|-----------------|-------------|--------|
| `array<f16>` | `W[i]` | ❌ Returns zeros |
| `array<vec2<f16>>` | `W[i/2].x / .y` | ❌ Returns zeros |
| `array<u32>` + `unpack2x16float()` | `unpack2x16float(W[i/2])` | ✅ Works |

The precision itself is fine — Phi-4's original weights are BF16, so
fp32→fp16 is lossless.

**Fix path**: Upload fp16 data as `u32` (viewing the same bytes) and
use `array<u32>` + `unpack2x16float()` in the shader.  This avoids
`f16` typed buffers entirely while still halving memory bandwidth.
The WGSL translator needs to emit `array<u32>` for `*fp16` pointers
with `unpack2x16float()` for loads and `pack2x16float()` for stores.
**Expected impact**: ~2× decode throughput (memory-bandwidth-bound).
**Future**: On-GPU INT4 dequantization kernels would eliminate the 12 GB RAM
overhead of pre-dequantization, but require fused dequant-matmul to amortize
the unpack cost.

### 1.2 Grouped Query Attention (GQA)

GQA uses fewer KV heads than Q heads (e.g., 24 Q heads, 8 KV heads for
Phi-4 mini).  This reduces:
- **KV cache size** by `n_head / n_kv_heads` (3× for Phi-4).
- **KV projection weight size** by the same factor.
- **Attention compute** for decode (fewer KV heads to expand).

**Implementation**: The Q/K/V projections are fused into a single QKV
weight `(q_dim + 2 * kv_dim, n_embd)`.  After projection, split the
output into Q, K, V tensors.  For attention, expand KV heads via
`np.repeat(K, n_rep, axis=head_axis)`.

### 1.3 Rotary Position Embeddings (RoPE)

RoPE encodes position information directly into Q and K vectors, avoiding
learned positional embeddings.

**Optimization**: Pre-compute cos/sin tables for all positions up to
`MAX_SEQ_LEN` at model init.  During decode, position lookup is a simple
array index — no trigonometric functions per token.

**Partial RoPE**: Some architectures (Phi-4) only rotate a fraction of
`head_dim` (e.g., `rotary_factor=0.75`).  The unrotated dimensions pass
through unchanged.

### 1.4 Fused Projections

Fusing multiple weight matrices into a single matmul reduces dispatch
count and improves memory locality.

| Fusion | Shape | Dispatch savings |
|--------|-------|-----------------|
| QKV projection | `(q_dim + 2*kv_dim, n_embd)` | 3→1 dispatches |
| Gate+Up projection (SwiGLU) | `(2*intermediate, n_embd)` | 2→1 dispatches |

Each avoided dispatch saves ~0.7 ms on D3D12 with warm caches.

### 1.5 Static KV Cache

The KV cache stores past key/value vectors so that decode only processes
one new token per step.  A *static* KV cache pre-allocates fixed-size
buffers at model init rather than growing dynamically.

#### Current implementation (CPU, pre-allocated numpy)

```python
# At first use per layer:
K_buf = np.zeros((MAX_SEQ_LEN, n_kv_heads, head_dim), dtype=np.float32)
V_buf = np.zeros((MAX_SEQ_LEN, n_kv_heads, head_dim), dtype=np.float32)

# Each decode step writes in-place:
K_buf[cur_len] = K_new   # no allocation, no copy
V_buf[cur_len] = V_new
cur_len += 1
```

This avoids `np.concatenate` (which allocates a new array every step)
and keeps the cache on CPU for the CPU attention path.

#### Target: GPU-resident static KV cache

To eliminate KV cache round-trips when running GPU decode, the cache
should live on GPU as fixed-size storage buffers:

```
Init:
  For each layer:
    K_cache_gpu = gpu_alloc(MAX_SEQ_LEN × n_kv_heads × head_dim × 4 bytes)
    V_cache_gpu = gpu_alloc(same)

Decode step:
  [GPU] QKV projection → Q, K_new, V_new (on GPU)
  [GPU] RoPE on Q, K_new (on GPU)
  [GPU] Write K_new → K_cache_gpu[cur_len]  (scatter write kernel)
  [GPU] Write V_new → V_cache_gpu[cur_len]
  [GPU] Attention: Q × K_cache_gpu[:cur_len+1] → scores → softmax → × V_cache_gpu
  [GPU] O projection
  ... (no CPU round-trip for attention)
```

**Memory cost** per layer (Phi-4 mini, MAX_SEQ_LEN=2048):

| Buffer | Shape | Size |
|--------|-------|------|
| K cache | (2048, 8, 128) | 8 MB |
| V cache | (2048, 8, 128) | 8 MB |
| **Per layer** | | **16 MB** |
| **All 32 layers** | | **512 MB** |

This fits comfortably in 16 GB VRAM alongside model weights (~1 MB norms
on GPU + ~384 MB per-layer weight streamed).

**Benefits**:
- Eliminates 2 CPU↔GPU round-trips per layer (readback QKV for CPU
  attention + upload attention output) — saves ~1.4 ms × 32 = ~45 ms/token.
- Enables fully pipelined GPU decode: norm → QKV → RoPE → cache update →
  attention → O proj → residual → norm → gate_up → SiLU → down → residual,
  all on GPU with no intermediate readback.
- Attention can use GPU-native parallelism across KV positions.

**Implementation steps**:
1. Pre-allocate `K_cache_gpu` and `V_cache_gpu` per layer at init.
2. Write a scatter kernel: `cache[pos, head, :] = new_vec[head, :]`.
3. Write a GPU decode-attention kernel that reads from the static cache
   buffers, computes Q·K scores, softmax, and V accumulation.
4. Pass `cur_len` as a scalar uniform to bound the attention loop.
5. Keep CPU KV cache as fallback for prefill (where GPU causal
   attention is already used per-head).

---

## 2. Op Optimization

### 2.1 GPU Limits and Workgroup Sizing

WebGPU imposes hard limits that constrain kernel design:

| Limit | Typical value | Impact |
|-------|--------------|--------|
| `maxComputeInvocationsPerWorkgroup` | 256 | Max threads per workgroup |
| `maxComputeWorkgroupSizeX` | 256 | Max X dimension of workgroup |
| `maxComputeWorkgroupsPerDimension` | 65535 | Max grid dimension |
| `maxStorageBufferBindingSize` | 128–256 MB | Max buffer per binding |
| `maxComputeWorkgroupStorageSize` | 16384 bytes | Shared memory per WG |

**Design rules**:
- **Workgroup size**: Use `num_warps * 32` threads.  For `BLOCK_K ≤ 256`,
  a single workgroup handles the full inner dimension.  For larger K,
  use loop-based kernels with `BLOCK_K=128` and iterate.
- **Grid dimensions**: When `N > 65535` (e.g., LM head with 200K vocab),
  chunk along N and concatenate results.
- **Buffer size**: When a weight matrix exceeds `maxStorageBufferBindingSize`,
  split into multiple buffers and dispatch per chunk.

**Always query actual adapter limits** rather than using WebGPU spec
minimums.  Request the full adapter limits during device creation:

```python
# Copy adapter's actual limits — not the conservative spec defaults
ctypes.memmove(ctypes.byref(required_limits),
               ctypes.byref(adapter_limits),
               ctypes.sizeof(WGPULimits))
```

### 2.2 Batch the Dispatches

Each GPU dispatch incurs overhead: command encoding, queue submission,
fence synchronization, and buffer readback.

| Scenario | Per-dispatch overhead |
|----------|---------------------|
| Cold (first run, pipeline creation) | ~11 ms |
| Warm (cached pipeline + buffers) | ~0.7 ms |

**Batching strategy**: Use `begin_batch()` / `end_batch()` to record
multiple compute passes into a single command encoder, submitting once:

```python
runner.begin_batch()
# Multiple dispatches share one command buffer
result1 = cache.run(kernel1, grid1, buffers1, gpu_outputs={'Y'})
result2 = cache.run(kernel2, grid2, buffers2, gpu_outputs={'Y'})
runner.end_batch(readback_buffers=[result2['Y']])
```

**Pipeline caching**: Compiled pipelines are keyed by WGSL source hash.
Identical kernels reuse the cached pipeline — no recompilation cost.

**Buffer caching**: GPU buffers are cached by `(name, size, usage)`.
Pre-upload model weights once at init; subsequent dispatches bind the
existing GPU buffer directly (zero upload cost).

### 2.3 CPU for Small Ops, GPU for Large Ops

The decision of where to run an operation depends on the ratio of
compute time to dispatch overhead.

| Category | Examples | Where to run | Why |
|----------|----------|-------------|-----|
| Large matmul | Linear projections (N×K with N,K ≥ 3072) | **GPU** | Compute-bound; GPU FLOPS dominate dispatch overhead |
| Small element-wise | RMSNorm, SiLU, residual add (3072 elements) | **CPU** | Dispatch overhead (~0.7 ms) exceeds compute time (~0.05 ms) |
| Attention (T=1) | Dot product Q·K, softmax, V accumulate | **CPU** | Small and sequential; numpy vectorized is faster than dispatch |
| LM head | (1, 3072) × (200064, 3072)^T | **CPU** | Exceeds `maxStorageBufferBindingSize`; chunked dispatch adds overhead |

**Rule of thumb**: If a single-threaded CPU op takes < 1 ms, run it on
CPU.  If the data is already on GPU and the op takes > 1 ms of compute,
keep it on GPU.

### 2.4 Once on GPU, Stay on GPU

Every CPU↔GPU transfer has cost: readback requires a buffer copy, a
fence wait, and a map operation.  The key principle is:

> **Minimize the number of GPU→CPU→GPU round-trips per token.**

**Decode pipeline design** (current best: 9.5 tok/s):

```
[CPU] RMSNorm (0.04ms) → tiny, not worth dispatch overhead
  ↓ upload x (once)
[GPU] QKV linear → readback for CPU attention
[CPU] Attention (0.1ms) → needs KV cache on CPU
[GPU] O projection → readback for CPU residual
[CPU] Residual + RMSNorm (0.05ms)
  ↓ upload
[GPU] Gate+Up linear → readback for CPU SiLU
[CPU] SiLU·mul (0.02ms)
  ↓ upload
[GPU] Down linear → readback for CPU residual
[CPU] Residual add
```

**Optimization opportunity**: Fuse norm/activation into adjacent GPU
kernels to eliminate intermediate round-trips.  For example:
- Fuse RMSNorm + QKV projection into one kernel (eliminates 1 upload)
- Fuse Gate+Up + SiLU·mul into one kernel (eliminates 1 readback + upload)
- Fuse Down projection + residual add (eliminates 1 readback)

Each eliminated round-trip saves ~0.7 ms × 32 layers = ~22 ms/token.

### 2.5 Subgroups

The WebGPU `Subgroups` feature enables warp-level communication
(`subgroupShuffleXor`, `subgroupAdd`, etc.) without shared memory
barriers.

**Benefits**:
- **Reductions**: Butterfly reduction via shuffle XOR (masks 16, 8, 4, 2, 1)
  without workgroup memory round-trips.
- **No barriers**: Intra-warp communication is implicit — no
  `workgroupBarrier()` needed.
- **Lower register pressure**: Values stay in registers instead of
  being spilled to shared memory.

**Availability**: Query `adapter.has_subgroups` at runtime.  When
unavailable, fall back to shared-memory emulation:

```wgsl
// Native subgroups (fast):
let result = subgroupShuffleXor(value, mask);

// Emulation fallback (slower, uses workgroup memory):
_shfl[local_id.x] = value;
workgroupBarrier();
let result = _shfl[local_id.x ^ mask];
workgroupBarrier();
```

The WGSL translator automatically chooses the correct path based on
adapter capabilities.

### 2.6 ShaderF16

When the `ShaderF16` feature is available, weights can be stored as
`f16` in storage buffers, halving memory bandwidth for weight reads.

**Impact**: For memory-bandwidth-bound operations (linear projections at
T=1), fp16 weights can theoretically 2× throughput.  In practice,
gains depend on whether the GPU's fp16 storage path is fully optimized
by the driver.

**Usage**: The `linear_loop_fp16w_kernel` accumulates in fp32 for
numerical stability while reading weights as fp16:

```
W stored as fp16 → load → cast to fp32 → accumulate → store fp32 output
```

### 2.7 Kernel Design for WebGPU

WebGPU compute shaders have specific constraints compared to CUDA:

- **No dynamic shared memory**: `var<workgroup>` sizes must be
  compile-time constants.
- **No warp-level primitives without Subgroups feature**: Must
  emulate with shared memory + barriers.
- **256-thread workgroup limit**: Kernels must fit within this;
  use loop-based iteration for larger dimensions.
- **No global synchronization**: Each dispatch is independent.  For
  multi-pass algorithms, use separate dispatches with intermediate
  buffers.
- **Uniform store guard**: On some backends (D3D12), concurrent UAV writes
  from multiple threads to the same address can silently lose data.
  Guard scalar stores with `if local_id.x == 0u { ... }`.

---

## 3. Profiling

### 3.1 Unified CPU+GPU Timeline Profiler

The most important tool for optimization is the **unified CPU+GPU timeline
profiler** (`common/profiler.py` + `common/profiler_html.py`). It correlates
CPU and GPU events on a single timeline so you can see exactly where time
is spent — and more critically, where the CPU is waiting on the GPU
or vice versa.

**Architecture**:

```
InferenceProfiler
  ├── CPUProfiler        — time.perf_counter_ns() scoped events
  ├── GPUTimestampProfiler  — WebGPU hardware timestamp queries
  └── GPUDispatchEvents  — CPU-timed GPU dispatch durations
```

The profiler records three types of events on a shared nanosecond clock:

| Event type | Source | What it measures |
|-----------|--------|-----------------|
| **CPU events** | `profiler.cpu("op_name")` context manager | CPU-side compute (norms, attention, activation) |
| **GPU dispatch events** | Automatically by `KernelCache.run()` | Wall-clock cost of each GPU dispatch (encode → submit → poll → readback) |
| **GPU hardware timestamps** | WebGPU `TimestampQuery` feature | Exact GPU-side kernel execution time (excludes dispatch overhead) |

**How it works**: When a GPU dispatch finishes, `KernelCache.run()` calls
`profiler.record_dispatch(name, begin_ns, end_ns)` with CPU-side
`perf_counter_ns()` timestamps. GPU hardware timestamps are correlated
to the CPU timeline by anchoring the first GPU timestamp's `begin_ns` to
its `cpu_submit_ns` and applying the offset to all subsequent GPU events.

**CPU-GPU clock calibration**: The current approach uses a simple anchor
(first GPU timestamp aligned to its CPU submit time), which can drift over
long runs.  A more accurate method is D3D12's
[`ID3D12CommandQueue::GetClockCalibration`](https://learn.microsoft.com/en-us/windows/win32/api/d3d12/nf-d3d12-id3d12commandqueue-getclockcalibration),
which returns a matched pair of GPU and CPU timestamps at the same instant.
This gives the exact relative offset between the two clocks, enabling
precise wall-time alignment of CPU and GPU timeline events without drift.
Dawn exposes the underlying D3D12 command queue, so this calibration could
be called at profiling start (and periodically during long runs) to
maintain accurate cross-clock alignment.

**Enabling profiling**:

```python
# In model code:
model.enable_profiling()   # sets model.profiler.enable()

# After inference:
model.profiler.report()    # console summary

# HTML timeline:
from common.profiler_html import generate_html_report
generate_html_report(model.profiler, "profile.html")
```

Most models support `--profile` flag:
```bash
python models/phi-4/model.py --profile --prompt "Hello" --max-tokens 10
```

### 3.2 The HTML Timeline — What It Shows

The `generate_html_report()` produces an interactive HTML page with:

- **Two swim lanes**: CPU events (top) and GPU events (bottom) on a
  synchronized time axis
- **Step markers**: vertical lines separating decode steps
- **Zoom/pan**: scroll to zoom, drag to pan — examine individual
  dispatches at microsecond granularity
- **Tooltips**: hover any bar to see exact duration
- **Summary tables**: aggregated time per operation across all steps

**Why this matters**: The unified timeline reveals bottlenecks that
per-op timing alone cannot:

1. **CPU-GPU pipeline bubbles**: If the CPU lane shows idle gaps while
   waiting for GPU readback, the bottleneck is readback latency, not
   compute. Fix: use `gpu_out=True` to eliminate unnecessary readbacks.

2. **Dispatch overhead dominance**: If GPU bars are tiny but the gap
   between them is large, dispatch overhead dominates. Fix: batch
   dispatches or move small ops to CPU.

3. **Serialization stalls**: If CPU and GPU never overlap, the pipeline
   is fully serial. Fix: overlap CPU ops (norm, activation) with
   in-flight GPU dispatches via batched command buffers.

4. **Memory bandwidth saturation**: If GPU hardware timestamps show
   near-identical kernel times regardless of compute intensity,
   the kernel is memory-bound. Fix: fp16 weights, quantization.

### 3.3 Per-Layer Profiling with CPU Scopes

For decode-path optimization, the profiler also integrates directly
into model code via CPU scopes:

```python
# In _decode_cpu() or _transformer_block():
p = self.profiler
if p and p.enabled:
    p._cpu.begin("rms_norm_1")
    rn1 = self._rms_norm_cpu(x, ...)
    p._cpu.end("rms_norm_1")

    p._cpu.begin("qkv_linear")
    qkv = self._cpu_matmul(rn1, ...)
    p._cpu.end("qkv_linear")
```

This produces per-op timing within each layer, making it easy to see
which operation is the bottleneck (matmul vs. attention vs. norm).

### 3.4 Identifying Bottlenecks

Use `--profile` to generate both a console report and an interactive
HTML timeline (`profile.html` in the model folder).

```bash
python models/phi-4/model.py --profile --prompt "Hello" --max-tokens 10
```

The profiler tracks:
- **CPU timeline**: `time.perf_counter_ns()` around each op
- **GPU timeline**: WebGPU timestamp queries (if `TimestampQuery` feature
  is available)

### 3.5 What to Look For

1. **Dispatch overhead dominating**: If CPU time per op is mostly
   dispatch overhead (encode → submit → fence → readback), consider
   batching dispatches or moving the op to CPU.
2. **Memory bandwidth saturation**: If GPU compute utilization is low
   but throughput doesn't improve with faster compute, the kernel is
   memory-bound.  Consider fp16 weights or quantized storage.
3. **Upload/download stalls**: If `upload_weights` appears in the
   hotspot list, ensure weights are pre-uploaded and cached on GPU.
4. **Small-op overhead**: If norms/activations take > 0.5 ms each,
   they're dominated by dispatch overhead — move to CPU or fuse.
5. **GPU bubbles between kernel executions**: In the HTML timeline,
   look for gaps between consecutive GPU kernel bars.  If GPU HW time
   is much smaller than CPU-timed dispatch time (e.g., 5.9ms vs 21.6ms
   per token), the GPU is idle while the CPU prepares the next dispatch.
   This is the strongest signal that the **fast decode path** should be
   enabled.

   **Diagnosis**: Compare the GPU Kernels (HW Timestamps) total with
   the GPU Dispatches (CPU-timed) total in the profiler report:

   ```
   GPU Dispatches (CPU-timed):  228ms (2432 dispatches)  ← includes bubbles
   GPU Kernels (HW Timestamps):  59ms (2432 kernels)     ← actual compute
   ```

   If HW time is < 50% of CPU-timed dispatch time, the GPU is bubble-
   bound.  The gap is CPU overhead per dispatch (~0.04ms × 219
   dispatches/token = ~9ms/token) plus fence/readback latency.

   **Fix**: Use the pre-recorded fast decode path (`_decode_fast`).
   This pre-creates all bind groups and dispatch lists at init time,
   then submits them via `submit_dispatches_pipelined()` with only
   3 buffer writes per token.  Example result (SmolLM2-1.7B):

   | Path | Dispatches/token | Decode tok/s |
   |------|-----------------|--------------|
   | Regular GPU decode | 219 individual | 93.6 |
   | Fast decode | 1 pipelined submit | 209 |

   The fast path reduces per-token dispatch from 219 individual Python-
   level dispatches to a single pipelined submit of pre-recorded
   commands, eliminating virtually all CPU-GPU bubbles.

### 3.6 Key Metrics

| Metric | How to compute | Target |
|--------|---------------|--------|
| Decode tok/s | 1000 / per_token_ms | Higher is better |
| Memory BW utilization | bytes_read / time / peak_bw | > 80% means BW-bound |
| Dispatch overhead ratio | dispatch_overhead / total_time | < 10% |
| GPU occupancy | active_threads / max_threads | > 50% |

### 3.7 TPS Measurement Methodology

The `generate()` function uses two timers:

- **Prefill (TTFT)**: from the very beginning (before prefill forward)
  to the output of the first token.  This includes prefill forward,
  fast decode pipeline init, 1st decode forward, and 1st sampling.
- **Decode**: from immediately after the first token is output to the
  last token.  Only tokens 2+ are counted.

```
Prefill (TTFT): 1573.1ms (5 prompt + 1st token)
Decode:  99 tokens in 838.9ms (118.0 tok/s)
  forward() only: 768.3ms (128.9 tok/s)
```

Two decode TPS numbers are reported:
- **forward() only**: time spent inside `model.forward()`, excludes
  sampling, tokenizer decode, and stdout flush.  This is the model’s
  actual throughput.
- **wall-clock**: total loop time including sampling and I/O.
  This is what the end-user experiences.

Weight upload happens during model loading, before `generate()` starts,
and is never included in either timer.

To reproduce, always use `≥ 50` decode tokens so that per-token
variance averages out.

---

## 4. Buffer Management

### 4.1 Keep Kernels on GPU

The most impactful optimization is ensuring that GPU buffers stay on GPU
between kernel invocations.  Every GPU→CPU readback incurs:
- `wgpuCommandEncoderCopyBufferToBuffer` to a staging buffer
- `wgpuQueueSubmit` for the copy
- `wgpuBufferMapAsync` + fence wait (blocks CPU)
- `memcpy` from mapped pointer

**Cost**: ~0.3–0.7 ms per readback.  Over 24 layers, unnecessary
round-trips add 7–17 ms per token.

**Strategy**: Use `gpu_out=True` on all intermediate operations so
results stay as `GPUBuffer` objects.  Only readback when CPU logic
requires it (e.g., MoE routing, argmax for token selection).

```python
# BAD: readback after every op
norm = self._rms_norm(x, w)              # returns numpy
qkv = self._linear_fp16w(norm, w, b, N, K) # uploads norm, returns numpy

# GOOD: stay on GPU
norm = self._rms_norm(x, w, gpu_out=True)      # returns GPUBuffer
qkv = self._linear_fp16w(norm, w, b, N, K, gpu_out=True)  # no upload
```

### 4.2 Buffer Reuse with Size-Class Pooling

GPU buffer allocation (`wgpuDeviceCreateBuffer`) is expensive.  The
dawn runner caches buffers by `(name, size, usage)`, but exact-size
matching misses reuse opportunities when sizes differ by small amounts.

**Strategy**: Round output buffer sizes up to the next size class
(powers of 2 or fixed buckets), so that differently-sized outputs can
reuse the same physical buffer.  This reduces total allocations and
improves cache hit rates.

```
Sizes used across MoE:       2880, 5760, 11520 floats
Rounded to next power of 2:  4096, 8192, 16384 floats
```

The toggle-pool already alternates between two buffers per `(name, size)`
to prevent read-write aliasing.  With size-class pooling, more
allocations map to the same `(name, rounded_size)` key.

### 4.3 Intermediate Buffer Lifetime Analysis

Beyond size-class pooling, full **lifetime analysis** of intermediate
tensors can dramatically reduce memory pressure.  Because neural network
execution is sequential, tensors with non-overlapping lifetimes can share
the same physical buffer.

```
Without sharing: 4.31 GB (all activations live simultaneously)
With Greedy-by-Size: 387 MB (93% reduction via sequential reuse)
```

The *Greedy by Size* strategy assigns the largest-first tensor to the
smallest available memory slot, minimizing total allocation.

**Our status**: We use a toggle-pool with 2 buffers per `(name, size)`.
Full lifetime analysis across the entire layer DAG could further reduce
allocation count.

### 4.4 Pre-Allocated Working Buffers

For operations with known maximum sizes (e.g., MoE accumulator, KV
cache), pre-allocate buffers at init time and reuse via
`write_buffer()` rather than `upload_to_gpu()` per call.

```python
# At init:
self._moe_x_gpu = runner.upload_to_gpu(zeros(E), "moe_x")
self._moe_acc_gpu = runner.upload_to_gpu(zeros(E), "moe_acc")

# Per call: overwrite contents, no allocation
runner.write_buffer(self._moe_x_gpu.handle, data.tobytes())
```

### 4.5 Memory Pool with Smart Heuristics

Balancing performance and memory usage requires a managed memory pool
rather than allocate-on-demand / destroy-on-free.  Key principles:

1. **Over-allocate to aligned sizes**: Allocate buffers rounded up to
   a **size class** (e.g., powers of 2, or fixed buckets like 4KB,
   16KB, 64KB, 256KB, 1MB, 4MB, ...).  When a buffer is freed, it
   returns to the pool's free list under its size class instead of
   being destroyed.  Future allocations of the same or smaller size
   reuse the pooled buffer with zero allocation cost.

2. **Free → pool, not destroy**: GPU buffer creation
   (`wgpuDeviceCreateBuffer`) is expensive (~0.1–0.5 ms).  By
   returning freed buffers to a per-size-class free list, subsequent
   allocations become O(1) lookups instead of GPU API calls.

3. **Smart size-class alignment**: Choose size classes that balance
   fragmentation vs. reuse:
   - Too many classes → buffers rarely match → low reuse
   - Too few classes → large waste per buffer → high memory overhead
   - Good default: powers of 2 above 4KB, with finer granularity
     (e.g., 1.5× steps) below 4KB for small norm/bias buffers

4. **Peak memory tracking**: Monitor high-water-mark to right-size
   the pool.  After warmup (first inference pass), the pool stabilizes
   and no further allocations should occur during steady-state decode.

```
Example size classes:
  4K, 8K, 16K, 32K, 64K, 128K, 256K, 512K,
  1M, 2M, 4M, 8M, 16M, 32M, 64M, 128M, 256M

A 3000-float buffer (12KB) → rounded to 16KB class
A 3100-float buffer (12.4KB) → same 16KB class → reuses freed buffer
```

**Current status**: The dawn runner uses exact `(name, size, usage)`
caching with a toggle pool.  Migrating to size-class pooling would
reduce allocation count by ~50% for models with varying activation
sizes (e.g., MoE, varying sequence lengths).

---

## 5. Matmul Optimization

### 5.1 Current Matmul Strategy

| Kernel | Weights | Inner loop | Use case |
|--------|---------|-----------|----------|
| `linear_loop_fp16w` | fp16→fp32 | Loop over K in blocks | Dense layers |
| `linear_q4` | INT4 + scales | Per-group dequant | Phi-4 quantized |
| `linear_mxfp4` | MXFP4 packed | Fused dequant | GPT-OSS experts |

All kernels are **memory-bandwidth bound** at T=1 (decode): the GPU
reads the full weight matrix (N×K) for a single vector multiply.

**Weight layout optimization**: Storing weights in 4-element SIMD-aligned
slices (e.g., `W[N/4, K, 4]` instead of `W[N, K]`) ensures each GPU
thread reads a contiguous vec4, maximizing memory bandwidth utilization.
This layout can yield **~20% matmul speedup** on GPUs with native vec4
load paths.  Our Triton kernels let the compiler handle layout; for
hand-crafted WGSL kernels, explicit vec4-aligned weight storage could
improve throughput.

### 5.2 DP4A / Dot Product Accumulate

D3D12 supports `dp4a` (Dot Product of 4 Accumulate) which computes
4 INT8 multiply-adds in a single instruction.  This is the key to
fast INT4/INT8 matmul on GPU.

**WebGPU status**: Available via `packed_4x8_integer_dot_product`
WGSL language feature.  Verified working on Dawn D3D12 (RTX 5080):

```wgsl
requires packed_4x8_integer_dot_product;
// dot4U8Packed: dot product of 4 packed unsigned bytes
let result: u32 = dot4U8Packed(a_packed, b_packed);
// dot4I8Packed: signed variant
let result: i32 = dot4I8Packed(a_packed, b_packed);
```

**Enabling** (browser API):
```javascript
if (!navigator.gpu.wgslLanguageFeatures.has("packed_4x8_integer_dot_product")) {
  throw new Error("DP4a built-in functions are not available");
}
```

**When DP4A helps**: DP4A accelerates the **integer arithmetic** part of
the dot product.  It processes 4 byte-level multiply-adds in a single
instruction.  The biggest wins come from **W8A8** (INT8 weights + INT8
activations) or **W4A8** scenarios where both operands are integers.

**When DP4A doesn't help**: For **W4_fp32** (INT4 weights, fp32
activations), the bottleneck is **memory bandwidth**, not ALU.  At
T=1 decode, each matmul reads the full N×K weight matrix regardless
of whether nibbles are extracted with shift+mask or DP4A.  The
compute is not the bottleneck.  Benchmarks show the Triton-compiled
kernel (0.17ms at N=4608, K=3072) outperforms hand-crafted WGSL DP4A
kernels (1.5ms) due to superior pipeline/buffer caching and deferred
`subgroupShuffleXor`-based reduction.

**Future**: W4A8 quantization (the approach used by ONNX Runtime WebGPU EP
with `int4_accuracy_level=4`):

1. **Offline**: Quantize weights to INT4 with per-group scales using
   `MatMulNBitsQuantizer` from `onnxruntime.quantization`, or
   `onnxruntime-genai` builder:
   ```bash
   python builder.py -m microsoft/phi-4-mini -o phi4_int4 \
     -p int4 -e webgpu --extra_options int4_block_size=32
   ```

2. **Runtime** (in the matmul kernel):
   - Upcast INT4 weights → INT8: expand 4 nibbles to 4 bytes in a `u32`
   - Quantize fp32 activations → INT8 per-group: `x_q = round(x / x_scale)`
   - Use `dot4U8Packed(x_packed, w_packed)` for 4 multiply-adds per instruction
   - De-scale result: `result * x_scale * w_scale`

```wgsl
// Per group of 4 elements:
// 1. Extract 4 INT4 nibbles into bytes
let w_byte0 = packed_w & 0xFFu;
let w_byte1 = (packed_w >> 8u) & 0xFFu;
let w_bytes = w_byte0 | (w_byte1 << 8u) | ...;

// 2. Quantize 4 activations to uint8
let x_scale = max(abs(x[0..3])) / 127.0;
let x_q = round(x / x_scale);
let x_packed = pack_4_u8(x_q[0], x_q[1], x_q[2], x_q[3]);

// 3. DP4A: 4 multiplies + accumulate in single instruction
acc += dot4U8Packed(x_packed, w_bytes);

// 4. De-scale
result = f32(acc) * x_scale * w_scale;
```

**Benefit**: DP4A processes 4 multiply-adds per instruction vs 4
separate fp32 multiplies.  On hardware with dedicated DP4A units
(NVIDIA Turing+, AMD RDNA3+), this can significantly improve
throughput for memory-bandwidth-bound decode operations.

**Stage-aware quantization**: For the memory-bound **decode** stage,
fuse activation quantization directly into the matmul kernel — the
quantization arithmetic is effectively "free" since the GPU is
bottlenecked on weight memory reads anyway.  For the compute-bound
**prefill** stage, use a separate quantization dispatch to prepare
INT8 activations, allowing the matmul kernel to use pure INT8
instructions without per-element branching overhead.

**Current implementation** (`--use-dp4a` flag optional):
Both Triton Q4 and DP4A fast decode paths now use the same hand-crafted
WSGL Q4 kernel with workgroup_size=256, TILE_N=8 (multi-output tiling),
`subgroupAdd` reduction, and pre-compiled fast decode pipeline.  This
achieves **~130 tok/s** forward-only (steady-state decode).  Each
workgroup produces 8 output elements using 8 warps, with L1/L2 cache
providing activation data reuse across warps.  The kernel dequantizes
INT4 weights to fp32 on-the-fly — no int8 activation quantization.
Full fp32 precision; output identical to Triton.

The previous Triton Q4 fast decode path used grid=(1, N) with 1 output
per workgroup, achieving only 95 tok/s.  Switching to the WGSL kernel
(grid=(1, N/8), 8 outputs/WG) gave a **37% speedup** from fewer
workgroup dispatches and native `subgroupAdd`.

**ORT's cooperative tiled matmul** (for prefill, M≥64):
ORT uses a 64×64 output tile with 256 threads, shared memory tiling,
and `subgroupShuffle` for register-level data sharing.  Both A (activations)
and B (weights) are quantized to INT8.  The `SDP8AI` helper performs
8× `dot4I8Packed` per subtile step.  This is designed for the
**compute-bound prefill** stage; for memory-bound decode (T=1), the
simpler per-thread serial approach is already optimal.

### 5.3 Subgroup Optimizations for Matmul

Subgroup operations (`subgroupAdd`, `subgroupShuffleXor`) enable
efficient partial-sum reduction across threads without shared memory:

```wgsl
// Reduce partial sums across 32 threads in a subgroup
var sum = partial_sum;
sum = subgroupAdd(sum);  // single instruction, no barrier
```

**Current**: The WGSL translator automatically emulates subgroup ops
via shared memory when the `Subgroups` feature is unavailable.

### 5.4 Shared Memory Tiling

For matmul with K > workgroup size, load tiles of X and W into shared
memory (`var<workgroup>`) to improve data reuse:

```
for tile in 0..K/BLOCK_K:
    Load X[row, tile*BK:(tile+1)*BK] → shared_x
    Load W[col, tile*BK:(tile+1)*BK] → shared_w
    barrier()
    acc += dot(shared_x, shared_w)
    barrier()
```

**Constraint**: WebGPU limits shared memory to 16384 bytes per
workgroup.  With f32, that's 4096 elements — enough for small tiles
but not large GEMM blocks.

---

## 6. Op Fusion

### 6.1 Principles

Each GPU dispatch incurs ~0.1–0.7 ms overhead (command encoding, bind
group creation, queue submission).  Over 24 layers, even small per-
dispatch savings compound: saving 1 dispatch × 24 layers × 10 tokens
= 240 avoided dispatches.

### 6.2 Existing Fusions

| Fusion | Saves | Implementation |
|--------|-------|---------------|
| Q+K+V → QKV | 2 dispatches/layer | Concatenated weight matrix |
| Gate+Up → GateUp | 1 dispatch/layer | Concatenated weight matrix |
| RoPE Q + RoPE K + KV scatter | 1 dispatch/layer | `fused_rope_qkv_kernel` |
| Attention (Q·K + softmax + V) | 2 dispatches | Single `gqa_decode_attn_kernel` |
| Residual + RMSNorm | 1 dispatch/layer | `add_rms_norm_loop_kernel` |
| SiLU·Gate (GPT-OSS) | 1 dispatch | `gptoss_gate_kernel` |

### 6.3 Fusion Opportunities

| Fusion | Expected saving | Complexity |
|--------|----------------|-----------|
| RMSNorm + QKV linear | 1 dispatch (norm becomes prologue of matmul) | Medium |
| O-proj + residual add | 1 dispatch (add bias + residual in epilogue) | Low |
| Down-proj + residual add | 1 dispatch (same pattern) | Low |
| RoPE + QKV layout reshape | 1 dispatch (apply RoPE and reshape Q/K/V layout in one pass) | Medium |

### 6.4 Batch-Level Fusion via Command Batching

Rather than fusing at the kernel level, batch multiple dispatches into
a single command encoder submission.  This eliminates per-submit overhead
while keeping individual kernel simplicity:

```python
runner.begin_batch()
# 8 dispatches, 1 submission
norm = self._rms_norm(x, w, gpu_out=True)
qkv = self._linear_fp16w(norm, w, b, N, K, gpu_out=True)
# ... more ops ...
runner.end_batch()
```

**Current**: GPT-OSS uses two-phase batching per layer:
- Phase 1 (attention): 8 dispatches → 1 submit
- Phase 2 (MoE): 17 dispatches → 1 submit

This reduced dispatch overhead from 1760ms → 337ms (5.2× speedup).

### 6.5 Fast Decode: Pre-Compiled Pipeline

The highest-performance decode path eliminates **all per-token Python
overhead** by pre-compiling everything at init time:

1. **Pre-compile pipelines**: Call `get_pipeline_info()` once per kernel
   to create the WGSL pipeline and bind group layout.
2. **Pre-create bind groups**: One per layer per op (32 layers × 12 ops
   = 384 bind groups), each binding the layer's weight buffers and
   shared intermediate buffers.
3. **Pre-upload params**: Constant scalar params (K, N, stride, eps)
   packed into GPU buffers — one per unique parameter set.
4. **Pre-allocate intermediates**: Fixed-size GPU buffers for norm_out,
   qkv_out, attn_out, etc., reused across all layers.

Per-token decode then reduces to:
```python
# Only 3 dynamic params change per token:
write_buffer(rope_params, pack(pos))          # position
write_buffer(rope_kv_params, pack(cache_off)) # KV cache offset
write_buffer(attn_params, pack(T_total))      # sequence length

# Submit all 384 pre-recorded dispatches in pipelined batches
logits = submit_dispatches_pipelined(all_batches, readback=logits_buf)
```

**Results** (phi4-mini, RTX 5080, steady-state decode, 99 tokens):
- Without fast decode: ~30 tok/s (per-dispatch Python overhead)
- Previous Triton Q4 fast decode: 95 tok/s (used Triton kernel, grid=(1,N))
- Unified WGSL Q4 fast decode: **130 tok/s** forward-only (both paths)

**Key insight**: The per-dispatch overhead is not GPU compute time —
it's Python-side bind group creation, buffer allocation, and command
encoding.  Pre-creating everything eliminates this entirely.

**Pitfall — params buffer overwrite during batching**: When multiple
matmuls with different shapes (QKV N=5120, O N=3072, gate_up N=16384)
share the same params buffer name, `wgpuQueueWriteBuffer` overwrites
previous values.  Fix: use separate pre-uploaded GPUBuffers per
projection shape, or unique buffer names per params set.

---

## 7. Weight Quantization Strategies

### 7.1 INT4 with Per-Group Scales

Store weights as 4-bit integers with per-group (group_size=128) float
scales and zero-points.  At decode time, dequantize in the matmul kernel:

```
w_fp32 = (w_q4 - zero) * scale
y += x * w_fp32
```

**Used by**: Phi-4 mini quantized mode.
**Compression**: 4× over fp16, 8× over fp32.

### 7.2 MXFP4 (Microscaling FP4)

Store weights as FP4 (E2M1) with per-block (32-element) E8M0 scales.
The FP4 lookup table converts nibbles to float values:

```
FP4 values: 0, ±0.5, ±1, ±1.5, ±2, ±3, ±4, ±6
Scale: 2^(byte - 127)
```

**Used by**: GPT-OSS-20B (32 MoE experts × 24 layers).
**Compression**: 4× over fp16.

### 7.3 Mixed Quantization (8/4/4)

Not all layers are equally sensitive to quantization.  A mixed-precision
strategy applies different bit widths per layer type:

| Layer type | Precision | Rationale |
|-----------|-----------|----------|
| Attention (Q, K, V, O) | INT8 | Accuracy-sensitive |
| Feed-forward / MLP | INT4 | Larger, more compressible |
| Embeddings | INT4 | Largest single weight matrix |

This gives ~1.8× decode speedup over uniform INT8, while preserving
attention accuracy.  Token prefill speed is largely unaffected by
quantization level (compute-bound).

**Our status**: Phi-4 uses uniform INT4 for all weights.  GPT-OSS uses
MXFP4 for MoE experts, fp16 for attention.  Mixed quantization could
improve quality by keeping attention layers at INT8 precision.

### 7.4 fp16 Weight Storage

For dense layers that don't benefit from quantization (norms, biases,
small projections), store as fp16 using the `u32` + `unpack2x16float()`
pattern to work around D3D12's typed buffer limitation:

```wgsl
// Works on all backends:
let packed: u32 = W_u32[idx / 2u];
let pair = unpack2x16float(packed);
let w: f32 = select(pair.x, pair.y, (idx & 1u) != 0u);
```

### 7.5 Precision Requirements by Model Type

Image generation models (diffusion transformers) are significantly more
sensitive to quantization than LLMs.  Small weight perturbations
accumulate across 28–50 denoising steps and produce visible artifacts:
color shifts, blurring, structural distortion.

| Model type | Recommended | Notes |
|-----------|------------|-------|
| LLM (text) | INT4 | Token prediction is robust to quantization |
| Image gen (DiT) | **fp16** | Visual quality degrades noticeably at INT8 and below |
| VAE decoder | **fp16** | Pixel-level reconstruction is precision-sensitive |
| Text encoder | fp16 or INT8 | Embedding quality tolerates mild quantization |

**Rule**: For image models, use fp16 weights (2 bytes/param) as the
default.  INT4/INT8 quantization may be acceptable for some layers
(e.g., FFN) but should be validated visually on representative prompts
before deploying.  The `u32` + `unpack2x16float()` pattern (§7.4)
enables fp16 storage on D3D12 without the `f16` typed buffer bug.

---

## 8. KV Cache Strategies

### 8.1 GPU-Resident Static Cache

Pre-allocate fixed-size GPU buffers per layer at init.  New K/V vectors
are scattered into the cache via a GPU kernel (no CPU involvement):

```python
# Init: pre-allocate for MAX_SEQ_LEN
K_cache = gpu_alloc(MAX_SEQ_LEN * n_kv * HD * 4)
V_cache = gpu_alloc(MAX_SEQ_LEN * n_kv * HD * 4)

# Decode: scatter via rope_kv_scatter_kernel
scatter(K_new → K_cache[cur_len])
scatter(V_new → V_cache[cur_len])
```

**Benefits**:
- Zero allocation during decode
- KV data never leaves GPU
- Supports sliding window attention (GPT-OSS: 128-token window)

### 8.2 KV Cache Memory Budget

| Model | Layers | n_kv | HD | MAX_SEQ | Per-layer | Total |
|-------|--------|------|-----|---------|-----------|-------|
| Phi-4 mini | 32 | 8 | 128 | 2048 | 16 MB | 512 MB |
| GPT-OSS-20B | 24 | 8 | 64 | 2048 | 4 MB | 96 MB |

---

## 9. Attention Optimization

### 9.1 Grouped Query Attention (GQA)

GQA reduces KV heads (e.g., 8) relative to Q heads (e.g., 64),
reducing both memory and compute.  The attention kernel repeats each
KV head for multiple Q heads:

```python
# n_rep = n_head // n_kv_heads = 8
kv_head = q_head // n_rep
# Each KV head serves 8 Q heads
```

**GPU kernel**: The `gqa_decode_attn_kernel` handles GQA natively by
computing `kv_head = pid / n_rep` in the grid dispatch.

### 9.2 Attention with Sinks and Sliding Window

GPT-OSS uses attention sinks (learnable bias per head) with alternating
full-attention and sliding-window layers:

```
Full attention:    attend to all T positions
Sliding window:    attend to last WINDOW positions only
Sinks:             additive bias on softmax logits
```

This is handled by `gqa_decode_attn_sink_kernel` which supports both
modes via the `kv_start` and `T_win` parameters.

### 9.3 RoPE (Rotary Position Embeddings)

Pre-compute cos/sin tables at init.  GPU kernels apply rotation
in-place:

```wgsl
// For each pair (x[2i], x[2i+1]):
y[2i]   = x[2i]   * cos[pos][i] - x[2i+1] * sin[pos][i]
y[2i+1] = x[2i+1] * cos[pos][i] + x[2i]   * sin[pos][i]
```

**Partial RoPE**: Phi-4 rotates only 75% of head_dim.
**Full RoPE**: GPT-OSS rotates all of head_dim (half_rot = HD/2).

**Cache layout for attention compatibility**: Storing K cache as
transposed `K^T` (so that `Q @ K_cache` computes `Q·K^T` directly)
and V cache with dimensions arranged for the desired attention output
layout can eliminate layout transforms during attention computation.
Our flat `[MAX_SEQ, n_kv, HD]` layout is already sequential-read
friendly for the attention kernel.

---

## 10. Decision Checklist

When adding a new operation or optimizing an existing one:

- [ ] **Is it large enough for GPU?**  If compute < 1 ms, consider CPU.
- [ ] **Can it be fused with an adjacent op?**  Each eliminated dispatch
      saves ~0.1–0.7 ms × n_layers.
- [ ] **Is the data already on GPU?**  Avoid CPU→GPU→CPU→GPU round-trips.
- [ ] **Does it exceed GPU limits?**  Check buffer size, workgroup size,
      and grid dimensions.
- [ ] **Can it use subgroups?**  For reductions and shuffles, native
      subgroups are significantly faster.
- [ ] **Can weights be fp16 or quantized?**  Halve or quarter memory
      bandwidth with appropriate storage format.
- [ ] **Can buffers be reused?**  Use pre-allocated or size-rounded
      buffers to avoid per-dispatch allocation.
- [ ] **Is the pipeline cached?**  Ensure identical kernels reuse
      compiled pipelines.
- [ ] **Can dispatches be batched?**  Group GPU-only ops into single
      `begin_batch()`/`end_batch()` submissions.
- [ ] **Is it prefill or decode?**  Use compute-optimized kernels for
      prefill, memory-optimized for decode (§5.2).
- [ ] **Can mixed quantization help?**  Keep accuracy-sensitive layers
      (attention) at INT8, compress FFN/embed to INT4 (§7.3).
- [ ] **Profile before and after.**  Use the HTML timeline to verify
      the optimization actually helps.
