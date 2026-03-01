# Triton WebGPU Backend

This document tracks the major achievements in adding WebGPU as a backend target for Triton, enabling GPU compute on any platform that supports WebGPU (Vulkan, Metal, D3D12) without requiring CUDA hardware.

## Architecture

```
Triton Kernel (.py)
        │
        ▼
    TTIR (Triton IR)
        │
        ▼
    TTGIR (Triton GPU IR)
        │
        ▼
    LLIR (LLVM IR, SPIR-V triple)
        │
        ▼
    WGSL (WebGPU Shading Language)
        │
        ▼
    Dawn / wgpu-py (WebGPU Runtime)
```

**Compilation pipeline**: Triton's existing frontend and MLIR passes produce LLVM IR targeting `spir64-unknown-unknown`. A custom LLVM IR → WGSL translator converts this into WebGPU compute shaders executed via the Dawn native WebGPU implementation.

## End-to-End: From Triton Kernel to Deployment

This section explains how the Triton WebGPU pipeline works in practice — the offline/runtime split, why Triton matters, and how to deploy as a standalone C++ binary.

### Offline vs Runtime

The pipeline separates into two distinct phases. The offline phase requires Python + Triton; the runtime phase needs only the generated WGSL files and a WebGPU implementation.

```
                     OFFLINE (build time)                    RUNTIME (inference)
    ┌─────────────────────────────────────────┐  ┌──────────────────────────────────┐
    │                                         │  │                                  │
    │  1. Triton Kernel (.py)                 │  │  5. Load WGSL → GPU pipelines    │
    │     @triton.jit                         │  │     Dawn/wgpu compiles to native │
    │     def my_kernel(X, W, ...)            │  │                                  │
    │              │                          │  │  6. Load weights → GPU            │
    │              ▼                          │  │                                  │
    │  2. triton.compile()                    │  │  7. Create bind groups            │
    │     Python AST → TTIR → TTGIR → LLIR   │  │     Pre-record buffer bindings   │
    │              │                          │  │                                  │
    │              ▼                          │  │  8. Dispatch loop                 │
    │  3. translate_llvm_to_wgsl()            │  │     WriteBuffer (dynamic params) │
    │     LLVM IR → WGSL compute shader       │  │     Encode dispatches            │
    │              │                          │  │     QueueSubmit                   │
    │              ▼                          │  │                                  │
    │  4. Export .wgsl files                  │  │  Runtime can be C++, Rust, JS,   │
    │     Plain text, no Python needed        │  │  or Python — any WebGPU client.  │
    │                                         │  │                                  │
    └─────────────────────────────────────────┘  └──────────────────────────────────┘
```

### Offline Steps (Python + Triton)

These steps run once per model architecture. The output is a set of `.wgsl` text files.

**Step 1: Define kernels in Python**

```python
@triton.jit
def rms_norm_kernel(X, Y, W, Rstd, stride, N, eps,
                    BLOCK_SIZE: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_SIZE)
    x = tl.load(X + row * stride + offs, mask=offs < N)
    rms = tl.sqrt(tl.sum(x * x) / N + eps)
    tl.store(Y + row * stride + offs, x / rms * tl.load(W + offs, mask=offs < N), mask=offs < N)
    tl.store(Rstd + row, 1.0 / rms)
```

A typical LLM needs ~10 unique kernels: normalization, linear projection (fp32/fp16/INT4), attention, RoPE, activation, residual add, embedding gather, argmax.

**Step 2: Compile to LLVM IR**

```python
compiled = triton.compile(
    ASTSource(fn=rms_norm_kernel, signature=sig, constexprs={'BLOCK_SIZE': 128}),
    target=GPUTarget('webgpu', 0, 32),
    options={'num_warps': 4})
llvm_ir = compiled.asm['llir']
```

Triton's MLIR pipeline handles tiling, memory coalescing, shared memory allocation, and warp-level scheduling — the same optimizations applied to CUDA targets.

**Step 3: Translate LLVM IR → WGSL**

```python
result = translate_llvm_to_wgsl(llvm_ir, signature,
    num_warps=4, warp_size=32, use_native_subgroups=True)
# result.wgsl: ~100-300 lines of WGSL per kernel
# result.buffer_bindings: [{name, binding, access, elem_type}, ...]
# result.param_fields: [{name, wgsl_type}, ...]
```

**Step 4: Export**

Save each kernel's WGSL to a file. The output is portable — any WebGPU runtime can consume it.

### Runtime Steps (Any Language)

No Triton or Python required. Just `.wgsl` files + weights + a WebGPU implementation.

**Load shaders → create pipelines**

```cpp
// C++ with Dawn
auto wgsl = readFile("rms_norm.wgsl");
auto shader = wgpuDeviceCreateShaderModule(device, &desc);
auto pipeline = wgpuDeviceCreateComputePipeline(device, &pipelineDesc);
```

Dawn compiles WGSL to the native GPU format: DXIL (D3D12), SPIR-V (Vulkan), or MSL (Metal).

**Pre-record bind groups**

```cpp
// Create once, reuse every inference step
for (int layer = 0; layer < numLayers; layer++) {
    normBG[layer] = makeBG(normPipeline, {xBuf, normBuf, normWeight[layer], paramsBuf});
    matmulBG[layer] = makeBG(matmulPipeline, {input, weights[layer], output, paramsBuf});
}
```

**Dispatch**

```cpp
// Encode all ops into a single command buffer
auto enc = wgpuDeviceCreateCommandEncoder(device, &desc);
for (int layer = 0; layer < numLayers; layer++) {
    dispatch(enc, normPipeline, normBG[layer], gridSize);
    dispatch(enc, matmulPipeline, matmulBG[layer], 1, outputDim);
    // ... more ops per layer
}
wgpuQueueSubmit(queue, 1, &cmdBuf);  // single GPU submit
```

### Why Triton?

| Aspect | Hand-written WGSL | With Triton |
|--------|-------------------|-------------|
| **Kernel authoring** | 100-300 lines WGSL per kernel | 10-30 lines Python |
| **Tiling & scheduling** | Manual workgroup size tuning | Automatic via MLIR passes |
| **Shared memory** | Manual `var<workgroup>` | Automatic from `tl.load`/`tl.store` |
| **Subgroup ops** | Manual `subgroupShuffleXor` + fallback | Automatic with feature detection |
| **Portability** | One WGSL works everywhere, but hard to write | Same Python → any WebGPU backend |
| **Iteration speed** | Edit WGSL → reload → test | Edit Python → `triton.compile()` → test |

The key insight: **Triton is an offline compiler**. It generates WGSL text files that can be consumed by any WebGPU runtime — Dawn (C++), wgpu (Rust), or browser WebGPU (JavaScript) — without Triton at runtime.

### Deploying as a Standalone C++ Binary

The offline/runtime separation enables deployment as a single executable with no Python dependencies:

```
Standalone binary layout:
  inference.exe       (~200 KB, model-specific dispatch logic)
  webgpu_dawn.dll     (~30 MB, Dawn WebGPU runtime)
  dxil.dll            (~1 MB, DXC shader compiler for D3D12)
  shaders/            (~50 KB, exported WGSL files)
  weights.npz         (model weights, INT4 quantized)
  vocab.bin + merges.bin  (tokenizer data)
```

The C++ runtime is straightforward (~700 lines for a full LLM):
- **NPZ reader**: Parse NumPy's uncompressed ZIP format (with ZIP64 for files > 4GB), ~150 lines
- **BPE tokenizer**: Encode text → token IDs, decode IDs → text, ~200 lines
- **GPU wrapper**: Dawn device setup, buffer management, dispatch helpers, ~150 lines
- **Model logic**: Pre-record bind groups, encode dispatch loop, ~200 lines

### Performance Example (Phi-4 mini, 3.8B, INT4, RTX 5080)

| Metric | Python | C++ |
|--------|--------|-----|
| Decode | 102 tok/s | 101 tok/s (no readback) / 89 tok/s (streaming) |
| Init | 3.4s | 1.8s |
| GPU memory | 2.8 GB | 2.8 GB |
| Dependencies | Python + Triton + NumPy | Dawn DLL only |

The GPU kernels are identical (same WGSL), so decode throughput is the same. The C++ streaming overhead (~12%) comes from per-token `wgpuBufferMapAsync` readback; Python avoids this by keeping the argmax → embedding feedback loop entirely on GPU.

## File Structure

| Path | Description |
|------|-------------|
| `third_party/webgpu/backend/compiler.py` | Compiler pipeline: stages from TTIR → WGSL |
| `third_party/webgpu/backend/llvm_to_wgsl.py` | LLVM IR → WGSL translator (~1800 lines) |
| `third_party/webgpu/backend/dawn_runner.py` | Dawn WebGPU runtime via ctypes (~1350 lines) |
| `third_party/webgpu/backend/wgpu_runner.py` | Alternative runner using wgpu-py |
| `third_party/webgpu/backend/driver.py` | Triton driver interface for WebGPU |
| `third_party/webgpu/backend/driver.c` | Minimal C stub for the launcher |
| `third_party/webgpu/backend/__init__.py` | Backend registration |
| `third_party/webgpu/lib/TritonWebGPUToLLVM/` | MLIR conversion pass (C++) |
| `third_party/webgpu/triton_webgpu.cc` | Backend plugin entry point |
| `third_party/webgpu/CMakeLists.txt` | CMake build for the C++ pass |
| `python/test/unit/language/test_webgpu_core.py` | 165-test comprehensive test suite |
| `python/test/unit/language/test_webgpu_run.py` | Runtime integration tests |
| `python/test/unit/language/test_webgpu_tutorials.py` | Tutorial-based tests |
| `python/examples/webgpu/gpt2/model.py` | GPT-2 inference on WebGPU via Triton |

## Major Achievements

### 1. WebGPU Backend Registration & Compilation Pipeline

- Registered `webgpu` as a Triton backend with `GPUTarget('webgpu', 0, 32)`
- Implemented the full compilation pipeline through all Triton stages (TTIR → TTGIR → LLIR → WGSL)
- Modified `CMakeLists.txt` and `setup.py` to integrate the WebGPU target into the build system
- Created the TritonGPU → LLVM MLIR conversion pass (`TritonGPUToLLVM.cpp`, `TargetInfo.cpp`) that lowers Triton GPU dialect operations to LLVM IR with SPIR-V calling conventions

### 2. LLVM IR → WGSL Translator

The translator (`llvm_to_wgsl.py`) converts LLVM IR from Triton's compilation pipeline into valid WGSL compute shaders. It handles:

- **Type mapping**: LLVM types (`float`, `i32`, `i64`, `half`, `i1`) → WGSL types (`f32`, `i32`, `bool`), with promotion for unsupported widths (i8/i16 → i32)
- **Buffer bindings**: `ptr addrspace(1)` arguments → `var<storage>` buffer bindings with automatic read/read_write access detection
- **Scalar parameters**: Non-pointer arguments → uniform struct fields
- **Arithmetic operations**: All integer and float binary ops, bitwise ops, shifts, comparisons (signed and unsigned)
- **Cast instructions**: `fptosi`, `sitofp`, `fptoui`, `uitofp`, `zext`, `sext`, `trunc`, `fpext`, `fptrunc`, `bitcast`
- **LLVM intrinsics**: `llvm.fabs`, `llvm.exp`, `llvm.log`, `llvm.sqrt`, `llvm.sin`, `llvm.cos`, `llvm.maxnum`, `llvm.minnum`, `llvm.smin`, `llvm.smax`, `llvm.fma`, and more → WGSL builtins
- **SPIR-V builtins**: `__spirv_BuiltInWorkgroupId`, `__spirv_BuiltInLocalInvocationId`, `__spirv_BuiltInNumWorkgroups` → WGSL `@builtin` inputs
- **GEP tracking**: `getelementptr` provenance tracking to map pointer SSA values back to buffer bindings with computed offsets
- **Atomic operations**: `atomicrmw` (add, sub, max, min, and, or, xor, xchg) → WGSL atomic builtins (`atomicAdd`, `atomicSub`, etc.); float atomics emulated via `atomicCompareExchangeWeak` CAS loop; atomic buffers auto-detected and declared as `array<atomic<i32>>`
- **Vector element ops**: `insertelement`/`extractelement` on `<1 x T>` pass-through (Triton wrapping pattern)

### 3. Reduction Support (Cross-Warp via Shared Memory)

Triton generates butterfly reductions using `__spirv_SubgroupShuffleXor` with intra-warp stages (masks 16, 8, 4, 2, 1) and cross-warp stages via shared memory (`addrspace(3)` with `__spirv_ControlBarrier`).

Since the Dawn D3D12 adapter does not support the `Subgroups` WebGPU feature, the backend emulates subgroup shuffles entirely through workgroup shared memory:

- **Shuffle emulation**: `__spirv_SubgroupShuffleXor(scope, val, mask)` → `_shfl[_lid.x] = val; workgroupBarrier(); result = _shfl[_lid.x ^ mask]; workgroupBarrier();`
- **Shared memory**: `@global_smem` (`addrspace(3)`) GEP/load/store → `var<workgroup> _smem: array<i32, N>` with byte-offset-to-index conversion and `bitcast<f32>`/`bitcast<i32>` for float data
- **Barriers**: `__spirv_ControlBarrier` → `workgroupBarrier()`
- **Uniform store detection**: Scalar stores to output buffers (e.g., reduction results, means, rstd) are guarded with `if _lid.x == 0u { ... }`. The translator tracks which values are "uniform" — derived from `tl.sum`, `tl.max`, `tl.min`, or scalar loads from uniform sources. This prevents a D3D12 bug where concurrent UAV writes from multiple threads within the same workgroup after a barrier can silently lose data.
- **Unused internal buffer pruning**: The translator scans the LLVM IR body to detect which `ptr addrspace(1)` arguments are actually referenced. Unreferenced internal buffers (e.g., memory allocation slots unused by certain kernels) are excluded from binding declarations, preventing binding index misalignment.
- Supports all reduction ops: `sum`, `max`, `min` across `float32` and `int32`, for block sizes 32, 64, 128, 256

### 4. Matrix Multiplication (tl.dot / GEMM)

Full `tl.dot` support for GEMM-style matrix multiplication on WebGPU:

- **Tiled GEMM**: `C[i,j] += A[i,:] * B[:,j]` via explicit loops over tiles, using 2D grid dispatch across `(M/BLOCK_M, N/BLOCK_N)` workgroups
- **Per-element dot product**: `Y[row, col] = dot(X[row, :], W[col, :])` via `tl.sum(x * w)` with 2D grid `(M, N)` — used for linear projections
- **8 GEMM tests**: covering sizes 16×16 to 32×64×16 with various block sizes

### 6. Atomic Operations

WGSL provides atomic built-in functions for `atomic<i32>` and `atomic<u32>` types (see [WGSL Atomic Builtins](https://www.w3.org/TR/WGSL/#atomic-builtin-functions)). The backend translates LLVM `atomicrmw` instructions:

- **Integer atomics**: `atomicrmw add/sub/max/min/and/or/xor/xchg` → direct WGSL `atomicAdd`/`atomicSub`/`atomicMax`/`atomicMin`/`atomicAnd`/`atomicOr`/`atomicXor`/`atomicExchange`
- **Float atomic add**: `atomicrmw fadd` emulated via CAS loop using `atomicCompareExchangeWeak` with `bitcast<f32>`/`bitcast<i32>` conversion
- **Atomic buffer detection**: Prescan identifies buffers with atomic operations; declares them as `array<atomic<i32>>` instead of `array<f32>` or `array<i32>`
- **Atomic load/store**: Regular loads/stores on atomic buffers automatically use `atomicLoad`/`atomicStore`

### 5. f16 and Subgroups Support (WebGPU Extensions)

f16 and subgroups are [WebGPU extensions](https://www.w3.org/TR/webgpu/) that require feature detection via `adapter.features.has("shader-f16")` and `adapter.features.has("subgroups")`:

- **f16 (half-precision)**: `enable f16;` WGSL directive auto-emitted when half types detected; LLVM `half` type → WGSL `f16`; `fptrunc float to half` → `f16(val)`; `fpext half to float` → `f32(val)`; f16 literal support in `_operand`; shared memory f16 pack/unpack via `vec2<f16>` and `bitcast`
- **Subgroups**: Dual-path for `__spirv_SubgroupShuffleXor` — native `subgroupShuffleXor()` with `enable subgroups;` when adapter supports it, or shared-memory emulation fallback
- **Feature negotiation**: Dawn runtime checks adapter capabilities via `wgpuAdapterHasFeature`; features requested only if available

- **Masked load**: Triton's `tl.load(ptr, mask, other)` generates `select i1 %mask, T %loaded, T %other` — translated to `select(other, loaded, mask)` in WGSL
- **Masked store**: `select i1 %cond, T %val, T undef` pattern tracked; subsequent stores emit `if cond { buf[idx] = val; }` instead of unconditional writes

### 7. Control Flow

- **If-else**: Multi-basic-block LLVM IR with `br i1 %cond, label %true, label %false` + phi nodes at merge points → structured WGSL `if { } else { }`
- **Pointer phi resolution**: Phi nodes on `ptr addrspace(1)` values (selecting between buffers) resolved to buffer load expressions in each branch
- **For loops**: Back-edge detection for structured loop translation
- **While loops**: Supported through the loop translation path

### 8. Dawn WebGPU Runtime

The runtime (`dawn_runner.py`) is a pure-Python ctypes wrapper around the Dawn native WebGPU library (`webgpu_dawn.dll`), providing:

- **Device management**: Adapter request, device creation with conditional feature negotiation (`Subgroups`, `ShaderF16` requested only if adapter supports them via `wgpuAdapterHasFeature`)
- **Buffer management**: Create storage buffers, upload/download data, map buffers for readback
- **Shader compilation**: `wgpuDeviceCreateShaderModule` with WGSL source
- **Pipeline creation**: Compute pipeline with auto-generated bind group layouts
- **Dispatch**: `wgpuComputePassEncoderDispatchWorkgroups` with proper grid dimensions
- **Synchronization**: Queue submission and `wgpuDevicePoll` for completion

### 9. Comprehensive Test Suite — 165 Tests Passing

All 165 tests in `test_webgpu_core.py` pass, covering:

| Category | Tests | Details |
|----------|-------|---------|
| Binary ops (float) | 5 | add, sub, mul, div, mod |
| Binary ops (int) | 4 | add, sub, mul, div (+ floordiv, modulo) |
| Bitwise ops | 5 | and, or, xor, lshift, rshift, not |
| Comparisons | 16 | eq, ne, lt, le, gt, ge × float32/int32 + NaN handling |
| Unary ops | 3 | neg (float, int), abs |
| Math functions | 10 | exp, log, sqrt, ceil, floor, round, sin, cos, exp2, log2 |
| Where/select | 2 | float32, int32 |
| Broadcasting | 2 | scalar, row |
| Tensor creation | 4 | full, arange, zeros (float32/int32) |
| Type casting | 2 | float→int, int→float |
| Reductions | 24 | sum/max/min × float32/int32 × block 32/64/128/256 |
| Memory ops | 12 | masked load (8 combos), masked store, copy, strided, indirect, same-ptr |
| Control flow | 5 | if-else, for loop, for accumulate, nested for, while loop |
| Multi-output | 1 | two output buffers |
| Statistics | 1 | mean + variance via reduction |
| Vector add | 6 | sizes 1, 100, 256, 1000, 1024, 8192 |
| Activations | 3 | ReLU, GELU, softmax |
| Normalization | 2 | vector norm, RMSNorm |
| Loss functions | 1 | cross-entropy |
| Scalar params | 4 | scalar multiply, scalar param, clamp, maximum/minimum |
| Constexprs | 2 | single constexpr, multiple constexprs |
| Integer ops | 2 | int add, int mul |
| Program ID | 2 | program_id, num_programs |
| WGSL structure | 3 | shader validation, asm output, compilation stages |
| Bandwidth | 1 | vector add throughput |
| Atomic ops | 4 | int add, int accumulate, float add, float accumulate |
| GEMM/dot | 8 | tl.dot matmul with various sizes and block configurations |
| LayerNorm (fused) | 6 | fused layer normalization, various sizes |
| Softmax | 5 | row-wise softmax, multi-warp, various sizes |
| RoPE | 2 | rotary positional embeddings |
| Cross-entropy loss | 2 | numerically-stable cross-entropy |
| Causal attention | 2 | FlashAttention-style online softmax |
| Linear projection | 2 | Y = X @ W^T + bias |
| GELU activation | 2 | tanh-approximation GELU via exp |
| Loop-based LN | 2 | loop-based layer norm for N=256, N=768 |
| Loop-based linear | 2 | loop-based linear for K=768→2304, K=3072→768 |
| GPT-2 pipeline | 2 | end-to-end transformer: small (64d) + full-scale (768d) |

### 10. GPT-2 Inference on WebGPU

A complete GPT-2 (124M) inference pipeline (`python/examples/webgpu/gpt2/model.py`) demonstrating end-to-end transformer execution on WebGPU, producing coherent English text from real HuggingFace weights:

- **Full-scale model**: 12 layers, 12 heads, 768 embedding dimensions — all running on WebGPU
- **Loop-based kernels**: LayerNorm and linear projections use loop-based GPU kernels that iterate in BLOCK=128 chunks, staying within WebGPU's 256-thread workgroup limit while handling any dimension
- **Online softmax attention**: FlashAttention-style per-query causal attention with running max/sum (no materialization of the T×T attention matrix)
- **KV-cache**: Prefill processes full prompt once; decode steps process single token with cached K/V tensors. 2.4× speedup over naive recomputation
- **Chunked LM head**: Splits vocabulary projection into ≤128 MB chunks to stay within WebGPU `maxStorageBufferBindingSize` limit
- **Verified exact match**: Forward pass output matches NumPy reference (max_diff ≤ 0.000168 across all 12 layers)
- **Weight loading**: Downloads GPT-2 (124M) weights from HuggingFace safetensors format, with Conv1D weight transpose (K,N→N,K) and BF16-to-F32 conversion
- **Token generation**: Top-k sampling with temperature control via tiktoken tokenizer
- **Sequential loop fix**: Fixed `process_block_terminator()` in WGSL translator to properly detect loop headers when multiple loops appear in sequence
- **num_warps consistency**: Discovered and fixed a critical bug where mismatched `num_warps` between `triton.compile()` and `translate_llvm_to_wgsl()` caused elements at certain indices to be silently unwritten

```bash
# Verify pipeline with random weights (no download)
python python/examples/webgpu/gpt2/model.py --verify

# Generate text with real GPT-2 weights (auto-downloads ~500MB)
python python/examples/webgpu/gpt2/model.py --prompt "The future of AI is"
```

## Prerequisites

- **Python**: 3.13+
- **Triton**: Built from source (triton-windows fork)
- **Dawn**: Built as shared library (`webgpu_dawn.dll`) at `third_party/webgpu/dawn/build/`
  - Build with: `cmake -G Ninja -DDAWN_BUILD_MONOLITHIC_LIBRARY=SHARED ...` then `ninja webgpu_dawn`
- **GPU**: Any GPU with D3D12, Vulkan, or Metal support (no CUDA required)

## Running Tests

```bash
cd python/test/unit/language
python -m pytest test_webgpu_core.py -v
```

## Known Limitations

- Single workgroup dispatch only for reduction kernels (WebGPU has no inter-workgroup synchronization — `workgroupBarrier()` and `var<workgroup>` shared memory only operate within a single workgroup, so cross-workgroup reductions would require multiple dispatch passes)
- `i64` values are truncated to `i32` (WGSL has no 64-bit integer type)
- `num_warps` must be passed consistently to both `triton.compile(options={'num_warps': N})` and `translate_llvm_to_wgsl(num_warps=N)` — mismatches cause silent element coverage gaps when BLOCK > num_warps × warp_size
- GPT-2 inference uses CPU for trivial element-wise operations (add, GELU) to avoid GPU transfer overhead; GPU is used for compute-heavy operations (linear projection, LayerNorm, attention)
