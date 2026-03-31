# Backpack High Level Design

Three-way comparison of ORT WebGPU Native (C++), llama.cpp, and Backpack.
Organized as: scenarios first, then problems, designs, and limitations.

---

## 1. Scenarios

What operational situations arise when running models on a GPU?

| # | Scenario | Description |
|---|----------|-------------|
| 1 | **Model loading** | Parse model file, upload weights to GPU, compile shaders |
| 2 | **Session creation** | Allocate execution context, KV cache buffers |
| 3 | **Inference execution** | Prefill prompt tokens, then decode one token at a time |
| 4 | **Continue conversation** | User sends follow-up after EOS — reuse existing KV cache, prefill new tokens |
| 5 | **New conversation** | Reset state, start fresh — reuse buffers without reallocation |
| 6 | **Abort mid-generation** | User cancels during decode — stop cleanly without corrupting state |
| 7 | **Concurrent sessions** | Multiple users or sequences sharing one model on one GPU |
| 8 | **Multi-model on same device** | Sequential or parallel models (e.g., text encoder → DiT → VAE pipeline) |
| 9 | **Session eviction** | Free GPU memory by evicting idle sessions under memory pressure |
| 10 | **Device lost / GPU recovery** | GPU reset, driver crash, or browser tab backgrounded |
| 11 | **Web deployment** | Run in browser via WASM + WebGPU |

### Scenarios Not Planned

The following were considered and explicitly excluded:

| Scenario | Why not supported |
|---|---|
| **Context size change** | KV cache size fixed at session creation. Create a new session for different context size. Shaders are already shape-independent — only KV reallocation and fast decode re-capture needed. Not worth the complexity of dynamic resize + bind group invalidation. |
| **Speculative decoding** | Requires two models (draft + verifier) with coordinated acceptance/rejection. Significant complexity for a niche optimization. |
| **Model hot-swap** | Loading new model version while sessions are active. Would require ref-counting via `shared_ptr`. Current design: release all sessions before releasing model. |
| **Batch inference (batch size > 1)** | Multiple independent prompts in one `Execute()`. Requires reworking dispatch dimensions and KV indexing. Future work. |
| **Device-lost recovery** | Detection is implemented (`Device::IsLost()`). Auto-recovery (re-request adapter, re-upload weights) is not — full teardown + reload is the expected path, same as ORT and llama.cpp. |

---

## 2. Problems

What does each project set out to solve?

| | llama.cpp | ORT WebGPU Native | Backpack |
|---|---|---|---|
| **Goal** | Fast LLM inference across many GPU backends | Run any ONNX model on any backend | Single-backend (WebGPU) runtime for on-device inference |
| **Model format** | GGUF only | ONNX only | ONNX (+ GGUF via ModelRunner) |
| **GPU backend** | ggml_backend (CUDA, Vulkan, Metal, WebGPU, 17+) | EP system (CUDA, TRT, WebGPU, etc.) | WebGPU only (via Dawn) |
| **Model scope** | LLM only | Any ONNX model | Any ONNX model |
| **Language** | C API (opaque structs) | C/C++ API (`Ort::` wrappers) | C++ (classes, DLL export) |
| **LLM layer** | Built into runtime | Separate library (onnxruntime-genai) | App-layer `LlmContext` |

**llama.cpp** is purpose-built for autoregressive LLMs. Rich KV cache management,
sampling, and context handling are baked into the runtime. Supports 17+ GPU backends
but each has different maturity.

**ORT** is a general-purpose ONNX runtime supporting many backends via Execution
Providers. LLM-specific concerns (tokenization, KV management, sampling) are
handled by a separate library, **onnxruntime-genai**. The WebGPU EP is one of many.

**Backpack** exploits the simplicity of targeting a single backend (WebGPU) to
make different trade-offs: eager shader compilation, always-GPU tensors, type-safe
config. Like ORT, the runtime is model-agnostic — LLM logic lives in the app layer.

---

## 3. Designs

How does each project solve these problems?

### Object Model

| Concept | llama.cpp | ORT Native | Backpack |
|---------|-----------|------------|----------|
| **Runtime env** | `llama_backend_init()` | `Ort::Env(level, "app")` | `bp::Device::Create(backend)` |
| **Model (weights)** | `llama_model` (opaque) | None — merged into Session | `bp::Model` |
| **Execution context** | `llama_context` (KV cache, compute bufs) | `Ort::Session` (model+context merged) | `bp::Session` (model-agnostic) |
| **Tensor** | `ggml_tensor` (internal) | `Ort::Value` | `bp::Tensor` |
| **KV cache** | `llama_memory_t` (inside context) | Explicit I/O tensors via `IoBinding` | App-layer (explicit I/O tensors) |
| **GPU binding** | Backend scheduler | `Ort::IoBinding` (pre-bind GPU buffers) | `session.SetInput/SetOutput` (always GPU) |

**llama.cpp** separates model from context — load weights once, create multiple
contexts with different `n_ctx`. But the entire API is LLM-specific.

**ORT** merges model+context into `Ort::Session`. A single Session can `Run()`
with different KV cache sizes since KV caches are just regular I/O tensors. For
concurrent independent execution (e.g., two threads), you need two Sessions, which
re-parses and re-optimizes the graph. `PrepackedWeightsContainer` shares weights
but not optimized graph state.

**Backpack** separates model from session — model holds weights + compiled shaders,
session is a lightweight execution context. LLM-specific state (KV cache, position
tracking) lives in the app layer, similar to how onnxruntime-genai wraps ORT.

### Model Loading

| Aspect | llama.cpp | ORT Native | Backpack |
|--------|-----------|------------|----------|
| **API** | `llama_model_load_from_file(path, params)` | `Ort::Session(env, path, opts)` | `bp::Model::Load(device, path, opts)` |
| **What happens** | Parse GGUF, upload weights | Parse ONNX, optimize graph, upload weights, register kernels | Parse ONNX, upload weights, **compile all shaders** |
| **Shader compilation** | N/A (pre-compiled backend kernels) | **Deferred** — compiled on first `Run()` | **Eager** — compiled async+parallel at load |
| **From memory buffer?** | Yes | Yes | Not yet (path only; web will use buffer) |
| **Weight sharing** | Multiple contexts share one `llama_model` | `PrepackedWeightsContainer` across sessions | Multiple sessions share one `bp::Model` |

#### Why Backpack Uses Eager Shader Compilation

ORT's WebGPU EP defers shader compilation to first `Run()`. However, all shader
sources can be fully determined from the parsed graph — op types, dtypes, model
architecture constants (head_dim, etc.). No runtime input shapes are needed:

- Dynamic dimensions (M, N, K, sequence length, batch size) are passed via
  `_params_` uniform buffers at dispatch time, not baked into shader source
- `patchShaderHD()` uses `headDim` from model metadata (known at load time)
- Tile sizes (`TILE_M`, `TILE_N`, `CHUNK`) are fixed algorithm constants
- Autotuning selects between pre-existing shader variants, not new shaders

ORT's laziness is likely a multi-EP architectural simplification (don't compile
until you know which EP handles a kernel), not a technical necessity. For a
single-backend system like Backpack, eager async+parallel compilation at
`Model::Load()` eliminates first-run latency spikes with no downside.

### Session Creation

| Aspect | llama.cpp | ORT Native | Backpack |
|--------|-----------|------------|----------|
| **API** | `llama_init_from_model(model, ctx_params)` | (merged into Session constructor) | `bp::Session::Create(model)` |
| **Context size** | Set via `ctx_params.n_ctx` | Implicit in tensor shapes — same Session handles different sizes | Implicit in tensor shapes (app-layer sets maxSeqLen) |
| **KV cache alloc** | Allocated inside context | Caller provides via `IoBinding` or `Run()` args | App-layer allocates based on maxSeqLen |
| **Multiple per model?** | Yes — each with different `n_ctx`, shared weights | One Session suffices for different KV sizes; multiple needed only for concurrent execution | Yes — shared weights + shaders, independent sessions |
| **KV cache type** | Configurable (`type_k`/`type_v`: f16, q8_0, q4_0) | Whatever dtype the model outputs | App-layer choice (currently f32 ONNX, f16 GGUF) |

### Inference Execution

| Aspect | llama.cpp | ORT Native | Backpack |
|--------|-----------|------------|----------|
| **API** | `llama_decode(ctx, batch)` | `session.Run(opts, names, vals, ...)` or `session.Run(opts, binding)` | `session.SetInput(...); session.Run()` |
| **Batch input** | `llama_batch` (tokens, positions, seq_ids, logits flags) | Named arrays or `IoBinding` | Named `SetInput`/`SetOutput` calls |
| **GPU I/O** | Automatic (backend scheduler) | CPU default; `Ort::IoBinding` needed for GPU stay | Always GPU (`bp::Tensor` = GPU buffer) |
| **Output location** | CPU (`llama_get_logits`) | CPU default, or GPU via `IoBinding` | GPU (`bp::Tensor` holds `GPUBuffer`) |
| **Session reuse** | Context reused across decode calls | Session reused; `IoBinding` cleared between runs | `Session::Reset()` clears bindings between runs |

### KV Cache Management

| Aspect | llama.cpp | ORT Native | Backpack |
|--------|-----------|------------|----------|
| **Who manages KV?** | Runtime (opaque `llama_memory_t`) | Caller (explicit I/O tensors) | App-layer (explicit I/O tensors via `SetInput/SetOutput`) |
| **Resize at runtime?** | No — fixed at context creation | Yes — pass different-sized KV tensors each `Run()` | Yes — pass different-sized tensors (app-layer manages) |
| **Cache ops** | Rich: `seq_rm`, `seq_cp`, `seq_add`, `seq_div`, `seq_keep` | None — caller manages | None in runtime — app-layer implements as needed |
| **Shader recompilation on resize?** | N/A | May recompile (depends on `inputDependencies`) | **No** — all shapes are runtime params |

#### Context Size Independence from Compiled Shaders

In Backpack, no shaders need recompilation when context size changes:

- All attention ops pass `total_seq`, `past_seq`, `max_seq` via params buffer
- All matmul ops pass `M, N, K` as runtime params
- Tile sizes are fixed constants independent of sequence length
- `HD` (head dimension) is patched at load time from model metadata

The caller controls context size by providing differently-sized KV cache tensors.
Same compiled model, same session, different buffer sizes — zero recompilation.

### Web Compatibility

| Aspect | llama.cpp | ORT Native | Backpack |
|--------|-----------|------------|----------|
| **Runs in browser?** | No | Yes (WASM as `onnxruntime-web`) | Not yet, but API designed for it |
| **WebGPU backend** | `ggml-webgpu` (experimental) | Mature WebGPU EP | Dawn native; Emscripten future |
| **Graph capture** | N/A | `enableGraphCapture` — record/replay dispatch list | Fast decode capture/replay |
| **Buffer caching** | N/A | Configurable: `disabled`, `lazyRelease`, `simple`, `bucket` | Power-of-2 bucket pool |

#### Web Compatibility Rules

The API follows rules to ensure future WASM+WebGPU compatibility:

| C++ (native) | Web (WASM + WebGPU) | Notes |
|---|---|---|
| `bp::Device::Create()` | `await bp.Device.create()` | Async in web (adapter request) |
| `bp::Model::Load(dev, path)` | `await bp.Model.load(dev, buffer)` | Web takes ArrayBuffer, not file path |
| `session.Run()` | `await session.run()` | Async in web (GPU fence) |

- **No file I/O in public API** — `Model::Load` takes path on native; web takes buffer
- **No blocking GPU waits** — `wgpuInstanceProcessEvents` maps to WebGPU's event model
- **No thread-dependent state** — web is single-threaded; current API is already single-threaded
- **No platform-specific types in public header** — uses only `<cstdint>`, `<string>`, `<vector>`

### The Two-Layer Pattern

Backpack follows the same layering as ORT's ecosystem:

| ORT ecosystem | Backpack | Role |
|---|---|---|
| `Ort::Session` | `bp::Session` | Model-agnostic execution |
| `OgaModel` + `OgaGenerator` | App-layer `LlmContext` | LLM-specific (KV, positions, sampling) |
| `Ort::Value` | `bp::Tensor` | Tensor |
| `Ort::Env` + EP config | `bp::Device` | Runtime environment |

**Layer 1 (bp:: runtime)** — model-agnostic, knows nothing about LLMs:

```
bp::Device → bp::Model → bp::Session → SetInput/SetOutput → Run
```

Works for LLM, TTS, image, audio — any ONNX model. Same API the image app uses today.

**Layer 2 (app layer)** — LLM-specific, manages KV cache and positions:

```
LlmContext { Model, Session, KV tensors, position counter }
    → Decode(token) → session.Reset/SetInput/SetOutput/Run
```

Multiple LlmContexts can share one Model (different context sizes, independent
KV caches, zero shader recompilation).

### Summary of Design Choices

| Decision | llama.cpp | ORT Native | **Backpack** | Rationale |
|----------|-----------|------------|--------------|-----------|
| Model/session split | Separate | Merged | **Separate** | Lightweight concurrent sessions without re-parsing. Currently limited: executor is per-model (needs refactor to per-session execution state). |
| Shader compilation | Pre-compiled | Lazy on first `Run()` | **Eager, async+parallel** | Predictable TTFT, no first-run spike |
| KV cache ownership | Runtime-managed (opaque) | Caller-managed (I/O tensors) | **App-layer managed** | Runtime is model-agnostic; avoids LLM lock-in |
| Context size | Fixed at creation | Implicit in tensor shapes | **Implicit in tensor shapes** | Runtime doesn't care; app-layer sets maxSeqLen |
| GPU I/O | Backend scheduler | Requires `IoBinding` | **Always GPU** | No extra API surface for GPU stay |
| Model scope | LLM-only | Any ONNX model | **Any ONNX model** | Same Session for LLM, TTS, image, audio |
| Batch API | `llama_batch` struct | Named arrays / `IoBinding` | **Named `SetInput/SetOutput`** | Simple, web-compatible, ONNX-native |

---

## 4. Limitations

What does each project get wrong or can't do?

### ORT: Intentional Trade-offs (Not Drawbacks)

ORT is designed to run **any ONNX model** across many backends. Most of its design
choices are intentional trade-offs for that generality. LLM-specific concerns are
handled by the separate **onnxruntime-genai** library.

| Design choice | Why it makes sense | Mitigation |
|---|---|---|
| **Merged model+context** (`Ort::Session`) | A model-agnostic runtime doesn't need a separate "context" — that's an LLM concept. One Session runs any graph. | GenAI's `OgaGenerator` provides the LLM context layer |
| **KV cache is caller-managed** (just I/O tensors) | `past_key_values` are regular ONNX inputs/outputs. Runtime shouldn't have special KV logic. | GenAI hides KV management inside `OgaGenerator` |
| **`IoBinding` for GPU stay** | Supports CPU, CUDA, TensorRT, WebGPU, OpenVINO from one API. Each EP has different memory semantics. | Necessary cost of multi-backend support |
| **No model introspection before Session** | Graph optimization needed to know final shapes. | GenAI uses `genai_config.json` for model metadata |

### ORT: Genuine Drawbacks

| Problem | Impact |
|---------|--------|
| **Lazy shader compilation** | All shaders can be determined from the parsed graph (op types, dtypes, model constants) — no runtime input shapes needed. Laziness is a multi-EP simplification, not a technical necessity. Causes unpredictable first-run latency for all model types. |
| **Per-Session graph optimization** | For concurrent execution, each Session re-parses and re-optimizes. `PrepackedWeightsContainer` shares weights but not optimized graph or compiled shaders. |
| **Stringly-typed EP config** | `{string: string}` key-value pairs. No compile-time validation, typos silently ignored. |

### llama.cpp: Drawbacks

| Problem | Impact |
|---------|--------|
| **LLM-only architecture** | Entire API designed for autoregressive LLMs. Cannot use for TTS, image gen, audio, or encoder-only models. |
| **GGUF-only format** | Can't run ONNX. Conversion from HuggingFace sometimes loses information. |
| **Opaque KV cache** | Can't share KV across contexts. Can't serialize/deserialize for pause/resume. Cache ops are LLM-specific and baked into the runtime. |
| **Fixed context size** | `n_ctx` immutable after context creation. Must destroy and recreate to grow. |
| **Backend fragmentation** (17+) | Each has different maturity, quantization support, performance. WebGPU backend is experimental. |
| **C API, manual lifetime** | Requires explicit `_free()`. No RAII. Easy to leak in error paths. |
| **Sampling tightly coupled** | `llama_sampler` chain baked into library. Arguably app-layer responsibility. |

### Scenario Comparison

How each runtime handles the operational scenarios from Section 1.

#### Concurrent Sessions

| | ORT WebGPU Native | llama.cpp | Backpack |
|---|---|---|---|
| **API support** | Multiple `Ort::Session` on same device via `context_id` ref-counting | Multiple `llama_context` from one `llama_model` (weights shared read-only) | `bp::Session` holds raw pointer to `Model::Impl` |
| **Concurrent GPU execution** | **No** — `ConcurrentRunSupported()` returns `false` for WebGPU EP. Runs serialized via `session_mutex_`. | **Not validated** — API allows it, but server uses single-context multi-slot pattern. No mutexes in core lib. GPU backends have global state. | **No** — single `GraphExecutor` per model. Shared mutable state (tensorStore, fastDecode, tensorPlan) would corrupt. |
| **Limitation** | WebGPU EP relies on global state; serialization is intentional | Thread safety of concurrent `llama_decode` across contexts is undocumented | Requires GraphExecutor split into GraphDef + ExecutionContext (Part 0 in plan) |

#### Continue Conversation (EOS → New Prompt)

| | ORT WebGPU Native | llama.cpp | Backpack |
|---|---|---|---|
| **Mechanism** | Caller passes previous `present.*` output KV tensors as next `past_key_values.*` inputs | Rich KV management: `llama_memory_seq_rm/cp/keep/add` for prefix reuse, position shifting, sequence forking | App-layer: KV cache kept, pos counter maintained, fast decode invalidated |
| **Prefix reuse** | Caller must implement (compare old vs new prompt, reuse matching KV prefix) | Built-in: server computes longest common prefix, removes only divergent tokens | App-layer responsibility — no prefix-aware KV management in runtime |
| **Context overflow** | Caller must handle (truncate or sliding window) | Built-in context shifting: `seq_rm` + `seq_add` to discard old tokens and shift positions | App-layer responsibility |
| **Limitation** | No helper API — caller handles all KV lifecycle logic | KV ops are LLM-specific and baked into runtime (can't use for non-LLM models) | No limitation per se — app-layer management is intentional (same as ORT) |

#### New Conversation (Reset)

| | ORT WebGPU Native | llama.cpp | Backpack |
|---|---|---|---|
| **Mechanism** | Caller passes zeroed/fresh KV tensors. No session reset API needed. | `llama_memory_clear()` or `llama_memory_seq_rm(mem, id, -1, -1)` per sequence | App-layer: reset pos=0, pass zeroed KV. `Session::Reset()` clears input/output bindings for reuse. |
| **KV buffer reuse** | Caller controls — can reuse same buffer with zeroed content | KV buffer is pre-allocated to `n_ctx`, always reused | KV buffer reused (same size), no reallocation |
| **Limitation** | None — stateless design makes reset trivial | None | None — `Session::Reset()` implemented |

#### Abort Mid-Generation

| | ORT WebGPU Native | llama.cpp | Backpack |
|---|---|---|---|
| **API** | `RunOptions::SetTerminate()` — first-class, thread-safe | `ggml_abort_callback` — polling callback checked between graph nodes | `Session::RequestStop()` — sets atomic flag, checked at top of `Run()` |
| **GPU support** | Stops CPU-side dispatch; already-queued GPU work completes | **CPU only** — GPU backends (CUDA, Vulkan, Metal) run to completion. Comment: "currently works only with CPU execution" | Cooperative — already-queued GPU work completes, next `Run()` is skipped |
| **Granularity** | Between kernel submissions | Between graph nodes (CPU only) | Between `Run()` calls (per decode step) |
| **Return value** | Error status from `Run()` | `llama_decode` returns `2` (aborted). Processed ubatches remain in KV cache. | `Run()` returns early without modifying state. `ClearStop()` to resume. |
| **Limitation** | GPU work already submitted still completes | No GPU cancellation. App must check between `llama_decode` calls. | No mid-dispatch abort (same as ORT/llama.cpp GPU). |

#### Device Lost / GPU Recovery

| | ORT WebGPU Native | llama.cpp | Backpack |
|---|---|---|---|
| **Detection** | Device-lost callback registered but **only logs**. Code has `// TODO: revise temporary device lost handling`. | **No handling** — Vulkan: `VK_CHECK` calls `exit(1)` on any error including device lost. CUDA: `GGML_ABORT()`. | `deviceLostCallbackInfo` registered in `GPUContext::init()`. Sets `deviceLost` flag + stores reason string. `Device::IsLost()` exposes to app layer. |
| **Recovery** | None — must destroy `Ort::Session` and `Ort::Env`, recreate everything | None — process terminates on GPU error | None — full teardown + reload required (detected but not auto-recovered) |
| **Limitation** | Device-lost is detected but not recoverable. Marked as TODO. | Fatal — any GPU error kills the process | Detection implemented. Auto-recovery not planned — same as ORT. |

#### Session Eviction / Memory Pressure

| | ORT WebGPU Native | llama.cpp | Backpack |
|---|---|---|---|
| **Memory tracking** | `AllocatorGetStats()` exists for CPU arena but **not for WebGPU `BufferManager`**. No GPU memory counters. | No API for memory queries. Server tracks slot states at app level. | `GPUContext` tracks `totalAllocatedBytes`, `peakAllocatedBytes`, `totalAllocCount`. `getMemoryStats()` API. Baseline JSON includes memory section. |
| **Eviction** | No eviction API. OOM propagates as exception. | Server-level LRU eviction: `llama_memory_seq_rm` frees idle slots. `llama_memory_defrag()` compacts fragmented KV. | Tiered eviction designed: warm caches → fast decode → KV cache → session. Requires per-session ExecutionContext (Part 0). |
| **KV defragmentation** | N/A (caller-managed KV) | `llama_memory_defrag()` compacts KV cache in-place | N/A (caller-managed KV; no internal KV fragmentation) |
| **Limitation** | No GPU memory visibility. Cannot make informed eviction decisions. | Pre-allocated KV to `n_ctx` — no dynamic growth/shrink. Defrag only within fixed allocation. | Eviction strategy designed but requires per-session ExecutionContext for full implementation. |

#### Multi-Model on Same Device

| | ORT WebGPU Native | llama.cpp | Backpack |
|---|---|---|---|
| **Support** | Multiple `Ort::Session` share same device via `context_id`. Shared `BufferManager`. | Per-model GPU state — no global barriers. Multi-GPU with `tensor_split`. | Multiple models share `GPUContext` (pool, pipeline cache). |
| **VRAM coordination** | No budget or priority system between sessions | No coordination — models compete for VRAM through allocator | Buffer pool shared — models can reuse freed buffers. `flushBufferPool()` between stages. |
| **Limitation** | No memory budget per session/model | No API to query available VRAM before loading second model | `buffers_` name collision fixed (only device-level entries registered in shared map). Pool and pipelines shared safely. |

### Gaps Summary

**ORT WebGPU Native gaps**:
- No concurrent GPU execution (serialized by design)
- Device-lost handling is a placeholder (TODO in code)
- No GPU memory tracking or eviction API
- Graph capture incompatible with variable KV sizes

**llama.cpp gaps**:
- Context size (`n_ctx`) fixed after creation — cannot resize
- Abort callback CPU-only — no GPU cancellation
- GPU errors are fatal (`exit(1)` / `GGML_ABORT`)
- No memory queries or budget API
- Concurrent contexts undocumented for thread safety

**Backpack gaps** (remaining):
- No concurrent sessions — requires GraphExecutor split (Plan Part 0)

**Backpack gaps resolved** (implemented in `55e6340`):
- ~~No abort API~~ → `Session::RequestStop()` / `ClearStop()` with atomic flag
- ~~No device-lost callback~~ → `deviceLostCallbackInfo` registered + `Device::IsLost()`
- ~~No memory tracking~~ → `GPUContext::getMemoryStats()` (peak/current/count)
- ~~No `Session::Reset()`~~ → Implemented, clears input/output bindings
- ~~`Tensor::Release()` buffer leak~~ → `Tensor::Impl` destructor returns buffer to pool
- ~~`buffers_` name collision~~ → Only `_`-prefixed device-level entries registered in shared map

---

## 5. Backpack Internals

Backpack-specific implementation details — not a comparison, but a deep dive into
buffer management, resource sharing, and concurrency constraints.

### Buffer Lifecycle

#### Buffer Categories by Lifetime

| Category | Lifetime | Size | Release trigger |
|----------|----------|------|-----------------|
| **Weights** | Model lifetime | Large (entire model) | `model.Release()` |
| **KV cache** | Session / conversation | Large (scales with context) | `session.Release()` |
| **Fast decode bind groups** | Between prefills | Medium (~400 intermediate buffers held) | `ReleaseCaptured()` on shape change |
| **Tensor plan intermediates** | Between shape changes | Medium (~350-380 buffers) | `InvalidateWarmCaches()` or memory pressure |
| **Param pool** | Executor lifetime | Small (~80-320 KB) | Never (fixed, round-robin) |
| **Transient intermediates** | Single `Execute()` call | Variable | Released to pool at end of `Execute()` |
| **Caller tensors** (input/output) | Caller-managed | Variable | `tensor.Release()` |

#### Release Granularity

From finest to coarsest:

| Level | API | What happens |
|-------|-----|-------------|
| **Single buffer** | `gpu->releaseBuffer(buf)` | Returns to power-of-2 pool (27 buckets, 16B-1GB, max 256/bucket). Overflow destroys. |
| **Single tensor** | `tensor.Release()` | Drops C++ Impl, returns GPU buffer to pool via destructor. |
| **Session** | `session.Release()` | Drops execution context. `Reset()` clears bindings for reuse. |
| **Model** | `model.Release()` | Frees all weight buffers + flushes pool. Full cleanup. |
| **Pool flush** | `gpu->flushBufferPool()` | Destroys all pooled buffers. Use between pipeline stages. |
| **Device** | `device.Release()` | Teardown everything. |

#### Usage Patterns

**LLM** (single model, long-lived):
```
Load model → Create session → Create KV tensors → Run in loop → Process exit
```

**Image** (sequential pipeline, memory-constrained):
```
Load text encoder → Run → model.Release() → flushPool()
→ Load DiT → Run diffusion → model.Release() → flushPool()
→ Load VAE → Run → Done
```

**Web** (WASM, GPU memory constrained):
Same patterns via JS. Pool-based recycling is especially important in browsers.

### Buffer Release Strategy

| Event | Action | Memory freed |
|-------|--------|-------------|
| Decode step (replay) | Nothing | 0 |
| Abort mid-generation | `Run()` returns early; state untouched | 0 |
| New user prompt (keep KV) | `ReleaseCaptured()` + `InvalidateWarmCaches()` | Captured + tensor plan → pool |
| New conversation | Above + reset pos=0 | Same (KV reused, not released) |
| Session eviction | `session.Release()` | All: KV, captured, plan, params → pool |
| Model unload | `model.Release()` | Weights → pool → `flushPool()` |
| Device lost | Full teardown — all GPU resources invalid | Everything (forced) |
| Between models (image) | `model.Release()` + `flushPool()` | Everything destroyed |

#### Session Eviction Priorities

For a serving scenario with many sessions, free memory by evicting idle sessions:

| Priority | What to release | Memory freed | Cost to resume |
|----------|----------------|-------------|----------------|
| 1st | Tensor plan (`InvalidateWarmCaches()`) | ~350 intermediate buffers | One cold Execute() to rebuild plan |
| 2nd | Fast decode (`ReleaseCaptured()`) | ~400 captured buffers | One capture Execute() to re-record |
| 3rd | KV cache (`session.Release()`) | All KV buffers | Conversation lost, must re-prefill |

### Shared Resources Across Models

All models on the same `bp::Device` share a single `GPUContext`:

| Resource | Scope | Shared? | Safe? |
|----------|-------|---------|-------|
| **Buffer pool** (`pool_[]`) | Per-device | Yes — freed buffers reusable across models | Yes |
| **Pipeline cache** (`pipelines_[]`) | Per-device | Yes — Model B reuses Model A's compiled shaders | Yes — pipelines are immutable |
| **Named buffer map** (`buffers_[]`) | Per-device | Only device-level resources (prefixed with `_`) | Yes — per-model buffers tracked in `tensorStore_` instead |
| **Weight buffers** | Per-model (`tensorStore_`) | No | Yes |
| **Intermediate buffers** | Per-model (`tensorStore_`) | No — cleaned up between `Execute()` | Yes |
| **Fast decode state** | Per-model (`GraphExecutor`) | No — but per-model, not per-session | See multi-session section |

#### Pipeline Sharing

If Model A already compiled `"matmul_f32"`, Model B gets it from cache instantly.
Compiled pipelines (`WGPUComputePipeline`, `WGPUShaderModule`, `WGPUBindGroupLayout`)
are immutable GPU objects — sharing is correct and beneficial.

#### Sequential Multi-Model Pattern (Image App)

The image app demonstrates correct sequential buffer management:

```
Text Encoder: Load → Run → Release() → waitForQueue() → flushBufferPool()
DiT:          Load → Run diffusion loop (kept alive)
VAE:          Load → Run → _exit(0)
```

The `flushBufferPool()` between text encoder and DiT is critical — without it,
the pool holds freed buffers, and the much larger DiT might fail to allocate.

### Multi-Session and Fast Decode

#### Current Limitation: Sessions Cannot Run Concurrently

The current implementation has a critical limitation: `Model::Impl` owns a single
`GraphExecutor` by value, and `Session::Run()` calls `executor.Execute()` directly
on it. Two sessions from the same model would corrupt shared mutable state.

```
Model::Impl
  └── GraphExecutor executor        (single instance)
        ├── tensorStore_             (intermediate buffers)
        ├── tensorPlan_              (warm execution cache)
        ├── shapeCacheValid_         (shape cache)
        ├── fastDecodeState_         (Off / Capturing / Replaying)
        ├── capturedFlushes_         (captured bind groups + dispatches)
        └── pendingDispatches_       (current dispatch queue)

Session A ──→ executor.Execute()  ──┐
                                    ├── CONFLICT: same mutable state
Session B ──→ executor.Execute()  ──┘
```

Three categories of conflict:

| State | What conflicts |
|-------|----------------|
| **Fast decode** | `fastDecodeState_`, `capturedFlushes_`, `capturedWrites_` are singleton — only one session can capture/replay |
| **Warm caches** | `tensorPlan_`, `shapeCacheValid_` — different input shapes between sessions corrupt cached buffers |
| **Execution** | `tensorStore_`, `pendingDispatches_` — interleaved `Execute()` calls mix intermediate tensors |

No mutexes or locking exist anywhere in the runtime.

#### Fast Decode Buffer Safety

During fast decode capture:
- Intermediate buffers are deliberately **not released to pool** — the captured
  bind groups hold `wgpuBindGroupAddRef` references keeping GPU memory alive
- If those buffers were released and recycled, the captured bind group would still
  hold a valid GPU handle (AddRef prevents driver-level free), but pool bookkeeping
  would be inconsistent
- `ReleaseCaptured()` calls `wgpuBindGroupRelease()` which drops the ref, allowing
  GPU to reclaim the underlying buffer resources

Fast decode buffer retention during capture:
- Intermediate buffers are **not released** to pool — captured bind groups hold
  `wgpuBindGroupAddRef` references keeping GPU memory alive
- Param pool is expanded from 512→2048 per bucket to prevent round-robin wrapping

During replay:
- **Zero allocation** — `ReplayWrites()` updates existing buffers, `ReplayDispatches()`
  re-encodes dispatches with same bind groups
- Only transient WebGPU command encoder/buffer objects are created (not data buffers)

#### Required Architecture: Per-Session Execution State

To support multiple sessions (including independent fast decode), `GraphExecutor`
must be split into immutable (shared via Model) and mutable (per-Session) parts:

```
Model::Impl  (immutable, shareable)
  ├── compiled pipelines          (GPUContext::pipelines_ cache)
  ├── weight buffers              (persistent tensors, read-only during Execute)
  ├── graph structure             (ops, edges, metadata)
  └── input/output names

Session::Impl  (mutable, independent)
  └── ExecutionContext
        ├── tensorStore_          (own intermediates)
        ├── tensorPlan_           (own warm cache)
        ├── nodeShapeCache_       (own shape cache)
        ├── paramPool_[]          (own param buffers, 4 buckets)
        ├── fastDecodeState_      (own capture/replay)
        ├── capturedFlushes_      (own AddRef'd bind groups)
        ├── replayParamUpdates_   (own param updates)
        └── capturedTokenIdBufs_  (own token ID buffers)
```

This mirrors llama.cpp's `llama_model` (immutable) vs `llama_context` (mutable)
split, and enables:

- Session A in fast decode replay while Session B runs normal prefill
- Independent warm caches per session (different input shapes)
- Independent buffer lifecycles per session
- Safe `session.Release()` — releases only that session's intermediates and
  captured bind groups

#### Buffer Release With Independent Sessions

| Action | What happens |
|--------|-------------|
| `session.Release()` | Release captured bind groups (if fast decode active), release intermediate buffers to pool, release tensor plan buffers |
| `session.Reset()` | Clear input/output bindings. Optionally invalidate warm caches. |
| `model.Release()` | Release weight buffers. Only safe after all sessions released. |
| Fast decode capture | Session A captures its own bind groups. Session B unaffected. |

### Future: Intra-Execute Buffer Reuse

Currently each intermediate tensor gets its own buffer. A liveness analysis could
assign the same buffer to non-overlapping intermediates (like a register allocator),
reducing peak memory during `Execute()` by ~30-50%. This is independent of the
multi-session refactor.
