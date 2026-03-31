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
| **Model format** | GGUF only | ONNX only | ONNX + GGUF |
| **GPU backend** | ggml_backend (CUDA, Vulkan, Metal, WebGPU, 17+) | EP system (CUDA, TRT, WebGPU, etc.) | WebGPU only (via Dawn) |
| **Model scope** | LLM only | Any ONNX model | Any ONNX or GGUF model |
| **Language** | C API (opaque structs) | C/C++ API (`Ort::` wrappers) | C++ (classes, DLL export) |
| **LLM layer** | Built into runtime | Separate library (onnxruntime-genai) | `bp::LmSession` (runtime Layer 2) |

**llama.cpp** is purpose-built for autoregressive LLMs. Rich KV cache management,
sampling, and context handling are baked into the runtime. Supports 17+ GPU backends
but each has different maturity.

**ORT** is a general-purpose ONNX runtime supporting many backends via Execution
Providers. LLM-specific concerns (tokenization, KV management, sampling) are
handled by a separate library, **onnxruntime-genai**. The WebGPU EP is one of many.

**Backpack** exploits the simplicity of targeting a single backend (WebGPU) to
make different trade-offs: eager shader compilation, always-GPU tensors, type-safe
config. Like ORT, the runtime is model-agnostic at Layer 1 — LLM logic lives in
the runtime's Layer 2 API (`bp::LmSession`).

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
| **KV cache** | `llama_memory_t` (inside context) | Explicit I/O tensors via `IoBinding` | `bp::LmSession` (internal KV management) |
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
tracking) lives in the runtime's Layer 2 (`bp::LmSession`), similar to how
onnxruntime-genai wraps ORT.

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
| **Context size** | Set via `ctx_params.n_ctx` | Implicit in tensor shapes — same Session handles different sizes | Implicit in tensor shapes (`LmSession` auto-computes maxSeqLen) |
| **KV cache alloc** | Allocated inside context | Caller provides via `IoBinding` or `Run()` args | `LmSession` allocates based on maxSeqLen |
| **Multiple per model?** | Yes — each with different `n_ctx`, shared weights | One Session suffices for different KV sizes; multiple needed only for concurrent execution | Yes — shared weights + shaders, independent sessions |
| **KV cache type** | Configurable (`type_k`/`type_v`: f16, q8_0, q4_0) | Whatever dtype the model outputs | `LmSession` choice (currently f32 ONNX, f16 GGUF) |

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
| **Who manages KV?** | Runtime (opaque `llama_memory_t`) | Caller (explicit I/O tensors) | `bp::LmSession` (internal, hidden from caller) |
| **Resize at runtime?** | No — fixed at context creation | Yes — pass different-sized KV tensors each `Run()` | Fixed at `LmSession::Create()` (auto-computed from GPU memory) |
| **Cache ops** | Rich: `seq_rm`, `seq_cp`, `seq_add`, `seq_div`, `seq_keep` | None — caller manages | `LmSession::Reset()` clears KV state |
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
| `OgaModel` + `OgaGenerator` | `bp::LmSession` | LLM-specific (KV, positions, sampling) |
| `Ort::Value` | `bp::Tensor` | Tensor |
| `Ort::Env` + EP config | `bp::Device` | Runtime environment |

**Layer 1 (bp:: runtime)** — model-agnostic, knows nothing about LLMs:

```
bp::Device → bp::Model → bp::Session → SetInput/SetOutput → Run
```

Works for LLM, TTS, image, audio — any ONNX or GGUF model. Same API the image app uses today.

**Layer 2 (`bp::LmSession` in runtime DLL)** — LLM-specific, manages KV cache and positions:

```
bp::LmSession::Create(device, modelPath)
    → Prefill(tokens) → Decode() → Generate(prompt, maxTokens, sampling, onToken)
```

`LmSession` is a pimpl class (`BP_EXPORT`) in the runtime DLL. Internally it dispatches
between two backends: `GenericOnnxState` (GraphExecutor, for non-standard architectures
like LFM2) and `StandardState` (ModelRunner, for standard transformers). The caller only
sees strings, token IDs, and logits — no GPU buffers, sessions, or internal state.

`apps/llm/main.cpp` is now a thin CLI (~200 lines) that parses CLI args and calls
`bp::LmSession` for all model loading, generation, benchmarking, and profiling.

Multiple LmSessions can share one Device (different models, independent KV caches,
zero shader recompilation).

### Summary of Design Choices

| Decision | llama.cpp | ORT Native | **Backpack** | Rationale |
|----------|-----------|------------|--------------|-----------|
| Model/session split | Separate | Merged | **Separate** | Lightweight concurrent sessions without re-parsing. `ExecutionContext` per session, `GraphExecutor` shared. |
| Shader compilation | Pre-compiled | Lazy on first `Run()` | **Eager, async+parallel** | Predictable TTFT, no first-run spike |
| KV cache ownership | Runtime-managed (opaque) | Caller-managed (I/O tensors) | **`bp::LmSession` managed** | Layer 1 model-agnostic; Layer 2 handles KV internally |
| Context size | Fixed at creation | Implicit in tensor shapes | **Auto-computed from GPU memory** | `LmSession` budgets 25% of maxBufferSize for KV |
| GPU I/O | Backend scheduler | Requires `IoBinding` | **Always GPU** | No extra API surface for GPU stay |
| Model scope | LLM-only | Any ONNX model | **Any ONNX or GGUF model** | Same Session for LLM, TTS, image, audio |
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
| **API support** | Multiple `Ort::Session` on same device via `context_id` ref-counting | Multiple `llama_context` from one `llama_model` (weights shared read-only) | `bp::Session` owns `ExecutionContext`; multiple sessions share one `Model` (shared `GraphExecutor` with independent per-session state) |
| **Concurrent GPU execution** | **No** — `ConcurrentRunSupported()` returns `false` for WebGPU EP. Runs serialized via `session_mutex_`. | **Not validated** — API allows it, but server uses single-context multi-slot pattern. No mutexes in core lib. GPU backends have global state. | **Interleaved** (single-threaded cooperative) — each session has its own `ExecutionContext` (intermediates, param pool, fast decode, warm caches). Sessions take turns calling `Execute()`. Not parallel (WebGPU queue not thread-safe). |
| **Limitation** | WebGPU EP relies on global state; serialization is intentional | Thread safety of concurrent `llama_decode` across contexts is undocumented | Single-threaded only — no mutexes on `GPUContext`. Only one session may be in fast decode capture at a time (callback on shared GPU context). |

#### Continue Conversation (EOS → New Prompt)

| | ORT WebGPU Native | llama.cpp | Backpack |
|---|---|---|---|
| **Mechanism** | Caller passes previous `present.*` output KV tensors as next `past_key_values.*` inputs | Rich KV management: `llama_memory_seq_rm/cp/keep/add` for prefix reuse, position shifting, sequence forking | `LmSession`: KV cache kept, pos counter maintained, fast decode invalidated |
| **Prefix reuse** | Caller must implement (compare old vs new prompt, reuse matching KV prefix) | Built-in: server computes longest common prefix, removes only divergent tokens | Not yet implemented — `LmSession::Reset()` clears all state |
| **Context overflow** | Caller must handle (truncate or sliding window) | Built-in context shifting: `seq_rm` + `seq_add` to discard old tokens and shift positions | Not yet implemented — maxSeqLen enforced at creation |
| **Limitation** | No helper API — caller handles all KV lifecycle logic | KV ops are LLM-specific and baked into runtime (can't use for non-LLM models) | No prefix reuse or context overflow handling yet |

#### New Conversation (Reset)

| | ORT WebGPU Native | llama.cpp | Backpack |
|---|---|---|---|
| **Mechanism** | Caller passes zeroed/fresh KV tensors. No session reset API needed. | `llama_memory_clear()` or `llama_memory_seq_rm(mem, id, -1, -1)` per sequence | `LmSession::Reset()` — clears KV caches, resets pos=0, invalidates fast decode. KV buffers reused, no reallocation. |
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
| **Eviction** | No eviction API. OOM propagates as exception. | Server-level LRU eviction: `llama_memory_seq_rm` frees idle slots. `llama_memory_defrag()` compacts fragmented KV. | Tiered eviction: warm caches → fast decode → KV cache → session. Each session's `ExecutionContext` can be independently evicted. |
| **KV defragmentation** | N/A (caller-managed KV) | `llama_memory_defrag()` compacts KV cache in-place | N/A (caller-managed KV; no internal KV fragmentation) |
| **Limitation** | No GPU memory visibility. Cannot make informed eviction decisions. | Pre-allocated KV to `n_ctx` — no dynamic growth/shrink. Defrag only within fixed allocation. | Per-session `ExecutionContext` enables independent eviction. No automatic eviction policy yet. |

#### Multi-Model on Same Device

| | ORT WebGPU Native | llama.cpp | Backpack |
|---|---|---|---|
| **Support** | Multiple `Ort::Session` share same device via `context_id`. Shared `BufferManager`. | Per-model GPU state — no global barriers. Multi-GPU with `tensor_split`. | Multiple models share `GPUContext` (pool, pipeline cache). |
| **VRAM coordination** | No budget or priority system between sessions | No coordination — models compete for VRAM through allocator | Buffer pool shared — models can reuse freed buffers. `flushBufferPool()` between stages. |
| **Limitation** | No budget or priority system between sessions | No coordination — models compete for VRAM through allocator | `buffers_` name collision fixed (only device-level entries registered in shared map). Pool and pipelines shared safely. |

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
- `ModelRunner` (GGUF path) not yet split — concurrent sessions only work for ONNX `GraphExecutor` path
- No parallel execution (single-threaded cooperative only — WebGPU queue limitation)

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
| **Param pool** | Session lifetime | Small (~80-320 KB per session) | `session.Release()` |
| **Transient intermediates** | Single `Execute()` call | Variable | Released to pool at end of `Execute()` |
| **Caller tensors** (input/output) | Caller-managed | Variable | `tensor.Release()` |

#### Release Granularity

From finest to coarsest:

| Level | API | What happens |
|-------|-----|-------------|
| **Single buffer** | `gpu->releaseBuffer(buf)` | Returns to power-of-2 pool (27 buckets, 16B-1GB, max 256/bucket). Overflow destroys. |
| **Single tensor** | `tensor.Release()` | Drops C++ Impl, returns GPU buffer to pool via destructor. |
| **Session** | `session.Release()` | Drops `ExecutionContext` (intermediates, param pool, fast decode, warm caches). `Reset()` clears bindings for reuse. |
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

For a serving scenario with many sessions sharing one model, free memory by
evicting idle sessions. Each session's `ExecutionContext` can be independently
evicted without affecting other sessions:

| Priority | What to release | Memory freed | Cost to resume |
|----------|----------------|-------------|----------------|
| 1st | Tensor plan (`execCtx.InvalidateWarmCaches()`) | ~350 intermediate buffers | One cold Execute() to rebuild plan |
| 2nd | Fast decode (`execCtx.ReleaseCaptured()`) | ~400 captured buffers | One capture Execute() to re-record |
| 3rd | KV cache (`session.Release()`) | All KV buffers | Conversation lost, must re-prefill |

### Shared Resources Across Models

All models on the same `bp::Device` share a single `GPUContext`:

| Resource | Scope | Shared? | Safe? |
|----------|-------|---------|-------|
| **Buffer pool** (`pool_[]`) | Per-device | Yes — freed buffers reusable across models | Yes |
| **Pipeline cache** (`pipelines_[]`) | Per-device | Yes — Model B reuses Model A's compiled shaders | Yes — pipelines are immutable |
| **Named buffer map** (`buffers_[]`) | Per-device | Only device-level resources (prefixed with `_`) | Yes — per-model buffers tracked in `weightStore_` instead |
| **Weight buffers** | Per-model (`weightStore_`) | No | Yes |
| **Intermediate buffers** | Per-session (`ExecutionContext::tensorStore_`) | No | Yes — independent per session |
| **Fast decode state** | Per-session (`ExecutionContext`) | No — each session has its own capture/replay | Yes |

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

#### Architecture: Per-Session Execution State

`GraphExecutor` is split into shared immutable state and per-session mutable state
via `ExecutionContext`. Multiple sessions share one `GraphExecutor` (graph, weights,
compiled pipelines) while each owning independent execution state.

```
Model::Impl  (shared, immutable after Load())
  └── GraphExecutor executor
        ├── graph_                   (ops, edges, metadata — immutable)
        ├── weightStore_             (weight tensors — read-only during Execute)
        ├── cachedExecOrder_         (topological sort — computed once)
        ├── fusedGroups_             (fusion info — computed once)
        └── compiled pipelines       (via GPUContext::pipelines_ cache)

Session A                            Session B
  └── ExecutionContext                 └── ExecutionContext
        ├── tensorStore_                     ├── tensorStore_
        ├── tensorPlan_                      ├── tensorPlan_
        ├── nodeShapeCache_                  ├── nodeShapeCache_
        ├── paramPool_[4]                    ├── paramPool_[4]
        ├── pendingDispatches_               ├── pendingDispatches_
        ├── fastDecodeState_                 ├── fastDecodeState_
        ├── capturedFlushes_                 ├── capturedFlushes_
        ├── replayParamUpdates_              ├── replayParamUpdates_
        ├── capturedTokenIdBufs_             ├── capturedTokenIdBufs_
        └── profileData_                     └── profileData_
```

This mirrors llama.cpp's `llama_model` (immutable) vs `llama_context` (mutable)
split, and enables:

- Session A in fast decode replay while Session B runs normal prefill
- Independent warm caches per session (different input shapes)
- Independent buffer lifecycles per session
- Safe `session.Release()` — releases only that session's intermediates and
  captured bind groups

#### Implementation Layers

Three layers use `ExecutionContext`:

| Layer | How it uses ExecutionContext |
|-------|----------------------------|
| **`bp::Session`** (runtime API) | `Session::Impl` owns `ExecutionContext execCtx`. `Run()` calls `executor.Execute(execCtx, inputs, outputs)`. |
| **`bp::LmSession`** (runtime Layer 2) | Owns `ExecutionContext execCtx` (via `GenericOnnxState` or `StandardState`). All per-session calls (capture, replay, dispatch, profiling) go through `execCtx`. Shared calls (pipelines, bind groups, graph queries) stay on `executor`. |
| **Op dispatch** | Ops receive `OpContext& ex` — a facade forwarding to both `GraphExecutor` (shared) and `ExecutionContext` (per-session). |

`OpContext` provides the same method names ops previously called on `GraphExecutor`,
routing each to the correct target:

```cpp
struct OpContext {
    GraphExecutor& graph;      // shared: GetPipeline, AllocTensor, weights
    ExecutionContext& exec;     // per-session: QueueDispatch, paramPool, etc.

    GPUContext* getGpu() const;
    GPUBuffer getParamBuffer(uint32_t size);     // → exec
    void QueueDispatch(...);                      // → exec
    const CompiledPipeline& GetPipeline(...);     // → graph
    GpuTensor AllocTensor(...);                   // → graph
    GpuTensor* GetTensor(const std::string& name); // exec.tensorStore_ then graph.weightStore_
};
```

#### Threading Model

**Single-threaded cooperative concurrency (interleaved, not parallel).**

- WebGPU's `wgpuQueueSubmit` is not thread-safe in Dawn
- No mutexes exist in the codebase — adding them is out of scope
- Primary use case: server with multiple chat sessions sharing one model,
  naturally interleaved (one session runs a decode step, then another)
- Matches ORT WebGPU EP's approach (`ConcurrentRunSupported() = false`)

#### Backward Compatibility

`GraphExecutor` retains a `defaultCtx_` and forwarding methods so all existing
callers (op tests, image app, `backpack_runtime` CLI) continue working unchanged
without migrating to explicit `ExecutionContext`:

```cpp
// Old API still works — forwards to defaultCtx_
executor.Execute(inputs, outputs);
executor.QueueDispatch(pipeline, bg, gx, gy, gz, name);
executor.getParamBuffer(16);
```

#### Remaining: `ModelRunner` Split

`ModelRunner` (the GGUF/standard transformer path) has the same singleton state
problem (KV cache, intermediate buffers, dispatch lists). The same
shared-vs-per-session pattern applies but is a separate effort since `ModelRunner`
uses its own pre-recorded command buffer architecture.

### Future: Intra-Execute Buffer Reuse

Currently each intermediate tensor gets its own buffer. A liveness analysis could
assign the same buffer to non-overlapping intermediates (like a register allocator),
reducing peak memory during `Execute()` by ~30-50%. This is independent of the
multi-session refactor.

---

## 6. Profiling and Benchmarking

### Overview

Backpack provides layered instrumentation from coarse benchmarking down to
per-kernel GPU hardware timestamps:

| Layer | Trigger | What it measures | Output |
|-------|---------|------------------|--------|
| **Benchmark** | `--benchmark` | Prefill/decode throughput across prompt lengths | Console table |
| **Baseline JSON** | `--save-baseline` | System info + model info + perf + memory + TTFT | `.json` file |
| **GPU profiling** | `--profile` | Per-dispatch nanosecond timestamps via WebGPU query sets | Console table + interactive `profile.html` |
| **CPU profiling** | `profilingEnabled` | Per-op-type wall-clock time in `Execute()` | Console table (stderr) |
| **GPU API timing** | `--bench-detail` | Encode/submit/map/wait/unmap breakdown | Console table |
| **Autotuning** | Automatic on first load | Optimal decode pipeline depth + kernel variant selection | Cache file |

### Benchmark Mode

`--benchmark` runs a prefill+decode sweep across prompt lengths
`{128, 256, 512, 1024, 2048, 4096}` (or a single length via `--bench-prompt-len`).

For each prompt length:
1. **Prefill**: Batch-process all prompt tokens, measure wall-clock time
2. **TTFT**: Time the first decode step after prefill (captures fast decode capture overhead)
3. **Warmup**: 2 additional decode steps to stabilize capture/replay
4. **Decode**: Time `--bench-gen-tokens` (default 128) decode steps

Results are printed as a table:

```
prompt_len   prefill_ms   pf_tok/s  decode_ms   dc_tok/s
128               12.3      10406       480.2      266.5
256               23.1      11082       481.0      266.1
```

### Baseline JSON

`--save-baseline [path]` writes a structured JSON file (default:
`apps/baseline/<arch>.json`) containing:

```json
{
  "system":    { "cpu": "...", "memory_gb": 32, "os": "Windows" },
  "gpu":       { "name": "...", "backend": "d3d12", "driver": "..." },
  "model":     { "name": "...", "format": "onnx_generic", "layers": 24, ... },
  "benchmark": {
    "decode_tokens": 128,
    "results": [
      { "input_tokens": 128, "prefill_ms": 12.3, ..., "ttft_ms": 8.5 },
      ...
    ]
  },
  "loading":   { "weight_ms": 1200, "shader_ms": 350, "total_ms": 1550 },
  "memory":    { "peak_bytes": 2147483648, "current_bytes": ..., "alloc_count": ... },
  "timestamp": "2026-03-31T..."
}
```

Data structures (`apps/common/app_common.h`):
- `BenchResultEntry`: per-prompt-length results including `ttftMs`
- `LoadingInfo`: sub-phase timing (weight upload, shader compilation, autotune)
- `MemoryInfo`: GPU memory counters from `GPUContext::getMemoryStats()`
- `SystemInfo`: CPU name, RAM, OS (platform-specific collection)

### GPU Hardware Profiling

`--profile` enables per-dispatch GPU timestamp queries using WebGPU's
`WGPUPassTimestampWrites`. Each dispatch gets its own compute pass with
begin/end timestamp writes into a query set (`GPUProfiler::MAX_TIMESTAMPS = 16384`,
supporting up to 8192 profiled dispatches).

#### How It Works

```
1. GPUProfiler::allocate("matmul_L5") → (beginIdx, endIdx)
2. Each dispatch gets its own compute pass with timestampWrites
3. After all dispatches: resolveQuerySet → copy to staging buffer
4. Map staging buffer → read nanosecond timestamps on CPU
5. Aggregate by kernel name → print table + generate HTML
```

Profiled dispatch submission (`submitOnlyProfiled`, `submitAndReadbackProfiled`)
uses one compute pass per dispatch instead of batching all dispatches into a single
pass. This is required because `WGPUPassTimestampWrites` is per-pass.

#### Clock Calibration

GPU timestamps are in device-local nanoseconds. To align them with CPU wall-clock
time (for the HTML timeline), `ClockCalibration` correlates the two clocks:

- **D3D12**: `ID3D12CommandQueue::GetClockCalibration()` returns correlated
  GPU tick + CPU QPC tick at the same instant
- **Vulkan**: `vkGetCalibratedTimestampsEXT` with `VK_TIME_DOMAIN_DEVICE_EXT` +
  `VK_TIME_DOMAIN_QUERY_PERFORMANCE_COUNTER_EXT`

Both are accessed through Dawn's internal APIs (loaded via mangled symbols from
`webgpu_dawn.dll`). The timestamp quantization Dawn toggle is disabled to ensure
high-resolution timestamps.

#### Profile HTML Report

`generateProfileHTML()` produces a self-contained HTML file with:

- **Interactive canvas timeline**: GPU lane showing each dispatch as a colored
  block (color-coded by kernel name). Optional CPU lane with hierarchical scope
  events from `CPUProfiler`.
- **Step markers**: Divides timeline into prefill and decode steps (detected by
  finding `argmax` dispatch boundaries)
- **Summary tables**: Per-kernel aggregate (total ms, count, avg ms, percentage)
- **Zoom/pan**: Mouse wheel zoom, click-drag pan, reset button
- **GPU bubble analysis**: Reports percentage of time where GPU is idle between
  dispatches (CPU overhead)

### CPU-Side Profiling

`ExecutionContext::profilingEnabled` enables per-op wall-clock timing within
`Execute()`. Each op dispatch is bracketed with `steady_clock::now()` calls.
Results are aggregated by op type and printed to stderr:

```
+-- Profile (247 ops, 4.8ms total) ----------------+
| Op                        ms    cnt       %       |
| MatMul                  2.1     48   43.8%        |
| Add                     0.5     24   10.4%        |
| ...                                               |
+---------------------------------------------------+
```

Also tracks GPU sync count (`flushCount_`) and sync sources per op — useful for
identifying ops that force CPU-GPU synchronization (e.g., shape readbacks).

### GPU API Timing

`GPUContext::timing` records nanosecond-precision breakdown of WebGPU API calls:

| Counter | What it measures |
|---------|------------------|
| `encode_ns` | Command encoder creation + dispatch recording |
| `submit_ns` | `wgpuQueueSubmit` call |
| `map_start_ns` | `wgpuBufferMapAsync` initiation |
| `wait_ns` | `wgpuInstanceWaitAny` (GPU completion fence) |
| `unmap_ns` | Buffer unmap + data copy |
| `write_buf_ns` | `wgpuQueueWriteBuffer` (param updates) |

Enabled via `--bench-detail`. Useful for diagnosing whether bottlenecks are in
GPU compute, CPU-side encoding, or synchronization.

### Autotuning

On first model load (D3D12 only), Backpack runs a two-phase autotune:

1. **Decode pipeline depth** (`autotuneDecodeDepth`): Tries depths 2-4, benchmarks
   24 tokens × 2 repeats each, selects lowest ms/tok with 1.5% epsilon (prefers
   shallower depth on ties)

2. **Decode kernel variants** (`autotuneDecodeKernels`): Tries all 8 combinations
   of fast vs. base kernels for QKV, output projection, and gate-up matmuls.
   Benchmarks 12 tokens × 1 repeat each, selects lowest ms/tok

Results are cached to `gitignore/models/<model>/decode_autotune_<backend>.txt`.
The cache key includes GPU name, model architecture, layer count, embedding size,
and hardware limits — a key mismatch triggers re-autotune.

```

---

## 7. Feature Comparison

Detailed feature-by-feature comparison across all three frameworks. Each subsection
focuses on one capability area with a comparison table.

### 7.1 Profiling

| Aspect | llama.cpp (Vulkan) | ORT WebGPU Native | Backpack |
|--------|-------------------|-------------------|----------|
| **GPU timestamp profiling** | Yes — Vulkan `QueryPool` timestamps per dispatch. `vk_perf_logger` aggregates per-op nanosecond durations with GFLOPS/s calculation. | Yes — `WGPUPassTimestampWrites` per dispatch. Three modes: `InsidePasses` (Chromium extension), `AtPasses` (standard), `None`. Offset-based calibration (TODO: full CPU-GPU alignment). | Yes — `WGPUPassTimestampWrites` per dispatch. Up to 8192 profiled dispatches. CPU-GPU clock calibration via D3D12 `GetClockCalibration()` or Vulkan `vkGetCalibratedTimestampsEXT`. |
| **CPU-side profiling** | Wall-clock timing via profiling names per dispatch. | Per-kernel metadata in `PendingKernelInfo`. | Per-op wall-clock timing in `Execute()`. Aggregated by op type. Tracks GPU sync count and sync sources per op. |
| **Profile visualization** | Console table with GFLOPS/s. | Raw timestamp readback (no visualization). | Interactive HTML timeline: zoom/pan canvas, color-coded kernel blocks, step markers, summary tables, GPU bubble analysis. |
| **Memory profiling** | `vk_memory_logger`: per-buffer allocation tracking, running device/host totals. | `AllocatorStats` via standard ORT interface. No buffer pool hit/miss counters. | `GPUContext::getMemoryStats()`: `totalAllocatedBytes`, `peakAllocatedBytes`, `totalAllocCount`. Baseline JSON includes memory section. |
| **PIX / RenderDoc** | RenderDoc capture support. | PIX capture via `ENABLE_PIX_FOR_WEBGPU_EP`. | Not integrated (Dawn's own capture mechanisms available). |

### 7.2 Benchmarking

| Aspect | llama.cpp | ORT WebGPU Native | Backpack |
|--------|-----------|-------------------|----------|
| **Built-in benchmark** | `llama-bench` tool: prefill/decode sweep across batch sizes and context lengths. | None built-in. `onnxruntime_perf_test` for general models but no LLM-specific benchmark. | `LmSession::Benchmark(promptLen, genTokens)`: prefill/decode throughput per prompt length. CLI: `--benchmark` sweeps `{128, 256, 512, 1024, 2048, 4096}`. |
| **TTFT measurement** | Not directly (first decode step not isolated). | Not available. | Yes — first decode step timed separately after prefill. `BenchmarkResult::ttftMs`. |
| **Warmup handling** | Separate warmup runs before timed runs. | None. | 3-step warmup: 1 warm prefill + 3 decode steps (capture/replay stabilization) before timed decode. |
| **Baseline recording** | Console output only. | Not available. | JSON baseline files with system info, GPU info, model info, per-prompt-length results, loading phases, memory stats. |
| **Loading phase timing** | Not broken down. | Not available. | `LoadingInfo`: weight upload ms, shader compilation ms, autotune ms, total ms. |

### 7.3 Buffer Management

| Aspect | llama.cpp (Vulkan) | ORT WebGPU Native | Backpack |
|--------|-------------------|-------------------|----------|
| **Pool architecture** | Pre-allocated scratch buffers (`prealloc_x/y/split_k`). Descriptor pool (256 sets). No general-purpose buffer pool. | Multi-tier system: 4 independent cache managers per usage type (storage, uniform, query resolve, default). 6 cache modes. | Power-of-2 bucket pool: 27 buckets (16B to 1GB), max 256 buffers per bucket. Single pool shared across models. |
| **Cache modes** | N/A — fixed pre-allocation. | `Disabled`, `LazyRelease`, `Simple` (exact-size), `Bucket` (26 predefined sizes, 64B-160MB), `Graph` (unlimited, for capture), `GraphSimple`. | Single mode: always-on bucketed pool. Overflow destroys immediately. OOM triggers full pool flush + retry. |
| **Buffer reuse** | Conversion caching: tracks `prealloc_y_last_pipeline_used` to avoid redundant requantization. | Per-usage-type caching. `Bucket` mode: `lower_bound` search for smallest fitting bucket. Refresh cycle before reuse. | Pool hit: pop from matching bucket. No refresh cycle — immediate reuse on release. |
| **Buffer alignment** | Hardware-specific alignment from Vulkan properties. | 16-byte aligned. | Rounded up to next power-of-2 bucket size. |
| **Cross-model sharing** | No — per-model GPU state. | Multiple Sessions share `BufferManager` via `context_id`. | Yes — single `GPUContext` pool shared across all models on a device. |

### 7.4 KV Cache

| Aspect | llama.cpp | ORT WebGPU Native | Backpack |
|--------|-----------|-------------------|----------|
| **Data types** | Any `ggml_type`: f32, f16, bf16, q8_0, q4_0, q4_1, q5_0, q5_1, all K-quants. Quantized V requires flash attention. | Whatever dtype the model outputs (typically f16). `past_present_share_buffer_` for in-place updates. | f16 (GGUF/ModelRunner path), f32 (ONNX/GraphExecutor path). No quantized KV cache. |
| **Allocation** | Pre-allocated to `n_ctx` at context creation. Zeroed to avoid NaN. | Caller-managed tensors. ORT creates/caches buffers as needed. | Pre-allocated to `maxSeqLen` at `LmSession::Create()`. Auto-computed: 25% of `maxBufferSize` budget, rounded to power-of-2. |
| **Ring buffer / shift** | Circular slot search with SWA eviction. Context shifting via RoPE-based K-shift (`seq_add`). | No built-in context management. | No ring buffer or context shifting. Fixed allocation, `Reset()` clears to zero. |
| **Cache ops** | Rich: `seq_rm`, `seq_cp`, `seq_keep`, `seq_add` (position shift), `seq_div` (context compression), `defrag`. | None — caller manages. | `LmSession::Reset()` only. No per-sequence ops. |
| **Window attention** | `local_window_size_` for bounded KV memory. Head sink for attention sink tokens. | `local_window_size_` in GQA kernel. | Not implemented. |
| **Multi-sequence** | `n_stream` parallel KV streams per context. Cross-stream copies. | Via batched `seqlen_k` tensor. | Single sequence per `LmSession`. |

### 7.5 Fast Decode / Graph Capture

| Aspect | llama.cpp | ORT WebGPU Native | Backpack |
|--------|-----------|-------------------|----------|
| **Mechanism** | No graph capture. Vulkan dispatches are re-recorded each step. | `enableGraphCapture`: records dispatch sequence during capture step, replays without CPU-side overhead. Bind groups AddRef'd during capture. | Fast decode: 3-stage state machine (normal → capture → replay). Captures dispatch list + param updates. Replays with updated GQA params + token ID. |
| **What changes per step** | Everything re-recorded. | Replay is opaque — same dispatches exactly. Dynamic inputs require exiting graph capture. | Only 3 things change: 24 GQA position params (6 layers × 4 words), 1 token ID buffer, conv cast dispatches (run after replay). |
| **Resource management** | N/A. | `Graph` cache mode: unlimited buffer retention during capture. `ReleaseGraphResources` for cleanup. | `ReleaseCaptured()` releases AddRef'd bind groups. `InvalidateWarmCaches()` drops tensor plan. |
| **Compatibility** | N/A. | Incompatible with variable KV sizes (shapes must not change during replay). | Compatible with growing KV cache — only position params update, buffer handles stay fixed (static KV buffers). |
| **Measured speedup** | N/A. | Not published. | +10% decode throughput (54 → 59 tok/s on RTX 5080, LFM2-8B, 975 dispatches). |

### 7.6 Warmup / Shader Compilation

| Aspect | llama.cpp (Vulkan) | ORT WebGPU Native | Backpack |
|--------|-------------------|-------------------|----------|
| **Shader language** | GLSL → SPIR-V (compiled at build time by `glslc`). 151 `.comp` files. | WGSL (generated at runtime by `ShaderHelper`). | WGSL (hand-written, embedded in binary via `wgsl_shaders.h`). 60+ kernel entries. |
| **Compilation timing** | SPIR-V embedded in binary (build time). Vulkan `VkPipeline` creation lazy on first use. | Lazy — compiled on first kernel execution. `CreateComputePipelineAsync` + immediate `Wait()`. | Eager — all pipelines compiled async+parallel at `Model::Load()` or `LmSession::Create()`. |
| **Parallel compilation** | Not documented for pipeline creation. | No — sequential (async creation + sync wait per shader). | Yes — `std::async` for shader modules (Dawn's `CreateShaderModule` is thread-safe), then async pipeline creation in parallel. |
| **Extension testing** | Build-time test programs for 4 Vulkan extensions (cooperative matrix, integer dot product, bfloat16). | Runtime feature detection. | Runtime feature detection (`supportsShaderF16`, `supportsSubgroups`, `supportsSubgroupMatrix`). |

### 7.7 Backend Support

| Aspect | llama.cpp | ORT WebGPU Native | Backpack |
|--------|-----------|-------------------|----------|
| **GPU backends** | 17+ via ggml_backend: CUDA, Vulkan, Metal, HIP, SYCL, CANN, OpenCL, WebGPU (experimental), etc. | WebGPU only (via Dawn on native, browser on web). Also CUDA, TensorRT, OpenVINO, etc. via other EPs. | WebGPU only (via Dawn). D3D12, Vulkan, Metal backends through Dawn's HAL. |
| **Backend selection** | Runtime backend selection. Multiple backends can be active simultaneously (e.g., CUDA + CPU). | EP selected at Session construction. One EP per Session. | `bp::Device::Create(backend)`. One backend per device. Default: D3D12 (Windows), Metal (macOS), Vulkan (Linux). |
| **D3D12-specific** | N/A (Vulkan only on Windows). | Dawn backend option. Power preference selection. | Dawn toggles: `skip_validation`, `disable_robustness`, `d3d_disable_ieee_strictness`. Backend-specific matmul kernels (DP4A, 256-thread tiled). |
| **Vulkan-specific** | Full Vulkan 1.3 target. Cooperative matrix (v1 + NVIDIA v2). BDA. External host memory. Timeline semaphores. | Dawn's Vulkan backend. | Dawn's Vulkan backend. MMA-based matmul and flash attention via `chromium_experimental_subgroup_matrix`. |
| **Metal-specific** | Dedicated Metal backend with Apple GPU optimizations. | Dawn's Metal backend. Subgroup matrix path for Apple GPUs. | Dawn's Metal backend. No Apple-specific kernel variants yet. |

### 7.8 Memory Management

| Aspect | llama.cpp | ORT WebGPU Native | Backpack |
|--------|-----------|-------------------|----------|
| **mmap support** | Yes — default weight loading path. `llama_mmap` abstraction (Windows `MapViewOfFile`, POSIX `mmap`). Prefetch control. `llama_mlock` for page pinning. | No mmap on native C++. `ExternalDataLoader` only for WASM/browser builds. Standard ORT initializer loading. | Yes — `MappedFile` abstraction (Windows `CreateFileMappingW/MapViewOfFile`, POSIX `mmap`). Weights read directly from mapped pointer, uploaded via `wgpuQueueWriteBuffer`. |
| **GPU memory tracking** | `vk_memory_logger`: per-buffer size tracking, device/host totals. | `AllocatorStats` via ORT allocator interface. No pool-level metrics. | `GPUContext::getMemoryStats()`: peak bytes, current bytes, alloc count. Exposed in baseline JSON. |
| **UMA support** | UMA-aware allocation (`uma` flag, `prefer_host_memory`). | `BufferMapExtendedUsages` feature for UMA. | Not explicitly handled (Dawn abstracts this). |
| **Async GPU upload** | Yes — 4 × 64MB pinned staging buffers with round-robin async transfer + events. | `mappedAtCreation = true` for efficient CPU→GPU during buffer creation. | Synchronous `wgpuQueueWriteBuffer`. No async staging pipeline. |
| **Eviction** | Server-level LRU eviction via `llama_memory_seq_rm`. `llama_memory_defrag()` for KV compaction. | No eviction API. OOM propagates as exception. | Tiered eviction: warm caches → fast decode → KV cache → session. Each session's `ExecutionContext` independently evictable. |

### 7.9 GPU-Specific Optimizations

| Aspect | llama.cpp (Vulkan) | ORT WebGPU Native | Backpack |
|--------|-------------------|-------------------|----------|
| **Subgroup operations** | Granular tracking: basic, arithmetic, shuffle, ballot, clustered, vote. Size control via `VK_EXT_subgroup_size_control`. | Used in MatMulNBits subgroup matrix path. Apple + Intel vendor-specific paths. | `enable subgroups` + `subgroupAdd` in all Q8 kernels. Vulkan MMA path via `chromium_experimental_subgroup_matrix`. |
| **Cooperative matrix** | `VK_KHR_cooperative_matrix` (standard) + `VK_NV_cooperative_matrix2` (NVIDIA v2). Multiple tile sizes with f16/f32 accumulators. Flash attention uses cooperative matrix. | `ChromiumExperimentalSubgroupMatrix` for MatMulNBits on Apple and Intel. | `subgroupMatrixMultiply` on Vulkan for MMA-based matmul and flash attention (128 threads, 16×16 MMA tiles). |
| **Integer dot product** | `VK_EXT_integer_dot_product` for Q8_1 matmul paths. Runtime activation quantization to Q8_1. | `dot4I8Packed` DP4A in MatMulNBits. Qualcomm-specific DP4A path. | `dot4I8Packed` DP4A on D3D12. Used in prequant matmul path when subgroup matrix unavailable. |
| **Vendor-specific kernels** | Architecture detection (AMD GCN/RDNA1/2/3, Intel Xe2, NVIDIA pre-Turing/Turing). Matvec column count tuning. Reduction mode selection. | Apple, Intel, Qualcomm, NVIDIA vendor flags. Intel Xe-2lpg subgroup matrix with reduced `tile_size_k_vec`. | D3D12 vs Vulkan kernel variants: `q8_matmul_d3d12` (256 threads, DP4A) vs `q8_matmul_vulkan` (512 threads, MMA). `flash_attn_vulkan` (MMA) vs `causal_attn` (subgroup). |
| **Flash attention** | 3 code paths: scalar, cooperative matrix v1, cooperative matrix v2 (NVIDIA). Split-K reduction. Configurable block sizes, shmem staging, occupancy tuning. | Separate prefill (`FlashAttentionProgram`) and decode (3-stage: QK^T, softmax+V, reduce). Indirect dispatch for GPU-driven workgroup sizing. | `flash_attn_vulkan` (MMA-based, Vulkan) and `causal_attn` (subgroup-based, D3D12). `gqa_fused_attn` + `gqa_prefill` for ONNX path. |

### 7.10 Quantization

| Aspect | llama.cpp (Vulkan) | ORT WebGPU Native | Backpack |
|--------|-------------------|-------------------|----------|
| **Weight formats** | Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q2_K, Q3_K, Q4_K, Q5_K, Q6_K, IQ1_M, IQ1_S, IQ2_S, IQ2_XS, IQ2_XXS, IQ3_S, IQ3_XXS, IQ4_NL, IQ4_XS, MXFP4, f32, f16, bf16. 151 shaders. | MatMulNBits: 2-bit, 4-bit, 8-bit with per-group scales. 4 kernel paths (subgroup matrix, DP4A, wide tile, default). | GGUF: Q8_0 (30 kernel variants), Q4_K, Q5_K, Q6_K (K-quant kernels), f16. ONNX: MatMulNBits INT4, MXFP4 (E2M1+E8M0). |
| **Runtime quantization** | `quantize_q8_1.comp` for activation quantization to Q8_1 (integer matmul path). | Activation quantization in DP4A path (1024-entry LUT for Q2 zero points). | No runtime quantization. Activations stay in native precision. |
| **Embedding dequant** | Dedicated per-type dequant shaders on GPU. | `GatherBlockQuantized` for quantized embedding lookups on GPU. | CPU-side dequant at load time for Q8_0, Q4_K, Q5_K, Q6_K embeddings → f32. |
| **bf16** | `VK_KHR_shader_bfloat16` extension support. | Not explicitly supported in WebGPU EP. | Not supported. |
| **MoE quantization** | Standard dequant in MoE path. | `QMoE` — quantized Mixture of Experts with MatMulNBits integration. | No quantized MoE. MoE matmuls use dequantized weights. |

### 7.11 Autotuning

| Aspect | llama.cpp | ORT WebGPU Native | Backpack |
|--------|-----------|-------------------|----------|
| **Kernel selection** | Architecture-based dispatch (AMD GCN/RDNA, Intel Xe2, NVIDIA). Multiple matmul tiers (large/medium/small). Matvec column count (up to 8). 3 reduction modes. | Deterministic selection by vendor, matrix dims, features. No runtime benchmarking. | Two-phase autotune at first load (D3D12 only): (1) decode pipeline depth (2-4, 24-token bench), (2) kernel variants (8 combos of fast/base QKV/Oproj/GateUp, 12-token bench). |
| **Cache** | No autotuning cache. | N/A. | Results cached to disk. Cache key: GPU name, backend, arch, layers, nEmbd, hardware limits. Mismatch triggers re-autotune. |
| **Heuristics** | Cooperative matrix support, subgroup capabilities, vendor ID → kernel variant. | GPU vendor → shader template. M threshold → tile variant. | Initial heuristics for pool depth based on invocations, WG memory, backend. Autotune refines. |

### 7.12 Pipelined / Async Decode

| Aspect | llama.cpp (Vulkan) | ORT WebGPU Native | Backpack |
|--------|-------------------|-------------------|----------|
| **Decode pipelining** | Dual queues (compute + transfer). Timeline semaphores. `almost_ready_fence` for overlapped preparation. Deferred submission. | No pipelining. Single command encoder, single compute pass. `ConcurrentRunSupported() = false`. | N-deep pipelined decode (ModelRunner path): pre-recorded command buffers, async argmax readback. While CPU reads slot `i`, slots `i+1..i+N-1` are in-flight. `decodePoolDepth` = 3-4 (autotuned). |
| **Command batching** | Up to 8192 nodes per graph. Operation fusion (up to 9-way add, MoE subgraph, RMS+Mul+RoPE). | Batch up to `max_num_pending_dispatches_` (default 16) dispatches per compute pass. | GGUF: pre-recorded CB pool (`decodeCbPoolBatch = 128` tokens per slot). ONNX: `pendingDispatches_` flushed after readback requests. |
| **Async readback** | Timeline semaphore synchronization. Pipeline parallelism when multiple GPUs present. | `MapAsync` + `Wait()` for profiling data only. Inference output is synchronous. | GGUF: `submitDecode()` starts async map (non-blocking), `readArgmax()` completes it (4 bytes). ONNX: `mapReadbackBuffer` after submit. |
| **GPU-driven dispatch** | Not in Vulkan backend. | Indirect dispatch in flash attention decode: GPU writes workgroup counts, avoids CPU-GPU sync. | Argmax on GPU: kernel writes single i32 to `argmaxResultBuf`, only 4 bytes read back (not full vocab logits). |

### 7.13 Logit Readback

| Aspect | llama.cpp | ORT WebGPU Native | Backpack |
|--------|-----------|-------------------|----------|
| **Default path** | Full logits copied to CPU (`llama_get_logits`). Float array of vocab_size. | CPU default. `IoBinding` needed for GPU-side results. | GGUF: argmax on GPU → 4-byte i32 readback (greedy). Full logits via `runner.decode()` when sampling needed. ONNX: full vocab-size f32 logits read back each step. |
| **Greedy optimization** | No GPU-side argmax. Full logits always transferred. | No GPU-side argmax. | GPU-side argmax kernel avoids transferring vocab_size × 4 bytes per token. 4-byte readback only. |
| **Sampling path** | CPU-side sampling from full logits. `llama_sampler` chain. | Caller handles sampling from CPU logits. | `LmSession::Decode()` uses GPU argmax (greedy). `LmSession::DecodeLogits()` returns full logits for temperature/top-k sampling. |
| **Readback buffer** | Single output buffer. | Standard `BufferManager::Download` (synchronous). | Reusable readback buffer (`getOrCreateReadbackBuf`). Readback folded into the same command buffer as compute dispatches — no extra submit. |
