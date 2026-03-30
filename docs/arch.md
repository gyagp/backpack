# Backpack Architecture: Design Comparison with ORT and llama.cpp

Three-way comparison of API and architecture design across ORT WebGPU Native (C++),
llama.cpp, and Backpack. Focuses on the design rationale — why each system is
structured the way it is, and what Backpack can do differently as a single-backend
(WebGPU-only) runtime.

---

## Architecture Overview

| Concept | llama.cpp | ORT Native (C++ WebGPU EP) | Backpack |
|---------|-----------|---------------------------|----------|
| **Language** | C API (opaque structs) | C/C++ API (`Ort::` wrappers) | C++ (classes, DLL export) |
| **Model format** | GGUF only | ONNX only | ONNX (+ GGUF via ModelRunner) |
| **GPU backend** | ggml_backend (CUDA, Vulkan, Metal, WebGPU, 17+) | EP system (CUDA, TRT, WebGPU, etc.) | WebGPU only (via Dawn) |
| **Model scope** | LLM only | Any ONNX model | Any ONNX model |

---

## Object Model

| Concept | llama.cpp | ORT Native | Backpack |
|---------|-----------|------------|----------|
| **Runtime env** | `llama_backend_init()` | `Ort::Env(level, "app")` | `bp::Device::Create(backend)` |
| **Model (weights)** | `llama_model` (opaque) | None — merged into Session | `bp::Model` |
| **Execution context** | `llama_context` (KV cache, compute bufs) | `Ort::Session` (model+context merged) | `bp::Session` (model-agnostic) |
| **Tensor** | `ggml_tensor` (internal) | `Ort::Value` | `bp::Tensor` |
| **KV cache** | `llama_memory_t` (inside context) | Explicit I/O tensors via `IoBinding` | App-layer (explicit I/O tensors) |
| **GPU binding** | Backend scheduler | `Ort::IoBinding` (pre-bind GPU buffers) | `session.SetInput/SetOutput` (always GPU) |
| **LLM layer** | Built into runtime | Separate library (onnxruntime-genai) | App-layer `LlmContext` |

### Key Structural Differences

**llama.cpp** separates model from context — load weights once, create multiple
contexts with different `n_ctx`. But the entire API is LLM-specific.

**ORT** merges model+context into `Ort::Session`. A single Session can `Run()`
with different KV cache sizes since KV caches are just regular I/O tensors. For
concurrent independent execution (e.g., two threads), you need two Sessions, which
re-parses and re-optimizes the graph. `PrepackedWeightsContainer` shares weights
but not optimized graph state. LLM-specific concerns (KV cache management,
tokenization, sampling) are handled by a separate library, **onnxruntime-genai**.

**Backpack** separates model from session — model holds weights + compiled shaders,
session is a lightweight execution context. LLM-specific state (KV cache, position
tracking) lives in the app layer, similar to how onnxruntime-genai wraps ORT.

---

## Stage 1: Model Loading

| Aspect | llama.cpp | ORT Native | Backpack |
|--------|-----------|------------|----------|
| **API** | `llama_model_load_from_file(path, params)` | `Ort::Session(env, path, opts)` | `bp::Model::Load(device, path, opts)` |
| **What happens** | Parse GGUF, upload weights | Parse ONNX, optimize graph, upload weights, register kernels | Parse ONNX, upload weights, **compile all shaders** |
| **Shader compilation** | N/A (pre-compiled backend kernels) | **Deferred** — compiled on first `Run()` | **Eager** — compiled async+parallel at load |
| **From memory buffer?** | Yes | Yes | Not yet (path only; web will use buffer) |
| **Weight sharing** | Multiple contexts share one `llama_model` | `PrepackedWeightsContainer` across sessions | Multiple sessions share one `bp::Model` |

### Why Backpack Uses Eager Shader Compilation

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

---

## Stage 2: Context / Session Creation

| Aspect | llama.cpp | ORT Native | Backpack |
|--------|-----------|------------|----------|
| **API** | `llama_init_from_model(model, ctx_params)` | (merged into Session constructor) | `bp::Session::Create(model)` |
| **Context size** | Set via `ctx_params.n_ctx` | Implicit in tensor shapes — same Session handles different sizes | Implicit in tensor shapes (app-layer sets maxSeqLen) |
| **KV cache alloc** | Allocated inside context | Caller provides via `IoBinding` or `Run()` args | App-layer allocates based on maxSeqLen |
| **Multiple per model?** | Yes — each with different `n_ctx`, shared weights | One Session suffices for different KV sizes; multiple needed only for concurrent execution | Yes — shared weights + shaders, independent sessions |
| **KV cache type** | Configurable (`type_k`/`type_v`: f16, q8_0, q4_0) | Whatever dtype the model outputs | App-layer choice (currently f32 ONNX, f16 GGUF) |

---

## Stage 3: Inference Execution

| Aspect | llama.cpp | ORT Native | Backpack |
|--------|-----------|------------|----------|
| **API** | `llama_decode(ctx, batch)` | `session.Run(opts, names, vals, ...)` or `session.Run(opts, binding)` | `session.SetInput(...); session.Run()` |
| **Batch input** | `llama_batch` (tokens, positions, seq_ids, logits flags) | Named arrays or `IoBinding` | Named `SetInput`/`SetOutput` calls |
| **GPU I/O** | Automatic (backend scheduler) | CPU default; `Ort::IoBinding` needed for GPU stay | Always GPU (`bp::Tensor` = GPU buffer) |
| **Output location** | CPU (`llama_get_logits`) | CPU default, or GPU via `IoBinding` | GPU (`bp::Tensor` holds `GPUBuffer`) |
| **Session reuse** | Context reused across decode calls | Session reused; `IoBinding` cleared between runs | `Reset()` clears bindings between runs |

---

## KV Cache Management

| Aspect | llama.cpp | ORT Native | Backpack |
|--------|-----------|------------|----------|
| **Who manages KV?** | Runtime (opaque `llama_memory_t`) | Caller (explicit I/O tensors) | App-layer (explicit I/O tensors via `SetInput/SetOutput`) |
| **Resize at runtime?** | No — fixed at context creation | Yes — pass different-sized KV tensors each `Run()` | Yes — pass different-sized tensors (app-layer manages) |
| **Cache ops** | Rich: `seq_rm`, `seq_cp`, `seq_add`, `seq_div`, `seq_keep` | None — caller manages | None in runtime — app-layer implements as needed |
| **Shader recompilation on resize?** | N/A | May recompile (depends on `inputDependencies`) | **No** — all shapes are runtime params |

### Context Size Independence from Compiled Shaders

In Backpack, no shaders need recompilation when context size changes:

- All attention ops pass `total_seq`, `past_seq`, `max_seq` via params buffer
- All matmul ops pass `M, N, K` as runtime params
- Tile sizes are fixed constants independent of sequence length
- `HD` (head dimension) is patched at load time from model metadata

The caller controls context size by providing differently-sized KV cache tensors.
Same compiled model, same session, different buffer sizes — zero recompilation.

---

## Web Compatibility

| Aspect | llama.cpp | ORT Native | Backpack |
|--------|-----------|------------|----------|
| **Runs in browser?** | No | Yes (WASM as `onnxruntime-web`) | Not yet, but API designed for it |
| **WebGPU backend** | `ggml-webgpu` (experimental) | Mature WebGPU EP | Dawn native; Emscripten future |
| **Graph capture** | N/A | `enableGraphCapture` — record/replay dispatch list | Fast decode capture/replay |
| **Buffer caching** | N/A | Configurable: `disabled`, `lazyRelease`, `simple`, `bucket` | Power-of-2 bucket pool |

### Web Compatibility Rules

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

---

## Design Analysis

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

---

## Backpack's Design Position

Rather than fixing drawbacks in ORT or llama.cpp, Backpack exploits the simplicity
of targeting a **single backend (WebGPU)** to make different trade-offs:

| Area | ORT / llama.cpp | Backpack | Why |
|---|---|---|---|
| **Shader compilation** | ORT: lazy (multi-EP simplification). llama.cpp: N/A | **Eager at `Model::Load()`, async+parallel** | Single backend → can eagerly compile without multi-EP complexity |
| **Model/Session split** | ORT: merged (re-parse for concurrent use). llama.cpp: separate | **Separate** (`Model` + `Session`) | Lightweight sessions share compiled shaders + weights |
| **GPU tensor default** | ORT: CPU default, `IoBinding` for GPU | **Always GPU** (`bp::Tensor` = GPU buffer) | Single backend → no CPU/GPU/NPU memory abstraction needed |
| **Model introspection** | ORT: requires Session creation | **`Model::GetInputInfo/GetOutputInfo`** at load time | Graph metadata extracted at `Load()` without full optimization |
| **Config** | ORT: `{string: string}` EP options | **Type-safe** `ModelOptions` struct | Single backend → no generic key-value config needed |
| **LLM support** | ORT: model-agnostic + GenAI. llama.cpp: LLM-only | **Model-agnostic runtime + app-layer LLM context** | Same layering as ORT + GenAI, simplified by single backend |

### The Two-Layer Pattern (shared with ORT + GenAI)

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

---

## Summary of Design Choices

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

## Buffer Lifecycle and Session Scenarios

### Buffer Categories by Lifetime

| Category | Lifetime | Size | Release trigger |
|----------|----------|------|-----------------|
| **Weights** | Model lifetime | Large (entire model) | `model.Release()` |
| **KV cache** | Session / conversation | Large (scales with context) | `session.Release()` or context resize |
| **Fast decode bind groups** | Between prefills | Medium (~400 intermediate buffers held) | `ReleaseCaptured()` on shape change |
| **Tensor plan intermediates** | Between shape changes | Medium (~350-380 buffers) | `InvalidateWarmCaches()` or memory pressure |
| **Param pool** | Executor lifetime | Small (~80-320 KB) | Never (fixed, round-robin) |
| **Transient intermediates** | Single `Execute()` call | Variable | Released to pool at end of `Execute()` |
| **Caller tensors** (input/output) | Caller-managed | Variable | `tensor.Release()` |

### Scenario: Continue Conversation (EOS → New Prompt)

```
Session: pos=500, fast decode replaying
  → EOS at pos=600
  → User sends new prompt (10 tokens)
  → Keep KV cache (0-600), prefill new tokens (601-610), resume decode
```

Buffer operations:
1. KV cache stays alive (positions 0-600 are valid)
2. `ReleaseCaptured()` — free captured bind groups (prefill has different shape than decode)
3. `InvalidateWarmCaches()` — release tensor plan buffers to pool (shapes change)
4. Prefill new tokens — `Execute()` with M>1, rebuilds tensor plan
5. First decode after prefill — new `CaptureBegin()`, captures fresh bind groups
6. Subsequent decodes — zero-allocation replay

### Scenario: New Conversation (Reset)

KV cache buffers can be **reused** (same size) — just reset `pos=0`. No need to
reallocate. Only release captured bind groups and warm caches.

### Scenario: Context Size Change

KV cache must be **reallocated** (new size). Old buffers returned to pool. Fast
decode and warm caches invalidated. Compiled shaders unchanged (all shapes are
runtime params).

### Scenario: Multiple Concurrent Sessions

```
Session A: maxSeqLen=2048, pos=500, fast decode replaying
Session B: maxSeqLen=8192, pos=100, fast decode replaying
Both share one Model
```

What must be independent per session:
- KV cache tensors (different sizes)
- Fast decode captured bind groups (reference different KV buffers)
- Tensor plan (intermediate sizes may differ)
- Param pool (captured bind groups reference specific param buffers)
- Position counter and param buffer contents

What is shared:
- Weight buffers (immutable)
- Compiled pipelines (immutable)
- Buffer pool (single-threaded, safe)

### Scenario: Session Eviction Under Memory Pressure

For a serving scenario with many sessions, can free memory by evicting idle sessions:

| Priority | What to release | Memory freed | Cost to resume |
|----------|----------------|-------------|----------------|
| 1st | Tensor plan (`InvalidateWarmCaches()`) | ~350 intermediate buffers | One cold Execute() to rebuild plan |
| 2nd | Fast decode (`ReleaseCaptured()`) | ~400 captured buffers | One capture Execute() to re-record |
| 3rd | KV cache (`session.Release()`) | All KV buffers | Conversation lost, must re-prefill |

### Fast Decode Buffer Retention

During capture:
- Intermediate buffers are **not released** to pool — captured bind groups hold
  `wgpuBindGroupAddRef` references keeping GPU memory alive
- Param pool is expanded from 512→2048 per bucket to prevent round-robin wrapping

During replay:
- **Zero allocation** — `ReplayWrites()` updates existing buffers, `ReplayDispatches()`
  re-encodes dispatches with same bind groups
- Only transient WebGPU command encoder/buffer objects are created (not data buffers)

### Buffer Release Strategy Summary

| Event | Action | Memory freed |
|-------|--------|-------------|
| Decode step (replay) | Nothing | 0 |
| Abort mid-generation | Nothing (cooperative; state stays valid) | 0 |
| New user prompt (keep KV) | `ReleaseCaptured()` + `InvalidateWarmCaches()` | Captured + tensor plan → pool |
| New conversation | Above + reset pos=0 | Same (KV reused, not released) |
| Context size change | Above + reallocate KV | Old KV + captured + plan → pool |
| Session eviction | `session.Release()` | All: KV, captured, plan, params → pool |
| Model unload | `model.Release()` | Weights → pool → `flushPool()` |
| Device lost | Full teardown — all GPU resources invalid | Everything (forced) |
| Between models (image) | `model.Release()` + `flushPool()` | Everything destroyed |

### Scenario: Abort Mid-Generation

User cancels during prefill or decode. No abort mechanism exists today — generation
loops terminate only on EOS, max token count, or stream callback returning `false`.

**Buffer impact**: If abort happens between `Execute()` calls (the decode loop in
app code), all state is consistent — tensor plan and fast decode remain valid.
If abort happened *during* `Execute()` (e.g., via signal handler), state would be
undefined. The safe approach is cooperative cancellation via a `shouldStop` flag
checked between decode steps.

**Fast decode**: Stays valid if aborted between `Execute()` calls. The captured
bind groups are unaffected — next generation can resume replay from wherever it
left off, or the app can invalidate and re-prefill.

**Recommended design**:
- `Session::RequestStop()` sets atomic flag, checked at top of `Execute()`
- Returns `bp::Status::Cancelled` without modifying any state
- No mid-dispatch abort (GPU work completes; only the next step is skipped)

### Scenario: WebGPU Device Lost

GPU reset, driver crash, or browser tab backgrounded. Currently no
`wgpuDeviceSetDeviceLostCallback` is registered — the event would surface through
the uncaptured error callback as `[DAWN Lost]` on stderr, with no recovery.

**Buffer impact**: All GPU resources become invalid — buffers, pipelines, bind
groups. The buffer pool, pipeline cache, and per-session state are all stale.
No resource can be reused.

**Recovery**: Full teardown + reload required:
1. Release all sessions and models (C++ side cleanup only — GPU objects already gone)
2. Release device
3. Create new device (re-request adapter)
4. Reload model(s) (re-upload weights, recompile shaders)
5. Re-create sessions

**Recommended design**:
- Register `deviceLostCallbackInfo` in `GPUContext::init()` with a flag + reason
- `Device::IsLost()` lets app code detect and handle
- In browser: listen for `visibilitychange` to preemptively pause before device loss

### Future: Intra-Execute Buffer Reuse

Currently each intermediate tensor gets its own buffer. A liveness analysis could
assign the same buffer to non-overlapping intermediates (like a register allocator),
reducing peak memory during `Execute()` by ~30-50%. This is independent of the
multi-session refactor.

---

## Shared Resources Across Models

All models on the same `bp::Device` share a single `GPUContext`:

| Resource | Scope | Shared? | Safe? |
|----------|-------|---------|-------|
| **Buffer pool** (`pool_[]`) | Per-device | Yes — freed buffers reusable across models | Yes |
| **Pipeline cache** (`pipelines_[]`) | Per-device | Yes — Model B reuses Model A's compiled shaders | Yes — pipelines are immutable |
| **Named buffer map** (`buffers_[]`) | Per-device | Yes — flat `unordered_map<string, GPUBuffer>` | **No** — name collisions (see below) |
| **Weight buffers** | Per-model (`tensorStore_`) | No | Yes |
| **Intermediate buffers** | Per-model (`tensorStore_`) | No — cleaned up between `Execute()` | Yes |
| **Fast decode state** | Per-model (`GraphExecutor`) | No — but per-model, not per-session | See multi-session section |

### Named Buffer Collision (`buffers_`)

`GPUContext::buffers_` is a flat map keyed by name. `createBuffer(name, size)`
unconditionally overwrites any previous entry with the same name. Two models with
identically named ONNX initializers clobber each other's entries.

The GPU buffers themselves remain valid (callers hold `GPUBuffer` handles directly,
the map is mainly for diagnostics), but the map becomes inconsistent. Fix: either
namespace names per-model, or stop using the shared map for per-model tracking
(just use `tensorStore_` which is already per-model).

### Pipeline Sharing

If Model A already compiled `"matmul_f32"`, Model B gets it from cache instantly.
Compiled pipelines (`WGPUComputePipeline`, `WGPUShaderModule`, `WGPUBindGroupLayout`)
are immutable GPU objects — sharing is correct and beneficial.

### Sequential Multi-Model Pattern (Image App)

The image app demonstrates correct sequential buffer management:

```
Text Encoder: Load → Run → Release() → waitForQueue() → flushBufferPool()
DiT:          Load → Run diffusion loop (kept alive)
VAE:          Load → Run → _exit(0)
```

The `flushBufferPool()` between text encoder and DiT is critical — without it,
the pool holds freed buffers, and the much larger DiT might fail to allocate.

---

## Multi-Session and Fast Decode

### Current Limitation: Sessions Cannot Run Concurrently

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

### Fast Decode Buffer Safety

During fast decode capture:
- Intermediate buffers are deliberately **not released to pool** — the captured
  bind groups hold `wgpuBindGroupAddRef` references keeping GPU memory alive
- If those buffers were released and recycled, the captured bind group would still
  hold a valid GPU handle (AddRef prevents driver-level free), but pool bookkeeping
  would be inconsistent
- `ReleaseCaptured()` calls `wgpuBindGroupRelease()` which drops the ref, allowing
  GPU to reclaim the underlying buffer resources

### Required Architecture: Per-Session Execution State

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

### Buffer Release With Independent Sessions

| Action | What happens |
|--------|-------------|
| `session.Release()` | Release captured bind groups (if fast decode active), release intermediate buffers to pool, release tensor plan buffers |
| `session.Reset()` | Clear input/output bindings. Optionally invalidate warm caches. |
| `model.Release()` | Release weight buffers. Only safe after all sessions released. |
| Fast decode capture | Session A captures its own bind groups. Session B unaffected. |

---

## GPU Buffer Lifecycle

### Current Release Granularity

From finest to coarsest:

| Level | API | What happens |
|-------|-----|-------------|
| **Single buffer** | `gpu->releaseBuffer(buf)` | Returns to power-of-2 pool (27 buckets, 16B–1GB, max 256/bucket). Overflow destroys. |
| **Single tensor** | `tensor.Release()` | Drops C++ Impl. Should return GPU buffer to pool (fix needed). |
| **Session** | `session.Release()` | Drops execution context. Doesn't free tensors (they may be shared). |
| **Model** | `model.Release()` | Frees all weight buffers + flushes pool. Full cleanup. |
| **Pool flush** | `gpu->flushBufferPool()` | Destroys all pooled buffers. Use between pipeline stages. |
| **Device** | `device.Release()` | Teardown everything. |

### Usage Patterns

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

---

## Scenario Comparison: ORT vs llama.cpp vs Backpack

Three-way comparison of how each runtime handles the operational scenarios
identified in the Buffer Lifecycle section.

### 1. Concurrent Sessions

| | ORT WebGPU Native | llama.cpp | Backpack |
|---|---|---|---|
| **API support** | Multiple `Ort::Session` on same device via `context_id` ref-counting | Multiple `llama_context` from one `llama_model` (weights shared read-only) | `bp::Session` holds raw pointer to `Model::Impl` |
| **Concurrent GPU execution** | **No** — `ConcurrentRunSupported()` returns `false` for WebGPU EP. Runs serialized via `session_mutex_`. | **Not validated** — API allows it, but server uses single-context multi-slot pattern. No mutexes in core lib. GPU backends have global state. | **No** — single `GraphExecutor` per model. Shared mutable state (tensorStore, fastDecode, tensorPlan) would corrupt. |
| **Limitation** | WebGPU EP relies on global state; serialization is intentional | Thread safety of concurrent `llama_decode` across contexts is undocumented | Requires GraphExecutor split into GraphDef + ExecutionContext (Part 0 in plan) |

### 2. Continue Conversation (EOS → New Prompt)

| | ORT WebGPU Native | llama.cpp | Backpack |
|---|---|---|---|
| **Mechanism** | Caller passes previous `present.*` output KV tensors as next `past_key_values.*` inputs | Rich KV management: `llama_memory_seq_rm/cp/keep/add` for prefix reuse, position shifting, sequence forking | App-layer: KV cache kept, pos counter maintained, fast decode invalidated |
| **Prefix reuse** | Caller must implement (compare old vs new prompt, reuse matching KV prefix) | Built-in: server computes longest common prefix, removes only divergent tokens | App-layer responsibility — no prefix-aware KV management in runtime |
| **Context overflow** | Caller must handle (truncate or sliding window) | Built-in context shifting: `seq_rm` + `seq_add` to discard old tokens and shift positions | App-layer responsibility |
| **Limitation** | No helper API — caller handles all KV lifecycle logic | KV ops are LLM-specific and baked into runtime (can't use for non-LLM models) | No limitation per se — app-layer management is intentional (same as ORT) |

### 3. New Conversation (Reset)

| | ORT WebGPU Native | llama.cpp | Backpack |
|---|---|---|---|
| **Mechanism** | Caller passes zeroed/fresh KV tensors. No session reset API needed. | `llama_memory_clear()` or `llama_memory_seq_rm(mem, id, -1, -1)` per sequence | App-layer: reset pos=0, pass zeroed KV. `Session::Reset()` clears bindings (planned). |
| **KV buffer reuse** | Caller controls — can reuse same buffer with zeroed content | KV buffer is pre-allocated to `n_ctx`, always reused | KV buffer reused (same size), no reallocation |
| **Limitation** | None — stateless design makes reset trivial | None | No `Session::Reset()` API yet — must create new session or manage manually |

### 4. Context Size Change

| | ORT WebGPU Native | llama.cpp | Backpack |
|---|---|---|---|
| **Dynamic resize** | **Yes** — same `Ort::Session` accepts differently-sized KV tensors across runs (dynamic ONNX shapes) | **No** — `n_ctx` fixed at `llama_context` creation. Must destroy and recreate. | **Yes** — KV is app-managed I/O. Same session, different-sized tensors. Shaders are shape-independent. |
| **Graph capture** | Incompatible with variable sizes — graph capture requires static shapes | N/A (no graph capture in Vulkan/WebGPU backends) | Fast decode must be re-captured after resize (bind groups reference old buffers) |
| **Limitation** | Must choose: dynamic sizes OR graph capture, not both | Hard limitation — full context teardown + rebuild for size change | None — but fast decode re-capture has warmup cost |

### 5. Abort Mid-Generation

| | ORT WebGPU Native | llama.cpp | Backpack |
|---|---|---|---|
| **API** | `RunOptions::SetTerminate()` — first-class, thread-safe | `ggml_abort_callback` — polling callback checked between graph nodes | **None** — no cancellation mechanism |
| **GPU support** | Stops CPU-side dispatch; already-queued GPU work completes | **CPU only** — GPU backends (CUDA, Vulkan, Metal) run to completion. Comment: "currently works only with CPU execution" | N/A |
| **Granularity** | Between kernel submissions | Between graph nodes (CPU only) | N/A |
| **Return value** | Error status from `Run()` | `llama_decode` returns `2` (aborted). Processed ubatches remain in KV cache. | N/A |
| **Limitation** | GPU work already submitted still completes | No GPU cancellation. App must check between `llama_decode` calls. | **Gap**: No abort API. Must be added (cooperative `Session::RequestStop()` + atomic flag). |

### 6. WebGPU Device Lost / GPU Recovery

| | ORT WebGPU Native | llama.cpp | Backpack |
|---|---|---|---|
| **Detection** | Device-lost callback registered but **only logs**. Code has `// TODO: revise temporary device lost handling`. | **No handling** — Vulkan: `VK_CHECK` calls `exit(1)` on any error including device lost. CUDA: `GGML_ABORT()`. | **No callback registered**. Would surface as `[DAWN Lost]` on stderr via uncaptured error callback. |
| **Recovery** | None — must destroy `Ort::Session` and `Ort::Env`, recreate everything | None — process terminates on GPU error | None — full teardown + reload required |
| **Limitation** | Device-lost is detected but not recoverable. Marked as TODO. | Fatal — any GPU error kills the process | **Gap**: No device-lost callback. Need `deviceLostCallbackInfo` in `GPUContext::init()` + `Device::IsLost()`. |

### 7. Session Eviction / Memory Pressure

| | ORT WebGPU Native | llama.cpp | Backpack |
|---|---|---|---|
| **Memory tracking** | `AllocatorGetStats()` exists for CPU arena but **not for WebGPU `BufferManager`**. No GPU memory counters. | No API for memory queries. Server tracks slot states at app level. | No memory tracking. `totalAllocatedBytes` planned (Part B1). |
| **Eviction** | No eviction API. OOM propagates as exception. | Server-level LRU eviction: `llama_memory_seq_rm` frees idle slots. `llama_memory_defrag()` compacts fragmented KV. | Planned tiered eviction: warm caches → fast decode → KV cache → session (Part 0 architecture). |
| **KV defragmentation** | N/A (caller-managed KV) | `llama_memory_defrag()` compacts KV cache in-place | N/A (caller-managed KV; no internal KV fragmentation) |
| **Limitation** | No GPU memory visibility. Cannot make informed eviction decisions. | Pre-allocated KV to `n_ctx` — no dynamic growth/shrink. Defrag only within fixed allocation. | **Gap**: No memory tracking (planned). Eviction strategy designed but requires per-session ExecutionContext. |

### 8. Multi-Model on Same Device

| | ORT WebGPU Native | llama.cpp | Backpack |
|---|---|---|---|
| **Support** | Multiple `Ort::Session` share same device via `context_id`. Shared `BufferManager`. | Per-model GPU state — no global barriers. Multi-GPU with `tensor_split`. | Multiple models share `GPUContext` (pool, pipeline cache). |
| **VRAM coordination** | No budget or priority system between sessions | No coordination — models compete for VRAM through allocator | Buffer pool shared — models can reuse freed buffers. `flushBufferPool()` between stages. |
| **Limitation** | No memory budget per session/model | No API to query available VRAM before loading second model | **Bug**: `buffers_` name collision — two models with same ONNX initializer names clobber map entries. Pool and pipelines are safe. |

### Summary: Gaps by Runtime

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

**Backpack gaps** (tracked in implementation plan):
- No concurrent sessions — requires GraphExecutor split (Plan Part 0)
- No abort API — need `Session::RequestStop()` (Plan Step 9)
- No device-lost callback — need `deviceLostCallbackInfo` (Plan Step 10)
- No memory tracking — need `GPUContext` instrumentation (Plan Step B1)
- No `Session::Reset()` API (Plan Step 6)
- `Tensor::Release()` buffer leak (Plan Step 6)
- `buffers_` name collision across models (Plan Step 7)

### Scenarios Not Planned

The following scenarios were considered and explicitly excluded from the roadmap:

| Scenario | Why not supported |
|---|---|
| **Speculative decoding** | Requires two models (draft + verifier) running coordinated inference with token acceptance/rejection logic. Significant complexity for a niche optimization. Can revisit if multi-model concurrent execution is needed. |
| **Model hot-swap** | Loading a new model version while sessions are active. Would require ref-counting Model lifetime via `shared_ptr`. Current design requires releasing all sessions before releasing a model — this ordering constraint is simple and sufficient. |
| **Batch inference (batch size > 1)** | Processing multiple independent prompts in a single `Execute()` call. Requires reworking matmul dispatch dimensions and KV cache indexing for multi-sequence batching. Listed in `docs/todo.md` as future work. |
| **Device-lost recovery** | Detecting device loss is planned (Step 10), but automatic recovery (re-request adapter, re-upload weights, rebuild sessions) is not. Full teardown + reload is the expected recovery path — same as ORT and llama.cpp. |
