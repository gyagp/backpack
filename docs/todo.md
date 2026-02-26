# WebGPU Backend - Next Steps

## Goal: Support multiple LLM architectures on WebGPU

### Current Status
- **GPT-2 124M** — 24.8 tok/s prefill, 62.6 tok/s decode on WebGPU
- **SmolLM2-135M** — 21.9 tok/s prefill, 22.2 tok/s decode on WebGPU (LLaMA architecture)
- Two distinct architectures running end-to-end via Triton→WGSL→Dawn
- 165 tests passing

### Completed Tasks

- [x] 1. Loop-based LayerNorm kernel — 3-pass (mean, variance, normalize) with BLOCK=128 chunks. Handles N=768 with max_diff=0.000000.
- [x] 2. Loop-based linear kernel — iterates over K in BLOCK_K=128 chunks, accumulates dot product. Handles any K dimension.
- [x] 3. Integrated loop-based kernels into GPT-2 pipeline — auto-selects between single-pass and loop-based kernels via `_needs_loop()`.
- [x] 4. Verified at GPT-2 scale with random weights — n_embd=768, n_head=12, 2 layers, max_diff=0.000001.
- [x] 5. Added tests — `test_loop_ln_256`, `test_loop_ln_768`, `test_loop_linear_768_to_2304`, `test_loop_linear_3072_to_768`, `test_gpt2_pipeline_768`.
- [x] 6. Real GPT-2 weight inference — downloaded 522MB safetensors, fixed Conv1D weight transpose (K,N→N,K), fixed 128MB SSBO limit for LM head via chunking. Output: coherent text at 0.3 tok/s.
- [x] 7. KV-cache — prefill processes full prompt, decode steps process single token with cached K/V. 2.4× speedup.
- [x] 8. Updated docs and committed.
- [x] 9. **Adapter limits** — query real hardware limits via `wgpuAdapterGetLimits()` instead of WebGPU conservative defaults. Copies full adapter limits to device descriptor (handles both max* and min* fields correctly).
- [x] 10. **Pipeline & buffer caching** — cache shader modules, compute pipelines, bind group layouts, pipeline layouts (keyed by WGSL SHA256 hash) and GPU buffers (keyed by name+size+usage). Only bind groups and command buffers created per dispatch. 10× speedup (0.3 → 3.0 tok/s).
- [x] 11. **Dynamic SSBO limit** — GPT-2 LM head now queries `runner.max_storage_buffer_binding_size` from adapter instead of hardcoded 128MB. D3D12 adapters typically support much larger SSBO, eliminating the need for LM head chunking.
- [x] 12. **Model caching** — safetensors file cached on disk (skip re-download if already exists). NPZ cache already existed.
- [x] 13. **GPU-resident weights (ML Drift §3.5)** — `GPUBuffer` class + `upload_to_gpu()` API. Pre-upload all 146 weight tensors (324 MB) to GPU at init. `_linear()` and `_layer_norm()` accept GPUBuffer inputs, skipping per-call `wgpuQueueWriteBuffer`. 6× decode speedup (3.0 → 17.5 tok/s). Inspired by ML Drift paper (arXiv:2505.00232) memory management §3.5.
- [x] 14. **GPU-resident output support** — `run_kernel(gpu_outputs=...)` returns `GPUBuffer` objects for specified outputs, keeping data on GPU without readback. Enables chaining dispatches without CPU round-trips (§3.6 operator fusion pattern). Infrastructure ready for future GPU→GPU intermediate chaining.
- [x] 15. **GPU→GPU intermediate chaining (ML Drift §3.6)** — Layer norm outputs stay on GPU and feed directly into subsequent linear projections without CPU readback. MLP intermediates (fc→GELU→proj) chain on GPU via new `gelu_kernel` and `add_kernel` Triton kernels. Residual adds run on GPU via `add_kernel`. Eliminates ~60 sync points per decode token.
- [x] 16. **Pre-upload LM head weights** — The wte weight matrix (50257×768, 147 MB) was re-uploaded from CPU every decode step. Pre-uploading to GPU eliminates this bottleneck, reducing LM head time from ~42 ms to ~1.5 ms. Also pre-uploads zero bias buffer for LM head. Overall: **2.3× speedup** (17.5 → 40–51 tok/s).
- [x] 17. **Cached GPU output buffer pool** — GPU-output buffers (from `gpu_outputs`) now use a toggle-cached pool instead of fresh allocations. Two buffers per (binding_name, size) alternate to prevent read-write aliasing while avoiding buffer creation/destruction overhead (96 buffer ops → 0 per token).
- [x] 18. **Prefill/decode performance reporting** — Separate timing for prefill and decode phases using `time.perf_counter()`. GPT-2: 24.8 tok/s prefill, 62.6 tok/s decode.
- [x] 19. **SmolLM2-135M support** — LLaMA-architecture model with new Triton kernels: `rms_norm_kernel`/`rms_norm_loop_kernel` (RMSNorm), `silu_mul_kernel` (fused SwiGLU activation). Supports RoPE (rotary position embeddings, CPU), GQA (grouped query attention: 9 Q heads / 3 KV heads), SwiGLU MLP (gate_proj + up_proj + SiLU + down_proj). Verified with random weights (max_diff=0.000000). 269 MB BF16 weights from HuggingFace, 513 MB GPU-resident (FP32). 21.9 tok/s prefill, 22.2 tok/s decode.

### Potential Future Optimizations (from ML Drift paper)
- **Batched command encoding** — record multiple dispatches into one command buffer, submit once. Currently ~9 separate submissions per block. Profile shows dispatch overhead is ~0.042 ms and readback is ~0.112 ms per call.
- **FP16 inference** — adapter supports ShaderF16. Half precision would halve weight memory and bandwidth. Requires WGSL shader changes.
- **Fused kernels** — residual_add + layer_norm fusion, QKV projection + RoPE (§3.6).
- **GPU KV cache** — keep KV cache on GPU to avoid CPU↔GPU transfers during attention. Currently KV cache lives on CPU for the decode-mode attention loop.
- **GPU decode attention** — move single-query attention to GPU (currently done on CPU with numpy).
- **Stage-aware kernels** — separate prefill (compute-bound, tiled matmul) and decode (memory-bound, fully connected) kernel variants (§3.7).
- **Weight quantization** — int8 / mixed 8/4/4 quantization for 2-4× memory bandwidth improvement (§3.7).

### Key Fixes
- **Sequential loop translation bug**: `process_block_terminator()` in `llvm_to_wgsl.py` didn't detect loop headers for consecutive loops, causing linearized WGSL output. Fixed with full loop detection + phi resolution.
- **Weight transpose**: GPT-2 Conv1D stores weights as (K_in, N_out); our linear kernel expects (N_out, K_in). Transpose applied during weight download.
- **128 MB SSBO limit**: WebGPU `maxStorageBufferBindingSize` = 128 MB. The wte embedding (50257×768×4B = 147 MB) exceeds this. Now uses adapter's actual limit instead of hardcoded 128 MB.
- **Adapter min* limits**: `minUniformBufferOffsetAlignment`, `minStorageBufferOffsetAlignment` must be ≥ adapter's supported minimum. Solved by copying full adapter limits struct via `ctypes.memmove`.
