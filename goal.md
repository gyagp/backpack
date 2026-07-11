# Goal

Fix all the correctness issues of **Qwen3.5** and **Gemma 4** (GGUF format), and
optimize their perf as much as possible, comparing against **llama.cpp Vulkan**
on Windows (RTX 5080, D3D12 + Vulkan backends).

## Status snapshot (2026-07-12)

| Model | Correctness | Perf (backpack D3D12 decode vs llama.cpp Vulkan) |
|-------|-------------|--------------------------------------------------|
| Qwen3.5-2B | ✅ Fixed — "2 + 2 = 4" | 234 tok/s vs 350 = **0.67x** |
| Qwen3.5-4B | ✅ Fixed — "2+2 equals 4." | 120 tok/s vs 194 = **0.62x** |
| Gemma 3/4 | ⚠️ Runs end-to-end, generates tokens, NOT yet coherent | — |

### Gemma breakthrough + remaining work
The fatal blocker is FIXED: a uniform buffer was bound to a storage descriptor
slot in the sandwich-norm down projection, which made Dawn's D3D12 backend
remove the device (DXGI_ERROR_DEVICE_REMOVED). Since then Gemma 3 and 4 load,
build all layers, and generate tokens.

Fixes landed for Gemma:
- Device-removal (uniform→storage params) — the critical fix.
- Sandwich-norm detection for Gemma 2/3 (post_attention_norm.weight) — was
  completely missing, so those norms were never applied.
- Per-layer head dims (256 sliding / 512 global) for rope + attention kernels
  and params.
- SWA-vs-global classification from per-layer head dim.
- Q-only attention path for Gemma 4 shared-KV layers 15-34.
- Serial decode + serial prefill routing for Gemma (pooled/batched paths don't
  carry sandwich dispatches or per-layer params).
- VRAM: fp16 / tied-Q8 embedding gather, exact-size large-buffer allocation.

Still NOT coherent. Symptom: top logits after prefill are nearly flat (~19 with
sub-1.0 spread) — the hidden state loses discriminative signal through the
layers. Remaining suspects to investigate (need layer-by-layer activation
diff vs llama.cpp):
- QK-norm application detail in fused_qknorm_rope (dimension / ordering).
- Query pre-attention scaling (Gemma may use a fixed query_pre_attn_scalar
  rather than 1/sqrt(headDim) per layer).
- PLE (per-layer embedding) injection correctness for Gemma 4.
- Approximate Q-only shared-KV attention quality.
- Sandwich-norm numeric exactness (eps, +1 bias, buffer targets).
- rope convention / dimension_count vs head_dim.

Debugging approach that would close it: dump per-layer hidden state (xBuf) for
the same prompt from both backpack and llama.cpp and bisect the first layer
where they diverge. backpack has BP_DUMP_BUFFER_STATS / BP_DUMP_TOP_LOGITS;
llama.cpp would need --verbose or an eval-callback build.

## Environment

### Repo / build
- Repo root: `D:\workspace\project\backpack`
- Runtime source: `runtime/` (C++), shaders embedded in `runtime/wgsl_shaders.h`,
  external shaders in `runtime/kernels/`
- CLI app built at `gitignore\runtime\build\backpack_llm.exe`
- Build (Ninja + MSVC), from a shell with vcvarsall sourced:
  ```
  call "C:\Program Files\Microsoft Visual Studio\2022\Enterprise\VC\Auxiliary\Build\vcvarsall.bat" x64
  cmake --build D:\workspace\project\backpack\gitignore\runtime\build --config Release
  ```
  (`scripts/build_backpack.bat` hard-codes a stale `E:\...` path — build the
  `gitignore\runtime\build` dir directly.)

### Models (GGUF)
- `D:\workspace\project\agents\ai-models\Qwen3.5-2B-GGUF\Qwen3.5-2B-Q4_K_M.gguf`
- `D:\workspace\project\agents\ai-models\Qwen3.5-4B-GGUF\Qwen3.5-4B-Q4_K_M.gguf`
- `D:\workspace\project\agents\ai-models\gemma-4-E2B-it-qat-GGUF\gemma-4-E2B-it-qat-UD-Q4_K_XL.gguf`
- `D:\workspace\project\agents\ai-models\gemma-3-1b-it\gguf\gemma-3-1b-it.Q4_K_M.gguf` (simpler Gemma to debug — no shared-KV/PLE, uniform head dim)

### Running backpack
```
backpack_llm.exe --model <path> --format gguf --chat "What is 2+2?" --max-tokens 8 --temperature 0
backpack_llm.exe --model <path> --format gguf --prompt "Hello" --max-tokens 10 --temperature 0
backpack_llm.exe --model <path> --format gguf --backend d3d12 --benchmark --bench-prompt-len 128
```
Debug env vars: BP_DUMP_BUFFER_STATS=1, BP_DUMP_TOP_LOGITS=N, BP_TRACE_PIPELINE=1.

### llama.cpp baseline (Vulkan)
- `D:\backup\x64\llamacpp\b8981\llama-bench.exe -m <gguf> -p 128 -n 128 -ngl 99`
  (tg128 = decode baseline). CLI: `llama-completion.exe -m <gguf> --temp 0 -ngl 99 -p "<prompt>" -n <N>`.

### Perf targets
Match/beat llama.cpp Vulkan decode tok/s on all models. Qwen3.5 gap (~0.62-0.67x)
is bounded by llama.cpp's NV_coopmat2 (tensor-core) matmul; Dawn reports
subgroup_matrix=no on this driver so backpack falls back to DP4A/scalar.

