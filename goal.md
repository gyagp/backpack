# Goal

Fix all the correctness issues of **Qwen3.5** and **Gemma 4** (GGUF format), and
optimize their perf as much as possible, comparing against **llama.cpp Vulkan**
on Windows (RTX 5080, D3D12 + Vulkan backends).

## Status snapshot (2026-07-12)

| Model | Correctness | Perf (backpack D3D12 decode vs llama.cpp Vulkan) |
|-------|-------------|--------------------------------------------------|
| Qwen3.5-2B | ✅ Fixed — "2 + 2 = 4" | 234 tok/s vs 350 = **0.67x** |
| Qwen3.5-4B | ✅ Fixed — "2+2 equals 4." | 120 tok/s vs 194 = **0.62x** |
| Gemma 3 (1B) | ✅ Coherent in prompt mode ("Paris is the capital of the world, and to the people,"); chat mode still off | — |
| Gemma 4 (E2B) | ⚠️ Runs end-to-end, stable/bounded numerics, output not coherent | — |

### The two Gemma breakthroughs
1. Device-removal fix: a uniform buffer was bound to a storage descriptor slot
   in the sandwich-norm down projection → Dawn removed the D3D12 device. Fixed.
2. Double +1 on norm weights: Gemma's RMSNorm is x*(1+w), but llama.cpp's GGUF
   conversion already bakes the +1 into the stored weights (they sit ~1-6, not
   ~0). We were adding +1 again, over-scaling every norm and exploding the
   residual (xBuf hit ~38000). Removing the extra +1 made Gemma 3 coherent.

### Gemma 4 remaining work (why it's still not coherent)
Gemma 3 works and shares the base path, so the remaining issues are the
Gemma-4-specific features:
- PLE (per-layer embeddings, Gemma 3n MatFormer): novel architecture, our
  injection (inp_gate / proj / per_layer_token_embd) is unverified against the
  reference and is likely the dominant remaining issue (Gemma 4 E2B needs it).
- Shared-KV Q-only attention (layers 15-34): implemented approximately (Q-proj
  + fused-rope with scratch KV write + attention against source cache); needs
  validation.
- Gemma 3 chat mode: SPM tokenization of the <start_of_turn>/<end_of_turn>
  template still yields wrong tokens (prompt mode is fine).

Continue by dumping per-layer xBuf for Gemma 4 vs llama.cpp and bisecting the
first divergent layer; check PLE math against the Gemma 3n reference.

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

