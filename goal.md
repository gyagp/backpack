# Goal

Fix all the correctness issues of **Qwen3.5** and **Gemma 4** (GGUF format), and
optimize their Backpack D3D12 performance as much as possible, comparing
against **llama.cpp Vulkan** as the reference backend on Windows (RTX 5080).

## Status snapshot (2026-07-12)

| Model | Correctness | Perf (backpack D3D12 decode vs llama.cpp Vulkan) |
|-------|-------------|--------------------------------------------------|
| Qwen3.5-2B | ✅ Fixed — "2 + 2 = 4" | 234 tok/s vs 350 = **0.67x** |
| Qwen3.5-4B | ✅ Fixed — "2+2 equals 4." | 120 tok/s vs 194 = **0.62x** |
| Gemma 3 (1B) | ✅ **Fixed** — "The capital of France is Paris." (Paris top logit) | — |
| Gemma 4 (E2B) | ⚠️ Runs, coherent-ish but factual retrieval still off ("...is the a to in") | — |

### The Gemma breakthroughs
1. Device-removal fix: a uniform buffer bound to a storage descriptor slot in
   the sandwich-norm down projection removed the D3D12 device. Fixed.
2. Double +1 on norm weights: GGUF already bakes Gemma's (1+w); we were adding
   it again and exploding the residual. Removed the extra +1.
3. Tokenizer: Gemma 4 declares tokenizer.ggml.model="gemma4" (Gemma 3 uses
   "llama"); map any "gemma*" to SentencePiece. Fixed Gemma 4 token salad.
4. PLE: implemented project_per_layer_inputs + the post_norm injection RMSNorm.
5. **Q8 matmul K-tail (the Gemma 3 fix):** every Q8 matmul kernel truncated its
   K-loop to `K/256`, dropping the last 128 of E=1152 dims on Gemma 3's QKV /
   gate-up / LM-head. Model stayed fluent but lost facts. Rounding up to
   `(K+255)/256` + tail guards made Gemma 3 say "Paris". Qwen (aligned dims +
   separate qwen35 path) was never affected. See memory q8-matmul-k-tail-bug.

### Gemma 4 remaining work (E=1536 is 256-aligned, so NOT the K-tail bug)
Factual retrieval still broken (generic function-word logits). Suspects are the
Gemma-4-only features, all only approximately implemented:
- Shared-KV Q-only attention (layers 15-34, 20 of 35 layers)
- PLE injection exact scales / gating
- Variable per-layer head dims (256 SWA / 512 global) attention
- Double-wide MLP
Localize by dumping per-layer xBuf and bisecting the first divergent layer.


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
