# Backpack Development Guidelines

## Auto-push on milestones
When any of the following milestones is achieved, automatically push the code:
- A model that was previously failing now runs successfully
- Benchmark shows measurable improvement (>5% tok/s) on any model vs previous results
- A new architecture is supported (granite, nemotron, etc.)
- A critical bug fix (correctness issue, crash fix)

## Key paths
- Runtime code: `runtime/`
- Embedded shaders: `runtime/wgsl_shaders.h`
- External shaders: `runtime/kernels/`
- CLI app: `apps/llm/main.cpp`
- Common utilities: `apps/common/app_common.h`
- Perf data: `apps/perf/` (per-model JSON with all backends)
- Perf collection script: `apps/perf/run_perf.py`
- Benchmark script: `benchmark_suite.py`
- Build scripts: `scripts/`

## Build
```
scripts/build_dawn.bat       # rebuild Dawn DLL (first time or after Dawn update)
scripts/build_backpack.bat   # incremental build
scripts/clean_build_backpack.bat  # clean rebuild from scratch
scripts/quick_test.bat       # build + benchmark TinyLlama
```

## Baselines (competitive targets)
- **llama.cpp Vulkan**: `D:\backup\x64\llamacpp\b8981\llama-bench.exe`
- **ORT WebGPU native**: `D:\backup\x64\ort\20260430\model_benchmark.exe` (default EP = WebGPU)
- Models: `E:\workspace\project\agents\ai-models\`
- Perf data recorded in `apps/perf/*.json` — use `apps/perf/run_perf.py` to collect

## Performance targets
**Goal: Beat llama.cpp Vulkan AND ORT WebGPU native on ALL benchmarks.**

Both GGUF and ONNX paths must be optimized. Both D3D12 and Vulkan backends must be competitive.

### GGUF path (vs llama.cpp Vulkan)
- Current: ~0.46x on D3D12 (consistently across all model sizes)
- Target: ≥1.0x (match or beat llama.cpp Vulkan decode tok/s)
- Bottleneck: K-quant matmul bandwidth + Dawn dispatch overhead
- Key kernel: `q4k_matmul` in `runtime/wgsl_shaders.h`

### ONNX path (vs ORT WebGPU native)
- Current: winning on ≤1.7B models (up to 3.0x on Qwen3-0.6B), losing on 7B+ (0.36-0.61x)
- Target: ≥1.0x on ALL model sizes
- Bottleneck: Q4→Q8 repack doubles weight bandwidth; Dawn writeBuffer serialization
- Key optimization: native Q4 decode (skip Q8 intermediate), per-slot param buffers

### D3D12 backend
- Currently faster than Vulkan on this GPU/driver (RTX 5080)
- Primary optimization target

### Vulkan backend
- Currently ~15% slower than D3D12 (Dawn Vulkan overhead)
- Must also be competitive with llama.cpp's native Vulkan

### How to measure
1. Run `apps/perf/run_perf.py` to collect perf for all 56 models across all 4 backends
2. Compare `apps/perf/*.json` files — each has `backpack_gguf`, `backpack_onnx`, `llamacpp_vulkan`, `ort_webgpu` entries
3. Profile hotspots with `--profile` flag to identify kernel bottlenecks
4. After any optimization, re-run perf and compare against baseline numbers in the JSON files
