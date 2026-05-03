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
- Benchmark script: `benchmark_suite.py`
- Perf collection script: `apps/perf/run_perf.py`

## Build
```
runtime/build_now.bat   # builds everything
cp runtime/build/Release/backpack.dll runtime/build/  # copy DLL
```

## Baselines
- llama.cpp Vulkan: `D:\backup\x64\llamacpp\b8981\llama-bench.exe`
- ORT WebGPU: `D:\backup\x64\ort\20260430\model_benchmark.exe` (default EP = WebGPU)
- Models: `E:\workspace\project\agents\ai-models\`

## Performance targets
- GGUF: beat llama.cpp Vulkan (currently 0.50x)
- ONNX: beat ORT WebGPU (currently winning on ≤1.7B, losing on 7B+)
