# Backpack

Backpack is a C++ WebGPU inference runtime for LLM and multimodal workloads. The runtime owns the model execution code, manually written WGSL kernels, profiling, and tests. Applications live under `apps/`, and performance experiments live under `benchmarks/`.

## Architecture

```
GGUF / ONNX weights -> Backpack runtime ops -> WGSL kernels -> Dawn WebGPU
```

Backpack no longer depends on Triton. Kernels are authored and maintained directly under `runtime/kernels/`, embedded into C++ with `runtime/gen_wgsl_shaders.py`, and executed through Dawn.

## Project Structure

```
backpack/
├── runtime/       # C++ runtime, ops, WGSL kernels, profiling, tests
├── apps/          # C++ applications such as LLM chat, ASR, and image generation
├── benchmarks/    # Performance tuning scripts and comparisons
├── docs/          # Design notes, investigations, and optimization notes
├── scripts/       # Helper scripts
├── third_party/   # External dependencies such as Dawn
└── gitignore/     # Local generated files, build output, models, logs, traces
```

## Prerequisites

| Tool | Notes |
|------|-------|
| Windows 10/11 | x86-64 |
| Python 3.13+ | Used for provisioning scripts and shader embedding |
| MSVC v143+ | Visual Studio 2022 Build Tools or Community |
| CMake + Ninja | `pip install cmake ninja` is sufficient |
| Git | Required to clone Dawn |

## Build

Use the Python build entry point from a normal PowerShell. It loads MSVC if needed and writes runtime build output to `gitignore/runtime/build/`.

```powershell
python build.py setup
python build.py runtime
```

Common targets:

```powershell
python build.py deps
python build.py clone
python build.py dawn
python build.py runtime
python build.py info
python build.py clean
```

GNU Make users can call the same targets through `make`, for example `make runtime`.

## Kernels

All runtime WGSL sources live in `runtime/kernels/`. Regenerate embedded shader constants after editing kernels:

```powershell
python runtime/gen_wgsl_shaders.py
```

Generated and temporary files belong under `gitignore/`; downloaded models should use `gitignore/models/` or another explicitly external model cache.
