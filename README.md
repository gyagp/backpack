# Backpack

LLM inference engine powered by Triton kernels compiled to WebGPU (WGSL) and executed on Dawn.

## Architecture

```
Triton Kernel → TTIR → TTGIR → LLIR (SPIR-V triple) → WGSL → Dawn
```

Unlike llama.cpp which uses hand-written kernel implementations, Backpack writes kernels in Triton, then leverages [triton-windows](https://github.com/gyagp/triton-windows) to convert LLVM IR to WGSL (WebGPU Shading Language), and runs them on [Dawn](https://dawn.googlesource.com/dawn) (Google's WebGPU implementation).

## Supported Models

All models run fully on Triton WebGPU — no PyTorch at inference time.

| Model | Type | Params | Precision | Performance | Status |
|-------|------|--------|-----------|-------------|--------|
| **GPT-2** | LLM | 124M | FP32 | 60ms TTFT, 97.5 tok/s decode | Done |
| **Phi-4** | LLM | 3.8B | INT4 | 458ms TTFT, 124.6 tok/s decode | WIP |
| **Qwen-2.5** | LLM | 1.5B | INT4 | 148ms TTFT, 220 tok/s decode | Done |
| **Qwen-3.5** | LLM | 27B | INT4 | 2.4s TTFT, 4.9 tok/s decode |  |
| **Gemma-3** | LLM | 4B | FP16 | | gated model |
| **SmolLM-2** | LLM | 1.7B | INT4 | 133ms TTFT, 208 tok/s decode | Done |
| **GPT-OSS** | LLM (MoE) | 20B | MXFP4 | 1.2s TTFT, 38.4 tok/s decode | Done |
| **Whisper** | Speech-to-Text | 39M | FP16 | 160ms encoder, 30 tok/s decode, 0.9s total | Done |
| **SAM-3** | Segmentation | 31M | FP16 | 2.3s encoder + 33ms decoder (1024×1024) | gated model |
| **Flux-Klein** | Image Gen | 4B | FP16 | 5.6s/step (512×512), DiT dual-stream |  |
| **SD-Turbo** | Image Gen | ~5B | FP16 | 7.1s/step (512×512), 1-step distilled |  |
| **Z-Image-Turbo** | Image Gen | ~12B | FP16 | 24s/step (512×512), DiT + Qwen3 |  |
| **SDXL** | Image Gen | ~5B | FP16 | 7.3s/step (512×512), 14.5s with CFG |  |

## Project Structure
```
backpack/
├── models/           # Model implementations
│   ├── common/       # Shared kernels, base classes, utilities
│   ├── gpt-2/        # GPT-2 inference
│   ├── gemma-3/      # Google Gemma 3
│   ├── phi-4/        # Microsoft Phi-4
│   ├── qwen-2.5/     # Alibaba Qwen 2.5
│   └── ...           # More models
├── docs/             # Documentation
├── third_party/
│   ├── triton-windows/   # Triton compiler with WebGPU backend (webgpu branch)
│   └── dawn/             # Google's WebGPU implementation
```

## Third-Party Dependencies

- **triton-windows** ([webgpu branch](https://github.com/gyagp/triton-windows/tree/webgpu)): Triton compiler fork with WebGPU backend support, based on [triton-lang/triton-windows](https://github.com/triton-lang/triton-windows) `release/3.6.x-windows`
- **Dawn**: Google's native WebGPU implementation providing D3D12/Vulkan/Metal backends

## Getting Started

### Prerequisites

| Tool | Version | Notes |
|------|---------|-------|
| **OS** | Windows 10/11 | x86-64 only |
| **Python** | 3.13+ | Disable path length limit during install |
| **MSVC** | v143+ (VS 2022) | [Build Tools](https://aka.ms/vs/17/release/vs_BuildTools.exe) or Community edition |
| **CMake** | 3.20+ | `pip install cmake` or system install |
| **Ninja** | 1.11+ | `pip install ninja` |
| **Git** | any | Required for Dawn dependency fetching |

### Quick Start (recommended)

```powershell
git clone https://github.com/gyagp/backpack.git
cd backpack
python build.py          # Full setup: deps, clone, build Dawn, build triton
python build.py verify   # Verify the pipeline works
```

The build script automates everything — MSVC detection, repo cloning, Dawn build, triton build, and deployment. It skips steps that are already done, so it's safe to re-run.

Individual targets are available for partial builds:

```powershell
python build.py help           # Show all targets
python build.py dawn           # Rebuild Dawn only
python build.py triton         # Rebuild triton only
python build.py clean          # Remove all build artifacts
python build.py info           # Show build status
```

### Manual Steps (if not using build.py)

<details>
<summary>Click to expand manual setup steps</summary>

#### Step 1: Clone the Repository

```powershell
git clone https://github.com/gyagp/backpack.git
cd backpack
```

#### Step 2: Clone Third-Party Dependencies

```powershell
# Dawn (WebGPU runtime)
git clone https://dawn.googlesource.com/dawn third_party/dawn

# triton-windows (Triton compiler with WebGPU backend)
git clone -b webgpu https://github.com/gyagp/triton-windows.git third_party/triton-windows
```

#### Step 3: Load MSVC Environment

Open **"x64 Native Tools Command Prompt for VS 2022"**, or from PowerShell:

```powershell
cmd /c '"C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1 && set' | ForEach-Object {
    if ($_ -match '^([^=]+)=(.*)$') {
        [System.Environment]::SetEnvironmentVariable($matches[1], $matches[2], 'Process')
    }
}

# Verify: should print "Microsoft (R) C/C++ Optimizing Compiler ..."
cl 2>&1 | Select-Object -First 1
```

#### Step 4: Install Python Dependencies

```powershell
pip install setuptools wheel cmake ninja pybind11 lit
pip install numpy tiktoken requests
```

#### Step 5: Build Dawn

Dawn produces `webgpu_dawn.dll` (the WebGPU runtime) plus DXC shader compiler DLLs.

```powershell
cd third_party/dawn

# Configure (fetches third-party deps automatically)
cmake -S . -B build -G Ninja `
    -DCMAKE_BUILD_TYPE=Release `
    -DDAWN_FETCH_DEPENDENCIES=ON `
    -DDAWN_BUILD_MONOLITHIC_LIBRARY=SHARED `
    -DBUILD_SHARED_LIBS=OFF `
    -DDAWN_USE_BUILT_DXC=ON `
    -DDAWN_ENABLE_D3D12=ON `
    -DDAWN_ENABLE_D3D11=ON `
    -DDAWN_ENABLE_VULKAN=ON `
    -DDAWN_ENABLE_NULL=ON `
    -DDAWN_ENABLE_INSTALL=ON `
    -DDAWN_FORCE_SYSTEM_COMPONENT_LOAD=ON `
    -DDAWN_ENABLE_DESKTOP_GL=OFF `
    -DDAWN_ENABLE_OPENGLES=OFF `
    -DDAWN_ENABLE_SWIFTSHADER=OFF `
    -DDAWN_BUILD_SAMPLES=OFF `
    -DDAWN_BUILD_TESTS=OFF `
    -DDAWN_BUILD_BENCHMARKS=OFF `
    -DDAWN_BUILD_NODE_BINDINGS=OFF `
    -DDAWN_BUILD_PROTOBUF=OFF `
    -DDAWN_WERROR=OFF `
    -DTINT_BUILD_CMD_TOOLS=OFF `
    -DTINT_BUILD_TESTS=OFF

# Build (10-30 minutes)
cmake --build build --target webgpu_dawn --target dxcompiler
cmake --build build --target third_party/copy_dxil_dll

cd ../..
```

#### Step 6: Deploy Dawn DLLs

The `dawn_runner.py` searches for DLLs at `triton-windows/third_party/webgpu/dawn/build/`:

```powershell
$dawn_dll_dir = "third_party/triton-windows/third_party/webgpu/dawn/build"
New-Item -ItemType Directory -Path $dawn_dll_dir -Force | Out-Null

Copy-Item third_party/dawn/build/webgpu_dawn.dll $dawn_dll_dir/
Copy-Item third_party/dawn/build/dxcompiler.dll   $dawn_dll_dir/
Copy-Item third_party/dawn/build/dxil.dll         $dawn_dll_dir/
```

#### Step 7: Build & Install triton-windows

```powershell
$env:TRITON_BUILD_PROTON = "0"
$env:TRITON_BUILD_UT = "0"

cd third_party/triton-windows
pip install --no-build-isolation --verbose -e .
cd ../..
```

> **Note:** The first build downloads a pre-compiled LLVM (~1 GB) and compiles the Triton C++ core (~15-45 min). Subsequent builds are incremental.

#### Step 8: Verify

```powershell
# Quick pipeline verification with random weights (no download needed)
python models/gpt-2/model.py --verify

# Full GPT-2 text generation (downloads ~500 MB weights on first run)
python models/gpt-2/model.py --prompt "The future of AI is"
```

</details>

### Troubleshooting

| Problem | Solution |
|---------|----------|
| `No module named 'triton'` | Re-run Step 7, or set `TRITON_BACKENDS_IN_TREE=1` |
| `No CMAKE_CXX_COMPILER found` | Load MSVC environment (Step 3) |
| `DynamicLib.Open: dxil.dll Error 126` | Re-run Step 6 to deploy DXC DLLs |
| `Failed to create Dawn WebGPU device` | Check GPU drivers; Dawn needs D3D12 or Vulkan capable GPU |
| `0 compatible backends for target (webgpu)` | Ensure `third_party/webgpu/backend/` exists in triton-windows |
| `LLVM pre-compiled image is not available` | Use `release/3.6.x-windows` branch or `webgpu` branch from gyagp fork |

## Documentation

See [docs/triton-webgpu.md](docs/triton-webgpu.md) for architecture details and [docs/opt-guide.md](docs/opt-guide.md) for optimization guidance.
