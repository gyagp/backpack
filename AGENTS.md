# AGENTS.md

## Project structure

- `runtime/` — C++ runtime (WebGPU inference engine, WGSL kernels, ops, profiling, tests)
- `apps/` — C++ applications (LLM chat/benchmark, image generation)
- `benchmarks/` — Performance tuning scripts and llama.cpp/ORT comparison harnesses
- `third_party/` — External dependencies (Dawn and other vendored libraries)

## Rules

### Target this device only

The active enablement and optimization target is **webgfx-104 / NVIDIA GeForce
RTX 5080**. Run model support, applications, correctness checks, profiling,
benchmarks, and reference measurements on this device only. Do not target other
devices, dispatch remote work, synchronize artifacts to other machines, or
require their results for acceptance. Historical multi-device documentation and
results do not expand this scope.

### Intermediate files go in `gitignore/`

All intermediate, generated, and temporary files must be placed under the `gitignore/` directory — never in the source tree. This includes:

- Build outputs (`gitignore/runtime/build/`)
- Downloaded models (`gitignore/models/`)
- Log files (`gitignore/logs/`)
- Debug dumps, traces, profiling outputs
- Any scratch or temp files created during development

The `gitignore/` directory is excluded from version control via `.gitignore`. Mirror the source tree structure inside it when appropriate (e.g., `gitignore/runtime/build/` for runtime builds).
