# AGENTS.md

## Project structure

- `runtime/` — C++ runtime (WebGPU inference engine, WGSL kernels, ops, profiling, tests)
- `app/` — C++ applications (LLM chat/benchmark, image generation)
- `compiler/` — Python compiler (Triton kernels, ONNX graph compilation)
- `third_party/` — External dependencies (Dawn, Triton)

## Rules

### Intermediate files go in `gitignore/`

All intermediate, generated, and temporary files must be placed under the `gitignore/` directory — never in the source tree. This includes:

- Build outputs (`gitignore/runtime/build/`)
- Downloaded models (`gitignore/models/`)
- Log files (`gitignore/logs/`)
- Debug dumps, traces, profiling outputs
- Any scratch or temp files created during development

The `gitignore/` directory is excluded from version control via `.gitignore`. Mirror the source tree structure inside it when appropriate (e.g., `gitignore/runtime/build/` for runtime builds).
