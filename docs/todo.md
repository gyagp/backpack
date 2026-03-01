# WebGPU Backend — Status & Roadmap

## Current Status

- **Phi-4 mini (3.8B, INT4)** — 35ms TTFT, 102 tok/s decode (RTX 5080)
- **Qwen-3.5 (27B, INT4)** — 2.4s TTFT, 4.9 tok/s decode
- **Whisper Large V3 Turbo (809M)** — 8.0s encoder, 11.2s total
- **SDXL-Turbo** — 8.6s/step image generation
- **SmolLM2 (1.7B, INT4)** — 133ms TTFT, 208 tok/s decode
- **GPT-2 (124M)** — 60ms TTFT, 97.5 tok/s decode
- **C++ standalone binary** — Phi-4 at 101 tok/s with Dawn, no Python dependency
- Interactive HTML profiler with flamechart timeline
- 165 Triton WebGPU backend tests passing

## Potential Future Work

- **Browser WebGPU** — Run the same WGSL shaders in Chrome/Firefox via navigator.gpu
- **Multi-batch inference** — Batch size > 1 for throughput-oriented workloads
- **Speculative decoding** — Draft model + verification for higher effective tok/s
- **Longer context** — Sliding window attention, RoPE scaling for >2K context
- **More models** — Gemma-3, Llama-3, Mistral, FLUX image generation
- **WASM deployment** — Emscripten + Dawn for browser-native C++ inference
