# WebGPU Backend — Roadmap

## Potential Future Work

- **Browser WebGPU** — Run the same WGSL shaders in Chrome/Firefox via navigator.gpu
- **Multi-batch inference** — Batch size > 1 for throughput-oriented workloads
- **Speculative decoding** — Draft model + verification for higher effective tok/s
- **Longer context** — Sliding window attention, RoPE scaling for >2K context
- **More models** — Gemma-3, Llama-3, Mistral, FLUX image generation
- **WASM deployment** — Emscripten + Dawn for browser-native C++ inference
