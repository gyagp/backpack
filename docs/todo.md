# WebGPU Backend — Roadmap

## Potential Future Work

- **GGUF parity path (llama.cpp-compatible)**
	- Implement native WebGPU matmul kernels for `Q4_K`, `Q5_K`, `Q6_K` block layouts
	- Keep GGUF blocks compressed in host memory, dequantize in-kernel or tile-local
	- Match llama.cpp accumulation order and scaling math for output parity
- **Qwen-3.5 GGUF quality/perf milestone**
	- Validate `--use-gguf-fp16` quality against llama.cpp references
	- Add layerwise tensor parity checks (Q/K/V, MLP outputs) for GGUF-loaded runs
	- Add performance profile for fp16-converted GGUF and native K-quant kernels
- **Browser WebGPU** — Run the same WGSL shaders in Chrome/Firefox via navigator.gpu
- **Multi-batch inference** — Batch size > 1 for throughput-oriented workloads
- **Speculative decoding** — Draft model + verification for higher effective tok/s
- **Longer context** — Sliding window attention, RoPE scaling for >2K context
- **More models** — Gemma-3, Llama-3, Mistral, FLUX image generation
- **WASM deployment** — Emscripten + Dawn for browser-native C++ inference
