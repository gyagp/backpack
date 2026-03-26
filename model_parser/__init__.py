"""
Shared model file parsing for compiler and runtime stages.

Provides format-agnostic model parsing:
  - GGUF: metadata, config extraction, tokenizer
  - ONNX: config.json + tokenizer.json
  - Auto-detection: resolve_model() finds and identifies format

Both compiler/ and runtimes/ import from here.
"""
