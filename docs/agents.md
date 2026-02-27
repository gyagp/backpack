# Coding Agent Instructions

Checklist and conventions for AI coding agents working on this project.

## Project Overview

- **What**: ML inference engine — Triton kernels compiled to WebGPU (WGSL), executed on Dawn
- **Language**: Python (models, kernels), WGSL (hand-tuned shaders), C++ (triton backend)
- **Goal**: Run real ML models fully on Triton WebGPU — no PyTorch at inference time

## Pre-Flight Checklist

- [ ] Read this file before making changes
- [ ] Understand which model you're modifying (check `models/<name>/model.py`)
- [ ] Run `--verify` on any model you change before and after
- [ ] Never break existing models — run all affected verify tests

## Model Implementation Checklist

When adding a new model:

- [ ] Create `models/<name>/model.py` with a class extending `WebGPUModel`
- [ ] Implement required methods: `_compile_model_kernels()`, `_upload_weights_to_gpu()`, `_attention_block()`, `_mlp_block()`, `_transformer_block()`, `forward()`
- [ ] Add `verify_with_random_weights()` with NumPy reference for correctness
- [ ] Add `--verify` flag in `main()` argument parser
- [ ] Create `convert_weights.py` if model needs weight conversion from HuggingFace
- [ ] Add weight download support via `common.utils.download_weights()`
- [ ] Test with real weights end-to-end
- [ ] Update the model status table in `README.md`
- [ ] Commit with descriptive message listing architecture and perf numbers

## Architecture Patterns

### Base class (`common/model_base.py`)
- `WebGPUModel` provides: linear, RMSNorm, LayerNorm, GELU, SiLU, attention, RoPE kernels
- Override `_compile_model_kernels()` to compile model-specific kernels (e.g., `_compile_rms_norm()`, `_compile_silu_mul()`)
- Use `gpu_out=True` to chain operations on GPU without CPU readback
- Use `_gpu_weights` dict for uploaded GPU buffers

### Weight naming
- Strip framework prefixes (e.g., `model.` → ``, `model.language_model.` → ``)
- Use HuggingFace weight names directly after stripping prefix
- Skip non-language weights (vision, MTP) via `key_transform` in `download_weights()`

### Quantization
- INT4 per-group quantization: `quantize_int4()` / `dequantize_int4()` with GROUP_SIZE=128
- Quantized keys: `{name}.q4`, `{name}.scales`, `{name}.zeros`, `{name}.K`
- Typical compression: ~4× (e.g., 55GB BF16 → 14GB INT4)

## Performance Optimization Checklist

- [ ] **GPU-chain ops**: Pass `gpu_out=True` to keep intermediates on GPU
- [ ] **Vectorize attention**: Use batched matmul (`Q_t @ K_t`) instead of per-head Python loops
- [ ] **CPU decode path**: For T=1 decode, use `_cpu_matmul()` to avoid GPU dispatch overhead (~5-11ms per dispatch)
- [ ] **Pre-allocate KV cache**: Write-in-place instead of `np.concatenate` per step
- [ ] **Pre-compute RoPE tables**: Compute cos/sin at init, not per forward pass
- [ ] **Vectorize norms**: Use numpy broadcasting instead of per-element loops
- [ ] **Fast GELU**: Use `x * sigmoid(1.702 * x)` instead of `tanh`-based (5× faster)
- [ ] **Window attention**: For vision models, partition → batched matmul → unpartition
- [ ] **Profile first**: Add `--bench` mode with per-layer timing before optimizing blindly

## Code Conventions

- All inference must work **without PyTorch** — only numpy + Triton WebGPU at runtime
- Weight conversion scripts may use PyTorch/transformers (offline, one-time)
- Use `time.perf_counter()` for timing (not `time.time()`)
- Conv1d/2d: implement via `as_strided` im2col + matmul (not Python loops)
- Mel spectrograms, tokenizers, image preprocessing: pure numpy
- Test audio/images can use system tools (Windows SAPI, PIL) for generation

## File Structure

```
models/<name>/
  model.py              # Main model — WebGPUModel subclass
  convert_weights.py    # Weight converter (HF → npz)
  weights/              # Downloaded/converted weights (gitignored)
```

## Git Conventions

- Commit after each working milestone (not WIP states)
- Include architecture summary and perf numbers in commit messages
- Example: `"Whisper: full Triton WebGPU inference with real weights\n\nEncoder: 960ms, Decoder: 27 tok/s"`

## Common Pitfalls

- **Don't use `np.repeat` for large expansions** — use reshape + broadcast multiply instead
- **Don't forget `gpu_out=True`** on intermediate ops — readback kills performance
- **Don't allocate in decode loops** — pre-allocate buffers at init
- **Multi-shard models**: pass `safetensors_files=[...]` list to `download_weights()`
- **Tied weights**: check if `lm_head` / `proj_out` is tied to `embed_tokens`
- **k_proj bias**: Whisper/some models have no bias on K projection — use zero_bias
- **Window attention padding**: pad spatial dims to be divisible by window_size, then crop output

## Running Models

```powershell
# Verify (random weights, no download)
python models/<name>/model.py --verify

# Convert weights (one-time, may need torch/transformers)
python models/<name>/convert_weights.py

# Run with real weights
python models/gpt2/model.py --prompt "Hello"
python models/whisper/model.py --audio file.wav
python models/sam3/model.py --image photo.jpg --point-x 128 --point-y 128

# Benchmark
python models/qwen3.5/model.py --bench
```
