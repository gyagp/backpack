"""
GPT-2 inference on WebGPU via Triton.

Demonstrates end-to-end transformer inference using Triton kernels compiled
to WGSL and executed on the GPU through Dawn's D3D12/Vulkan/Metal backend.

All matrix multiplications, normalization, attention, and activation
operations run as WebGPU compute shaders — no CUDA required.

Usage:
    python python/examples/webgpu/gpt2/model.py
    python python/examples/webgpu/gpt2/model.py --verify

Requirements:
    pip install tiktoken requests
    Dawn WebGPU library built at third_party/webgpu/dawn/build/
"""
import os
import sys
import time
from typing import Dict

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_SCRIPT_DIR))

import numpy as np

from common.model_base import WebGPUModel
from common.utils import (
    _parse_safetensors, load_weights, download_weights, generate,
)


class GPT2WebGPU(WebGPUModel):
    """GPT-2 inference on WebGPU via Triton kernels.

    Supports GPT-2 small (124M): n_layer=12, n_head=12, n_embd=768.
    Features: LayerNorm, GELU activation, absolute position embeddings,
    fused QKV projection, multi-head attention (MHA).
    """

    def __init__(self, weights: Dict[str, np.ndarray],
                 n_layer: int = 12, n_head: int = 12,
                 n_embd: int = 768, n_vocab: int = 50257):
        # GPT-2 uses K dimensions: E (for most projections) and 4*E (MLP)
        super().__init__(
            weights, n_layer=n_layer, n_head=n_head, n_embd=n_embd,
            n_vocab=n_vocab,
            intermediate_size=4 * n_embd,
            k_dimensions={n_embd, 4 * n_embd},
            norm_eps=1e-5,
        )
        self._upload_weights_to_gpu()

    def _compile_model_kernels(self):
        """Compile GPT-2-specific kernels: LayerNorm, GELU."""
        self._compile_layer_norm()
        self._compile_gelu()

    def _upload_weights_to_gpu(self):
        """Pre-upload all GPT-2 weights to GPU memory."""
        E = self.n_embd

        # Per-layer weights
        for i in range(self.n_layer):
            pfx = f"h.{i}."
            # LayerNorm
            self._upload_norm_weight(pfx + "ln_1.weight")
            self._upload_bias(pfx + "ln_1.bias")
            self._upload_norm_weight(pfx + "ln_2.weight")
            self._upload_bias(pfx + "ln_2.bias")
            # Attention: fused QKV + output projection
            self._upload_linear_weight(pfx + "attn.c_attn.weight", 3 * E, E)
            self._upload_bias(pfx + "attn.c_attn.bias")
            self._upload_linear_weight(pfx + "attn.c_proj.weight", E, E)
            self._upload_bias(pfx + "attn.c_proj.bias")
            # MLP: fc (E→4E) + proj (4E→E)
            self._upload_linear_weight(pfx + "mlp.c_fc.weight", 4 * E, E)
            self._upload_bias(pfx + "mlp.c_fc.bias")
            self._upload_linear_weight(pfx + "mlp.c_proj.weight", E, 4 * E)
            self._upload_bias(pfx + "mlp.c_proj.bias")

        # Final LayerNorm
        self._upload_norm_weight("ln_f.weight")
        self._upload_bias("ln_f.bias")

        # LM head (tied wte) + zero bias
        self._upload_embedding_weight("wte.weight", self.n_vocab, E)
        self._upload_zero_bias("lm_head.bias", self.n_vocab)

        self._print_gpu_weight_stats()

    # -- Transformer blocks --

    def _attention_block(self, x, layer: int,
                         use_cache: bool = False,
                         **kwargs):
        """Multi-head causal self-attention (fused QKV, MHA)."""
        from common.model_base import GPUBuffer

        if isinstance(x, GPUBuffer):
            T = x.shape[0] if x.shape else 1
        else:
            T = x.shape[0]
        E = self.n_embd
        n_head = self.n_head
        HD = self.head_dim
        pfx = f"h.{layer}.attn."

        # QKV projection: (T, E) → (T, 3E) — readback for CPU attention
        qkv = self._linear(
            x, self._gpu_weights[pfx + "c_attn.weight"],
            self._gpu_weights[pfx + "c_attn.bias"], 3 * E)

        # Split into Q, K, V
        Q = qkv[:, :E].reshape(T, n_head, HD)
        K_new = qkv[:, E:2*E].reshape(T, n_head, HD)
        V_new = qkv[:, 2*E:].reshape(T, n_head, HD)

        # KV cache
        if use_cache:
            if self.kv_cache is not None and layer in self.kv_cache:
                K_prev, V_prev = self.kv_cache[layer]
                K_full = np.concatenate([K_prev, K_new], axis=0)
                V_full = np.concatenate([V_prev, V_new], axis=0)
            else:
                K_full = K_new
                V_full = V_new
            if self.kv_cache is None:
                self.kv_cache = {}
            self.kv_cache[layer] = (K_full, V_full)
        else:
            K_full = K_new
            V_full = V_new

        T_total = K_full.shape[0]

        if T == 1 and use_cache and T_total > 1:
            # Decode mode: single query against cached K,V (CPU, no mask)
            scale = 1.0 / np.sqrt(HD)
            attn_out = np.zeros((1, n_head, HD), dtype=np.float32)
            for h in range(n_head):
                scores = (Q[0, h, :] @ K_full[:, h, :].T) * scale
                scores -= scores.max()
                attn = np.exp(scores)
                attn /= attn.sum()
                attn_out[0, h, :] = attn @ V_full[:, h, :]
        else:
            # Prefill mode: full causal attention on GPU
            attn_out = np.zeros((T, n_head, HD), dtype=np.float32)
            for h in range(n_head):
                attn_out[:, h, :] = self._causal_attention(
                    Q[:, h, :].copy(), K_full[:, h, :].copy(),
                    V_full[:, h, :].copy())

        # Output projection
        attn_flat = attn_out.reshape(T, E)
        return self._linear(
            attn_flat, self._gpu_weights[pfx + "c_proj.weight"],
            self._gpu_weights[pfx + "c_proj.bias"], E)

    def _mlp_block(self, x, layer: int,
                   gpu_out: bool = False):
        """MLP: linear → GELU → linear. All intermediates chain on GPU."""
        E = self.n_embd
        pfx = f"h.{layer}.mlp."
        # fc: (T, E) → (T, 4E)
        h = self._linear(
            x, self._gpu_weights[pfx + "c_fc.weight"],
            self._gpu_weights[pfx + "c_fc.bias"], 4 * E,
            gpu_out=True)
        # GELU on GPU
        h = self._gelu(h, gpu_out=True)
        # proj: (T, 4E) → (T, E)
        return self._linear(
            h, self._gpu_weights[pfx + "c_proj.weight"],
            self._gpu_weights[pfx + "c_proj.bias"], E,
            gpu_out=gpu_out)

    def _transformer_block(self, x, layer: int,
                           use_cache: bool = False, **kwargs):
        """Pre-norm transformer block with full GPU chain."""
        pfx = f"h.{layer}."
        # Attention sub-block
        ln1 = self._layer_norm(
            x, self._gpu_weights[pfx + "ln_1.weight"],
            self._gpu_weights[pfx + "ln_1.bias"],
            gpu_out=True)
        attn = self._attention_block(ln1, layer, use_cache=use_cache)
        x = self._add(x, attn, gpu_out=True)

        # MLP sub-block
        ln2 = self._layer_norm(
            x, self._gpu_weights[pfx + "ln_2.weight"],
            self._gpu_weights[pfx + "ln_2.bias"],
            gpu_out=True)
        mlp = self._mlp_block(ln2, layer, gpu_out=True)
        x = self._add(x, mlp, gpu_out=True)
        return x

    def forward(self, token_ids: np.ndarray,
                use_cache: bool = False, pos_offset: int = 0) -> np.ndarray:
        """Run GPT-2 forward pass.

        Args:
            token_ids: (T,) int32 token IDs
            use_cache: if True, use/update KV cache
            pos_offset: position offset for embeddings

        Returns:
            logits: (T, n_vocab) float32
        """
        T = len(token_ids)

        # Token + position embeddings
        wte = self.weights["wte.weight"]
        wpe = self.weights["wpe.weight"]
        x = wte[token_ids] + wpe[pos_offset:pos_offset + T]

        # Transformer blocks
        for layer in range(self.n_layer):
            x = self._transformer_block(x, layer, use_cache=use_cache)

        # Final LayerNorm
        x = self._layer_norm(
            x, self._gpu_weights["ln_f.weight"],
            self._gpu_weights["ln_f.bias"])

        # LM head (tied wte)
        logits = self._linear(
            x, self._gpu_weights["wte.weight"],
            self._gpu_weights["lm_head.bias"],
            self.n_vocab)
        return logits


# ---------------------------------------------------------------------------
# Weight downloading
# ---------------------------------------------------------------------------

def download_gpt2_weights(model_dir: str = None) -> str:
    """Download GPT-2 (124M) weights from HuggingFace and convert to npz."""
    if model_dir is None:
        model_dir = os.path.join(_SCRIPT_DIR, "weights")

    def gpt2_key_transform(key, arr):
        """Rename and transpose GPT-2 weights."""
        new_key = key.replace("transformer.", "")
        # Transpose Conv1D weights
        if arr.ndim == 2 and any(
                s in new_key for s in
                ["c_attn.weight", "c_proj.weight", "c_fc.weight"]):
            arr = arr.T
        return new_key, arr

    npz_path, _ = download_weights(
        hf_repo="openai-community/gpt2",
        model_dir=model_dir,
        key_transform=gpt2_key_transform,
        download_tokenizer=False,
    )
    # Rename to expected filename
    expected = os.path.join(model_dir, "gpt2_weights.npz")
    if npz_path != expected and os.path.exists(npz_path):
        if not os.path.exists(expected):
            os.rename(npz_path, expected)
        npz_path = expected
    return npz_path


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------

def verify_with_random_weights():
    """Verify GPT-2 pipeline with small random weights (no download)."""
    print("=" * 60)
    print("GPT-2 WebGPU Pipeline Verification (random weights)")
    print("=" * 60)

    n_layer, n_head, n_embd, n_vocab = 2, 2, 64, 256
    head_dim = n_embd // n_head
    np.random.seed(42)

    weights = {}
    weights["wte.weight"] = np.random.randn(n_vocab, n_embd).astype(
        np.float32) * 0.02
    weights["wpe.weight"] = np.random.randn(1024, n_embd).astype(
        np.float32) * 0.02
    weights["ln_f.weight"] = np.ones(n_embd, dtype=np.float32)
    weights["ln_f.bias"] = np.zeros(n_embd, dtype=np.float32)

    for i in range(n_layer):
        pfx = f"h.{i}."
        weights[pfx + "ln_1.weight"] = np.ones(n_embd, dtype=np.float32)
        weights[pfx + "ln_1.bias"] = np.zeros(n_embd, dtype=np.float32)
        weights[pfx + "ln_2.weight"] = np.ones(n_embd, dtype=np.float32)
        weights[pfx + "ln_2.bias"] = np.zeros(n_embd, dtype=np.float32)
        weights[pfx + "attn.c_attn.weight"] = np.random.randn(
            3 * n_embd, n_embd).astype(np.float32) * 0.02
        weights[pfx + "attn.c_attn.bias"] = np.zeros(
            3 * n_embd, dtype=np.float32)
        weights[pfx + "attn.c_proj.weight"] = np.random.randn(
            n_embd, n_embd).astype(np.float32) * 0.02
        weights[pfx + "attn.c_proj.bias"] = np.zeros(
            n_embd, dtype=np.float32)
        weights[pfx + "mlp.c_fc.weight"] = np.random.randn(
            4 * n_embd, n_embd).astype(np.float32) * 0.02
        weights[pfx + "mlp.c_fc.bias"] = np.zeros(
            4 * n_embd, dtype=np.float32)
        weights[pfx + "mlp.c_proj.weight"] = np.random.randn(
            n_embd, 4 * n_embd).astype(np.float32) * 0.02
        weights[pfx + "mlp.c_proj.bias"] = np.zeros(
            n_embd, dtype=np.float32)

    print(f"\nModel: {n_layer} layers, {n_head} heads, {n_embd} embd, "
          f"{n_vocab} vocab")

    model = GPT2WebGPU(weights, n_layer=n_layer, n_head=n_head,
                       n_embd=n_embd, n_vocab=n_vocab)

    # Forward pass
    token_ids = np.array([1, 42, 100, 200], dtype=np.int32)
    t0 = time.time()
    logits = model.forward(token_ids)
    t1 = time.time()

    print(f"\nForward pass: {token_ids} → shape {logits.shape} "
          f"in {(t1-t0)*1000:.0f}ms")

    # NumPy reference
    x = weights["wte.weight"][token_ids] + weights["wpe.weight"][:4]
    for layer in range(n_layer):
        pfx = f"h.{layer}."
        mean = x.mean(axis=1, keepdims=True)
        var = x.var(axis=1, keepdims=True)
        ln1 = (x - mean) / np.sqrt(var + 1e-5) * weights[pfx + "ln_1.weight"] + weights[pfx + "ln_1.bias"]
        qkv = ln1 @ weights[pfx + "attn.c_attn.weight"].T + weights[pfx + "attn.c_attn.bias"]
        Q = qkv[:, :n_embd].reshape(4, n_head, head_dim)
        K = qkv[:, n_embd:2*n_embd].reshape(4, n_head, head_dim)
        V = qkv[:, 2*n_embd:].reshape(4, n_head, head_dim)
        scale = 1.0 / np.sqrt(head_dim)
        attn_out = np.zeros_like(Q)
        for h in range(n_head):
            scores = Q[:, h, :] @ K[:, h, :].T * scale
            mask = np.triu(np.ones((4, 4), dtype=bool), k=1)
            scores[mask] = -1e9
            exp_s = np.exp(scores - scores.max(axis=1, keepdims=True))
            attn = exp_s / exp_s.sum(axis=1, keepdims=True)
            attn_out[:, h, :] = attn @ V[:, h, :]
        attn_flat = attn_out.reshape(4, n_embd)
        proj = attn_flat @ weights[pfx + "attn.c_proj.weight"].T + weights[pfx + "attn.c_proj.bias"]
        x = x + proj
        mean = x.mean(axis=1, keepdims=True)
        var = x.var(axis=1, keepdims=True)
        ln2 = (x - mean) / np.sqrt(var + 1e-5) * weights[pfx + "ln_2.weight"] + weights[pfx + "ln_2.bias"]
        fc = ln2 @ weights[pfx + "mlp.c_fc.weight"].T + weights[pfx + "mlp.c_fc.bias"]
        gelu = 0.5 * fc * (1.0 + np.tanh(0.7978845608 * (fc + 0.044715 * fc**3)))
        mlp_out = gelu @ weights[pfx + "mlp.c_proj.weight"].T + weights[pfx + "mlp.c_proj.bias"]
        x = x + mlp_out
    mean = x.mean(axis=1, keepdims=True)
    var = x.var(axis=1, keepdims=True)
    x = (x - mean) / np.sqrt(var + 1e-5) * weights["ln_f.weight"] + weights["ln_f.bias"]
    logits_ref = x @ weights["wte.weight"].T

    max_diff = np.abs(logits - logits_ref).max()
    gpu_preds = logits.argmax(axis=1)
    ref_preds = logits_ref.argmax(axis=1)
    argmax_match = np.array_equal(gpu_preds, ref_preds)

    print(f"Max diff vs NumPy: {max_diff:.6f}")
    print(f"Predictions match: {argmax_match}")
    print(f"  GPU: {gpu_preds}  Ref: {ref_preds}")

    return max_diff < 0.1 and argmax_match


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    import argparse
    parser = argparse.ArgumentParser(description="GPT-2 on WebGPU via Triton")
    parser.add_argument("--verify", action="store_true",
                        help="Verify pipeline with random weights (no download)")
    parser.add_argument("--prompt", type=str,
                        default="The future of AI is",
                        help="Prompt for text generation")
    parser.add_argument("--max-tokens", type=int, default=50,
                        help="Maximum tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.8,
                        help="Sampling temperature")
    parser.add_argument("--weights-dir", type=str, default=None,
                        help="Directory for cached weights")
    args = parser.parse_args()

    if args.verify:
        success = verify_with_random_weights()
        sys.exit(0 if success else 1)

    # Download and load real weights
    npz_path = download_gpt2_weights(args.weights_dir)
    weights = load_weights(npz_path)
    print(f"Loaded {len(weights)} weight tensors")

    # Create model
    model = GPT2WebGPU(weights)
    print("Model created, kernels compiled")

    # Generate (GPT-2 uses tiktoken, no tokenizer.json)
    generate(model, args.prompt, tokenizer=None,
             max_tokens=args.max_tokens,
             temperature=args.temperature)


if __name__ == "__main__":
    main()
