"""
SmolLM2 inference on WebGPU via Triton.

Demonstrates LLaMA-architecture LLM inference using Triton kernels compiled
to WGSL and executed on the GPU through Dawn's D3D12/Vulkan/Metal backend.

SmolLM2 is a LLaMA-family model featuring:
  - RoPE (rotary position embeddings)
  - RMSNorm (root mean square normalization)
  - GQA (grouped query attention)
  - SwiGLU MLP (SiLU-gated linear unit)

All matrix multiplications, normalization, attention, and activation
operations run as WebGPU compute shaders — no CUDA required.

Usage:
    python python/examples/webgpu/smollm2/model.py
    python python/examples/webgpu/smollm2/model.py --model 360M --prompt "Hello"

Requirements:
    pip install requests tokenizers
    Dawn WebGPU library built at third_party/webgpu/dawn/build/
"""
import os
import sys
import time
from typing import Dict, Tuple

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_SCRIPT_DIR))

import numpy as np

from common.model_base import WebGPUModel
from common.utils import (
    _parse_safetensors, load_weights, download_weights,
    load_tokenizer, generate,
)


# SmolLM2 model configs
SMOLLM2_CONFIGS = {
    "135M": {
        "n_layer": 30, "n_head": 9, "n_kv_heads": 3,
        "n_embd": 576, "intermediate_size": 1536,
        "n_vocab": 49152, "rope_theta": 100000.0,
        "rms_norm_eps": 1e-5,
        "hf_repo": "HuggingFaceTB/SmolLM2-135M",
    },
    "360M": {
        "n_layer": 32, "n_head": 15, "n_kv_heads": 5,
        "n_embd": 960, "intermediate_size": 2560,
        "n_vocab": 49152, "rope_theta": 100000.0,
        "rms_norm_eps": 1e-5,
        "hf_repo": "HuggingFaceTB/SmolLM2-360M",
    },
}


class SmolLM2WebGPU(WebGPUModel):
    """SmolLM2 inference on WebGPU via Triton kernels.

    Supports SmolLM2-135M and SmolLM2-360M (LLaMA architecture).
    Features: RoPE, RMSNorm, GQA (grouped query attention), SwiGLU MLP.
    """

    def __init__(self, weights: Dict[str, np.ndarray],
                 n_layer: int = 30, n_head: int = 9,
                 n_kv_heads: int = 3, n_embd: int = 576,
                 intermediate_size: int = 1536,
                 n_vocab: int = 49152,
                 rope_theta: float = 100000.0,
                 rms_norm_eps: float = 1e-5):
        # SmolLM2 uses K dimensions: E and intermediate_size
        super().__init__(
            weights, n_layer=n_layer, n_head=n_head, n_embd=n_embd,
            n_vocab=n_vocab,
            n_kv_heads=n_kv_heads,
            intermediate_size=intermediate_size,
            rope_theta=rope_theta,
            rms_norm_eps=rms_norm_eps,
            k_dimensions={n_embd, intermediate_size},
        )
        self._upload_weights_to_gpu()

    def _compile_model_kernels(self):
        """Compile SmolLM2-specific kernels: RMSNorm, SiLU*mul."""
        self._compile_rms_norm()
        self._compile_silu_mul()

    def _upload_weights_to_gpu(self):
        """Pre-upload all SmolLM2 weights to GPU memory."""
        E = self.n_embd
        IM = self.intermediate_size

        # Per-layer weights
        for i in range(self.n_layer):
            pfx = f"layers.{i}."
            # RMSNorm weights
            self._upload_norm_weight(pfx + "input_layernorm.weight")
            self._upload_norm_weight(pfx + "post_attention_layernorm.weight")
            # Attention Q, K, V, O projections (no bias in LLaMA)
            self._upload_linear_weight(
                pfx + "self_attn.q_proj.weight", E, E)
            self._upload_linear_weight(
                pfx + "self_attn.k_proj.weight", self.kv_dim, E)
            self._upload_linear_weight(
                pfx + "self_attn.v_proj.weight", self.kv_dim, E)
            self._upload_linear_weight(
                pfx + "self_attn.o_proj.weight", E, E)
            # SwiGLU MLP
            self._upload_linear_weight(
                pfx + "mlp.gate_proj.weight", IM, E)
            self._upload_linear_weight(
                pfx + "mlp.up_proj.weight", IM, E)
            self._upload_linear_weight(
                pfx + "mlp.down_proj.weight", E, IM)

        # Final RMSNorm
        self._upload_norm_weight("norm.weight")

        # Embedding / LM head (tied weights)
        self._upload_embedding_weight(
            "embed_tokens.weight", self.n_vocab, E)

        # Zero bias buffers (LLaMA has no biases)
        self._upload_zero_bias("zero_bias_E", E)
        self._upload_zero_bias("zero_bias_KV", self.kv_dim)
        self._upload_zero_bias("zero_bias_IM", IM)
        self._upload_zero_bias("zero_bias_V", self.n_vocab)

        self._print_gpu_weight_stats()

    # -- Transformer blocks --

    def _attention_block(self, x, layer: int,
                         use_cache: bool = False,
                         positions: np.ndarray = None,
                         **kwargs):
        """GQA (Grouped Query Attention) with RoPE."""
        from common.model_base import GPUBuffer

        if isinstance(x, GPUBuffer):
            T = x.shape[0] if x.shape else 1
        else:
            T = x.shape[0]
        E = self.n_embd
        HD = self.head_dim
        n_head = self.n_head
        n_kv = self.n_kv_heads
        n_rep = self.n_rep
        pfx = f"layers.{layer}.self_attn."

        # Separate Q, K, V projections — readback for CPU attention
        q = self._linear(
            x, self._gpu_weights[pfx + "q_proj.weight"],
            self._gpu_weights["zero_bias_E"], E)
        k = self._linear(
            x, self._gpu_weights[pfx + "k_proj.weight"],
            self._gpu_weights["zero_bias_KV"], self.kv_dim)
        v = self._linear(
            x, self._gpu_weights[pfx + "v_proj.weight"],
            self._gpu_weights["zero_bias_KV"], self.kv_dim)

        # Reshape to heads
        Q = q.reshape(T, n_head, HD)
        K_new = k.reshape(T, n_kv, HD)
        V_new = v.reshape(T, n_kv, HD)

        # Apply RoPE
        if positions is None:
            positions = np.arange(T, dtype=np.int32)
        Q = self._apply_rope(Q, positions)
        K_new = self._apply_rope(K_new, positions)

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
                kv_h = h // n_rep
                scores = (Q[0, h, :] @ K_full[:, kv_h, :].T) * scale
                scores -= scores.max()
                attn = np.exp(scores)
                attn /= attn.sum()
                attn_out[0, h, :] = attn @ V_full[:, kv_h, :]
        else:
            # Prefill mode: full causal attention on GPU
            attn_out = np.zeros((T, n_head, HD), dtype=np.float32)
            for h in range(n_head):
                kv_h = h // n_rep
                attn_out[:, h, :] = self._causal_attention(
                    Q[:, h, :].copy(), K_full[:, kv_h, :].copy(),
                    V_full[:, kv_h, :].copy())

        # O projection
        attn_flat = attn_out.reshape(T, E)
        return self._linear(
            attn_flat, self._gpu_weights[pfx + "o_proj.weight"],
            self._gpu_weights["zero_bias_E"], E)

    def _mlp_block(self, x, layer: int,
                   gpu_out: bool = False):
        """SwiGLU MLP: down_proj(SiLU(gate_proj(x)) * up_proj(x)).

        All intermediates chain on GPU.
        """
        E = self.n_embd
        IM = self.intermediate_size
        pfx = f"layers.{layer}.mlp."

        # gate_proj: (T, E) → (T, IM)
        gate = self._linear(
            x, self._gpu_weights[pfx + "gate_proj.weight"],
            self._gpu_weights["zero_bias_IM"], IM,
            gpu_out=True)
        # up_proj: (T, E) → (T, IM)
        up = self._linear(
            x, self._gpu_weights[pfx + "up_proj.weight"],
            self._gpu_weights["zero_bias_IM"], IM,
            gpu_out=True)
        # SiLU(gate) * up on GPU
        h = self._silu_mul(gate, up, gpu_out=True)
        # down_proj: (T, IM) → (T, E)
        return self._linear(
            h, self._gpu_weights[pfx + "down_proj.weight"],
            self._gpu_weights["zero_bias_E"], E,
            gpu_out=gpu_out)

    def _transformer_block(self, x, layer: int,
                           use_cache: bool = False,
                           positions: np.ndarray = None,
                           **kwargs):
        """Pre-norm transformer block with full GPU chain."""
        pfx = f"layers.{layer}."

        # Attention sub-block
        rn1 = self._rms_norm(
            x, self._gpu_weights[pfx + "input_layernorm.weight"],
            gpu_out=True)
        attn = self._attention_block(rn1, layer, use_cache=use_cache,
                                     positions=positions)
        x = self._add(x, attn, gpu_out=True)

        # MLP sub-block
        rn2 = self._rms_norm(
            x, self._gpu_weights[pfx + "post_attention_layernorm.weight"],
            gpu_out=True)
        mlp = self._mlp_block(rn2, layer, gpu_out=True)
        x = self._add(x, mlp, gpu_out=True)
        return x

    def forward(self, token_ids: np.ndarray,
                use_cache: bool = False,
                pos_offset: int = 0) -> np.ndarray:
        """Run SmolLM2 forward pass.

        Args:
            token_ids: (T,) int32 token IDs
            use_cache: if True, use/update KV cache
            pos_offset: position offset for RoPE

        Returns:
            logits: (T, n_vocab) float32
        """
        T = len(token_ids)

        # Token embeddings (no position embeddings — RoPE applied later)
        wte = self.weights["embed_tokens.weight"]
        x = wte[token_ids]

        # Position indices for RoPE
        positions = np.arange(pos_offset, pos_offset + T, dtype=np.int32)

        # Transformer blocks
        for layer in range(self.n_layer):
            x = self._transformer_block(x, layer, use_cache=use_cache,
                                        positions=positions)

        # Final RMSNorm
        x = self._rms_norm(x, self._gpu_weights["norm.weight"])

        # LM head (tied embed_tokens)
        logits = self._linear(
            x, self._gpu_weights["embed_tokens.weight"],
            self._gpu_weights["zero_bias_V"],
            self.n_vocab)
        return logits


# ---------------------------------------------------------------------------
# Weight downloading
# ---------------------------------------------------------------------------

def download_smollm2_weights(model_size: str = "135M",
                              model_dir: str = None) -> Tuple[str, str]:
    """Download SmolLM2 weights and tokenizer from HuggingFace."""
    config = SMOLLM2_CONFIGS[model_size]
    hf_repo = config["hf_repo"]
    if model_dir is None:
        model_dir = os.path.join(_SCRIPT_DIR, "weights")

    def smollm2_key_transform(key, arr):
        """Strip 'model.' prefix, handle tied weights."""
        new_key = key.replace("model.", "")
        # lm_head is tied with embed_tokens — skip duplicate
        if new_key == "lm_head.weight":
            return None
        return new_key, arr

    npz_path, tokenizer_path = download_weights(
        hf_repo=hf_repo,
        model_dir=model_dir,
        key_transform=smollm2_key_transform,
        download_tokenizer=True,
    )
    return npz_path, tokenizer_path


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------

def verify_with_random_weights():
    """Verify SmolLM2 pipeline with small random weights (no download)."""
    print("=" * 60)
    print("SmolLM2 WebGPU Pipeline Verification (random weights)")
    print("=" * 60)

    n_layer, n_head, n_kv_heads = 2, 6, 2
    n_embd, intermediate_size, n_vocab = 96, 256, 512
    head_dim = n_embd // n_head
    kv_dim = n_kv_heads * head_dim
    n_rep = n_head // n_kv_heads
    rope_theta = 10000.0
    eps = 1e-5
    np.random.seed(42)

    weights = {}
    weights["embed_tokens.weight"] = np.random.randn(
        n_vocab, n_embd).astype(np.float32) * 0.02
    weights["norm.weight"] = np.ones(n_embd, dtype=np.float32)

    for i in range(n_layer):
        pfx = f"layers.{i}."
        weights[pfx + "input_layernorm.weight"] = np.ones(
            n_embd, dtype=np.float32)
        weights[pfx + "post_attention_layernorm.weight"] = np.ones(
            n_embd, dtype=np.float32)
        weights[pfx + "self_attn.q_proj.weight"] = np.random.randn(
            n_embd, n_embd).astype(np.float32) * 0.02
        weights[pfx + "self_attn.k_proj.weight"] = np.random.randn(
            kv_dim, n_embd).astype(np.float32) * 0.02
        weights[pfx + "self_attn.v_proj.weight"] = np.random.randn(
            kv_dim, n_embd).astype(np.float32) * 0.02
        weights[pfx + "self_attn.o_proj.weight"] = np.random.randn(
            n_embd, n_embd).astype(np.float32) * 0.02
        weights[pfx + "mlp.gate_proj.weight"] = np.random.randn(
            intermediate_size, n_embd).astype(np.float32) * 0.02
        weights[pfx + "mlp.up_proj.weight"] = np.random.randn(
            intermediate_size, n_embd).astype(np.float32) * 0.02
        weights[pfx + "mlp.down_proj.weight"] = np.random.randn(
            n_embd, intermediate_size).astype(np.float32) * 0.02

    print(f"\nModel: {n_layer} layers, {n_head} Q heads, {n_kv_heads} KV heads, "
          f"{n_embd} embd, {intermediate_size} intermediate, {n_vocab} vocab")

    model = SmolLM2WebGPU(
        weights, n_layer=n_layer, n_head=n_head, n_kv_heads=n_kv_heads,
        n_embd=n_embd, intermediate_size=intermediate_size,
        n_vocab=n_vocab, rope_theta=rope_theta, rms_norm_eps=eps)

    # Forward pass
    token_ids = np.array([1, 42, 100, 200], dtype=np.int32)
    T = len(token_ids)
    t0 = time.time()
    logits = model.forward(token_ids)
    t1 = time.time()

    print(f"\nForward pass: {token_ids} → shape {logits.shape} "
          f"in {(t1-t0)*1000:.0f}ms")

    # --- NumPy reference ---
    def _rope_numpy(x, positions, theta, hd):
        half = hd // 2
        inv_freq = 1.0 / (theta ** (
            np.arange(0, hd, 2, dtype=np.float32) / hd))
        angles = positions[:, None].astype(np.float32) * inv_freq[None, :]
        cos_v = np.cos(angles)[:, None, :]
        sin_v = np.sin(angles)[:, None, :]
        x1, x2 = x[..., :half], x[..., half:]
        out = np.empty_like(x)
        out[..., :half] = x1 * cos_v - x2 * sin_v
        out[..., half:] = x2 * cos_v + x1 * sin_v
        return out

    positions = np.arange(T, dtype=np.int32)
    x = weights["embed_tokens.weight"][token_ids]

    for layer in range(n_layer):
        pfx = f"layers.{layer}."
        rms = np.sqrt(np.mean(x ** 2, axis=-1, keepdims=True) + eps)
        ln1 = x / rms * weights[pfx + "input_layernorm.weight"]

        q = ln1 @ weights[pfx + "self_attn.q_proj.weight"].T
        k = ln1 @ weights[pfx + "self_attn.k_proj.weight"].T
        v = ln1 @ weights[pfx + "self_attn.v_proj.weight"].T

        Q = q.reshape(T, n_head, head_dim)
        K_ = k.reshape(T, n_kv_heads, head_dim)
        V_ = v.reshape(T, n_kv_heads, head_dim)

        Q = _rope_numpy(Q, positions, rope_theta, head_dim)
        K_ = _rope_numpy(K_, positions, rope_theta, head_dim)

        scale = 1.0 / np.sqrt(head_dim)
        attn_out = np.zeros_like(Q)
        for h in range(n_head):
            kv_h = h // n_rep
            scores = Q[:, h, :] @ K_[:, kv_h, :].T * scale
            mask = np.triu(np.ones((T, T), dtype=bool), k=1)
            scores[mask] = -1e9
            exp_s = np.exp(scores - scores.max(axis=-1, keepdims=True))
            attn = exp_s / exp_s.sum(axis=-1, keepdims=True)
            attn_out[:, h, :] = attn @ V_[:, kv_h, :]

        attn_flat = attn_out.reshape(T, n_embd)
        proj = attn_flat @ weights[pfx + "self_attn.o_proj.weight"].T
        x = x + proj

        rms = np.sqrt(np.mean(x ** 2, axis=-1, keepdims=True) + eps)
        ln2 = x / rms * weights[pfx + "post_attention_layernorm.weight"]

        gate = ln2 @ weights[pfx + "mlp.gate_proj.weight"].T
        up_val = ln2 @ weights[pfx + "mlp.up_proj.weight"].T
        silu_gate = gate / (1.0 + np.exp(-gate))
        mlp_h = silu_gate * up_val
        mlp_out = mlp_h @ weights[pfx + "mlp.down_proj.weight"].T
        x = x + mlp_out

    rms = np.sqrt(np.mean(x ** 2, axis=-1, keepdims=True) + eps)
    x = x / rms * weights["norm.weight"]
    logits_ref = x @ weights["embed_tokens.weight"].T

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
    parser = argparse.ArgumentParser(
        description="SmolLM2 on WebGPU via Triton")
    parser.add_argument("--verify", action="store_true",
                        help="Verify pipeline with random weights")
    parser.add_argument("--model", type=str, default="135M",
                        choices=["135M", "360M"],
                        help="Model size: 135M or 360M")
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
    config = SMOLLM2_CONFIGS[args.model]
    npz_path, tokenizer_path = download_smollm2_weights(
        args.model, args.weights_dir)
    weights = load_weights(npz_path)
    print(f"Loaded {len(weights)} weight tensors")

    tokenizer = load_tokenizer(tokenizer_path)

    # Create model
    model = SmolLM2WebGPU(
        weights,
        n_layer=config["n_layer"],
        n_head=config["n_head"],
        n_kv_heads=config["n_kv_heads"],
        n_embd=config["n_embd"],
        intermediate_size=config["intermediate_size"],
        n_vocab=config["n_vocab"],
        rope_theta=config["rope_theta"],
        rms_norm_eps=config["rms_norm_eps"])
    print("Model created, kernels compiled")

    # Generate
    generate(model, args.prompt, tokenizer,
             max_tokens=args.max_tokens,
             temperature=args.temperature)


if __name__ == "__main__":
    main()
