"""
Qwen2.5 inference on WebGPU via Triton.

Demonstrates Qwen-family LLM inference using Triton kernels compiled
to WGSL and executed on the GPU through Dawn's D3D12/Vulkan/Metal backend.

Qwen2.5 is a LLaMA-family model featuring:
  - RoPE (rotary position embeddings)
  - RMSNorm (root mean square normalization)
  - GQA (grouped query attention)
  - SwiGLU MLP (SiLU-gated linear unit)
  - Attention biases on Q, K, V projections (unlike SmolLM2/LLaMA)
  - Separate (untied) lm_head

All matrix multiplications, normalization, attention, and activation
operations run as WebGPU compute shaders — no CUDA required.

Usage:
    python python/examples/webgpu/qwen/model.py --verify
    python python/examples/webgpu/qwen/model.py --prompt "Hello"

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
    load_weights, download_weights, load_tokenizer, generate,
)


# Qwen2.5 model configs
QWEN_CONFIGS = {
    "0.5B": {
        "n_layer": 24, "n_head": 14, "n_kv_heads": 2,
        "n_embd": 896, "intermediate_size": 4864,
        "n_vocab": 151936, "rope_theta": 1000000.0,
        "rms_norm_eps": 1e-6, "head_dim": 64,
        "hf_repo": "Qwen/Qwen2.5-0.5B",
        "attention_bias": True,
        "tie_word_embeddings": True,
    },
    "1.5B": {
        "n_layer": 28, "n_head": 12, "n_kv_heads": 2,
        "n_embd": 1536, "intermediate_size": 8960,
        "n_vocab": 151936, "rope_theta": 1000000.0,
        "rms_norm_eps": 1e-6, "head_dim": 128,
        "hf_repo": "Qwen/Qwen2.5-1.5B",
        "attention_bias": True,
        "tie_word_embeddings": True,
    },
}


class QwenWebGPU(WebGPUModel):
    """Qwen2.5 inference on WebGPU via Triton kernels.

    Supports Qwen2.5-0.5B and Qwen2.5-1.5B.
    Key difference from SmolLM2: attention biases on Q/K/V projections.
    """

    def __init__(self, weights: Dict[str, np.ndarray],
                 n_layer: int = 24, n_head: int = 14,
                 n_kv_heads: int = 2, n_embd: int = 896,
                 intermediate_size: int = 4864,
                 n_vocab: int = 151936,
                 rope_theta: float = 1000000.0,
                 rms_norm_eps: float = 1e-6,
                 head_dim: int = 64,
                 attention_bias: bool = True,
                 tie_word_embeddings: bool = True):
        self.attention_bias = attention_bias
        self.tie_word_embeddings = tie_word_embeddings
        super().__init__(
            weights, n_layer=n_layer, n_head=n_head, n_embd=n_embd,
            n_vocab=n_vocab,
            n_kv_heads=n_kv_heads,
            intermediate_size=intermediate_size,
            head_dim=head_dim,
            rope_theta=rope_theta,
            rms_norm_eps=rms_norm_eps,
            k_dimensions={n_embd, intermediate_size},
        )
        self._upload_weights_to_gpu()

    def _compile_model_kernels(self):
        """Compile Qwen-specific kernels: RMSNorm, SiLU*mul."""
        self._compile_rms_norm()
        self._compile_silu_mul()

    def _upload_weights_to_gpu(self):
        """Pre-upload all Qwen weights to GPU memory."""
        E = self.n_embd
        HD = self.head_dim
        IM = self.intermediate_size

        for i in range(self.n_layer):
            pfx = f"layers.{i}."
            # RMSNorm weights
            self._upload_norm_weight(pfx + "input_layernorm.weight")
            self._upload_norm_weight(pfx + "post_attention_layernorm.weight")
            # Attention Q, K, V, O projections
            self._upload_linear_weight(
                pfx + "self_attn.q_proj.weight", E, E)
            self._upload_linear_weight(
                pfx + "self_attn.k_proj.weight", self.kv_dim, E)
            self._upload_linear_weight(
                pfx + "self_attn.v_proj.weight", self.kv_dim, E)
            self._upload_linear_weight(
                pfx + "self_attn.o_proj.weight", E, self.n_head * HD)
            # Attention biases
            if self.attention_bias:
                self._upload_bias(pfx + "self_attn.q_proj.bias")
                self._upload_bias(pfx + "self_attn.k_proj.bias")
                self._upload_bias(pfx + "self_attn.v_proj.bias")
            # SwiGLU MLP
            self._upload_linear_weight(
                pfx + "mlp.gate_proj.weight", IM, E)
            self._upload_linear_weight(
                pfx + "mlp.up_proj.weight", IM, E)
            self._upload_linear_weight(
                pfx + "mlp.down_proj.weight", E, IM)

        # Final RMSNorm
        self._upload_norm_weight("norm.weight")

        # Embedding
        self._upload_embedding_weight(
            "embed_tokens.weight", self.n_vocab, E)

        # LM head
        if not self.tie_word_embeddings:
            self._upload_embedding_weight(
                "lm_head.weight", self.n_vocab, E)

        # Zero bias buffers
        self._upload_zero_bias("zero_bias_E", E)
        n_head_dim = self.n_head * HD
        if n_head_dim != E:
            self._upload_zero_bias("zero_bias_QO", n_head_dim)
        self._upload_zero_bias("zero_bias_KV", self.kv_dim)
        self._upload_zero_bias("zero_bias_IM", IM)
        self._upload_zero_bias("zero_bias_V", self.n_vocab)

        self._print_gpu_weight_stats()

    def _attention_block(self, x, layer: int,
                         use_cache: bool = False,
                         positions: np.ndarray = None,
                         **kwargs):
        """GQA with RoPE and optional attention biases."""
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

        # Q, K, V projections with biases
        q_bias = (self._gpu_weights[pfx + "q_proj.bias"]
                  if self.attention_bias
                  else self._gpu_weights["zero_bias_E"])
        k_bias = (self._gpu_weights[pfx + "k_proj.bias"]
                  if self.attention_bias
                  else self._gpu_weights["zero_bias_KV"])
        v_bias = (self._gpu_weights[pfx + "v_proj.bias"]
                  if self.attention_bias
                  else self._gpu_weights["zero_bias_KV"])

        q = self._linear(
            x, self._gpu_weights[pfx + "q_proj.weight"],
            q_bias, E)
        k = self._linear(
            x, self._gpu_weights[pfx + "k_proj.weight"],
            k_bias, self.kv_dim)
        v = self._linear(
            x, self._gpu_weights[pfx + "v_proj.weight"],
            v_bias, self.kv_dim)

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
            attn_out = np.zeros((T, n_head, HD), dtype=np.float32)
            for h in range(n_head):
                kv_h = h // n_rep
                attn_out[:, h, :] = self._causal_attention(
                    Q[:, h, :].copy(), K_full[:, kv_h, :].copy(),
                    V_full[:, kv_h, :].copy())

        # O projection (no bias)
        attn_flat = attn_out.reshape(T, n_head * HD)
        o_bias = (self._gpu_weights.get("zero_bias_QO",
                                         self._gpu_weights["zero_bias_E"]))
        return self._linear(
            attn_flat,
            self._gpu_weights[pfx + "o_proj.weight"],
            o_bias, E)

    def _mlp_block(self, x, layer: int, gpu_out: bool = False):
        """SwiGLU MLP."""
        E = self.n_embd
        IM = self.intermediate_size
        pfx = f"layers.{layer}.mlp."

        gate = self._linear(
            x, self._gpu_weights[pfx + "gate_proj.weight"],
            self._gpu_weights["zero_bias_IM"], IM, gpu_out=True)
        up = self._linear(
            x, self._gpu_weights[pfx + "up_proj.weight"],
            self._gpu_weights["zero_bias_IM"], IM, gpu_out=True)
        h = self._silu_mul(gate, up, gpu_out=True)
        return self._linear(
            h, self._gpu_weights[pfx + "down_proj.weight"],
            self._gpu_weights["zero_bias_E"], E, gpu_out=gpu_out)

    def _transformer_block(self, x, layer: int,
                           use_cache: bool = False,
                           positions: np.ndarray = None, **kwargs):
        """Pre-norm transformer block."""
        pfx = f"layers.{layer}."

        rn1 = self._rms_norm(
            x, self._gpu_weights[pfx + "input_layernorm.weight"],
            gpu_out=True)
        attn = self._attention_block(rn1, layer, use_cache=use_cache,
                                     positions=positions)
        x = self._add(x, attn, gpu_out=True)

        rn2 = self._rms_norm(
            x, self._gpu_weights[pfx + "post_attention_layernorm.weight"],
            gpu_out=True)
        mlp = self._mlp_block(rn2, layer, gpu_out=True)
        x = self._add(x, mlp, gpu_out=True)
        return x

    def forward(self, token_ids: np.ndarray,
                use_cache: bool = False,
                pos_offset: int = 0) -> np.ndarray:
        """Run Qwen2.5 forward pass."""
        T = len(token_ids)
        wte = self.weights["embed_tokens.weight"]
        x = wte[token_ids]
        positions = np.arange(pos_offset, pos_offset + T, dtype=np.int32)

        for layer in range(self.n_layer):
            x = self._transformer_block(x, layer, use_cache=use_cache,
                                        positions=positions)

        x = self._rms_norm(x, self._gpu_weights["norm.weight"])

        # LM head
        lm_key = ("embed_tokens.weight" if self.tie_word_embeddings
                   else "lm_head.weight")
        logits = self._linear(
            x, self._gpu_weights[lm_key],
            self._gpu_weights["zero_bias_V"], self.n_vocab)
        return logits


# ---------------------------------------------------------------------------
# Weight downloading
# ---------------------------------------------------------------------------

def download_qwen_weights(model_size: str = "0.5B",
                          model_dir: str = None) -> Tuple[str, str]:
    """Download Qwen2.5 weights and tokenizer from HuggingFace."""
    config = QWEN_CONFIGS[model_size]
    hf_repo = config["hf_repo"]
    if model_dir is None:
        model_dir = os.path.join(_SCRIPT_DIR, "weights")

    def qwen_key_transform(key, arr):
        new_key = key.replace("model.", "")
        if new_key == "lm_head.weight" and config.get("tie_word_embeddings"):
            return None
        return new_key, arr

    npz_path, tokenizer_path = download_weights(
        hf_repo=hf_repo,
        model_dir=model_dir,
        key_transform=qwen_key_transform,
        download_tokenizer=True,
    )
    return npz_path, tokenizer_path


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------

def verify_with_random_weights():
    """Verify Qwen2.5 pipeline with small random weights."""
    print("=" * 60)
    print("Qwen2.5 WebGPU Pipeline Verification (random weights)")
    print("=" * 60)

    n_layer, n_head, n_kv_heads = 2, 4, 2
    n_embd, intermediate_size, n_vocab = 64, 128, 256
    head_dim = n_embd // n_head  # 16
    kv_dim = n_kv_heads * head_dim
    n_rep = n_head // n_kv_heads
    rope_theta = 10000.0
    eps = 1e-6
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
        # Attention biases
        weights[pfx + "self_attn.q_proj.bias"] = np.random.randn(
            n_embd).astype(np.float32) * 0.01
        weights[pfx + "self_attn.k_proj.bias"] = np.random.randn(
            kv_dim).astype(np.float32) * 0.01
        weights[pfx + "self_attn.v_proj.bias"] = np.random.randn(
            kv_dim).astype(np.float32) * 0.01
        # MLP
        weights[pfx + "mlp.gate_proj.weight"] = np.random.randn(
            intermediate_size, n_embd).astype(np.float32) * 0.02
        weights[pfx + "mlp.up_proj.weight"] = np.random.randn(
            intermediate_size, n_embd).astype(np.float32) * 0.02
        weights[pfx + "mlp.down_proj.weight"] = np.random.randn(
            n_embd, intermediate_size).astype(np.float32) * 0.02

    print(f"\nModel: {n_layer} layers, {n_head} Q heads, {n_kv_heads} KV heads, "
          f"{n_embd} embd, {intermediate_size} intermediate, {n_vocab} vocab")

    model = QwenWebGPU(
        weights, n_layer=n_layer, n_head=n_head, n_kv_heads=n_kv_heads,
        n_embd=n_embd, intermediate_size=intermediate_size,
        n_vocab=n_vocab, rope_theta=rope_theta, rms_norm_eps=eps,
        head_dim=head_dim, attention_bias=True, tie_word_embeddings=True)

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

        q = ln1 @ weights[pfx + "self_attn.q_proj.weight"].T + \
            weights[pfx + "self_attn.q_proj.bias"]
        k = ln1 @ weights[pfx + "self_attn.k_proj.weight"].T + \
            weights[pfx + "self_attn.k_proj.bias"]
        v = ln1 @ weights[pfx + "self_attn.v_proj.weight"].T + \
            weights[pfx + "self_attn.v_proj.bias"]

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
        mlp_out = (silu_gate * up_val) @ \
                  weights[pfx + "mlp.down_proj.weight"].T
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
        description="Qwen2.5 on WebGPU via Triton")
    parser.add_argument("--verify", action="store_true",
                        help="Verify pipeline with random weights")
    parser.add_argument("--model", type=str, default="0.5B",
                        choices=["0.5B", "1.5B"],
                        help="Model size")
    parser.add_argument("--prompt", type=str,
                        default="The future of AI is",
                        help="Prompt for text generation")
    parser.add_argument("--max-tokens", type=int, default=50)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--weights-dir", type=str, default=None)
    args = parser.parse_args()

    if args.verify:
        success = verify_with_random_weights()
        sys.exit(0 if success else 1)

    config = QWEN_CONFIGS[args.model]
    npz_path, tokenizer_path = download_qwen_weights(
        args.model, args.weights_dir)
    weights = load_weights(npz_path)
    print(f"Loaded {len(weights)} weight tensors")

    tokenizer = load_tokenizer(tokenizer_path)

    model = QwenWebGPU(
        weights,
        n_layer=config["n_layer"],
        n_head=config["n_head"],
        n_kv_heads=config["n_kv_heads"],
        n_embd=config["n_embd"],
        intermediate_size=config["intermediate_size"],
        n_vocab=config["n_vocab"],
        rope_theta=config["rope_theta"],
        rms_norm_eps=config["rms_norm_eps"],
        head_dim=config["head_dim"],
        attention_bias=config.get("attention_bias", True),
        tie_word_embeddings=config.get("tie_word_embeddings", True))
    print("Model created, kernels compiled")

    generate(model, args.prompt, tokenizer,
             max_tokens=args.max_tokens,
             temperature=args.temperature)


if __name__ == "__main__":
    main()
