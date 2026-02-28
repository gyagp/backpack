"""
Gemma 3 inference on WebGPU via Triton.

Demonstrates Google's Gemma-family LLM inference using Triton kernels compiled
to WGSL and executed on the GPU through Dawn's D3D12/Vulkan/Metal backend.

Gemma 3 is a LLaMA-family model featuring:
  - RoPE (rotary position embeddings)
  - RMSNorm (root mean square normalization) with pre AND post norms
  - GQA (grouped query attention)
  - GeGLU MLP (GELU-gated linear unit, unlike SwiGLU in LLaMA)
  - Pre/post attention layernorms and pre/post feedforward layernorms

All matrix multiplications, normalization, attention, and activation
operations run as WebGPU compute shaders — no CUDA required.

Usage:
    python python/examples/webgpu/gemma/model.py --verify
    python python/examples/webgpu/gemma/model.py --prompt "Hello"

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


# Gemma model configs
GEMMA_CONFIGS = {
    "2B": {
        "n_layer": 26, "n_head": 8, "n_kv_heads": 4,
        "n_embd": 2304, "intermediate_size": 9216,
        "n_vocab": 256000, "rope_theta": 10000.0,
        "rms_norm_eps": 1e-6, "head_dim": 256,
        "hf_repo": "unsloth/gemma-3-2b",
        "query_pre_attn_scalar": 256,  # = head_dim
        "attn_logit_softcapping": 50.0,
        "final_logit_softcapping": 30.0,
    },
}


class GemmaWebGPU(WebGPUModel):
    """Gemma 3 inference on WebGPU via Triton kernels.

    Key differences from SmolLM2/LLaMA:
    - GeGLU activation (GELU * up) instead of SwiGLU (SiLU * up)
    - Pre AND post attention/feedforward RMSNorm (4 norms per layer)
    - query_pre_attn_scalar for attention score scaling
    """

    def __init__(self, weights: Dict[str, np.ndarray],
                 n_layer: int = 26, n_head: int = 8,
                 n_kv_heads: int = 4, n_embd: int = 2304,
                 intermediate_size: int = 9216,
                 n_vocab: int = 256128,
                 rope_theta: float = 10000.0,
                 rms_norm_eps: float = 1e-6,
                 head_dim: int = 256,
                 query_pre_attn_scalar: int = 256,
                 attn_logit_softcapping: float = 50.0,
                 final_logit_softcapping: float = 30.0):
        self.query_pre_attn_scalar = query_pre_attn_scalar
        self.attn_logit_softcapping = attn_logit_softcapping
        self.final_logit_softcapping = final_logit_softcapping
        super().__init__(
            weights, n_layer=n_layer, n_head=n_head, n_embd=n_embd,
            n_vocab=n_vocab,
            n_kv_heads=n_kv_heads,
            intermediate_size=intermediate_size,
            head_dim=head_dim,
            rope_theta=rope_theta,
            rms_norm_eps=rms_norm_eps,
            k_dimensions={n_embd, intermediate_size, n_head * head_dim},
        )
        self._upload_weights_to_gpu()

    def _compile_model_kernels(self):
        """Compile Gemma-specific kernels: RMSNorm, GeGLU."""
        self._compile_rms_norm()
        self._compile_gelu_mul()  # GeGLU instead of SwiGLU

    def _upload_norm_weight(self, name: str):
        """Upload norm weight with Gemma's +1 offset.

        Gemma RMSNorm: output = x/rms * (1 + weight)
        The stored weights are offsets from 1.0, so we add 1.0 before
        uploading to GPU so the standard RMSNorm kernel works correctly.
        """
        runner = self.cache.runner
        w = self.weights[name].astype(np.float32) + 1.0
        buf = runner.upload_to_gpu(w, name)
        self._gpu_weights[name] = buf
        return buf

    def _upload_weights_to_gpu(self):
        """Pre-upload all Gemma weights to GPU memory."""
        E = self.n_embd
        HD = self.head_dim
        IM = self.intermediate_size
        QO_dim = self.n_head * HD  # May differ from E

        for i in range(self.n_layer):
            pfx = f"layers.{i}."
            # 4 RMSNorm weights per layer (pre/post attn + pre/post ffn)
            self._upload_norm_weight(pfx + "input_layernorm.weight")
            self._upload_norm_weight(pfx + "post_attention_layernorm.weight")
            self._upload_norm_weight(pfx + "pre_feedforward_layernorm.weight")
            self._upload_norm_weight(pfx + "post_feedforward_layernorm.weight")
            # Attention QKV + O
            self._upload_linear_weight(
                pfx + "self_attn.q_proj.weight", QO_dim, E)
            self._upload_linear_weight(
                pfx + "self_attn.k_proj.weight", self.kv_dim, E)
            self._upload_linear_weight(
                pfx + "self_attn.v_proj.weight", self.kv_dim, E)
            self._upload_linear_weight(
                pfx + "self_attn.o_proj.weight", E, QO_dim)
            # GeGLU MLP
            self._upload_linear_weight(
                pfx + "mlp.gate_proj.weight", IM, E)
            self._upload_linear_weight(
                pfx + "mlp.up_proj.weight", IM, E)
            self._upload_linear_weight(
                pfx + "mlp.down_proj.weight", E, IM)

        # Final RMSNorm
        self._upload_norm_weight("norm.weight")

        # Embedding (tied with LM head)
        self._upload_embedding_weight(
            "embed_tokens.weight", self.n_vocab, E)

        # Zero biases
        self._upload_zero_bias("zero_bias_E", E)
        QO_dim = self.n_head * HD
        if QO_dim != E:
            self._upload_zero_bias("zero_bias_QO", QO_dim)
        self._upload_zero_bias("zero_bias_KV", self.kv_dim)
        self._upload_zero_bias("zero_bias_IM", IM)
        self._upload_zero_bias("zero_bias_V", self.n_vocab)

        self._print_gpu_weight_stats()

    def _attention_block(self, x, layer: int,
                         use_cache: bool = False,
                         positions: np.ndarray = None,
                         **kwargs):
        """GQA with RoPE and query_pre_attn_scalar-based scaling."""
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
        QO_dim = n_head * HD
        pfx = f"layers.{layer}.self_attn."

        q_bias = (self._gpu_weights.get("zero_bias_QO",
                                         self._gpu_weights["zero_bias_E"]))
        q = self._linear(
            x, self._gpu_weights[pfx + "q_proj.weight"],
            q_bias, QO_dim)
        k = self._linear(
            x, self._gpu_weights[pfx + "k_proj.weight"],
            self._gpu_weights["zero_bias_KV"], self.kv_dim)
        v = self._linear(
            x, self._gpu_weights[pfx + "v_proj.weight"],
            self._gpu_weights["zero_bias_KV"], self.kv_dim)

        Q = q.reshape(T, n_head, HD)
        K_new = k.reshape(T, n_kv, HD)
        V_new = v.reshape(T, n_kv, HD)

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
        # Gemma uses query_pre_attn_scalar for scaling
        scale = 1.0 / np.sqrt(float(self.query_pre_attn_scalar))

        if T == 1 and use_cache and T_total > 1:
            attn_out = np.zeros((1, n_head, HD), dtype=np.float32)
            for h in range(n_head):
                kv_h = h // n_rep
                scores = (Q[0, h, :] @ K_full[:, kv_h, :].T) * scale
                # gemma-3 attention logit soft-capping
                if self.attn_logit_softcapping:
                    cap = self.attn_logit_softcapping
                    scores = np.tanh(scores / cap) * cap
                scores -= scores.max()
                attn = np.exp(scores)
                attn /= attn.sum()
                attn_out[0, h, :] = attn @ V_full[:, kv_h, :]
        else:
            attn_out = np.zeros((T, n_head, HD), dtype=np.float32)
            for h in range(n_head):
                kv_h = h // n_rep
                # CPU-based attention with non-standard scale
                scores = Q[:, h, :] @ K_full[:, kv_h, :].T * scale
                # gemma-3 attention logit soft-capping
                if self.attn_logit_softcapping:
                    cap = self.attn_logit_softcapping
                    scores = np.tanh(scores / cap) * cap
                mask = np.triu(np.ones((T, T_total), dtype=bool), k=1)
                scores[mask[:T, :T_total]] = -1e9
                scores -= scores.max(axis=-1, keepdims=True)
                attn = np.exp(scores)
                attn /= attn.sum(axis=-1, keepdims=True)
                attn_out[:, h, :] = attn @ V_full[:, kv_h, :]

        attn_flat = attn_out.reshape(T, QO_dim)
        return self._linear(
            attn_flat, self._gpu_weights[pfx + "o_proj.weight"],
            self._gpu_weights["zero_bias_E"], E)

    def _mlp_block(self, x, layer: int, gpu_out: bool = False):
        """GeGLU MLP: down_proj(GELU(gate_proj(x)) * up_proj(x))."""
        E = self.n_embd
        IM = self.intermediate_size
        pfx = f"layers.{layer}.mlp."

        gate = self._linear(
            x, self._gpu_weights[pfx + "gate_proj.weight"],
            self._gpu_weights["zero_bias_IM"], IM)
        up = self._linear(
            x, self._gpu_weights[pfx + "up_proj.weight"],
            self._gpu_weights["zero_bias_IM"], IM)
        h = self._gelu_mul(gate, up)
        return self._linear(
            h, self._gpu_weights[pfx + "down_proj.weight"],
            self._gpu_weights["zero_bias_E"], E, gpu_out=gpu_out)

    def _transformer_block(self, x, layer: int,
                           use_cache: bool = False,
                           positions: np.ndarray = None, **kwargs):
        """Gemma transformer block with pre+post norms."""
        pfx = f"layers.{layer}."

        # Attention sub-block (pre + post norm)
        rn1 = self._rms_norm(
            x, self._gpu_weights[pfx + "input_layernorm.weight"])
        attn = self._attention_block(rn1, layer, use_cache=use_cache,
                                     positions=positions)
        attn = self._rms_norm(
            attn, self._gpu_weights[pfx + "post_attention_layernorm.weight"])
        x = self._add(x, attn)

        # MLP sub-block (pre + post norm)
        rn2 = self._rms_norm(
            x, self._gpu_weights[pfx + "pre_feedforward_layernorm.weight"])
        mlp = self._mlp_block(rn2, layer)
        mlp = self._rms_norm(
            mlp, self._gpu_weights[pfx + "post_feedforward_layernorm.weight"])
        x = self._add(x, mlp)
        return x

    def forward(self, token_ids: np.ndarray,
                use_cache: bool = False,
                pos_offset: int = 0) -> np.ndarray:
        """Run Gemma 3 forward pass."""
        T = len(token_ids)
        wte = self.weights["embed_tokens.weight"]
        # Gemma normalizes embeddings by sqrt(hidden_size)
        x = (wte[token_ids] * np.sqrt(float(self.n_embd))).astype(np.float32)
        positions = np.arange(pos_offset, pos_offset + T, dtype=np.int32)

        for layer in range(self.n_layer):
            x = self._transformer_block(x, layer, use_cache=use_cache,
                                        positions=positions)

        x = self._rms_norm(x, self._gpu_weights["norm.weight"])

        logits = self._linear(
            x, self._gpu_weights["embed_tokens.weight"],
            self._gpu_weights["zero_bias_V"], self.n_vocab)

        # gemma-3 final logit soft-capping
        if self.final_logit_softcapping:
            cap = self.final_logit_softcapping
            logits = np.tanh(logits / cap) * cap

        return logits


# ---------------------------------------------------------------------------
# Weight downloading
# ---------------------------------------------------------------------------

def download_gemma_weights(model_size: str = "2B",
                           model_dir: str = None) -> Tuple[str, str]:
    """Download Gemma weights and tokenizer from HuggingFace."""
    config = GEMMA_CONFIGS[model_size]
    hf_repo = config["hf_repo"]
    if model_dir is None:
        model_dir = os.path.join(_SCRIPT_DIR, "..", "..", "gitignore", "models", os.path.basename(_SCRIPT_DIR), "weights")

    def gemma_key_transform(key, arr):
        new_key = key.replace("model.", "")
        if new_key == "lm_head.weight":
            return None  # tied
        return new_key, arr

    npz_path, tokenizer_path = download_weights(
        hf_repo=hf_repo,
        model_dir=model_dir,
        safetensors_files=["model.safetensors"],
        key_transform=gemma_key_transform,
        download_tokenizer=True,
    )
    return npz_path, tokenizer_path


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------

def verify_with_random_weights():
    """Verify Gemma 3 pipeline with small random weights."""
    print("=" * 60)
    print("Gemma 3 WebGPU Pipeline Verification (random weights)")
    print("=" * 60)

    n_layer, n_head, n_kv_heads = 2, 4, 2
    n_embd = 64
    head_dim = n_embd // n_head  # 16
    intermediate_size = 128
    n_vocab = 256
    kv_dim = n_kv_heads * head_dim
    n_rep = n_head // n_kv_heads
    rope_theta = 10000.0
    eps = 1e-6
    query_pre_attn_scalar = head_dim
    np.random.seed(42)

    weights = {}
    weights["embed_tokens.weight"] = np.random.randn(
        n_vocab, n_embd).astype(np.float32) * 0.02
    # Gemma stores norm weights as offsets from 1.0;
    # 0.0 means "no change" (scaling by 1.0+0.0 = 1.0)
    weights["norm.weight"] = np.zeros(n_embd, dtype=np.float32)

    for i in range(n_layer):
        pfx = f"layers.{i}."
        weights[pfx + "input_layernorm.weight"] = np.zeros(
            n_embd, dtype=np.float32)
        weights[pfx + "post_attention_layernorm.weight"] = np.zeros(
            n_embd, dtype=np.float32)
        weights[pfx + "pre_feedforward_layernorm.weight"] = np.zeros(
            n_embd, dtype=np.float32)
        weights[pfx + "post_feedforward_layernorm.weight"] = np.zeros(
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

    model = GemmaWebGPU(
        weights, n_layer=n_layer, n_head=n_head, n_kv_heads=n_kv_heads,
        n_embd=n_embd, intermediate_size=intermediate_size,
        n_vocab=n_vocab, rope_theta=rope_theta, rms_norm_eps=eps,
        head_dim=head_dim, query_pre_attn_scalar=query_pre_attn_scalar)

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

    def _gelu_numpy(x):
        inner = 0.7978845608 * (x + 0.044715 * x**3)
        return 0.5 * x * (1.0 + np.tanh(inner))

    positions = np.arange(T, dtype=np.int32)
    x = (weights["embed_tokens.weight"][token_ids] * np.sqrt(
        float(n_embd))).astype(np.float32)

    for layer in range(n_layer):
        pfx = f"layers.{layer}."
        # Pre-attention RMSNorm (Gemma adds 1 to weight)
        rms = np.sqrt(np.mean(x ** 2, axis=-1, keepdims=True) + eps)
        ln1 = x / rms * (weights[pfx + "input_layernorm.weight"] + 1.0)

        q = ln1 @ weights[pfx + "self_attn.q_proj.weight"].T
        k = ln1 @ weights[pfx + "self_attn.k_proj.weight"].T
        v = ln1 @ weights[pfx + "self_attn.v_proj.weight"].T

        Q = q.reshape(T, n_head, head_dim)
        K_ = k.reshape(T, n_kv_heads, head_dim)
        V_ = v.reshape(T, n_kv_heads, head_dim)

        Q = _rope_numpy(Q, positions, rope_theta, head_dim)
        K_ = _rope_numpy(K_, positions, rope_theta, head_dim)

        scale = 1.0 / np.sqrt(float(query_pre_attn_scalar))
        attn_out = np.zeros_like(Q)
        for h in range(n_head):
            kv_h = h // n_rep
            scores = Q[:, h, :] @ K_[:, kv_h, :].T * scale
            # Attention logit soft-capping
            scores = np.tanh(scores / 50.0) * 50.0
            mask = np.triu(np.ones((T, T), dtype=bool), k=1)
            scores[mask] = -1e9
            exp_s = np.exp(scores - scores.max(axis=-1, keepdims=True))
            attn = exp_s / exp_s.sum(axis=-1, keepdims=True)
            attn_out[:, h, :] = attn @ V_[:, kv_h, :]

        attn_flat = attn_out.reshape(T, n_embd)
        proj = attn_flat @ weights[pfx + "self_attn.o_proj.weight"].T

        # Post-attention RMSNorm (Gemma adds 1 to weight)
        rms = np.sqrt(np.mean(proj ** 2, axis=-1, keepdims=True) + eps)
        proj = proj / rms * (weights[pfx + "post_attention_layernorm.weight"] + 1.0)
        x = x + proj

        # Pre-feedforward RMSNorm (Gemma adds 1 to weight)
        rms = np.sqrt(np.mean(x ** 2, axis=-1, keepdims=True) + eps)
        ln2 = x / rms * (weights[pfx + "pre_feedforward_layernorm.weight"] + 1.0)

        gate = ln2 @ weights[pfx + "mlp.gate_proj.weight"].T
        up_val = ln2 @ weights[pfx + "mlp.up_proj.weight"].T
        mlp_out = _gelu_numpy(gate) * up_val
        mlp_out = mlp_out @ weights[pfx + "mlp.down_proj.weight"].T

        # Post-feedforward RMSNorm (Gemma adds 1 to weight)
        rms = np.sqrt(np.mean(mlp_out ** 2, axis=-1, keepdims=True) + eps)
        mlp_out = mlp_out / rms * (weights[pfx + "post_feedforward_layernorm.weight"] + 1.0)
        x = x + mlp_out

    rms = np.sqrt(np.mean(x ** 2, axis=-1, keepdims=True) + eps)
    x = x / rms * (weights["norm.weight"] + 1.0)
    logits_ref = x @ weights["embed_tokens.weight"].T
    # Final logit soft-capping
    logits_ref = np.tanh(logits_ref / 30.0) * 30.0

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
        description="Gemma 3 on WebGPU via Triton")
    parser.add_argument("--verify", action="store_true",
                        help="Verify pipeline with random weights")
    parser.add_argument("--model", type=str, default="2B",
                        choices=["2B"],
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

    config = GEMMA_CONFIGS[args.model]
    npz_path, tokenizer_path = download_gemma_weights(
        args.model, args.weights_dir)
    weights = load_weights(npz_path)
    print(f"Loaded {len(weights)} weight tensors")

    tokenizer = load_tokenizer(tokenizer_path)

    model = GemmaWebGPU(
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
        query_pre_attn_scalar=config["query_pre_attn_scalar"],
        attn_logit_softcapping=config["attn_logit_softcapping"],
        final_logit_softcapping=config["final_logit_softcapping"])
    print("Model created, kernels compiled")

    generate(model, args.prompt, tokenizer,
             max_tokens=args.max_tokens,
             temperature=args.temperature)


if __name__ == "__main__":
    main()
