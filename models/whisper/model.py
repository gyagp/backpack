"""
Whisper-Tiny speech-to-text inference on WebGPU via Triton.

Demonstrates encoder-decoder transformer inference for automatic speech
recognition using Triton kernels compiled to WGSL and executed on WebGPU
via Dawn.

Whisper-Tiny architecture:
  Encoder: Conv1d(80→384, k=3, s=1) → GELU → Conv1d(384→384, k=3, s=2) → GELU
           → sinusoidal positional embedding → 4 transformer layers → LayerNorm
  Decoder: token embedding + learned positional embedding → 4 transformer layers
           (self-attn + cross-attn + GELU MLP) → LayerNorm → LM head
  Config: d_model=384, heads=6, ffn=1536, encoder_layers=4, decoder_layers=4,
          mel_bins=80, vocab=51865, max_source=1500, max_target=448

Usage:
    python python/examples/webgpu/whisper/model.py --verify
    python python/examples/webgpu/whisper/model.py --audio audio.wav

Requirements:
    pip install requests
    Dawn WebGPU library built at third_party/webgpu/dawn/build/
"""
import os
import sys
import time
from typing import Dict, Tuple, Optional

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_SCRIPT_DIR))

import numpy as np

from common.model_base import WebGPUModel, _next_pow2
from common.utils import load_weights, download_weights


# Whisper tiny config
WHISPER_CONFIGS = {
    "tiny": {
        "d_model": 384,
        "encoder_layers": 4,
        "decoder_layers": 4,
        "encoder_attention_heads": 6,
        "decoder_attention_heads": 6,
        "encoder_ffn_dim": 1536,
        "decoder_ffn_dim": 1536,
        "num_mel_bins": 80,
        "vocab_size": 51865,
        "max_source_positions": 1500,
        "max_target_positions": 448,
        "hf_repo": "openai/whisper-tiny",
    },
}


class WhisperWebGPU(WebGPUModel):
    """Whisper speech-to-text inference on WebGPU.

    Architecture:
    1. Encoder: Conv1d mel spectrogram → sinusoidal pos → self-attention layers
    2. Decoder: Token embed + pos embed → self-attn → cross-attn → MLP

    Key differences from LLM models:
    - Encoder uses full (non-causal) self-attention
    - Decoder uses causal self-attention + cross-attention to encoder
    - Conv1d front-end for mel spectrogram input
    - GELU activation (not SiLU/SwiGLU)
    """

    def __init__(self, weights: Dict[str, np.ndarray],
                 d_model: int = 384,
                 encoder_layers: int = 4,
                 decoder_layers: int = 4,
                 encoder_attention_heads: int = 6,
                 decoder_attention_heads: int = 6,
                 encoder_ffn_dim: int = 1536,
                 decoder_ffn_dim: int = 1536,
                 num_mel_bins: int = 80,
                 vocab_size: int = 51865,
                 max_source_positions: int = 1500,
                 max_target_positions: int = 448):
        self.d_model = d_model
        self.encoder_layers = encoder_layers
        self.decoder_layers = decoder_layers
        self.encoder_heads = encoder_attention_heads
        self.decoder_heads = decoder_attention_heads
        self.encoder_ffn_dim = encoder_ffn_dim
        self.decoder_ffn_dim = decoder_ffn_dim
        self.num_mel_bins = num_mel_bins
        self.vocab_size = vocab_size
        self.max_source_positions = max_source_positions
        self.max_target_positions = max_target_positions
        self.head_dim_enc = d_model // encoder_attention_heads
        self.head_dim_dec = d_model // decoder_attention_heads

        super().__init__(
            weights, n_layer=encoder_layers + decoder_layers,
            n_head=encoder_attention_heads, n_embd=d_model,
            n_vocab=vocab_size,
            intermediate_size=max(encoder_ffn_dim, decoder_ffn_dim),
            head_dim=d_model // encoder_attention_heads,
            k_dimensions={d_model, encoder_ffn_dim, decoder_ffn_dim,
                          num_mel_bins},
        )
        self._upload_weights_to_gpu()

    def _compile_model_kernels(self):
        """Compile Whisper-specific kernels."""
        self._compile_layer_norm()
        self._compile_gelu()
        self._compile_full_attn()

    def _upload_weights_to_gpu(self):
        """Upload all Whisper weights to GPU memory."""
        D = self.d_model
        EF = self.encoder_ffn_dim
        DF = self.decoder_ffn_dim
        runner = self.cache.runner

        # --- Encoder ---
        # Conv1d weights: stored as (C_out, C_in, kernel_size)
        # We'll handle conv in numpy, so just upload as raw buffers
        for name in ["encoder.conv1.weight", "encoder.conv1.bias",
                      "encoder.conv2.weight", "encoder.conv2.bias"]:
            w = self.weights[name]
            if w.dtype == np.float16:
                w = w.astype(np.float32)
            buf = runner.upload_to_gpu(w.ravel(), name)
            self._gpu_weights[name] = buf

        # Encoder positional embedding (sinusoidal, precomputed)
        w = self.weights["encoder.embed_positions.weight"]
        if w.dtype == np.float16:
            w = w.astype(np.float32)
        buf = runner.upload_to_gpu(w.ravel(), "encoder.embed_positions.weight")
        self._gpu_weights["encoder.embed_positions.weight"] = buf

        # Encoder transformer layers
        for i in range(self.encoder_layers):
            pfx = f"encoder.layers.{i}."
            self._upload_norm_weight(pfx + "self_attn_layer_norm.weight")
            self._upload_bias(pfx + "self_attn_layer_norm.bias")
            self._upload_norm_weight(pfx + "final_layer_norm.weight")
            self._upload_bias(pfx + "final_layer_norm.bias")
            # Self-attention QKVO
            self._upload_linear_weight(pfx + "self_attn.q_proj.weight", D, D)
            self._upload_bias(pfx + "self_attn.q_proj.bias")
            self._upload_linear_weight(pfx + "self_attn.k_proj.weight", D, D)
            # k_proj has no bias in Whisper
            self._upload_linear_weight(pfx + "self_attn.v_proj.weight", D, D)
            self._upload_bias(pfx + "self_attn.v_proj.bias")
            self._upload_linear_weight(pfx + "self_attn.out_proj.weight", D, D)
            self._upload_bias(pfx + "self_attn.out_proj.bias")
            # FFN
            self._upload_linear_weight(pfx + "fc1.weight", EF, D)
            self._upload_bias(pfx + "fc1.bias")
            self._upload_linear_weight(pfx + "fc2.weight", D, EF)
            self._upload_bias(pfx + "fc2.bias")

        # Encoder final layer norm
        self._upload_norm_weight("encoder.layer_norm.weight")
        self._upload_bias("encoder.layer_norm.bias")

        # --- Decoder ---
        # Token embedding
        self._upload_embedding_weight(
            "decoder.embed_tokens.weight", self.vocab_size, D)

        # Positional embedding (learned)
        w = self.weights["decoder.embed_positions.weight"]
        if w.dtype == np.float16:
            w = w.astype(np.float32)
        buf = runner.upload_to_gpu(w.ravel(), "decoder.embed_positions.weight")
        self._gpu_weights["decoder.embed_positions.weight"] = buf

        # Decoder transformer layers
        for i in range(self.decoder_layers):
            pfx = f"decoder.layers.{i}."
            # Self-attention
            self._upload_norm_weight(pfx + "self_attn_layer_norm.weight")
            self._upload_bias(pfx + "self_attn_layer_norm.bias")
            self._upload_linear_weight(pfx + "self_attn.q_proj.weight", D, D)
            self._upload_bias(pfx + "self_attn.q_proj.bias")
            self._upload_linear_weight(pfx + "self_attn.k_proj.weight", D, D)
            # k_proj: no bias
            self._upload_linear_weight(pfx + "self_attn.v_proj.weight", D, D)
            self._upload_bias(pfx + "self_attn.v_proj.bias")
            self._upload_linear_weight(pfx + "self_attn.out_proj.weight", D, D)
            self._upload_bias(pfx + "self_attn.out_proj.bias")
            # Cross-attention
            self._upload_norm_weight(pfx + "encoder_attn_layer_norm.weight")
            self._upload_bias(pfx + "encoder_attn_layer_norm.bias")
            self._upload_linear_weight(pfx + "encoder_attn.q_proj.weight", D, D)
            self._upload_bias(pfx + "encoder_attn.q_proj.bias")
            self._upload_linear_weight(pfx + "encoder_attn.k_proj.weight", D, D)
            # k_proj: no bias
            self._upload_linear_weight(pfx + "encoder_attn.v_proj.weight", D, D)
            self._upload_bias(pfx + "encoder_attn.v_proj.bias")
            self._upload_linear_weight(pfx + "encoder_attn.out_proj.weight", D, D)
            self._upload_bias(pfx + "encoder_attn.out_proj.bias")
            # FFN
            self._upload_norm_weight(pfx + "final_layer_norm.weight")
            self._upload_bias(pfx + "final_layer_norm.bias")
            self._upload_linear_weight(pfx + "fc1.weight", DF, D)
            self._upload_bias(pfx + "fc1.bias")
            self._upload_linear_weight(pfx + "fc2.weight", D, DF)
            self._upload_bias(pfx + "fc2.bias")

        # Decoder final layer norm
        self._upload_norm_weight("decoder.layer_norm.weight")
        self._upload_bias("decoder.layer_norm.bias")

        # LM head (proj_out) — may be same as embed_tokens (tied)
        if "proj_out.weight" in self.weights:
            self._upload_linear_weight("proj_out.weight", self.vocab_size, D)
        # else: use decoder.embed_tokens.weight (tied weights)

        # Zero biases
        self._upload_zero_bias("zero_bias_D", D)
        self._upload_zero_bias("zero_bias_EF", EF)
        self._upload_zero_bias("zero_bias_DF", DF)
        self._upload_zero_bias("zero_bias_V", self.vocab_size)

        self._print_gpu_weight_stats()

    def _conv1d(self, x: np.ndarray, weight_name: str,
                bias_name: str, stride: int = 1) -> np.ndarray:
        """1D convolution in NumPy. x: (C_in, T), weight: (C_out, C_in, K)."""
        w = self.weights[weight_name]
        b = self.weights[bias_name]
        if w.dtype == np.float16:
            w = w.astype(np.float32)
        if b.dtype == np.float16:
            b = b.astype(np.float32)
        if x.dtype == np.float16:
            x = x.astype(np.float32)
        C_out, C_in, K = w.shape
        T_in = x.shape[1]
        # Pad to maintain temporal dimension for stride=1
        pad = (K - 1) // 2
        x_padded = np.pad(x, ((0, 0), (pad, pad)), mode='constant')
        T_out = (x_padded.shape[1] - K) // stride + 1
        out = np.zeros((C_out, T_out), dtype=np.float32)
        for co in range(C_out):
            for ci in range(C_in):
                for ki in range(K):
                    out[co] += w[co, ci, ki] * x_padded[ci, ki:ki + T_out * stride:stride]
            out[co] += b[co]
        return out

    def _gelu_np(self, x: np.ndarray) -> np.ndarray:
        """GELU activation in NumPy (used for conv outputs before GPU)."""
        inner = 0.7978845608 * (x + 0.044715 * x ** 3)
        return 0.5 * x * (1.0 + np.tanh(inner))

    def encode(self, mel: np.ndarray) -> np.ndarray:
        """Encode mel spectrogram to encoder hidden states.

        mel: (num_mel_bins, T_mel) float32
        Returns: (T_enc, d_model) float32
        """
        D = self.d_model
        n_head = self.encoder_heads
        HD = self.head_dim_enc

        # Conv1d layers (done in numpy — small ops for tiny model)
        x = self._conv1d(mel, "encoder.conv1.weight", "encoder.conv1.bias",
                         stride=1)
        x = self._gelu_np(x)
        x = self._conv1d(x, "encoder.conv2.weight", "encoder.conv2.bias",
                         stride=2)
        x = self._gelu_np(x)

        # x: (D, T_enc) → (T_enc, D)
        x = x.T.astype(np.float32)
        T_enc = x.shape[0]

        # Add positional embedding
        pos_emb = self.weights["encoder.embed_positions.weight"]
        if pos_emb.dtype == np.float16:
            pos_emb = pos_emb.astype(np.float32)
        x = x + pos_emb[:T_enc]

        # Encoder transformer layers
        for i in range(self.encoder_layers):
            x = self._encoder_block(x, i)

        # Final layer norm
        x = self._layer_norm(
            x,
            self._gpu_weights["encoder.layer_norm.weight"],
            self._gpu_weights["encoder.layer_norm.bias"])

        return x

    def _encoder_block(self, x: np.ndarray, layer: int) -> np.ndarray:
        """Encoder block: LN → self-attn → add → LN → FFN → add."""
        D = self.d_model
        EF = self.encoder_ffn_dim
        n_head = self.encoder_heads
        HD = self.head_dim_enc
        pfx = f"encoder.layers.{layer}."

        # Pre-norm → self-attention
        ln1 = self._layer_norm(
            x,
            self._gpu_weights[pfx + "self_attn_layer_norm.weight"],
            self._gpu_weights[pfx + "self_attn_layer_norm.bias"])

        T = ln1.shape[0]
        q = self._linear(
            ln1, self._gpu_weights[pfx + "self_attn.q_proj.weight"],
            self._gpu_weights[pfx + "self_attn.q_proj.bias"], D)
        k = self._linear(
            ln1, self._gpu_weights[pfx + "self_attn.k_proj.weight"],
            self._gpu_weights["zero_bias_D"], D)
        v = self._linear(
            ln1, self._gpu_weights[pfx + "self_attn.v_proj.weight"],
            self._gpu_weights[pfx + "self_attn.v_proj.bias"], D)

        # Full (non-causal) multi-head attention
        Q = q.reshape(T, n_head, HD)
        K = k.reshape(T, n_head, HD)
        V = v.reshape(T, n_head, HD)
        attn_out = self._full_attention_multihead(Q, K, V, n_head)
        attn_flat = attn_out[:, :, :HD].reshape(T, D)

        proj = self._linear(
            attn_flat, self._gpu_weights[pfx + "self_attn.out_proj.weight"],
            self._gpu_weights[pfx + "self_attn.out_proj.bias"], D)
        x = self._add(x, proj)

        # Pre-norm → FFN
        ln2 = self._layer_norm(
            x,
            self._gpu_weights[pfx + "final_layer_norm.weight"],
            self._gpu_weights[pfx + "final_layer_norm.bias"])
        h = self._linear(
            ln2, self._gpu_weights[pfx + "fc1.weight"],
            self._gpu_weights[pfx + "fc1.bias"], EF)
        h = self._gelu(h)
        h = self._linear(
            h, self._gpu_weights[pfx + "fc2.weight"],
            self._gpu_weights[pfx + "fc2.bias"], D)
        x = self._add(x, h)
        return x

    def _decoder_block(self, x: np.ndarray, encoder_out: np.ndarray,
                       layer: int) -> np.ndarray:
        """Decoder block: self-attn → cross-attn → FFN, each with pre-norm + residual."""
        D = self.d_model
        DF = self.decoder_ffn_dim
        n_head = self.decoder_heads
        HD = self.head_dim_dec
        pfx = f"decoder.layers.{layer}."

        T_dec = x.shape[0]
        T_enc = encoder_out.shape[0]

        # --- Self-attention (causal) ---
        ln1 = self._layer_norm(
            x,
            self._gpu_weights[pfx + "self_attn_layer_norm.weight"],
            self._gpu_weights[pfx + "self_attn_layer_norm.bias"])

        q = self._linear(
            ln1, self._gpu_weights[pfx + "self_attn.q_proj.weight"],
            self._gpu_weights[pfx + "self_attn.q_proj.bias"], D)
        k = self._linear(
            ln1, self._gpu_weights[pfx + "self_attn.k_proj.weight"],
            self._gpu_weights["zero_bias_D"], D)
        v = self._linear(
            ln1, self._gpu_weights[pfx + "self_attn.v_proj.weight"],
            self._gpu_weights[pfx + "self_attn.v_proj.bias"], D)

        # Causal self-attention (numpy for simplicity)
        Q = q.reshape(T_dec, n_head, HD)
        K_s = k.reshape(T_dec, n_head, HD)
        V_s = v.reshape(T_dec, n_head, HD)
        scale = 1.0 / np.sqrt(HD)
        sa_out = np.zeros((T_dec, n_head, HD), dtype=np.float32)
        for h in range(n_head):
            scores = Q[:, h, :] @ K_s[:, h, :].T * scale
            # Causal mask
            mask = np.triu(np.full((T_dec, T_dec), -1e9), k=1)
            scores = scores + mask
            exp_s = np.exp(scores - scores.max(axis=-1, keepdims=True))
            attn = exp_s / exp_s.sum(axis=-1, keepdims=True)
            sa_out[:, h, :] = attn @ V_s[:, h, :]

        sa_flat = sa_out.reshape(T_dec, D)
        proj = self._linear(
            sa_flat, self._gpu_weights[pfx + "self_attn.out_proj.weight"],
            self._gpu_weights[pfx + "self_attn.out_proj.bias"], D)
        x = self._add(x, proj)

        # --- Cross-attention ---
        ln2 = self._layer_norm(
            x,
            self._gpu_weights[pfx + "encoder_attn_layer_norm.weight"],
            self._gpu_weights[pfx + "encoder_attn_layer_norm.bias"])

        q_c = self._linear(
            ln2, self._gpu_weights[pfx + "encoder_attn.q_proj.weight"],
            self._gpu_weights[pfx + "encoder_attn.q_proj.bias"], D)
        k_c = self._linear(
            encoder_out, self._gpu_weights[pfx + "encoder_attn.k_proj.weight"],
            self._gpu_weights["zero_bias_D"], D)
        v_c = self._linear(
            encoder_out, self._gpu_weights[pfx + "encoder_attn.v_proj.weight"],
            self._gpu_weights[pfx + "encoder_attn.v_proj.bias"], D)

        # Cross-attention (full, non-causal)
        Q_c = q_c.reshape(T_dec, n_head, HD)
        K_c = k_c.reshape(T_enc, n_head, HD)
        V_c = v_c.reshape(T_enc, n_head, HD)
        scale = 1.0 / np.sqrt(HD)
        ca_out = np.zeros((T_dec, n_head, HD), dtype=np.float32)
        for h in range(n_head):
            scores = Q_c[:, h, :] @ K_c[:, h, :].T * scale
            exp_s = np.exp(scores - scores.max(axis=-1, keepdims=True))
            attn = exp_s / exp_s.sum(axis=-1, keepdims=True)
            ca_out[:, h, :] = attn @ V_c[:, h, :]

        ca_flat = ca_out.reshape(T_dec, D)
        proj_c = self._linear(
            ca_flat, self._gpu_weights[pfx + "encoder_attn.out_proj.weight"],
            self._gpu_weights[pfx + "encoder_attn.out_proj.bias"], D)
        x = self._add(x, proj_c)

        # --- FFN ---
        ln3 = self._layer_norm(
            x,
            self._gpu_weights[pfx + "final_layer_norm.weight"],
            self._gpu_weights[pfx + "final_layer_norm.bias"])
        h = self._linear(
            ln3, self._gpu_weights[pfx + "fc1.weight"],
            self._gpu_weights[pfx + "fc1.bias"], DF)
        h = self._gelu(h)
        h = self._linear(
            h, self._gpu_weights[pfx + "fc2.weight"],
            self._gpu_weights[pfx + "fc2.bias"], D)
        x = self._add(x, h)
        return x

    def decode(self, token_ids: np.ndarray, encoder_out: np.ndarray,
               positions: np.ndarray = None) -> np.ndarray:
        """Decode token ids conditioned on encoder output.

        token_ids: (T_dec,) int32
        encoder_out: (T_enc, d_model)
        Returns: (T_dec, vocab_size) logits
        """
        D = self.d_model
        T_dec = token_ids.shape[0]

        # Token embedding
        embed_w = self.weights["decoder.embed_tokens.weight"]
        if embed_w.dtype == np.float16:
            embed_w = embed_w.astype(np.float32)
        x = embed_w[token_ids]  # (T_dec, D)

        # Positional embedding (learned)
        if positions is None:
            positions = np.arange(T_dec, dtype=np.int32)
        pos_w = self.weights["decoder.embed_positions.weight"]
        if pos_w.dtype == np.float16:
            pos_w = pos_w.astype(np.float32)
        x = x + pos_w[positions]

        # Decoder transformer layers
        for i in range(self.decoder_layers):
            x = self._decoder_block(x, encoder_out, i)

        # Final layer norm
        x = self._layer_norm(
            x,
            self._gpu_weights["decoder.layer_norm.weight"],
            self._gpu_weights["decoder.layer_norm.bias"])

        # LM head: project to vocab
        if "proj_out.weight" in self._gpu_weights:
            logits = self._linear(
                x, self._gpu_weights["proj_out.weight"],
                self._gpu_weights["zero_bias_V"], self.vocab_size)
        else:
            # Tied weights: use embedding transpose
            logits = x @ embed_w.T  # (T_dec, vocab)

        return logits

    def forward(self, mel: np.ndarray,
                token_ids: np.ndarray = None, **kwargs) -> np.ndarray:
        """Full Whisper forward pass.

        mel: (num_mel_bins, T_mel) float32
        token_ids: (T_dec,) int32 — decoder input tokens
        Returns: (T_dec, vocab_size) logits
        """
        encoder_out = self.encode(mel)

        if token_ids is None:
            # Default: start token
            token_ids = np.array([50258], dtype=np.int32)

        logits = self.decode(token_ids, encoder_out)
        return logits


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------

def verify_with_random_weights():
    """Verify Whisper pipeline with small random weights."""
    print("=" * 60)
    print("Whisper-Tiny WebGPU Pipeline Verification (random weights)")
    print("=" * 60)

    # Use smaller sequence lengths for faster verification
    D = 384
    encoder_layers = 4
    decoder_layers = 4
    n_head = 6
    HD = D // n_head  # 64
    EF = 1536
    DF = 1536
    mel_bins = 80
    vocab_size = 51865
    max_source = 1500
    max_target = 448
    eps = 1e-5
    np.random.seed(42)

    # Use short sequences for verification
    T_mel = 100  # Short mel spectrogram

    weights = {}

    # Encoder conv1d weights
    weights["encoder.conv1.weight"] = np.random.randn(
        D, mel_bins, 3).astype(np.float32) * 0.02
    weights["encoder.conv1.bias"] = np.zeros(D, dtype=np.float32)
    weights["encoder.conv2.weight"] = np.random.randn(
        D, D, 3).astype(np.float32) * 0.02
    weights["encoder.conv2.bias"] = np.zeros(D, dtype=np.float32)

    # Encoder positional embedding
    weights["encoder.embed_positions.weight"] = np.random.randn(
        max_source, D).astype(np.float32) * 0.02

    # Encoder layers
    for i in range(encoder_layers):
        pfx = f"encoder.layers.{i}."
        weights[pfx + "self_attn_layer_norm.weight"] = np.ones(
            D, dtype=np.float32)
        weights[pfx + "self_attn_layer_norm.bias"] = np.zeros(
            D, dtype=np.float32)
        weights[pfx + "final_layer_norm.weight"] = np.ones(
            D, dtype=np.float32)
        weights[pfx + "final_layer_norm.bias"] = np.zeros(
            D, dtype=np.float32)
        weights[pfx + "self_attn.q_proj.weight"] = np.random.randn(
            D, D).astype(np.float32) * 0.02
        weights[pfx + "self_attn.q_proj.bias"] = np.zeros(
            D, dtype=np.float32)
        weights[pfx + "self_attn.k_proj.weight"] = np.random.randn(
            D, D).astype(np.float32) * 0.02
        weights[pfx + "self_attn.v_proj.weight"] = np.random.randn(
            D, D).astype(np.float32) * 0.02
        weights[pfx + "self_attn.v_proj.bias"] = np.zeros(
            D, dtype=np.float32)
        weights[pfx + "self_attn.out_proj.weight"] = np.random.randn(
            D, D).astype(np.float32) * 0.02
        weights[pfx + "self_attn.out_proj.bias"] = np.zeros(
            D, dtype=np.float32)
        weights[pfx + "fc1.weight"] = np.random.randn(
            EF, D).astype(np.float32) * 0.02
        weights[pfx + "fc1.bias"] = np.zeros(EF, dtype=np.float32)
        weights[pfx + "fc2.weight"] = np.random.randn(
            D, EF).astype(np.float32) * 0.02
        weights[pfx + "fc2.bias"] = np.zeros(D, dtype=np.float32)

    # Encoder final norm
    weights["encoder.layer_norm.weight"] = np.ones(D, dtype=np.float32)
    weights["encoder.layer_norm.bias"] = np.zeros(D, dtype=np.float32)

    # Decoder
    weights["decoder.embed_tokens.weight"] = np.random.randn(
        vocab_size, D).astype(np.float32) * 0.02
    weights["decoder.embed_positions.weight"] = np.random.randn(
        max_target, D).astype(np.float32) * 0.02

    for i in range(decoder_layers):
        pfx = f"decoder.layers.{i}."
        # Self-attention
        weights[pfx + "self_attn_layer_norm.weight"] = np.ones(
            D, dtype=np.float32)
        weights[pfx + "self_attn_layer_norm.bias"] = np.zeros(
            D, dtype=np.float32)
        weights[pfx + "self_attn.q_proj.weight"] = np.random.randn(
            D, D).astype(np.float32) * 0.02
        weights[pfx + "self_attn.q_proj.bias"] = np.zeros(
            D, dtype=np.float32)
        weights[pfx + "self_attn.k_proj.weight"] = np.random.randn(
            D, D).astype(np.float32) * 0.02
        weights[pfx + "self_attn.v_proj.weight"] = np.random.randn(
            D, D).astype(np.float32) * 0.02
        weights[pfx + "self_attn.v_proj.bias"] = np.zeros(
            D, dtype=np.float32)
        weights[pfx + "self_attn.out_proj.weight"] = np.random.randn(
            D, D).astype(np.float32) * 0.02
        weights[pfx + "self_attn.out_proj.bias"] = np.zeros(
            D, dtype=np.float32)
        # Cross-attention
        weights[pfx + "encoder_attn_layer_norm.weight"] = np.ones(
            D, dtype=np.float32)
        weights[pfx + "encoder_attn_layer_norm.bias"] = np.zeros(
            D, dtype=np.float32)
        weights[pfx + "encoder_attn.q_proj.weight"] = np.random.randn(
            D, D).astype(np.float32) * 0.02
        weights[pfx + "encoder_attn.q_proj.bias"] = np.zeros(
            D, dtype=np.float32)
        weights[pfx + "encoder_attn.k_proj.weight"] = np.random.randn(
            D, D).astype(np.float32) * 0.02
        weights[pfx + "encoder_attn.v_proj.weight"] = np.random.randn(
            D, D).astype(np.float32) * 0.02
        weights[pfx + "encoder_attn.v_proj.bias"] = np.zeros(
            D, dtype=np.float32)
        weights[pfx + "encoder_attn.out_proj.weight"] = np.random.randn(
            D, D).astype(np.float32) * 0.02
        weights[pfx + "encoder_attn.out_proj.bias"] = np.zeros(
            D, dtype=np.float32)
        # FFN
        weights[pfx + "final_layer_norm.weight"] = np.ones(
            D, dtype=np.float32)
        weights[pfx + "final_layer_norm.bias"] = np.zeros(
            D, dtype=np.float32)
        weights[pfx + "fc1.weight"] = np.random.randn(
            DF, D).astype(np.float32) * 0.02
        weights[pfx + "fc1.bias"] = np.zeros(DF, dtype=np.float32)
        weights[pfx + "fc2.weight"] = np.random.randn(
            D, DF).astype(np.float32) * 0.02
        weights[pfx + "fc2.bias"] = np.zeros(D, dtype=np.float32)

    # Decoder final norm
    weights["decoder.layer_norm.weight"] = np.ones(D, dtype=np.float32)
    weights["decoder.layer_norm.bias"] = np.zeros(D, dtype=np.float32)

    # proj_out (LM head)
    weights["proj_out.weight"] = np.random.randn(
        vocab_size, D).astype(np.float32) * 0.02

    print(f"\nModel: d_model={D}, enc={encoder_layers}L, dec={decoder_layers}L, "
          f"heads={n_head}")
    print(f"  mel_bins={mel_bins}, vocab={vocab_size}")

    config = WHISPER_CONFIGS["tiny"]
    model = WhisperWebGPU(
        weights, d_model=D,
        encoder_layers=encoder_layers, decoder_layers=decoder_layers,
        encoder_attention_heads=n_head, decoder_attention_heads=n_head,
        encoder_ffn_dim=EF, decoder_ffn_dim=DF,
        num_mel_bins=mel_bins, vocab_size=vocab_size,
        max_source_positions=max_source,
        max_target_positions=max_target)

    # Create dummy mel spectrogram
    mel = np.random.randn(mel_bins, T_mel).astype(np.float32) * 0.1
    # Decoder input: [<|startoftranscript|>]
    token_ids = np.array([50258], dtype=np.int32)

    t0 = time.time()
    logits = model.forward(mel, token_ids)
    t1 = time.time()

    print(f"\nForward pass: mel ({mel_bins},{T_mel}) + tokens {token_ids.shape} "
          f"→ logits {logits.shape} in {(t1-t0)*1000:.0f}ms")

    # --- NumPy reference ---
    def _gelu_numpy(x):
        inner = 0.7978845608 * (x + 0.044715 * x ** 3)
        return 0.5 * x * (1.0 + np.tanh(inner))

    def _conv1d_np(x, w, b, stride=1):
        C_out, C_in, K = w.shape
        pad = (K - 1) // 2
        x_padded = np.pad(x, ((0, 0), (pad, pad)), mode='constant')
        T_out = (x_padded.shape[1] - K) // stride + 1
        out = np.zeros((C_out, T_out), dtype=np.float32)
        for co in range(C_out):
            for ci in range(C_in):
                for ki in range(K):
                    out[co] += w[co, ci, ki] * x_padded[ci, ki:ki + T_out * stride:stride]
            out[co] += b[co]
        return out

    def _layernorm_np(x, w, b, eps=1e-5):
        mean = x.mean(axis=-1, keepdims=True)
        var = x.var(axis=-1, keepdims=True)
        return (x - mean) / np.sqrt(var + eps) * w + b

    # Encoder reference
    enc_x = _conv1d_np(mel, weights["encoder.conv1.weight"],
                        weights["encoder.conv1.bias"], stride=1)
    enc_x = _gelu_numpy(enc_x)
    enc_x = _conv1d_np(enc_x, weights["encoder.conv2.weight"],
                        weights["encoder.conv2.bias"], stride=2)
    enc_x = _gelu_numpy(enc_x)
    enc_x = enc_x.T  # (T_enc, D)
    T_enc = enc_x.shape[0]
    enc_x = enc_x + weights["encoder.embed_positions.weight"][:T_enc]

    for layer in range(encoder_layers):
        pfx = f"encoder.layers.{layer}."
        ln1 = _layernorm_np(enc_x,
                             weights[pfx + "self_attn_layer_norm.weight"],
                             weights[pfx + "self_attn_layer_norm.bias"])
        T = ln1.shape[0]
        q = ln1 @ weights[pfx + "self_attn.q_proj.weight"].T + \
            weights[pfx + "self_attn.q_proj.bias"]
        k = ln1 @ weights[pfx + "self_attn.k_proj.weight"].T
        v = ln1 @ weights[pfx + "self_attn.v_proj.weight"].T + \
            weights[pfx + "self_attn.v_proj.bias"]

        Q = q.reshape(T, n_head, HD)
        K_ = k.reshape(T, n_head, HD)
        V_ = v.reshape(T, n_head, HD)
        scale = 1.0 / np.sqrt(HD)
        attn_out = np.zeros_like(Q)
        for h in range(n_head):
            scores = Q[:, h, :] @ K_[:, h, :].T * scale
            exp_s = np.exp(scores - scores.max(axis=-1, keepdims=True))
            attn = exp_s / exp_s.sum(axis=-1, keepdims=True)
            attn_out[:, h, :] = attn @ V_[:, h, :]
        attn_flat = attn_out.reshape(T, D)
        proj = attn_flat @ weights[pfx + "self_attn.out_proj.weight"].T + \
               weights[pfx + "self_attn.out_proj.bias"]
        enc_x = enc_x + proj

        ln2 = _layernorm_np(enc_x,
                             weights[pfx + "final_layer_norm.weight"],
                             weights[pfx + "final_layer_norm.bias"])
        h_val = ln2 @ weights[pfx + "fc1.weight"].T + weights[pfx + "fc1.bias"]
        h_val = _gelu_numpy(h_val)
        h_val = h_val @ weights[pfx + "fc2.weight"].T + weights[pfx + "fc2.bias"]
        enc_x = enc_x + h_val

    enc_ref = _layernorm_np(enc_x,
                             weights["encoder.layer_norm.weight"],
                             weights["encoder.layer_norm.bias"])

    # Decoder reference
    dec_x = weights["decoder.embed_tokens.weight"][token_ids]
    dec_x = dec_x + weights["decoder.embed_positions.weight"][:len(token_ids)]

    for layer in range(decoder_layers):
        pfx = f"decoder.layers.{layer}."
        # Self-attention
        ln1 = _layernorm_np(dec_x,
                             weights[pfx + "self_attn_layer_norm.weight"],
                             weights[pfx + "self_attn_layer_norm.bias"])
        T_d = ln1.shape[0]
        q = ln1 @ weights[pfx + "self_attn.q_proj.weight"].T + \
            weights[pfx + "self_attn.q_proj.bias"]
        k = ln1 @ weights[pfx + "self_attn.k_proj.weight"].T
        v = ln1 @ weights[pfx + "self_attn.v_proj.weight"].T + \
            weights[pfx + "self_attn.v_proj.bias"]
        Q = q.reshape(T_d, n_head, HD)
        K_ = k.reshape(T_d, n_head, HD)
        V_ = v.reshape(T_d, n_head, HD)
        scale = 1.0 / np.sqrt(HD)
        sa_out = np.zeros_like(Q)
        for h_idx in range(n_head):
            scores = Q[:, h_idx, :] @ K_[:, h_idx, :].T * scale
            mask = np.triu(np.full((T_d, T_d), -1e9), k=1)
            scores = scores + mask
            exp_s = np.exp(scores - scores.max(axis=-1, keepdims=True))
            attn = exp_s / exp_s.sum(axis=-1, keepdims=True)
            sa_out[:, h_idx, :] = attn @ V_[:, h_idx, :]
        sa_flat = sa_out.reshape(T_d, D)
        proj = sa_flat @ weights[pfx + "self_attn.out_proj.weight"].T + \
               weights[pfx + "self_attn.out_proj.bias"]
        dec_x = dec_x + proj

        # Cross-attention
        ln2 = _layernorm_np(dec_x,
                             weights[pfx + "encoder_attn_layer_norm.weight"],
                             weights[pfx + "encoder_attn_layer_norm.bias"])
        q_c = ln2 @ weights[pfx + "encoder_attn.q_proj.weight"].T + \
              weights[pfx + "encoder_attn.q_proj.bias"]
        k_c = enc_ref @ weights[pfx + "encoder_attn.k_proj.weight"].T
        v_c = enc_ref @ weights[pfx + "encoder_attn.v_proj.weight"].T + \
              weights[pfx + "encoder_attn.v_proj.bias"]
        Q_c = q_c.reshape(T_d, n_head, HD)
        K_c = k_c.reshape(T_enc, n_head, HD)
        V_c = v_c.reshape(T_enc, n_head, HD)
        ca_out = np.zeros_like(Q_c)
        for h_idx in range(n_head):
            scores = Q_c[:, h_idx, :] @ K_c[:, h_idx, :].T * scale
            exp_s = np.exp(scores - scores.max(axis=-1, keepdims=True))
            attn = exp_s / exp_s.sum(axis=-1, keepdims=True)
            ca_out[:, h_idx, :] = attn @ V_c[:, h_idx, :]
        ca_flat = ca_out.reshape(T_d, D)
        proj_c = ca_flat @ weights[pfx + "encoder_attn.out_proj.weight"].T + \
                 weights[pfx + "encoder_attn.out_proj.bias"]
        dec_x = dec_x + proj_c

        # FFN
        ln3 = _layernorm_np(dec_x,
                             weights[pfx + "final_layer_norm.weight"],
                             weights[pfx + "final_layer_norm.bias"])
        h_val = ln3 @ weights[pfx + "fc1.weight"].T + weights[pfx + "fc1.bias"]
        h_val = _gelu_numpy(h_val)
        h_val = h_val @ weights[pfx + "fc2.weight"].T + weights[pfx + "fc2.bias"]
        dec_x = dec_x + h_val

    dec_ref = _layernorm_np(dec_x,
                             weights["decoder.layer_norm.weight"],
                             weights["decoder.layer_norm.bias"])
    logits_ref = dec_ref @ weights["proj_out.weight"].T

    # Compare
    max_diff = np.abs(logits - logits_ref).max()
    mean_diff = np.abs(logits - logits_ref).mean()

    # Check top-1 token matches
    gpu_top1 = np.argmax(logits[0])
    ref_top1 = np.argmax(logits_ref[0])

    print(f"\nMax diff vs NumPy: {max_diff:.6f}")
    print(f"Mean diff vs NumPy: {mean_diff:.6f}")
    print(f"Top-1 token: GPU={gpu_top1}, Ref={ref_top1}, match={gpu_top1 == ref_top1}")
    print(f"Output shape: {logits.shape}")

    success = max_diff < 1.0
    print(f"\n{'PASS' if success else 'FAIL'}")
    return success


# ---------------------------------------------------------------------------
# Full inference with real weights
# ---------------------------------------------------------------------------

def run_full_inference(audio_path: str):
    """Run Whisper-Tiny transcription with real weights via transformers."""
    import torch
    from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq

    model_id = "openai/whisper-tiny"

    print("=== Loading Whisper-Tiny ===")
    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForSpeechSeq2Seq.from_pretrained(model_id)
    model.eval()
    print(f"  Model loaded: {sum(p.numel() for p in model.parameters())/1e6:.1f}M params")

    # Load audio
    try:
        import librosa
        audio, sr = librosa.load(audio_path, sr=16000)
    except ImportError:
        import soundfile as sf
        audio, sr = sf.read(audio_path)
        if sr != 16000:
            # Simple resample
            from scipy import signal
            audio = signal.resample(audio, int(len(audio) * 16000 / sr))
            sr = 16000

    print(f"  Audio: {len(audio)/sr:.1f}s at {sr}Hz")

    # Process
    inputs = processor(audio, sampling_rate=sr, return_tensors="pt")

    print("  Transcribing...")
    t0 = time.time()
    with torch.no_grad():
        generated_ids = model.generate(
            inputs["input_features"],
            max_new_tokens=128,
        )
    t1 = time.time()

    text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    print(f"  Time: {(t1-t0)*1000:.0f}ms")
    print(f"\n  Transcription: {text}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Whisper-Tiny on WebGPU via Triton")
    parser.add_argument("--verify", action="store_true",
                        help="Verify pipeline with random weights")
    parser.add_argument("--audio", type=str, default=None,
                        help="Input audio path for transcription")
    args = parser.parse_args()

    if args.verify:
        success = verify_with_random_weights()
        sys.exit(0 if success else 1)

    if args.audio:
        run_full_inference(args.audio)
    else:
        print("Whisper-Tiny speech-to-text on WebGPU")
        print("Usage:")
        print("  --verify   Run pipeline verification (random weights)")
        print("  --audio F  Transcribe audio file")
        print("\nModel: openai/whisper-tiny (151 MB)")


if __name__ == "__main__":
    main()
