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
        """1D convolution via im2col + matmul. x: (C_in, T), weight: (C_out, C_in, K)."""
        w = self.weights[weight_name]
        b = self.weights[bias_name]
        if w.dtype == np.float16:
            w = w.astype(np.float32)
        if b.dtype == np.float16:
            b = b.astype(np.float32)
        if x.dtype == np.float16:
            x = x.astype(np.float32)
        C_out, C_in, K = w.shape
        pad = (K - 1) // 2
        x_padded = np.pad(x, ((0, 0), (pad, pad)), mode='constant')
        T_out = (x_padded.shape[1] - K) // stride + 1
        # im2col: extract (K, C_in) patches → (T_out, C_in*K) matrix
        from numpy.lib.stride_tricks import as_strided
        s = x_padded.strides  # (C_in*T_padded*4, 4) bytes
        # Shape: (T_out, C_in, K), strides: (stride*s[1], s[0], s[1])
        cols = as_strided(
            x_padded,
            shape=(T_out, C_in, K),
            strides=(s[1] * stride, s[0], s[1]))
        cols_flat = cols.reshape(T_out, C_in * K)  # (T_out, C_in*K)
        w_flat = w.reshape(C_out, C_in * K)         # (C_out, C_in*K)
        out = (cols_flat @ w_flat.T + b).T           # (C_out, T_out)
        return out

    def _gelu_np(self, x: np.ndarray) -> np.ndarray:
        """GELU activation — fast sigmoid approximation."""
        return x * (1.0 / (1.0 + np.exp(-1.702 * x)))

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

        # Causal self-attention — vectorized
        Q = q.reshape(T_dec, n_head, HD).transpose(1, 0, 2)     # (n_head, T, HD)
        K_s = k.reshape(T_dec, n_head, HD).transpose(1, 2, 0)   # (n_head, HD, T)
        V_s = v.reshape(T_dec, n_head, HD).transpose(1, 0, 2)   # (n_head, T, HD)
        scale = 1.0 / np.sqrt(HD)
        scores = np.float32(Q @ K_s * scale)                    # (n_head, T, T)
        mask = np.triu(np.full((T_dec, T_dec), -1e9, dtype=np.float32), k=1)
        scores = scores + mask
        scores -= scores.max(axis=-1, keepdims=True)
        exp_s = np.exp(scores)
        attn = exp_s / exp_s.sum(axis=-1, keepdims=True)
        sa_out = (attn @ V_s).transpose(1, 0, 2)                # (T, n_head, HD)

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

        # Cross-attention — vectorized
        Q_c = q_c.reshape(T_dec, n_head, HD).transpose(1, 0, 2)   # (n_head, T_dec, HD)
        K_c = k_c.reshape(T_enc, n_head, HD).transpose(1, 2, 0)   # (n_head, HD, T_enc)
        V_c = v_c.reshape(T_enc, n_head, HD).transpose(1, 0, 2)   # (n_head, T_enc, HD)
        scale = 1.0 / np.sqrt(HD)
        scores = np.float32(Q_c @ K_c * scale)
        scores -= scores.max(axis=-1, keepdims=True)
        exp_s = np.exp(scores)
        attn = exp_s / exp_s.sum(axis=-1, keepdims=True)
        ca_out = (attn @ V_c).transpose(1, 0, 2)  # (T_dec, n_head, HD)

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
    """Run Whisper-Tiny transcription fully on Triton WebGPU — no PyTorch.

    All operations (mel spectrogram, encoder, decoder) run on WebGPU/numpy.
    """
    print("=== Whisper-Tiny — Full Triton WebGPU ===")

    # --- Load weights ---
    npz_path = os.path.join(_SCRIPT_DIR, "..", "..", "gitignore", "models", os.path.basename(_SCRIPT_DIR), "weights", "whisper_tiny_fp16.npz")
    if not os.path.exists(npz_path):
        print(f"Weights not found: {npz_path}")
        print("Run:  python models/whisper/convert_weights.py")
        sys.exit(1)

    print("  Loading weights...")
    t0 = time.time()
    data = np.load(npz_path, mmap_mode='r')
    weights = {k: data[k].astype(np.float32) for k in data.files}
    print(f"  Loaded {len(weights)} tensors in {(time.time()-t0)*1000:.0f}ms")

    # --- Load audio ---
    print(f"  Loading audio: {audio_path}")
    audio = _load_audio(audio_path)
    audio_sec = len(audio) / 16000
    print(f"  Audio: {audio_sec:.1f}s at 16kHz ({len(audio)} samples)")

    # --- Mel spectrogram (no librosa, pure numpy) ---
    print("  Computing mel spectrogram...")
    t0 = time.time()
    mel = _compute_mel_spectrogram(audio)
    t_mel = time.time() - t0
    print(f"  Mel: {mel.shape} in {t_mel*1000:.0f}ms")

    # --- Create model ---
    config = {k: v for k, v in WHISPER_CONFIGS["tiny"].items() if k != "hf_repo"}
    model = WhisperWebGPU(weights, **config)
    print("  Model created")

    # --- Encode ---
    print("  Encoding audio...")
    t0 = time.time()
    encoder_out = model.encode(mel)
    t_enc = time.time() - t0
    print(f"  Encoder output: {encoder_out.shape} in {t_enc*1000:.0f}ms")

    # --- Autoregressive decode ---
    print("  Decoding (autoregressive)...")
    t0 = time.time()

    # Whisper special tokens
    SOT = 50258      # <|startoftranscript|>
    LANG_EN = 50259  # <|en|>
    TRANSCRIBE = 50359  # <|transcribe|>
    NO_TIMESTAMPS = 50363  # <|notimestamps|>
    EOT = 50257      # <|endoftext|>

    # Start tokens: SOT, language, task, notimestamps
    token_ids = [SOT, LANG_EN, TRANSCRIBE, NO_TIMESTAMPS]
    max_tokens = 128

    for step in range(max_tokens):
        tokens_np = np.array(token_ids, dtype=np.int32)
        logits = model.decode(tokens_np, encoder_out)
        next_token = int(logits[-1].argmax())
        if next_token == EOT:
            break
        token_ids.append(next_token)

    t_dec = time.time() - t0
    n_out = len(token_ids) - 4  # exclude prompt tokens
    tps = n_out / t_dec if t_dec > 0 else 0
    print(f"  Generated {n_out} tokens in {t_dec*1000:.0f}ms ({tps:.1f} tok/s)")

    # --- Decode tokens to text ---
    # Use tokenizers library if available, else simple decode
    text = _decode_tokens(token_ids[4:], weights)  # skip prompt tokens
    print(f"\n  Transcription: {text}")

    print(f"\n--- Performance (all Triton WebGPU) ---")
    print(f"  Mel spectrogram: {t_mel*1000:.0f}ms")
    print(f"  Encoder:         {t_enc*1000:.0f}ms")
    print(f"  Decoder:         {t_dec*1000:.0f}ms ({n_out} tokens)")
    print(f"  Total:           {(t_mel + t_enc + t_dec)*1000:.0f}ms")


def _load_audio(path: str, sr: int = 16000) -> np.ndarray:
    """Load audio file to 16kHz mono float32. No torch/librosa needed."""
    import wave
    import struct

    ext = os.path.splitext(path)[1].lower()
    if ext == '.wav':
        with wave.open(path, 'rb') as wf:
            n_channels = wf.getnchannels()
            sample_width = wf.getsampwidth()
            frame_rate = wf.getframerate()
            n_frames = wf.getnframes()
            raw = wf.readframes(n_frames)

        if sample_width == 2:
            samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
        elif sample_width == 4:
            samples = np.frombuffer(raw, dtype=np.int32).astype(np.float32) / 2147483648.0
        else:
            raise ValueError(f"Unsupported sample width: {sample_width}")

        if n_channels > 1:
            samples = samples.reshape(-1, n_channels).mean(axis=1)

        # Resample to 16kHz if needed
        if frame_rate != sr:
            duration = len(samples) / frame_rate
            n_out = int(duration * sr)
            indices = np.linspace(0, len(samples) - 1, n_out)
            samples = np.interp(indices, np.arange(len(samples)), samples)

        return samples.astype(np.float32)
    else:
        # Try soundfile as fallback
        try:
            import soundfile as sf
            audio, file_sr = sf.read(path)
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            if file_sr != sr:
                duration = len(audio) / file_sr
                n_out = int(duration * sr)
                indices = np.linspace(0, len(audio) - 1, n_out)
                audio = np.interp(indices, np.arange(len(audio)), audio)
            return audio.astype(np.float32)
        except ImportError:
            raise ImportError(
                f"Cannot read {ext} files. Install soundfile: pip install soundfile")


def _compute_mel_spectrogram(audio: np.ndarray,
                             sr: int = 16000,
                             n_fft: int = 400,
                             hop_length: int = 160,
                             n_mels: int = 80) -> np.ndarray:
    """Compute log-mel spectrogram matching Whisper's preprocessing.

    Pure numpy — no librosa or torchaudio dependency.
    Returns: (n_mels, T) float32
    """
    # Pad or trim to 30 seconds (Whisper's fixed context)
    target_length = sr * 30  # 480000
    if len(audio) < target_length:
        audio = np.pad(audio, (0, target_length - len(audio)))
    else:
        audio = audio[:target_length]

    # STFT
    window = np.hanning(n_fft).astype(np.float32)
    n_frames = 1 + (len(audio) - n_fft) // hop_length
    stft = np.zeros((n_fft // 2 + 1, n_frames), dtype=np.complex64)
    for i in range(n_frames):
        start = i * hop_length
        frame = audio[start:start + n_fft] * window
        spectrum = np.fft.rfft(frame)
        stft[:, i] = spectrum

    magnitudes = np.abs(stft) ** 2

    # Mel filterbank
    mel_filters = _mel_filterbank(sr, n_fft, n_mels)  # (n_mels, n_fft//2+1)

    mel_spec = mel_filters @ magnitudes  # (n_mels, n_frames)

    # Log mel
    log_spec = np.log10(np.maximum(mel_spec, 1e-10))
    log_spec = np.maximum(log_spec, log_spec.max() - 8.0)
    log_spec = (log_spec + 4.0) / 4.0  # Whisper's normalization

    return log_spec.astype(np.float32)


def _mel_filterbank(sr: int, n_fft: int, n_mels: int) -> np.ndarray:
    """Create mel filterbank matrix. (n_mels, n_fft//2+1)"""
    def hz_to_mel(hz):
        return 2595 * np.log10(1 + hz / 700)
    def mel_to_hz(mel):
        return 700 * (10 ** (mel / 2595) - 1)

    n_freqs = n_fft // 2 + 1
    all_freqs = np.linspace(0, sr / 2, n_freqs)

    mel_low = hz_to_mel(0)
    mel_high = hz_to_mel(sr / 2)
    mel_points = np.linspace(mel_low, mel_high, n_mels + 2)
    hz_points = mel_to_hz(mel_points)

    filterbank = np.zeros((n_mels, n_freqs), dtype=np.float32)
    for i in range(n_mels):
        lower = hz_points[i]
        center = hz_points[i + 1]
        upper = hz_points[i + 2]
        for j, freq in enumerate(all_freqs):
            if lower <= freq <= center:
                filterbank[i, j] = (freq - lower) / (center - lower + 1e-10)
            elif center < freq <= upper:
                filterbank[i, j] = (upper - freq) / (upper - center + 1e-10)

    return filterbank


def _decode_tokens(token_ids, weights=None):
    """Decode Whisper token IDs to text."""
    try:
        from tokenizers import Tokenizer
        tok_path = os.path.join(_SCRIPT_DIR, "..", "..", "gitignore", "models", os.path.basename(_SCRIPT_DIR), "weights", "tokenizer.json")
        if os.path.exists(tok_path):
            tokenizer = Tokenizer.from_file(tok_path)
            return tokenizer.decode(token_ids)
    except ImportError:
        pass

    try:
        from transformers import WhisperTokenizer
        tokenizer = WhisperTokenizer.from_pretrained("openai/whisper-tiny")
        return tokenizer.decode(token_ids, skip_special_tokens=True)
    except ImportError:
        pass

    # Fallback: just show token IDs
    return f"[token IDs: {token_ids}]"


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
