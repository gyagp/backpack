"""
Z-Image-Turbo inference on WebGPU via Triton.

Architecture: Single-stream DiT transformer (Tongyi-MAI/Z-Image-Turbo)
  - 30 transformer blocks with adaLN modulation
  - 2 context refiner blocks
  - 2 noise refiner blocks
  - dim=3840, 30 heads, head_dim=128, in_channels=16
  - Qwen3 text encoder (cap_feat_dim=2560)
  - AutoencoderKL VAE
  - FlowMatchEulerDiscreteScheduler
  - 3D RoPE: axes_dims=[32,48,48], theta=256.0

Usage:
    python python/examples/webgpu/z-image-turbo/model.py --verify
    python python/examples/webgpu/z-image-turbo/model.py --prompt "a cat" --steps 4

Requirements:
    pip install torch diffusers transformers safetensors pillow accelerate
    Dawn WebGPU library built at third_party/webgpu/dawn/build/
"""
import argparse
import math
import os
import sys
import time
from typing import Dict, List, Optional, Tuple

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_SCRIPT_DIR))

import numpy as np

from common.model_base import WebGPUModel, _next_pow2
from common.utils import add_device_arg, apply_device_arg
from triton.backends.webgpu.dawn_runner import GPUBuffer

# ---------------------------------------------------------------------------
# Z-Image-Turbo architecture constants
# ---------------------------------------------------------------------------
DIM = 3840
NUM_HEADS = 30
N_KV_HEADS = 30
HEAD_DIM = DIM // NUM_HEADS   # 128
IN_CHANNELS = 16              # VAE latent channels
PATCH_SIZE = 2                # all_patch_size=[2]
F_PATCH_SIZE = 1              # all_f_patch_size=[1]
CAP_FEAT_DIM = 2560           # Qwen3 hidden size
AXES_DIMS = [32, 48, 48]      # RoPE axes dimensions
AXES_LENS = [1536, 512, 512]  # max position values
ROPE_THETA = 256.0
N_LAYERS = 30
N_REFINER_LAYERS = 2
NORM_EPS = 1e-5
T_SCALE = 1000.0
FF_DIM = DIM * 4              # 15360 (SwiGLU hidden)
TIMESTEP_DIM = 256            # sinusoidal embedding dim

# Text encoder
QWEN3_HIDDEN_SIZE = 2560


# ---------------------------------------------------------------------------
# CPU helper functions
# ---------------------------------------------------------------------------

def get_timestep_embedding(timesteps: np.ndarray, dim: int = 256) -> np.ndarray:
    """Sinusoidal timestep embedding."""
    half = dim // 2
    freqs = np.exp(-math.log(10000.0) / half * np.arange(half, dtype=np.float64))
    args = timesteps[:, None].astype(np.float64) * freqs[None, :]
    emb = np.concatenate([np.cos(args), np.sin(args)], axis=-1)
    return emb.astype(np.float32)


def compute_rope_freqs(ids: np.ndarray, axes_dim: List[int],
                       theta: float = 256.0) -> Tuple[np.ndarray, np.ndarray]:
    """Compute RoPE cos/sin for 3D position IDs.

    ids: (S, 3) position IDs [t, h, w]
    Returns: (cos, sin) each (S, sum(axes_dim)) = (S, 128)
    """
    cos_parts, sin_parts = [], []
    for i, dim in enumerate(axes_dim):
        pos = ids[:, i].astype(np.float64)
        half_dim = dim // 2
        freq = 1.0 / (theta ** (np.arange(half_dim, dtype=np.float64) / half_dim))
        angles = pos[:, None] * freq[None, :]
        cos_val = np.cos(angles)
        sin_val = np.sin(angles)
        cos_interleaved = np.repeat(cos_val, 2, axis=-1)
        sin_interleaved = np.repeat(sin_val, 2, axis=-1)
        cos_parts.append(cos_interleaved)
        sin_parts.append(sin_interleaved)
    return (np.concatenate(cos_parts, axis=-1).astype(np.float32),
            np.concatenate(sin_parts, axis=-1).astype(np.float32))


def apply_rotary_emb(x: np.ndarray, cos: np.ndarray,
                     sin: np.ndarray) -> np.ndarray:
    """Apply rotary embedding to (S, H, D) using interleaved pairs."""
    cos_ = cos[:, None, :]
    sin_ = sin[:, None, :]
    x_pairs = x.reshape(*x.shape[:-1], -1, 2)
    x_real = x_pairs[..., 0]
    x_imag = x_pairs[..., 1]
    x_rotated = np.stack([-x_imag, x_real], axis=-1).reshape(x.shape)
    return (x.astype(np.float32) * cos_ +
            x_rotated.astype(np.float32) * sin_).astype(np.float32)


def rms_norm_cpu(x: np.ndarray, w: np.ndarray,
                 eps: float = 1e-5) -> np.ndarray:
    """RMSNorm on last dimension. x: (..., D), w: (D,)."""
    x_f = x.astype(np.float32)
    rms = np.sqrt(np.mean(x_f ** 2, axis=-1, keepdims=True) + eps)
    return (x_f / rms * w.astype(np.float32)).astype(np.float32)


def prepare_latent_ids(height: int, width: int) -> np.ndarray:
    """3D position IDs for image latents: (t=0, h, w)."""
    num_tokens = height * width
    ids = np.zeros((num_tokens, 3), dtype=np.float32)
    hh, ww = np.meshgrid(np.arange(height), np.arange(width), indexing='ij')
    ids[:, 1] = hh.ravel()
    ids[:, 2] = ww.ravel()
    return ids


def prepare_text_ids(seq_len: int) -> np.ndarray:
    """3D position IDs for text tokens: all zeros (text has no spatial pos)."""
    return np.zeros((seq_len, 3), dtype=np.float32)


# ---------------------------------------------------------------------------
# Z-Image-Turbo WebGPU Model
# ---------------------------------------------------------------------------

class ZImageTurboWebGPU(WebGPUModel):
    """Z-Image-Turbo transformer on WebGPU.

    Architecture:
    1. Timestep embedding (sinusoidal → 2-layer MLP)
    2. Caption embedding (Linear → RMSNorm → Linear)
    3. Input projection (x_embedder: latent → DIM)
    4. 30 main transformer blocks (adaLN + self-attention + SwiGLU FF)
    5. 2 noise refiner blocks (same structure)
    6. Final layer (adaLN modulation + projection)

    Context refinement for text is done separately (2 context_refiner blocks).
    """

    def __init__(self, weights: Dict[str, np.ndarray]):
        # Collect K dimensions for kernel compilation
        k_dims = {
            IN_CHANNELS * PATCH_SIZE * PATCH_SIZE,  # 64: patchify input
            TIMESTEP_DIM,  # 256: sinusoidal embed
            DIM,           # 3840: main dimension
            CAP_FEAT_DIM,  # 2560: caption input
            FF_DIM,        # 15360: SwiGLU hidden
            FF_DIM * 2,    # 30720: fused gate+up
            DIM * 6,       # 23040: adaLN modulation output (6*D for main, 3*D for refiner)
            DIM * 3,       # 11520: adaLN modulation for refiners
            DIM * 2,       # 7680: final layer adaLN
        }

        super().__init__(
            weights,
            n_layer=N_LAYERS + N_REFINER_LAYERS * 2,
            n_head=NUM_HEADS,
            n_embd=DIM,
            n_vocab=1,
            head_dim=HEAD_DIM,
            intermediate_size=FF_DIM,
            k_dimensions=k_dims,
            norm_eps=NORM_EPS,
        )

        self._upload_weights_to_gpu()
        self._fuse_qkv_weights()
        self._fuse_gate_up_weights()
        self._precompute_caches()

    def _fuse_qkv_weights(self):
        """Fuse separate Q/K/V projection weights into single QKV matrices."""
        runner = self.cache.runner
        to_fp16 = lambda w: w.astype(np.float16) if w.dtype != np.float16 else w
        count = 0

        # Helper to fuse Q/K/V for a given block prefix
        def _fuse_block(pfx):
            nonlocal count
            q_key = f"{pfx}.attention.to_q.weight"
            k_key = f"{pfx}.attention.to_k.weight"
            v_key = f"{pfx}.attention.to_v.weight"
            fused_key = f"{pfx}.attention.qkv.weight"
            if q_key not in self.weights:
                return
            q_w = self.weights[q_key]
            k_w = self.weights[k_key]
            v_w = self.weights[v_key]
            fused = np.concatenate([q_w, k_w, v_w], axis=0)
            self._gpu_weights[fused_key] = runner.upload_to_gpu(
                to_fp16(fused), fused_key)
            count += 1

        for i in range(N_LAYERS):
            _fuse_block(f"layers.{i}")
        for i in range(N_REFINER_LAYERS):
            _fuse_block(f"context_refiner.{i}")
            _fuse_block(f"noise_refiner.{i}")
        print(f"  Fused & uploaded {count} QKV weight tensors")

    def _fuse_gate_up_weights(self):
        """Fuse gate (w1) + up (w3) weights into single gate_up matrices."""
        runner = self.cache.runner
        to_fp16 = lambda w: w.astype(np.float16) if w.dtype != np.float16 else w
        count = 0

        def _fuse_block(pfx):
            nonlocal count
            g_key = f"{pfx}.feed_forward.w1.weight"
            u_key = f"{pfx}.feed_forward.w3.weight"
            fused_key = f"{pfx}.feed_forward.gate_up.weight"
            if g_key not in self.weights:
                return
            g_w = self.weights[g_key]
            u_w = self.weights[u_key]
            fused = np.concatenate([g_w, u_w], axis=0)
            self._gpu_weights[fused_key] = runner.upload_to_gpu(
                to_fp16(fused), fused_key)
            count += 1

        for i in range(N_LAYERS):
            _fuse_block(f"layers.{i}")
        for i in range(N_REFINER_LAYERS):
            _fuse_block(f"context_refiner.{i}")
            _fuse_block(f"noise_refiner.{i}")
        print(f"  Fused & uploaded {count} gate+up weight tensors")

    def _compile_model_kernels(self):
        """Compile Z-Image-specific kernels."""
        self._compile_layer_norm()
        self._compile_rms_norm()
        self._compile_silu_mul()
        self._compile_full_attn()
        self._compile_qk_norm_rope()

    def _precompute_caches(self):
        """Pre-cache fp32 norm weights and zero biases."""
        # Pre-upload zero biases for common sizes
        runner = self.cache.runner
        self._zero_biases = {}
        for sz in (DIM, FF_DIM, DIM * 6, DIM * 3, DIM * 2,
                   IN_CHANNELS * PATCH_SIZE * PATCH_SIZE, CAP_FEAT_DIM):
            self._zero_biases[sz] = runner.upload_to_gpu(
                np.zeros(sz, dtype=np.float32), f"zero_bias_{sz}")

        # Pre-cache fp32 conversions of norm weights and upload to GPU
        self._fp32_cache = {}
        self._gpu_norm_weights = {}
        for name, w in self.weights.items():
            if "norm" in name and w.ndim == 1:
                w32 = w.astype(np.float32) if w.dtype != np.float32 else w
                self._fp32_cache[name] = w32
                if name not in self._gpu_weights:
                    self._gpu_norm_weights[name] = runner.upload_to_gpu(
                        w32, f"norm_{name}")
                else:
                    self._gpu_norm_weights[name] = self._gpu_weights[name]

    def _upload_weights_to_gpu(self):
        """Upload all weights to GPU."""
        runner = self.cache.runner
        count = 0
        for name, w in self.weights.items():
            if w.ndim == 2 and w.size >= 256:
                fp16 = w if w.dtype == np.float16 else w.astype(np.float16)
                self._gpu_weights[name] = runner.upload_to_gpu(fp16, name)
                count += 1
            elif w.ndim == 1:
                w32 = w.astype(np.float32) if w.dtype != np.float32 else w
                self._gpu_weights[name] = runner.upload_to_gpu(w32, name)
                count += 1
        print(f"  Uploaded {count} weight tensors "
              f"({sum(g.size for g in self._gpu_weights.values()) // (1024**2)} MB) "
              f"to GPU")

    # ------------------------------------------------------------------
    # Primitive wrappers
    # ------------------------------------------------------------------

    def _w(self, name: str):
        """Get weight as GPUBuffer."""
        return self._gpu_weights[name]

    def _get_fp32(self, name: str):
        """Get weight as fp32 (using pre-cached conversion if available)."""
        cached = self._fp32_cache.get(name)
        if cached is not None:
            return cached
        w = self.weights[name]
        return w.astype(np.float32) if w.dtype != np.float32 else w

    def _linear_proj(self, x, weight_name: str, out_features: int,
                     gpu_out: bool = False):
        """Bias-free linear projection using fp16 weights."""
        w = self._w(weight_name)
        T = x.shape[0] if not isinstance(x, GPUBuffer) else x.shape[0]
        K = x.shape[1] if not isinstance(x, GPUBuffer) else x.shape[1]
        bias = self._zero_biases.get(out_features)
        if bias is None:
            bias = np.zeros(out_features, dtype=np.float32)
        return self._linear_fp16w(x, w, bias, out_features, K=K,
                                  gpu_out=gpu_out)

    def _linear_with_bias(self, x, weight_name: str, bias_name: str,
                          out_features: int):
        """Linear with bias."""
        w = self._w(weight_name)
        b = self.weights[bias_name]
        if b.dtype == np.float16:
            b = b.astype(np.float32)
        K = x.shape[1] if not isinstance(x, GPUBuffer) else x.shape[1]
        return self._linear_fp16w(x, w, b, out_features, K=K)

    # ------------------------------------------------------------------
    # Timestep embedding
    # ------------------------------------------------------------------

    def _compute_temb(self, timestep: float) -> np.ndarray:
        """Compute timestep embedding: sinusoidal → MLP (CPU for tiny vectors)."""
        t_scaled = np.array([timestep * T_SCALE], dtype=np.float32)
        t_proj = get_timestep_embedding(t_scaled, TIMESTEP_DIM)  # (1, 256)

        # t_embedder MLP on CPU (tiny: 1×256 → 1×3840)
        w1 = self._get_fp32("t_embedder.mlp.0.weight")
        b1 = self._get_fp32("t_embedder.mlp.0.bias")
        t_emb = t_proj @ w1.T + b1
        t_emb = t_emb * (1.0 / (1.0 + np.exp(-t_emb)))  # SiLU
        w2 = self._get_fp32("t_embedder.mlp.2.weight")
        b2 = self._get_fp32("t_embedder.mlp.2.bias")
        t_emb = t_emb @ w2.T + b2
        return t_emb  # (1, DIM)

    # ------------------------------------------------------------------
    # Caption embedding
    # ------------------------------------------------------------------

    def _embed_caption(self, caption_embeds: np.ndarray) -> np.ndarray:
        """Project caption features: RMSNorm → Linear(with_bias).

        cap_embedder.0: RMSNorm (weight only, no bias)
        cap_embedder.1: Linear(CAP_FEAT_DIM → DIM, bias)

        caption_embeds: (S, CAP_FEAT_DIM)
        Returns: (S, DIM)
        """
        # RMSNorm
        x = rms_norm_cpu(caption_embeds,
                         self._get_fp32("cap_embedder.0.weight"),
                         NORM_EPS)
        # Linear projection to DIM
        x = self._linear_with_bias(x,
                                   "cap_embedder.1.weight",
                                   "cap_embedder.1.bias", DIM)
        return x

    # ------------------------------------------------------------------
    # Transformer block
    # ------------------------------------------------------------------

    def _transformer_block(self, x, c,
                           temb: np.ndarray, cos: np.ndarray,
                           sin: np.ndarray, layer: int):
        """One main transformer block with adaLN modulation (GPU-resident).

        x: (T, DIM) GPUBuffer — combined image + text hidden states
        c: not used (kept for API compat)
        temb: (1, DIM) — timestep embedding (CPU)
        Returns: (T, DIM) GPUBuffer
        """
        pfx = f"layers.{layer}"
        D = DIM
        n_head = NUM_HEADS
        HD = HEAD_DIM
        runner = self.cache.runner
        _p = self._profiling

        # adaLN modulation: SiLU(temb) → Linear → chunk into 6 (CPU, tiny)
        if _p: self._begin_cpu(f"L{layer}/adaLN_cpu")
        if _p: self._set_gpu_op(f"L{layer}/adaLN")
        mod = temb * (1.0 / (1.0 + np.exp(-temb)))  # SiLU
        mod = self._linear_with_bias(
            mod, f"{pfx}.adaLN_modulation.0.weight",
            f"{pfx}.adaLN_modulation.0.bias", D * 6)
        chunks = np.split(mod.ravel(), 6)
        shift_attn, scale_attn, gate_attn = chunks[0], chunks[1], chunks[2]
        shift_ff, scale_ff, gate_ff = chunks[3], chunks[4], chunks[5]

        # Upload modulation params to GPU
        scale_attn_gpu = runner.upload_to_gpu(scale_attn.astype(np.float32), f"_tb{layer}_sa_sc")
        shift_attn_gpu = runner.upload_to_gpu(shift_attn.astype(np.float32), f"_tb{layer}_sa_sh")
        gate_attn_gpu = runner.upload_to_gpu(gate_attn.astype(np.float32), f"_tb{layer}_sa_gt")
        scale_ff_gpu = runner.upload_to_gpu(scale_ff.astype(np.float32), f"_tb{layer}_ff_sc")
        shift_ff_gpu = runner.upload_to_gpu(shift_ff.astype(np.float32), f"_tb{layer}_ff_sh")
        gate_ff_gpu = runner.upload_to_gpu(gate_ff.astype(np.float32), f"_tb{layer}_ff_gt")
        if _p: self._end_cpu(f"L{layer}/adaLN_cpu")

        T = x.shape[0]

        # --- Attention ---
        if _p: self._set_gpu_op(f"L{layer}/norm1+mod")
        norm_w = self._gpu_norm_weights[f"{pfx}.attention_norm1.weight"]
        norm_x = self._rms_norm(x, norm_w, eps=NORM_EPS, gpu_out=True)
        norm_x = self._mod_scale_shift(norm_x, scale_attn_gpu, shift_attn_gpu,
                                       D, gpu_out=True)
        norm_x.shape = (T, D)

        if _p: self._set_gpu_op(f"L{layer}/qkv")
        qkv_gpu = self._linear_proj(norm_x, f"{pfx}.attention.qkv.weight",
                                    3 * D, gpu_out=True)
        qkv_gpu.shape = (T, 3 * D)

        if _p: self._set_gpu_op(f"L{layer}/qk_norm_rope")
        q_gpu, k_gpu, v_gpu = self._qk_norm_rope(
            qkv_gpu,
            self._get_fp32(f"{pfx}.attention.norm_q.weight"),
            self._get_fp32(f"{pfx}.attention.norm_k.weight"),
            cos, sin, n_head, T, eps=NORM_EPS, gpu_out=True)

        if _p: self._set_gpu_op(f"L{layer}/attn")
        attn_out = self._full_attention_multihead(q_gpu, k_gpu, v_gpu,
                                                  n_head, gpu_out=True)
        attn_out.shape = (T, D)

        if _p: self._set_gpu_op(f"L{layer}/o_proj")
        attn_out = self._linear_proj(attn_out,
                                     f"{pfx}.attention.to_out.0.weight",
                                     D, gpu_out=True)

        if _p: self._set_gpu_op(f"L{layer}/norm2+res")
        norm2_w = self._gpu_norm_weights[f"{pfx}.attention_norm2.weight"]
        attn_out = self._rms_norm(attn_out, norm2_w, eps=NORM_EPS, gpu_out=True)
        self._gate_residual_add(x, gate_attn_gpu, attn_out, D)

        # --- Feed-forward (SwiGLU) ---
        if _p: self._set_gpu_op(f"L{layer}/ff_norm+mod")
        ffn_norm_w = self._gpu_norm_weights[f"{pfx}.ffn_norm1.weight"]
        norm_x = self._rms_norm(x, ffn_norm_w, eps=NORM_EPS, gpu_out=True)
        norm_x = self._mod_scale_shift(norm_x, scale_ff_gpu, shift_ff_gpu,
                                       D, gpu_out=True)
        norm_x.shape = (T, D)

        if _p: self._set_gpu_op(f"L{layer}/gate_up")
        gate_up = self._linear_proj(norm_x,
                                    f"{pfx}.feed_forward.gate_up.weight",
                                    FF_DIM * 2, gpu_out=True)

        if _p: self._set_gpu_op(f"L{layer}/silu_mul")
        ff_out = self._silu_mul_fused(gate_up, FF_DIM, gpu_out=True)

        if _p: self._set_gpu_op(f"L{layer}/down")
        ff_out = self._linear_proj(ff_out,
                                   f"{pfx}.feed_forward.w2.weight",
                                   D, gpu_out=True)

        if _p: self._set_gpu_op(f"L{layer}/ff_norm2+res")
        ffn_norm2_w = self._gpu_norm_weights[f"{pfx}.ffn_norm2.weight"]
        ff_out = self._rms_norm(ff_out, ffn_norm2_w, eps=NORM_EPS, gpu_out=True)
        self._gate_residual_add(x, gate_ff_gpu, ff_out, D)

        if _p: self._clear_gpu_op()
        return x

    # ------------------------------------------------------------------
    # Context refiner block
    # ------------------------------------------------------------------

    def _context_refiner_block(self, x,
                               cos: np.ndarray, sin: np.ndarray,
                               layer: int):
        """Context refiner block (no modulation, GPU-resident)."""
        pfx = f"context_refiner.{layer}"
        D = DIM
        n_head = NUM_HEADS
        HD = HEAD_DIM
        _p = self._profiling

        T = x.shape[0]

        if _p: self._set_gpu_op(f"CR{layer}/norm1")
        norm_w = self._gpu_norm_weights[f"{pfx}.attention_norm1.weight"]
        norm_x = self._rms_norm(x, norm_w, eps=NORM_EPS, gpu_out=True)
        norm_x.shape = (T, D)

        if _p: self._set_gpu_op(f"CR{layer}/qkv")
        qkv_gpu = self._linear_proj(norm_x, f"{pfx}.attention.qkv.weight",
                                    3 * D, gpu_out=True)
        qkv_gpu.shape = (T, 3 * D)

        if _p: self._set_gpu_op(f"CR{layer}/qk_norm_rope")
        q_gpu, k_gpu, v_gpu = self._qk_norm_rope(
            qkv_gpu,
            self._get_fp32(f"{pfx}.attention.norm_q.weight"),
            self._get_fp32(f"{pfx}.attention.norm_k.weight"),
            cos, sin, n_head, T, eps=NORM_EPS, gpu_out=True)

        if _p: self._set_gpu_op(f"CR{layer}/attn")
        attn_out = self._full_attention_multihead(q_gpu, k_gpu, v_gpu,
                                                  n_head, gpu_out=True)
        attn_out.shape = (T, D)

        if _p: self._set_gpu_op(f"CR{layer}/o_proj+norm2+res")
        attn_out = self._linear_proj(attn_out,
                                     f"{pfx}.attention.to_out.0.weight",
                                     D, gpu_out=True)
        norm2_w = self._gpu_norm_weights[f"{pfx}.attention_norm2.weight"]
        attn_out = self._rms_norm(attn_out, norm2_w, eps=NORM_EPS, gpu_out=True)
        self._add_inplace(x, attn_out)

        if _p: self._set_gpu_op(f"CR{layer}/ff_norm")
        ffn_norm_w = self._gpu_norm_weights[f"{pfx}.ffn_norm1.weight"]
        norm_x = self._rms_norm(x, ffn_norm_w, eps=NORM_EPS, gpu_out=True)
        norm_x.shape = (T, D)

        if _p: self._set_gpu_op(f"CR{layer}/gate_up")
        gate_up = self._linear_proj(norm_x,
                                    f"{pfx}.feed_forward.gate_up.weight",
                                    FF_DIM * 2, gpu_out=True)

        if _p: self._set_gpu_op(f"CR{layer}/silu_mul+down")
        ff_out = self._silu_mul_fused(gate_up, FF_DIM, gpu_out=True)
        ff_out = self._linear_proj(ff_out,
                                   f"{pfx}.feed_forward.w2.weight",
                                   D, gpu_out=True)

        if _p: self._set_gpu_op(f"CR{layer}/ff_norm2+res")
        ffn_norm2_w = self._gpu_norm_weights[f"{pfx}.ffn_norm2.weight"]
        ff_out = self._rms_norm(ff_out, ffn_norm2_w, eps=NORM_EPS, gpu_out=True)
        self._add_inplace(x, ff_out)

        if _p: self._clear_gpu_op()
        return x

    # ------------------------------------------------------------------
    # Noise refiner block
    # ------------------------------------------------------------------

    def _noise_refiner_block(self, x, temb: np.ndarray,
                             cos: np.ndarray, sin: np.ndarray,
                             layer: int):
        """Noise refiner block (with adaLN modulation, GPU-resident)."""
        pfx = f"noise_refiner.{layer}"
        D = DIM
        n_head = NUM_HEADS
        HD = HEAD_DIM
        runner = self.cache.runner
        _p = self._profiling

        # adaLN modulation (CPU, tiny)
        if _p: self._begin_cpu(f"NR{layer}/adaLN_cpu")
        if _p: self._set_gpu_op(f"NR{layer}/adaLN")
        mod = temb * (1.0 / (1.0 + np.exp(-temb)))
        mod = self._linear_with_bias(
            mod, f"{pfx}.adaLN_modulation.0.weight",
            f"{pfx}.adaLN_modulation.0.bias", D * 6)
        chunks = np.split(mod.ravel(), 6)
        shift_attn, scale_attn, gate_attn = chunks[0], chunks[1], chunks[2]
        shift_ff, scale_ff, gate_ff = chunks[3], chunks[4], chunks[5]

        # Upload modulation params to GPU
        scale_attn_gpu = runner.upload_to_gpu(scale_attn.astype(np.float32), f"_nr{layer}_sa_sc")
        shift_attn_gpu = runner.upload_to_gpu(shift_attn.astype(np.float32), f"_nr{layer}_sa_sh")
        gate_attn_gpu = runner.upload_to_gpu(gate_attn.astype(np.float32), f"_nr{layer}_sa_gt")
        scale_ff_gpu = runner.upload_to_gpu(scale_ff.astype(np.float32), f"_nr{layer}_ff_sc")
        shift_ff_gpu = runner.upload_to_gpu(shift_ff.astype(np.float32), f"_nr{layer}_ff_sh")
        gate_ff_gpu = runner.upload_to_gpu(gate_ff.astype(np.float32), f"_nr{layer}_ff_gt")
        if _p: self._end_cpu(f"NR{layer}/adaLN_cpu")

        T = x.shape[0]

        if _p: self._set_gpu_op(f"NR{layer}/norm1+mod")
        norm_w = self._gpu_norm_weights[f"{pfx}.attention_norm1.weight"]
        norm_x = self._rms_norm(x, norm_w, eps=NORM_EPS, gpu_out=True)
        norm_x = self._mod_scale_shift(norm_x, scale_attn_gpu, shift_attn_gpu,
                                       D, gpu_out=True)
        norm_x.shape = (T, D)

        if _p: self._set_gpu_op(f"NR{layer}/qkv")
        qkv_gpu = self._linear_proj(norm_x, f"{pfx}.attention.qkv.weight",
                                    3 * D, gpu_out=True)
        qkv_gpu.shape = (T, 3 * D)

        if _p: self._set_gpu_op(f"NR{layer}/qk_norm_rope")
        q_gpu, k_gpu, v_gpu = self._qk_norm_rope(
            qkv_gpu,
            self._get_fp32(f"{pfx}.attention.norm_q.weight"),
            self._get_fp32(f"{pfx}.attention.norm_k.weight"),
            cos, sin, n_head, T, eps=NORM_EPS, gpu_out=True)

        if _p: self._set_gpu_op(f"NR{layer}/attn")
        attn_out = self._full_attention_multihead(q_gpu, k_gpu, v_gpu,
                                                  n_head, gpu_out=True)
        attn_out.shape = (T, D)

        if _p: self._set_gpu_op(f"NR{layer}/o_proj+norm2+res")
        attn_out = self._linear_proj(attn_out,
                                     f"{pfx}.attention.to_out.0.weight",
                                     D, gpu_out=True)
        norm2_w = self._gpu_norm_weights[f"{pfx}.attention_norm2.weight"]
        attn_out = self._rms_norm(attn_out, norm2_w, eps=NORM_EPS, gpu_out=True)
        self._gate_residual_add(x, gate_attn_gpu, attn_out, D)

        if _p: self._set_gpu_op(f"NR{layer}/ff_norm+mod")
        ffn_norm_w = self._gpu_norm_weights[f"{pfx}.ffn_norm1.weight"]
        norm_x = self._rms_norm(x, ffn_norm_w, eps=NORM_EPS, gpu_out=True)
        norm_x = self._mod_scale_shift(norm_x, scale_ff_gpu, shift_ff_gpu,
                                       D, gpu_out=True)
        norm_x.shape = (T, D)

        if _p: self._set_gpu_op(f"NR{layer}/gate_up")
        gate_up = self._linear_proj(norm_x,
                                    f"{pfx}.feed_forward.gate_up.weight",
                                    FF_DIM * 2, gpu_out=True)

        if _p: self._set_gpu_op(f"NR{layer}/silu_mul+down")
        ff_out = self._silu_mul_fused(gate_up, FF_DIM, gpu_out=True)
        ff_out = self._linear_proj(ff_out,
                                   f"{pfx}.feed_forward.w2.weight",
                                   D, gpu_out=True)

        if _p: self._set_gpu_op(f"NR{layer}/ff_norm2+res")
        ffn_norm2_w = self._gpu_norm_weights[f"{pfx}.ffn_norm2.weight"]
        ff_out = self._rms_norm(ff_out, ffn_norm2_w, eps=NORM_EPS, gpu_out=True)
        self._gate_residual_add(x, gate_ff_gpu, ff_out, D)

        if _p: self._clear_gpu_op()
        return x

    # ------------------------------------------------------------------
    # Output layer
    # ------------------------------------------------------------------

    def _output_layer(self, x: np.ndarray,
                      temb: np.ndarray) -> np.ndarray:
        """Final layer: adaLN modulation + projection.

        all_final_layer.2-1: adaLN_modulation (scale+shift) + linear
        """
        D = DIM
        out_dim = IN_CHANNELS * PATCH_SIZE * PATCH_SIZE  # 64

        # adaLN modulation
        mod = temb * (1.0 / (1.0 + np.exp(-temb)))
        mod = self._linear_with_bias(
            mod, "all_final_layer.2-1.adaLN_modulation.1.weight",
            "all_final_layer.2-1.adaLN_modulation.1.bias", D * 2)
        scale, shift = np.split(mod, 2, axis=-1)

        # LayerNorm (no affine) + adaLN
        x_f = x.astype(np.float32)
        mean = np.mean(x_f, axis=-1, keepdims=True)
        var = np.var(x_f, axis=-1, keepdims=True)
        norm_x = (x_f - mean) / np.sqrt(var + NORM_EPS)
        norm_x = norm_x * (1.0 + scale) + shift

        # Linear projection to patch space
        out = self._linear_with_bias(
            norm_x, "all_final_layer.2-1.linear.weight",
            "all_final_layer.2-1.linear.bias", out_dim)
        return out

    # ------------------------------------------------------------------
    # Full forward pass
    # ------------------------------------------------------------------

    def forward(self, latents: np.ndarray,
                encoder_hidden_states: np.ndarray,
                timestep: float,
                img_ids: np.ndarray,
                txt_ids: np.ndarray) -> np.ndarray:
        """Run one denoising step.

        latents: (T_img, IN_CHANNELS * P * P) packed image latents
        encoder_hidden_states: (T_txt, CAP_FEAT_DIM) text features
        timestep: fractional timestep in [0, 1]
        img_ids: (T_img, 3) position IDs [t, h, w]
        txt_ids: (T_txt, 3) position IDs
        Returns: (T_img, IN_CHANNELS * P * P) noise prediction
        """
        T_img = latents.shape[0]
        T_txt = encoder_hidden_states.shape[0]
        runner = self.cache.runner
        _p = self._profiling

        # 1. Timestep embedding
        if _p: self._begin_cpu("temb")
        temb = self._compute_temb(timestep)  # (1, DIM)
        if _p: self._end_cpu("temb")

        # 2. Caption embedding
        if _p: self._begin_cpu("cap_embed")
        cap = self._embed_caption(encoder_hidden_states)  # (T_txt, DIM)
        if _p: self._end_cpu("cap_embed")

        # 3. Context refinement (2 blocks on text only)
        if _p: self._begin_cpu("ctx_refiner")
        text_cos, text_sin = compute_rope_freqs(txt_ids, AXES_DIMS, ROPE_THETA)
        cap_gpu = runner.upload_to_gpu(
            cap.astype(np.float32).ravel(), "ctx_refiner_input")
        cap_gpu.shape = (T_txt, DIM)
        for i in range(N_REFINER_LAYERS):
            cap_gpu = self._context_refiner_block(cap_gpu, text_cos, text_sin, i)
        if _p: self._end_cpu("ctx_refiner")

        # 4. Input projection: patchified latents → DIM
        if _p: self._begin_cpu("input_proj")
        x = self._linear_with_bias(
            latents, "all_x_embedder.2-1.weight",
            "all_x_embedder.2-1.bias", DIM)
        if _p: self._end_cpu("input_proj")

        # 5. Concatenate: [text | image] for joint attention
        if _p: self._begin_cpu("concat+rope")
        x_gpu = runner.upload_to_gpu(
            x.astype(np.float32).ravel(), "main_input_img")
        x_gpu.shape = (T_img, DIM)
        x_gpu = self._concat_gpu(cap_gpu, x_gpu)
        T_total = T_txt + T_img
        x_gpu.shape = (T_total, DIM)

        # 6. Compute RoPE for concatenated sequence
        all_ids = np.concatenate([txt_ids, img_ids], axis=0)
        all_cos, all_sin = compute_rope_freqs(all_ids, AXES_DIMS, ROPE_THETA)
        if _p: self._end_cpu("concat+rope")

        # 7. Main transformer blocks (30 layers, all GPU-resident)
        if _p: self._begin_cpu("main_blocks")
        for i in range(N_LAYERS):
            if _p: self._begin_cpu(f"L{i}")
            x_gpu = self._transformer_block(x_gpu, None, temb,
                                            all_cos, all_sin, i)
            if _p: self._end_cpu(f"L{i}")
        if _p: self._end_cpu("main_blocks")

        # 8. Extract image tokens (GPU split)
        if _p: self._begin_cpu("split")
        N_txt_elems = T_txt * DIM
        N_img_elems = T_img * DIM
        x_gpu = self._split_gpu(x_gpu, N_txt_elems, N_img_elems)
        x_gpu.shape = (T_img, DIM)
        if _p: self._end_cpu("split")

        # 9. Noise refiner blocks (2 layers, image-only, GPU-resident)
        if _p: self._begin_cpu("noise_refiner")
        img_cos, img_sin = compute_rope_freqs(img_ids, AXES_DIMS, ROPE_THETA)
        for i in range(N_REFINER_LAYERS):
            if _p: self._begin_cpu(f"NR{i}")
            x_gpu = self._noise_refiner_block(x_gpu, temb,
                                              img_cos, img_sin, i)
            if _p: self._end_cpu(f"NR{i}")
        if _p: self._end_cpu("noise_refiner")

        # 10. Output layer (readback to CPU for final projection)
        if _p: self._begin_cpu("output_layer")
        x = runner.readback(x_gpu).reshape(T_img, DIM)
        output = self._output_layer(x, temb)
        if _p: self._end_cpu("output_layer")
        return output


# ---------------------------------------------------------------------------
# Text encoding
# ---------------------------------------------------------------------------

def encode_prompt(prompt: str, tokenizer, text_encoder,
                  device="cpu", max_seq_len: int = 256) -> np.ndarray:
    """Encode prompt using Qwen3 text encoder.

    Returns: (seq_len, CAP_FEAT_DIM) prompt embeddings.
    """
    import torch

    inputs = tokenizer(
        prompt, return_tensors="pt", padding="max_length",
        truncation=True, max_length=max_seq_len)
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    with torch.no_grad():
        output = text_encoder(
            input_ids=input_ids, attention_mask=attention_mask,
            output_hidden_states=True, use_cache=False)

    # Use last hidden state
    hidden = output.hidden_states[-1]  # (1, seq, 2560)
    return hidden.float().squeeze(0).cpu().numpy()  # (seq, 2560)


# ---------------------------------------------------------------------------
# Denoising loop
# ---------------------------------------------------------------------------

def generate_image(model: ZImageTurboWebGPU,
                   prompt_embeds: np.ndarray,
                   height: int, width: int,
                   num_steps: int = 4,
                   seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """Run the full denoising loop.

    Returns: (latents, latent_ids) as numpy arrays.
    """
    # Compute latent dimensions (VAE divides by 8, patch divides by 2)
    lat_h = height // 8 // PATCH_SIZE
    lat_w = width // 8 // PATCH_SIZE
    patch_dim = IN_CHANNELS * PATCH_SIZE * PATCH_SIZE  # 64

    # Generate noise
    rng = np.random.RandomState(seed)
    noise = rng.randn(lat_h * lat_w, patch_dim).astype(np.float32)

    latents = noise
    img_ids = prepare_latent_ids(lat_h, lat_w)
    txt_ids = prepare_text_ids(prompt_embeds.shape[0])

    # Scheduler
    from diffusers.schedulers.scheduling_flow_match_euler_discrete import (
        FlowMatchEulerDiscreteScheduler,
    )
    scheduler = FlowMatchEulerDiscreteScheduler(shift=3.0)
    scheduler.set_timesteps(num_steps)
    timesteps = scheduler.timesteps.numpy()
    sigmas = scheduler.sigmas.numpy()

    print(f"  Latent: {lat_h}x{lat_w} = {lat_h * lat_w} tokens, patch_dim={patch_dim}")
    print(f"  Text: {prompt_embeds.shape}")
    print(f"  Steps: {num_steps}")

    for i, t in enumerate(timesteps):
        t0 = time.time()
        sigma = sigmas[i]
        timestep_frac = t.item() / 1000.0

        noise_pred = model.forward(
            latents, prompt_embeds,
            timestep=timestep_frac,
            img_ids=img_ids, txt_ids=txt_ids)

        sigma_next = sigmas[i + 1] if i + 1 < len(sigmas) else 0.0
        dt = sigma_next - sigma
        latents = latents + dt * noise_pred

        elapsed = time.time() - t0
        print(f"  Step {i+1}/{num_steps}: t={t.item():.1f}, "
              f"sigma={sigma:.4f}, dt={dt:.4f}, "
              f"pred std={noise_pred.std():.4f}, "
              f"time={elapsed:.2f}s")

    return latents, img_ids


# ---------------------------------------------------------------------------
# Verification with random weights
# ---------------------------------------------------------------------------

def verify_with_random_weights():
    """Quick verification with scaled-down random weights.

    Uses a smaller model (4 layers, dim=384) to fit in memory while
    exercising the full pipeline: timestep embed → caption embed →
    context refiner → main blocks → noise refiner → output layer.
    """
    print("=" * 60)
    print("Z-Image-Turbo WebGPU Pipeline Verification (random weights)")
    print("=" * 60)
    rng = np.random.RandomState(42)
    s = 0.02

    # Scaled-down config for verification
    D = 384           # dim (real: 3840)
    n_heads = 6       # (real: 30)
    hd = D // n_heads  # 64
    n_layers = 4      # (real: 30)
    n_refiner = 2     # same as real
    ff = D * 4        # 1536 (real: 15360)
    cap_dim = 256     # (real: 2560)
    in_ch = IN_CHANNELS
    ps = PATCH_SIZE
    patch_dim = in_ch * ps * ps  # 64
    td = TIMESTEP_DIM  # 256

    W = {}

    # Timestep embedder
    W["t_embedder.mlp.0.weight"] = rng.randn(D, td).astype(np.float32) * s
    W["t_embedder.mlp.0.bias"] = np.zeros(D, dtype=np.float32)
    W["t_embedder.mlp.2.weight"] = rng.randn(D, D).astype(np.float32) * s
    W["t_embedder.mlp.2.bias"] = np.zeros(D, dtype=np.float32)

    # Caption embedder: RMSNorm(cap_dim) → Linear(cap_dim → D, bias)
    W["cap_embedder.0.weight"] = np.ones(cap_dim, dtype=np.float32)
    W["cap_embedder.1.weight"] = rng.randn(D, cap_dim).astype(np.float32) * s
    W["cap_embedder.1.bias"] = np.zeros(D, dtype=np.float32)

    # Input projection
    W["all_x_embedder.2-1.weight"] = rng.randn(D, patch_dim).astype(np.float32) * s
    W["all_x_embedder.2-1.bias"] = np.zeros(D, dtype=np.float32)

    # Main transformer blocks
    for i in range(n_layers):
        pfx = f"layers.{i}"
        W[f"{pfx}.adaLN_modulation.0.weight"] = rng.randn(D * 6, D).astype(np.float32) * s
        W[f"{pfx}.adaLN_modulation.0.bias"] = np.zeros(D * 6, dtype=np.float32)
        W[f"{pfx}.attention_norm1.weight"] = np.ones(D, dtype=np.float32)
        W[f"{pfx}.attention_norm2.weight"] = np.ones(D, dtype=np.float32)
        W[f"{pfx}.attention.to_q.weight"] = rng.randn(D, D).astype(np.float32) * s
        W[f"{pfx}.attention.to_k.weight"] = rng.randn(D, D).astype(np.float32) * s
        W[f"{pfx}.attention.to_v.weight"] = rng.randn(D, D).astype(np.float32) * s
        W[f"{pfx}.attention.to_out.0.weight"] = rng.randn(D, D).astype(np.float32) * s
        W[f"{pfx}.attention.norm_q.weight"] = np.ones(hd, dtype=np.float32)
        W[f"{pfx}.attention.norm_k.weight"] = np.ones(hd, dtype=np.float32)
        W[f"{pfx}.ffn_norm1.weight"] = np.ones(D, dtype=np.float32)
        W[f"{pfx}.ffn_norm2.weight"] = np.ones(D, dtype=np.float32)
        W[f"{pfx}.feed_forward.w1.weight"] = rng.randn(ff, D).astype(np.float32) * s
        W[f"{pfx}.feed_forward.w2.weight"] = rng.randn(D, ff).astype(np.float32) * s
        W[f"{pfx}.feed_forward.w3.weight"] = rng.randn(ff, D).astype(np.float32) * s

    # Context refiner
    for i in range(n_refiner):
        pfx = f"context_refiner.{i}"
        W[f"{pfx}.attention_norm1.weight"] = np.ones(D, dtype=np.float32)
        W[f"{pfx}.attention_norm2.weight"] = np.ones(D, dtype=np.float32)
        W[f"{pfx}.attention.to_q.weight"] = rng.randn(D, D).astype(np.float32) * s
        W[f"{pfx}.attention.to_k.weight"] = rng.randn(D, D).astype(np.float32) * s
        W[f"{pfx}.attention.to_v.weight"] = rng.randn(D, D).astype(np.float32) * s
        W[f"{pfx}.attention.to_out.0.weight"] = rng.randn(D, D).astype(np.float32) * s
        W[f"{pfx}.attention.norm_q.weight"] = np.ones(hd, dtype=np.float32)
        W[f"{pfx}.attention.norm_k.weight"] = np.ones(hd, dtype=np.float32)
        W[f"{pfx}.ffn_norm1.weight"] = np.ones(D, dtype=np.float32)
        W[f"{pfx}.ffn_norm2.weight"] = np.ones(D, dtype=np.float32)
        W[f"{pfx}.feed_forward.w1.weight"] = rng.randn(ff, D).astype(np.float32) * s
        W[f"{pfx}.feed_forward.w2.weight"] = rng.randn(D, ff).astype(np.float32) * s
        W[f"{pfx}.feed_forward.w3.weight"] = rng.randn(ff, D).astype(np.float32) * s

    # Noise refiner
    for i in range(n_refiner):
        pfx = f"noise_refiner.{i}"
        W[f"{pfx}.adaLN_modulation.0.weight"] = rng.randn(D * 6, D).astype(np.float32) * s
        W[f"{pfx}.adaLN_modulation.0.bias"] = np.zeros(D * 6, dtype=np.float32)
        W[f"{pfx}.attention_norm1.weight"] = np.ones(D, dtype=np.float32)
        W[f"{pfx}.attention_norm2.weight"] = np.ones(D, dtype=np.float32)
        W[f"{pfx}.attention.to_q.weight"] = rng.randn(D, D).astype(np.float32) * s
        W[f"{pfx}.attention.to_k.weight"] = rng.randn(D, D).astype(np.float32) * s
        W[f"{pfx}.attention.to_v.weight"] = rng.randn(D, D).astype(np.float32) * s
        W[f"{pfx}.attention.to_out.0.weight"] = rng.randn(D, D).astype(np.float32) * s
        W[f"{pfx}.attention.norm_q.weight"] = np.ones(hd, dtype=np.float32)
        W[f"{pfx}.attention.norm_k.weight"] = np.ones(hd, dtype=np.float32)
        W[f"{pfx}.ffn_norm1.weight"] = np.ones(D, dtype=np.float32)
        W[f"{pfx}.ffn_norm2.weight"] = np.ones(D, dtype=np.float32)
        W[f"{pfx}.feed_forward.w1.weight"] = rng.randn(ff, D).astype(np.float32) * s
        W[f"{pfx}.feed_forward.w2.weight"] = rng.randn(D, ff).astype(np.float32) * s
        W[f"{pfx}.feed_forward.w3.weight"] = rng.randn(ff, D).astype(np.float32) * s

    # Final layer
    W["all_final_layer.2-1.adaLN_modulation.1.weight"] = rng.randn(D * 2, D).astype(np.float32) * s
    W["all_final_layer.2-1.adaLN_modulation.1.bias"] = np.zeros(D * 2, dtype=np.float32)
    W["all_final_layer.2-1.linear.weight"] = rng.randn(patch_dim, D).astype(np.float32) * s
    W["all_final_layer.2-1.linear.bias"] = np.zeros(patch_dim, dtype=np.float32)

    print(f"\nModel (scaled-down): dim={D}, layers={n_layers}, heads={n_heads}, "
          f"ff={ff}, refiners={n_refiner}")
    print(f"  Created {len(W)} weight tensors")

    # Temporarily override module-level constants for scaled-down verify
    global DIM, NUM_HEADS, HEAD_DIM, FF_DIM, N_LAYERS, N_REFINER_LAYERS, CAP_FEAT_DIM, AXES_DIMS
    saved = (DIM, NUM_HEADS, HEAD_DIM, FF_DIM, N_LAYERS, N_REFINER_LAYERS, CAP_FEAT_DIM, AXES_DIMS)
    DIM, NUM_HEADS, HEAD_DIM = D, n_heads, hd
    FF_DIM, N_LAYERS, N_REFINER_LAYERS, CAP_FEAT_DIM = ff, n_layers, n_refiner, cap_dim
    AXES_DIMS = [16, 24, 24]  # sum=64=hd

    model = ZImageTurboWebGPU(W)

    # Test inputs
    T_img = 16  # 4x4 latent grid
    T_txt = 4
    latents = rng.randn(T_img, patch_dim).astype(np.float32) * 0.1
    ctx = rng.randn(T_txt, cap_dim).astype(np.float32) * 0.1
    img_ids = prepare_latent_ids(4, 4)
    txt_ids = prepare_text_ids(T_txt)

    t0 = time.time()
    out = model.forward(latents, ctx, timestep=0.5,
                        img_ids=img_ids, txt_ids=txt_ids)
    t1 = time.time()

    print(f"\nForward pass: ({T_img}, {patch_dim}) → {out.shape} "
          f"in {(t1-t0)*1000:.0f}ms")
    print(f"Output range: [{out.min():.4f}, {out.max():.4f}]")
    print(f"Output mean: {out.mean():.6f}, std: {out.std():.6f}")

    is_finite = np.all(np.isfinite(out))
    has_signal = out.std() > 1e-6
    correct_shape = out.shape == (T_img, patch_dim)

    print(f"\nAll finite: {is_finite}")
    print(f"Has signal: {has_signal}")
    print(f"Correct shape: {correct_shape}")

    success = is_finite and has_signal and correct_shape
    print(f"\n{'PASS' if success else 'FAIL'}")

    # Restore module-level constants
    DIM, NUM_HEADS, HEAD_DIM, FF_DIM, N_LAYERS, N_REFINER_LAYERS, CAP_FEAT_DIM, AXES_DIMS = saved

    return success


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Z-Image-Turbo on WebGPU via Triton")
    parser.add_argument("--verify", action="store_true",
                        help="Verify pipeline with random weights")
    parser.add_argument("--prompt", type=str,
                        default="a fluffy white cat sitting on a windowsill")
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    default_output = os.path.join(_SCRIPT_DIR, "..", "..", "gitignore",
                                   "output.png")
    parser.add_argument("--output", type=str, default=default_output)
    parser.add_argument("--profile", action="store_true",
                        help="Profile the denoising loop")
    add_device_arg(parser)
    args = parser.parse_args()
    apply_device_arg(args)

    if args.verify:
        success = verify_with_random_weights()
        sys.exit(0 if success else 1)

    # --- Full inference ---
    import torch
    weights_dir = os.path.join(_SCRIPT_DIR, "..", "..", "gitignore", "models", os.path.basename(_SCRIPT_DIR), "weights")
    hf_cache = os.path.join(weights_dir, "hf_cache")

    # Step 1: Load text encoder
    print("=== Loading text encoder (Qwen3) ===")
    from transformers import AutoTokenizer, AutoModelForCausalLM

    tokenizer = AutoTokenizer.from_pretrained(
        os.path.join(hf_cache, "tokenizer"), local_files_only=True)
    text_encoder = AutoModelForCausalLM.from_pretrained(
        os.path.join(hf_cache, "text_encoder"),
        torch_dtype=torch.bfloat16, local_files_only=True)

    print(f"  Text encoder: "
          f"{sum(p.numel() for p in text_encoder.parameters()) / 1e9:.2f}B params")

    print(f"\n=== Encoding prompt: '{args.prompt}' ===")
    prompt_embeds = encode_prompt(args.prompt, tokenizer, text_encoder)
    print(f"  Prompt embeddings: {prompt_embeds.shape}")

    del text_encoder, tokenizer
    import gc; gc.collect()

    # Step 2: Load VAE
    print("\n=== Loading VAE ===")
    from diffusers.models import AutoencoderKL
    vae = AutoencoderKL.from_pretrained(
        os.path.join(hf_cache, "vae"), local_files_only=True)
    vae.eval()
    print("  VAE loaded")

    # Step 3: Load transformer weights
    print("\n=== Loading transformer weights ===")
    npz_path = os.path.join(weights_dir, "transformer_fp16.npz")
    if not os.path.exists(npz_path):
        print(f"  NPZ not found, running conversion...")
        from convert_weights import convert
        convert()

    t0 = time.time()
    weights = dict(np.load(npz_path, mmap_mode='r'))
    print(f"  Loaded {len(weights)} tensors in {time.time()-t0:.1f}s")

    print("\n=== Initializing WebGPU model ===")
    model = ZImageTurboWebGPU(weights)
    if args.profile:
        model.enable_profiling()
        print(f"Profiling enabled (GPU timestamps: {model.profiler.gpu_enabled})")
    # Step 4: Denoise
    print(f"\n=== Generating {args.height}x{args.width} image "
          f"({args.steps} steps) ===")
    latents, img_ids = generate_image(
        model, prompt_embeds,
        args.height, args.width,
        num_steps=args.steps, seed=args.seed)

    # Step 5: VAE decode
    print("\n=== VAE decoding ===")
    lat_h = args.height // 8 // PATCH_SIZE
    lat_w = args.width // 8 // PATCH_SIZE

    # Unpatchify: (T, C*P*P) → (1, C, H*P, W*P)
    C = IN_CHANNELS
    P = PATCH_SIZE
    lat = latents.reshape(lat_h, lat_w, C, P, P)
    lat = lat.transpose(2, 0, 3, 1, 4).reshape(1, C, lat_h * P, lat_w * P)

    lat_tensor = torch.from_numpy(lat).to(vae.dtype)
    with torch.no_grad():
        img = vae.decode(lat_tensor, return_dict=False)[0]

    img = img.float().squeeze(0).permute(1, 2, 0).cpu().numpy()
    img = ((img + 1.0) / 2.0 * 255.0).clip(0, 255).astype(np.uint8)
    print(f"  Image shape: {img.shape}")

    from PIL import Image
    Image.fromarray(img).save(args.output)
    print(f"\n  Saved to {args.output}")

    if args.profile:
        model.save_profile(_SCRIPT_DIR, "Z-Image-Turbo")


if __name__ == "__main__":
    main()
