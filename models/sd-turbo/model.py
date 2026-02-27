"""
Stable Diffusion Turbo (SD-Turbo) inference on WebGPU via Triton.

SD-Turbo UNet with:
  - Same architecture as SDXL (self-contained, no cross-model imports)
  - 1-4 denoising steps (adversarial distillation)
  - No classifier-free guidance needed (cfg=0)
  - 512×512 default resolution
  - ResNet blocks + cross-attention + self-attention
  - GroupNorm for normalization
  - Conv1x1 via linear projection (reusing linear kernel)
  - SiLU activation (GPU-accelerated)
  - Timestep + text conditioning
  - Skip connections between down/up paths
  - GPU full multi-head attention
  - fp16 weight storage for bandwidth optimization

Supports both verification mode (random weights) and full inference
with real SDXL/SDXL-Turbo weights via diffusers for text encoding + VAE.

Usage:
    python python/examples/webgpu/sdxl/model.py --verify
    python python/examples/webgpu/sdxl/model.py --prompt "a cat" --steps 4

Requirements:
    pip install torch diffusers transformers safetensors pillow
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
from triton.backends.webgpu.dawn_runner import GPUBuffer


# ---------------------------------------------------------------------------
# SDXL configs
# ---------------------------------------------------------------------------
SDXL_CONFIGS = {
    "tiny": {
        "in_channels": 4,
        "out_channels": 4,
        "model_channels": 64,
        "channel_mult": [1, 2],
        "n_head": 4,
        "n_res_blocks": 1,
        "spatial_size": 8,
        "time_emb_dim": 128,
        "num_groups": 8,
        "context_dim": 0,
        "attn_levels": [],
    },
    "sdxl-turbo": {
        "in_channels": 4,
        "out_channels": 4,
        "model_channels": 320,
        "channel_mult": [1, 2, 4],
        "n_head": -1,  # head_dim=64 based
        "head_dim": 64,
        "n_res_blocks": 2,
        "spatial_size": 64,
        "time_emb_dim": 1280,
        "num_groups": 32,
        "context_dim": 2048,
        "attn_levels": [1, 2],
        "transformer_depth": [0, 2, 10],  # per down block
    },
    "sdxl-base": {
        "in_channels": 4,
        "out_channels": 4,
        "model_channels": 320,
        "channel_mult": [1, 2, 4],
        "n_head": -1,
        "head_dim": 64,
        "n_res_blocks": 2,
        "spatial_size": 128,  # 1024x1024 default
        "time_emb_dim": 1280,
        "num_groups": 32,
        "context_dim": 2048,
        "attn_levels": [1, 2],
        "transformer_depth": [0, 2, 10],
    },
}

VAE_SCALE_FACTOR = 8


# ---------------------------------------------------------------------------
# CPU helpers
# ---------------------------------------------------------------------------
def _w(weights, name):
    """Get weight as fp32."""
    w = weights[name]
    return w.astype(np.float32) if w.dtype != np.float32 else w


def group_norm_cpu(x, w, b, num_groups, eps=1e-5):
    """GroupNorm on (C,) or (C, H, W) — vectorized."""
    shape = x.shape
    C = shape[0]
    G = min(num_groups, C)
    cpg = C // G
    x32 = x.astype(np.float32) if x.dtype != np.float32 else x
    # Reshape: (G, cpg, ...) for per-group stats
    spatial = shape[1:]
    x_g = x32.reshape(G, cpg, *spatial)
    # Reduce over channels-per-group and spatial dims
    reduce_axes = tuple(range(1, x_g.ndim))
    mean = x_g.mean(axis=reduce_axes, keepdims=True)
    var = x_g.var(axis=reduce_axes, keepdims=True)
    x_g = (x_g - mean) / np.sqrt(var + eps)
    out = x_g.reshape(shape)
    # Apply affine per channel
    w_shape = (C,) + (1,) * len(spatial)
    return out * w.reshape(w_shape) + b.reshape(w_shape)


# ---------------------------------------------------------------------------
# SDXL UNet
# ---------------------------------------------------------------------------
class SDXLWebGPU(WebGPUModel):
    """SDXL UNet inference on WebGPU.

    Architecture:
    1. Time embedding: sinusoidal → MLP
    2. Input conv: in_channels → model_channels
    3. Down blocks: ResNet blocks + optional transformer blocks (self+cross attn)
    4. Mid block: ResNet + transformer + ResNet
    5. Up blocks: ResNet + optional transformer with skip connections
    6. Output: GroupNorm → SiLU → Conv → out_channels
    """

    def __init__(self, weights: Dict[str, np.ndarray],
                 in_channels: int = 4,
                 out_channels: int = 4,
                 model_channels: int = 64,
                 channel_mult: List[int] = None,
                 n_head: int = 4,
                 head_dim: int = None,
                 n_res_blocks: int = 1,
                 spatial_size: int = 8,
                 time_emb_dim: int = 128,
                 num_groups: int = 8,
                 context_dim: int = 0,
                 attn_levels: List[int] = None,
                 transformer_depth: List[int] = None):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.model_channels = model_channels
        self.channel_mult = channel_mult or [1, 2]
        self.n_res_blocks = n_res_blocks
        self.spatial_size = spatial_size
        self.time_emb_dim = time_emb_dim
        self.num_groups = num_groups
        self.context_dim = context_dim
        self.attn_levels = attn_levels or []
        self.transformer_depth = transformer_depth or [0] * len(self.channel_mult)

        # Determine head count
        if head_dim is not None:
            self._head_dim = head_dim
            n_head = model_channels // head_dim
        else:
            self._head_dim = model_channels // n_head

        # Collect all channel dimensions for kernel compilation
        ch_dims = set()
        ch_dims.add(model_channels)
        ch_dims.add(time_emb_dim)
        ch_dims.add(in_channels)
        ch_dims.add(out_channels)
        for mult in self.channel_mult:
            ch_dims.add(model_channels * mult)
        if context_dim > 0:
            ch_dims.add(context_dim)

        super().__init__(
            weights,
            n_layer=0,
            n_head=n_head,
            n_embd=model_channels,
            n_vocab=1,
            head_dim=self._head_dim,
            intermediate_size=model_channels * 4,
            k_dimensions=ch_dims,
        )
        self._upload_weights_to_gpu()

    def _compile_model_kernels(self):
        """Compile SDXL-specific kernels."""
        self._compile_silu()
        self._compile_full_attn()

    def _upload_weights_to_gpu(self):
        """Upload all UNet weights to GPU."""
        has_fp16 = getattr(self, '_has_fp16_linear', False)
        for name, w in self.weights.items():
            if w.size < 64:
                continue
            if w.ndim == 2 and w.size >= 256:
                if has_fp16:
                    fp16 = w if w.dtype == np.float16 else w.astype(np.float16)
                    self._gpu_weights[name] = self.cache.runner.upload_to_gpu(
                        fp16, name)
                else:
                    w32 = w.astype(np.float32) if w.dtype != np.float32 else w
                    self._gpu_weights[name] = self.cache.runner.upload_to_gpu(
                        w32, name)
            elif w.ndim == 1:
                w32 = w.astype(np.float32) if w.dtype != np.float32 else w
                self._gpu_weights[name] = self.cache.runner.upload_to_gpu(
                    w32, name)

        total_mb = sum(g.size for g in self._gpu_weights.values()) / (1024**2)
        print(f"  Uploaded {len(self._gpu_weights)} tensors "
              f"({total_mb:.0f} MB) to GPU")

    # ------------------------------------------------------------------
    # Primitives
    # ------------------------------------------------------------------

    def _linear_w(self, x, w_name, b_name=None, N=None, K=None,
                  gpu_out=False):
        """Linear projection using GPU weights (fp16 or fp32) or CPU fallback."""
        w_gpu = self._gpu_weights.get(w_name)
        if w_gpu is not None:
            if N is None:
                N = self.weights[w_name].shape[0]
            if K is None:
                K = x.shape[-1] if not isinstance(x, GPUBuffer) else K
            b_gpu = self._gpu_weights.get(b_name) if b_name else None
            if b_gpu is None and b_name and b_name in self.weights:
                b_gpu = _w(self.weights, b_name)
            elif b_gpu is None:
                b_gpu = np.zeros(N, dtype=np.float32)
            is_gpu = isinstance(x, GPUBuffer)
            if not is_gpu:
                orig = x.shape
                x2 = x.reshape(-1, K).astype(np.float32)
            else:
                x2 = x
            has_fp16 = getattr(self, '_has_fp16_linear', False)
            if has_fp16:
                out = self._linear_fp16w(x2, w_gpu, b_gpu, N, K,
                                         gpu_out=gpu_out)
            else:
                out = self._linear(x2, w_gpu, b_gpu, N,
                                   gpu_out=gpu_out)
            if gpu_out:
                return out
            return out.reshape(*orig[:-1], N)
        else:
            W = _w(self.weights, w_name)
            x32 = x.astype(np.float32) if x.dtype != np.float32 else x
            out = x32.reshape(-1, x32.shape[-1]) @ W.T
            if b_name and b_name in self.weights:
                out = out + _w(self.weights, b_name)
            return out.reshape(*x32.shape[:-1], W.shape[0])

    def _conv1x1(self, x, w_name, b_name, out_ch):
        """1×1 convolution: (C_in, H, W) → (C_out, H, W)."""
        C, H, W = x.shape
        x_flat = x.transpose(1, 2, 0).reshape(H * W, C)
        out_flat = self._linear_w(x_flat, w_name, b_name, N=out_ch, K=C)
        return out_flat.reshape(H, W, out_ch).transpose(2, 0, 1)

    def _group_norm(self, x, w_name, b_name):
        """GroupNorm on (C, ...) via CPU."""
        w = _w(self.weights, w_name)
        b = _w(self.weights, b_name)
        return group_norm_cpu(x, w, b, self.num_groups)

    def _silu_act(self, x):
        """SiLU activation using GPU kernel."""
        return self._silu(x)

    def _conv2d(self, x, w_name, b_name, out_ch, stride=1):
        """2D convolution via im2col + matmul — handles 1×1, 3×3, stride 1 & 2.

        x: (C_in, H, W)
        w: (C_out, C_in, kH, kW)  or  (C_out, C_in) for 1×1
        Returns: (C_out, H_out, W_out)
        """
        w = _w(self.weights, w_name)
        b = _w(self.weights, b_name) if b_name and b_name in self.weights else None

        if w.ndim == 2:
            # 1×1 conv — use linear projection path
            C, H, W = x.shape
            x_flat = x.transpose(1, 2, 0).reshape(H * W, C)
            out_flat = self._linear_w(x_flat, w_name, b_name,
                                      N=out_ch, K=C)
            return out_flat.reshape(H, W, out_ch).transpose(2, 0, 1)

        C_out, C_in, kH, kW = w.shape
        C, H, W = x.shape
        assert C == C_in, f"Channel mismatch: {C} vs {C_in}"

        # Pad for same-size output (padding = kH//2 for odd kernels)
        pad = kH // 2
        if pad > 0:
            x_pad = np.pad(x, ((0, 0), (pad, pad), (pad, pad)),
                           mode='constant', constant_values=0)
        else:
            x_pad = x

        _, H_p, W_p = x_pad.shape
        H_out = (H_p - kH) // stride + 1
        W_out = (W_p - kW) // stride + 1

        # im2col: extract patches → (H_out*W_out, C_in*kH*kW)
        from numpy.lib.stride_tricks import as_strided
        s = x_pad.strides  # (C_in_stride, H_stride, W_stride)
        col = as_strided(
            x_pad,
            shape=(H_out, W_out, C_in, kH, kW),
            strides=(s[1] * stride, s[2] * stride, s[0], s[1], s[2])
        )
        col_2d = col.reshape(H_out * W_out, C_in * kH * kW)

        # matmul with kernel reshaped to (C_out, C_in*kH*kW)
        w_2d = w.reshape(C_out, C_in * kH * kW).astype(np.float32)
        out = col_2d.astype(np.float32) @ w_2d.T  # (H_out*W_out, C_out)
        if b is not None:
            out = out + b
        return out.reshape(H_out, W_out, C_out).transpose(2, 0, 1)

    # ------------------------------------------------------------------
    # Building blocks
    # ------------------------------------------------------------------

    def _time_embedding(self, timestep, pooled_text_embeds=None,
                        time_ids=None):
        """Sinusoidal timestep embedding → MLP → (1, TED).

        For SDXL: adds pooled text embeddings + time_ids via add_embedding.
        """
        MC = self.model_channels
        half = MC // 2
        freqs = np.exp(
            -np.log(10000.0) * np.arange(0, half, dtype=np.float32) / half)
        args = np.float32(timestep) * freqs
        emb = np.concatenate([np.cos(args), np.sin(args)]).reshape(1, MC)

        h = self._linear_w(emb, "time_embed.0.weight", "time_embed.0.bias",
                           N=self.time_emb_dim, K=MC)
        h = self._silu_act(h)
        h = self._linear_w(h, "time_embed.2.weight", "time_embed.2.bias",
                           N=self.time_emb_dim, K=self.time_emb_dim)

        # SDXL add_embedding: pooled text + time_ids
        if pooled_text_embeds is not None and "add_embed.0.weight" in self.weights:
            # Encode time_ids: each of the 6 values gets its own sinusoidal
            # embedding (same as timestep embedding, dim=256 per value)
            if time_ids is not None:
                time_ids_flat = time_ids.ravel().astype(np.float32)
                tid_dim = 256  # add_time_proj uses dim=256
                tid_half = tid_dim // 2
                tid_freqs = np.exp(
                    -np.log(10000.0) * np.arange(tid_half, dtype=np.float32)
                    / tid_half)
                tid_embeds = []
                for tid in time_ids_flat:
                    args = np.float32(tid) * tid_freqs
                    tid_embeds.append(np.concatenate([np.cos(args),
                                                      np.sin(args)]))
                add_time_embed = np.concatenate(tid_embeds).reshape(1, -1)
            else:
                add_time_embed = np.zeros((1, 0), dtype=np.float32)

            # Concat pooled text + time embeddings
            pooled = pooled_text_embeds.reshape(1, -1)
            add_input = np.concatenate([pooled, add_time_embed], axis=-1)

            # MLP: Linear → SiLU → Linear
            add_w0 = self.weights["add_embed.0.weight"]
            K_add = add_input.shape[1]
            if K_add != add_w0.shape[1]:
                # Pad or truncate to match weight shape
                target_K = add_w0.shape[1]
                if K_add < target_K:
                    add_input = np.pad(add_input,
                                       ((0, 0), (0, target_K - K_add)))
                else:
                    add_input = add_input[:, :target_K]

            add_h = self._linear_w(add_input, "add_embed.0.weight",
                                   "add_embed.0.bias",
                                   N=self.time_emb_dim, K=add_input.shape[1])
            add_h = add_h * (1.0 / (1.0 + np.exp(-add_h)))  # SiLU
            add_h = self._linear_w(add_h, "add_embed.2.weight",
                                   "add_embed.2.bias",
                                   N=self.time_emb_dim,
                                   K=self.time_emb_dim)
            h = h + add_h

        return h  # (1, TED)

    def _resnet_block(self, x, temb, pfx, ch_in, ch):
        """ResNet block: GN → SiLU → Conv → +time → GN → SiLU → Conv."""
        C, H, W = x.shape
        h = self._group_norm(x, pfx + "norm1.weight", pfx + "norm1.bias")

        # Detect if conv weights are 3×3 or 1×1
        conv1_w = self.weights[pfx + "conv1.weight"]
        use_conv2d = (conv1_w.ndim == 4 and conv1_w.shape[2] > 1)

        if use_conv2d:
            # 3×3 path: apply SiLU on spatial tensor, then conv2d
            h_flat = h.transpose(1, 2, 0).reshape(H * W, C)
            h_flat = self._silu_act(h_flat)
            h = h_flat.reshape(H, W, C).transpose(2, 0, 1)
            h = self._conv2d(h, pfx + "conv1.weight",
                             pfx + "conv1.bias", ch)
        else:
            h_flat = h.transpose(1, 2, 0).reshape(H * W, C)
            h_flat = self._silu_act(h_flat)
            h_flat = self._linear_w(h_flat, pfx + "conv1.weight",
                                    pfx + "conv1.bias", N=ch, K=C,
                                    gpu_out=True)
            if isinstance(h_flat, GPUBuffer):
                h_flat = self.cache.runner.readback(h_flat).reshape(H * W, ch)
            h = h_flat.reshape(H, W, ch).transpose(2, 0, 1)

        # Time conditioning: SiLU(temb) → Linear → broadcast add
        temb_silu = self._silu_act(temb) if temb.ndim == 2 else \
            (temb * (1.0 / (1.0 + np.exp(-temb))))
        temb_proj = self._linear_w(
            temb_silu, pfx + "temb.weight", pfx + "temb.bias",
            N=ch, K=self.time_emb_dim)
        h = h + temb_proj.ravel()[:, None, None]

        h = self._group_norm(h, pfx + "norm2.weight", pfx + "norm2.bias")

        if use_conv2d:
            h_flat = h.transpose(1, 2, 0).reshape(H * W, ch)
            h_flat = self._silu_act(h_flat)
            h = h_flat.reshape(H, W, ch).transpose(2, 0, 1)
            h = self._conv2d(h, pfx + "conv2.weight",
                             pfx + "conv2.bias", ch)
        else:
            h_flat = h.transpose(1, 2, 0).reshape(H * W, ch)
            h_flat = self._silu_act(h_flat)
            h_flat = self._linear_w(h_flat, pfx + "conv2.weight",
                                    pfx + "conv2.bias", N=ch, K=ch,
                                    gpu_out=True)
            if isinstance(h_flat, GPUBuffer):
                h_flat = self.cache.runner.readback(h_flat).reshape(H * W, ch)
            h = h_flat.reshape(H, W, ch).transpose(2, 0, 1)

        # Skip projection (handles both 1×1 and 3×3)
        if ch_in != ch:
            proj_name = pfx + "proj.weight"
            if proj_name in self.weights:
                proj_w = self.weights[proj_name]
                if proj_w.ndim == 4 and proj_w.shape[2] > 1:
                    x = self._conv2d(x, proj_name, pfx + "proj.bias", ch)
                else:
                    x = self._conv1x1(x, proj_name, pfx + "proj.bias", ch)
        return x + h

    def _self_attn(self, x_flat, ch, prefix):
        """Self-attention on flat spatial tokens (S, C).

        Uses GPU full multi-head attention kernel.
        """
        S, C = x_flat.shape
        hd = self._head_dim
        n_head = ch // hd

        q = self._linear_w(x_flat, prefix + "to_q.weight",
                           prefix + "to_q.bias", N=ch, K=C)
        k = self._linear_w(x_flat, prefix + "to_k.weight",
                           prefix + "to_k.bias", N=ch, K=C)
        v = self._linear_w(x_flat, prefix + "to_v.weight",
                           prefix + "to_v.bias", N=ch, K=C)

        Q = q.reshape(S, n_head, hd).astype(np.float32)
        K_t = k.reshape(S, n_head, hd).astype(np.float32)
        V_t = v.reshape(S, n_head, hd).astype(np.float32)

        out = self._full_attention_multihead(Q, K_t, V_t, n_head=n_head)
        out = out.reshape(S, ch)

        out = self._linear_w(out, prefix + "to_out.0.weight",
                             prefix + "to_out.0.bias", N=ch, K=ch)
        return out

    def _cross_attn(self, x_flat, context, ch, prefix):
        """Cross-attention: Q from x, K/V from context."""
        S, C = x_flat.shape
        S_ctx = context.shape[0]
        hd = self._head_dim
        n_head = ch // hd

        q = self._linear_w(x_flat, prefix + "to_q.weight",
                           prefix + "to_q.bias", N=ch, K=C)
        k = self._linear_w(context, prefix + "to_k.weight",
                           prefix + "to_k.bias", N=ch,
                           K=self.context_dim)
        v = self._linear_w(context, prefix + "to_v.weight",
                           prefix + "to_v.bias", N=ch,
                           K=self.context_dim)

        Q = q.reshape(S, n_head, hd).astype(np.float32)
        K_t = k.reshape(S_ctx, n_head, hd).astype(np.float32)
        V_t = v.reshape(S_ctx, n_head, hd).astype(np.float32)

        # Vectorized cross-attention: batched matmul over heads
        scale = 1.0 / np.sqrt(float(hd))
        Q_h = Q.transpose(1, 0, 2)      # (n_head, S, hd)
        K_h = K_t.transpose(1, 2, 0)    # (n_head, hd, S_ctx)
        V_h = V_t.transpose(1, 0, 2)    # (n_head, S_ctx, hd)
        scores = np.float32(Q_h @ K_h * scale)  # (n_head, S, S_ctx)
        scores -= scores.max(axis=-1, keepdims=True)
        exp_s = np.exp(scores)
        attn = exp_s / exp_s.sum(axis=-1, keepdims=True)
        attn_out = (attn @ V_h).transpose(1, 0, 2)  # (S, n_head, hd)

        out = attn_out.reshape(S, ch)
        out = self._linear_w(out, prefix + "to_out.0.weight",
                             prefix + "to_out.0.bias", N=ch, K=ch)
        return out

    def _transformer_block(self, x, context, ch, prefix):
        """Single transformer block: LayerNorm → self-attn → LayerNorm → cross-attn → LayerNorm → FFN."""
        S, C = x.shape

        # Self-attention
        norm1_w = prefix + "norm1.weight"
        if norm1_w in self.weights:
            xn = self._layer_norm_1d(x, norm1_w, prefix + "norm1.bias")
            x = x + self._self_attn(xn, ch, prefix + "attn1.")

        # Cross-attention
        norm2_w = prefix + "norm2.weight"
        if norm2_w in self.weights and context is not None:
            xn = self._layer_norm_1d(x, norm2_w, prefix + "norm2.bias")
            x = x + self._cross_attn(xn, context, ch, prefix + "attn2.")

        # FFN (GEGLU)
        norm3_w = prefix + "norm3.weight"
        if norm3_w in self.weights:
            xn = self._layer_norm_1d(x, norm3_w, prefix + "norm3.bias")
            # GEGLU: split linear output into gate and value, gate = GELU(gate)
            ff_dim = self.weights[prefix + "ff.net.0.proj.weight"].shape[0]
            ff = self._linear_w(xn, prefix + "ff.net.0.proj.weight",
                                prefix + "ff.net.0.proj.bias",
                                N=ff_dim, K=ch)
            # GEGLU split: first half is gate (apply GELU), second half is value
            half = ff_dim // 2
            gate = ff[:, :half]
            val = ff[:, half:]
            # GELU activation
            gate = gate * 0.5 * (1.0 + np.tanh(
                np.sqrt(2.0 / np.pi) * (gate + 0.044715 * gate ** 3)))
            ff_out = gate * val
            ff_out = self._linear_w(ff_out, prefix + "ff.net.2.weight",
                                    prefix + "ff.net.2.bias",
                                    N=ch, K=half)
            x = x + ff_out

        return x

    def _layer_norm_1d(self, x, w_name, b_name, eps=1e-5):
        """LayerNorm on last dim via CPU."""
        x32 = x.astype(np.float32) if x.dtype != np.float32 else x
        mean = x32.mean(axis=-1, keepdims=True)
        var = x32.var(axis=-1, keepdims=True)
        xn = (x32 - mean) / np.sqrt(var + eps)
        w = _w(self.weights, w_name)
        b = _w(self.weights, b_name)
        return xn * w + b

    def _attn_block_simple(self, x, prefix, ch):
        """Simple self-attention block for mid block (no cross-attn)."""
        C, H, W = x.shape
        S = H * W
        h = self._group_norm(x, prefix + "norm.weight",
                             prefix + "norm.bias")
        h_flat = h.transpose(1, 2, 0).reshape(S, C).astype(np.float32)

        hd = self._head_dim
        n_head = ch // hd

        q = self._linear_w(h_flat, prefix + "q.weight",
                           prefix + "q.bias", N=ch, K=C)
        k = self._linear_w(h_flat, prefix + "k.weight",
                           prefix + "k.bias", N=ch, K=C)
        v = self._linear_w(h_flat, prefix + "v.weight",
                           prefix + "v.bias", N=ch, K=C)

        Q = q.reshape(S, n_head, hd).astype(np.float32)
        K_t = k.reshape(S, n_head, hd).astype(np.float32)
        V_t = v.reshape(S, n_head, hd).astype(np.float32)

        out = self._full_attention_multihead(Q, K_t, V_t, n_head=n_head)
        out = out.reshape(S, ch)

        proj = self._linear_w(out, prefix + "proj.weight",
                              prefix + "proj.bias", N=ch, K=ch)
        return x + proj.reshape(H, W, ch).transpose(2, 0, 1)

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward(self, latent, timestep=500, context=None,
                pooled_text_embeds=None, time_ids=None, **kwargs):
        """UNet forward pass for one denoising step.

        latent: (in_channels, H, W) noisy latent
        timestep: diffusion timestep (int or float)
        context: (S_ctx, context_dim) text encoder output, or None
        pooled_text_embeds: (1280,) pooled CLIP text embeddings (SDXL)
        time_ids: (6,) original/crop/target size IDs (SDXL)
        Returns: (out_channels, H, W) predicted noise
        """
        MC = self.model_channels
        temb = self._time_embedding(timestep, pooled_text_embeds, time_ids)

        # Input conv (handles both 1×1 and 3×3)
        input_w = self.weights["input_conv.weight"]
        if input_w.ndim == 4 and input_w.shape[2] > 1:
            x = self._conv2d(latent, "input_conv.weight",
                             "input_conv.bias", MC)
        else:
            x = self._conv1x1(latent, "input_conv.weight",
                              "input_conv.bias", MC)

        # --- Down path ---
        skips = [x.copy()]  # Save input conv output as first skip
        for level, mult in enumerate(self.channel_mult):
            ch = MC * mult
            ch_in = x.shape[0]
            for rb in range(self.n_res_blocks):
                pfx = f"down.{level}.res{rb}."
                x = self._resnet_block(x, temb, pfx, ch_in, ch)
                ch_in = ch

                # Transformer blocks at this level
                if level in self.attn_levels and context is not None:
                    C, H, W = x.shape
                    S = H * W
                    x_flat = x.transpose(1, 2, 0).reshape(S, C).astype(
                        np.float32)
                    n_depth = (self.transformer_depth[level]
                               if level < len(self.transformer_depth) else 1)
                    norm_pfx = f"down.{level}.attn{rb}."
                    if norm_pfx + "norm.weight" in self.weights:
                        x_flat = self._group_norm_flat(
                            x_flat, C, H, W,
                            norm_pfx + "norm.weight",
                            norm_pfx + "norm.bias")
                    # proj_in linear
                    proj_in = norm_pfx + "proj_in.weight"
                    if proj_in in self.weights:
                        x_flat = self._linear_w(
                            x_flat, proj_in, norm_pfx + "proj_in.bias",
                            N=ch, K=ch)
                    for td in range(n_depth):
                        tpfx = f"down.{level}.attn{rb}.t{td}."
                        x_flat = self._transformer_block(
                            x_flat, context, ch, tpfx)
                    # Proj out
                    proj_name = norm_pfx + "proj_out.weight"
                    if proj_name in self.weights:
                        x_flat = self._linear_w(
                            x_flat, proj_name, norm_pfx + "proj_out.bias",
                            N=ch, K=ch)
                    x = x_flat.reshape(H, W, C).transpose(2, 0, 1)

                # Save skip (after resnet + optional transformer)
                skips.append(x.copy())

            # Downsample between levels (except last)
            ds_w = f"down.{level}.downsample.weight"
            if ds_w in self.weights:
                x = self._conv2d(x, ds_w, f"down.{level}.downsample.bias",
                                 ch, stride=2)
                skips.append(x.copy())

        # --- Mid block ---
        ch_mid = x.shape[0]
        x = self._resnet_block(x, temb, "mid.res1.", ch_mid, ch_mid)

        # Mid attention (simple self-attn or transformer)
        mid_has_transformer = "mid.attn.t0.norm1.weight" in self.weights
        if "mid.attn.norm.weight" in self.weights and not mid_has_transformer:
            x = self._attn_block_simple(x, "mid.attn.", ch_mid)
        elif context is not None and mid_has_transformer:
            C, H, W = x.shape
            S = H * W
            x_flat = x.transpose(1, 2, 0).reshape(S, C).astype(np.float32)
            if "mid.attn.norm.weight" in self.weights:
                x_flat = self._group_norm_flat(
                    x_flat, C, H, W,
                    "mid.attn.norm.weight", "mid.attn.norm.bias")
            if "mid.attn.proj_in.weight" in self.weights:
                x_flat = self._linear_w(
                    x_flat, "mid.attn.proj_in.weight",
                    "mid.attn.proj_in.bias", N=ch_mid, K=ch_mid)
            # Count mid transformer blocks
            td = 0
            while f"mid.attn.t{td}.norm1.weight" in self.weights:
                x_flat = self._transformer_block(
                    x_flat, context, ch_mid, f"mid.attn.t{td}.")
                td += 1
            if "mid.attn.proj_out.weight" in self.weights:
                x_flat = self._linear_w(
                    x_flat, "mid.attn.proj_out.weight",
                    "mid.attn.proj_out.bias", N=ch_mid, K=ch_mid)
            x = x_flat.reshape(H, W, C).transpose(2, 0, 1)

        x = self._resnet_block(x, temb, "mid.res2.", ch_mid, ch_mid)

        # --- Up path ---
        for level, mult in enumerate(reversed(self.channel_mult)):
            ch = MC * mult
            # Detect up block resnet count from weights
            n_up_res = 0
            while f"up.{level}.res{n_up_res}.norm1.weight" in self.weights:
                n_up_res += 1
            if n_up_res == 0:
                n_up_res = self.n_res_blocks

            for rb in range(n_up_res):
                pfx = f"up.{level}.res{rb}."

                # Skip connection
                if skips:
                    skip = skips.pop()
                    # Concat-skip when resnet has proj (conv_shortcut)
                    proj_name = pfx + "proj.weight"
                    if proj_name in self.weights:
                        x = np.concatenate([x, skip], axis=0)
                    else:
                        # Additive skip: project skip if channels differ
                        spfx = pfx + "skip_proj."
                        if skip.shape[0] != x.shape[0]:
                            if spfx + "weight" in self.weights:
                                skip = self._conv1x1(
                                    skip, spfx + "weight",
                                    spfx + "bias", x.shape[0])
                        # Project x if channels differ
                        ipfx = pfx + "in_proj.weight"
                        if ipfx in self.weights:
                            x = self._conv1x1(
                                x, ipfx, pfx + "in_proj.bias", ch)
                        if skip.shape[0] == x.shape[0]:
                            x = x + skip

                ch_in = x.shape[0]
                x = self._resnet_block(x, temb, pfx, ch_in, ch)

                # Transformer at this level
                up_idx = len(self.channel_mult) - 1 - level
                if up_idx in self.attn_levels and context is not None:
                    attn_pfx = f"up.{level}.attn{rb}."
                    if attn_pfx + "norm.weight" in self.weights:
                        C, H, W = x.shape
                        S = H * W
                        x_flat = x.transpose(1, 2, 0).reshape(
                            S, C).astype(np.float32)
                        n_depth = (self.transformer_depth[up_idx]
                                   if up_idx < len(self.transformer_depth)
                                   else 1)
                        x_flat = self._group_norm_flat(
                            x_flat, C, H, W,
                            attn_pfx + "norm.weight",
                            attn_pfx + "norm.bias")
                        proj_in = attn_pfx + "proj_in.weight"
                        if proj_in in self.weights:
                            x_flat = self._linear_w(
                                x_flat, proj_in,
                                attn_pfx + "proj_in.bias",
                                N=ch, K=ch)
                        for td in range(n_depth):
                            tpfx = f"up.{level}.attn{rb}.t{td}."
                            x_flat = self._transformer_block(
                                x_flat, context, ch, tpfx)
                        proj_out = attn_pfx + "proj_out.weight"
                        if proj_out in self.weights:
                            x_flat = self._linear_w(
                                x_flat, proj_out,
                                attn_pfx + "proj_out.bias",
                                N=ch, K=ch)
                        x = x_flat.reshape(H, W, C).transpose(2, 0, 1)

            # Upsample between levels (except last)
            us_w = f"up.{level}.upsample.weight"
            if us_w in self.weights:
                C, H, W = x.shape
                x = np.repeat(np.repeat(x, 2, axis=1), 2, axis=2)
                x = self._conv2d(x, us_w, f"up.{level}.upsample.bias", ch)

        # Output: GroupNorm → SiLU → Conv
        if "out_norm.weight" in self.weights:
            x = self._group_norm(x, "out_norm.weight", "out_norm.bias")
            C, H, W = x.shape
            x_flat = x.transpose(1, 2, 0).reshape(H * W, C)
            x_flat = self._silu_act(x_flat)
            x = x_flat.reshape(H, W, C).transpose(2, 0, 1)

        out_w = self.weights["output_conv.weight"]
        if out_w.ndim == 4 and out_w.shape[2] > 1:
            x = self._conv2d(x, "output_conv.weight",
                             "output_conv.bias", self.out_channels)
        else:
            x = self._conv1x1(x, "output_conv.weight",
                              "output_conv.bias", self.out_channels)
        return x

    def _group_norm_flat(self, x_flat, C, H, W, w_name, b_name):
        """GroupNorm on spatial tokens reshaped from (S, C) → (C, H, W) → (S, C)."""
        x_spatial = x_flat.reshape(H, W, C).transpose(2, 0, 1)
        x_normed = self._group_norm(x_spatial, w_name, b_name)
        return x_normed.transpose(1, 2, 0).reshape(H * W, C)


# ---------------------------------------------------------------------------
# Diffusion Scheduler (Euler)
# ---------------------------------------------------------------------------
class EulerDiscreteScheduler:
    """Simplified Euler scheduler for SDXL."""

    def __init__(self, num_train_timesteps=1000,
                 beta_start=0.00085, beta_end=0.012):
        self.num_train_timesteps = num_train_timesteps
        betas = np.linspace(beta_start**0.5, beta_end**0.5,
                            num_train_timesteps, dtype=np.float64) ** 2
        alphas = 1.0 - betas
        self.alphas_cumprod = np.cumprod(alphas).astype(np.float32)
        self.sigmas = ((1 - self.alphas_cumprod) /
                       self.alphas_cumprod) ** 0.5
        self.sigmas = np.append(self.sigmas, 0.0).astype(np.float32)

    def get_timesteps(self, num_steps):
        """Get timesteps using 'trailing' spacing (matches HF default).

        trailing: step_ratio = N // num_steps
        timesteps = arange(N, 0, -step_ratio) - 1
        """
        step_ratio = self.num_train_timesteps // num_steps
        timesteps = (np.arange(self.num_train_timesteps, 0, -step_ratio)
                     - 1).astype(np.float32)
        return timesteps

    def get_sigmas(self, timesteps):
        """Get sigma values for given timesteps."""
        sigmas = np.interp(timesteps, np.arange(len(self.sigmas) - 1),
                           self.sigmas[:-1])
        sigmas = np.append(sigmas, 0.0).astype(np.float32)
        return sigmas

    def scale_input(self, sample, sigma):
        """Scale model input."""
        return sample / ((sigma**2 + 1) ** 0.5)

    def step(self, model_output, sigma, sigma_next, sample):
        """One Euler step."""
        pred_x0 = sample - sigma * model_output
        d = (sample - pred_x0) / sigma
        sample = sample + d * (sigma_next - sigma)
        return sample


# ---------------------------------------------------------------------------
# Text encoding + VAE (via diffusers)
# ---------------------------------------------------------------------------
def load_pipeline_components(model_id="stabilityai/sdxl-turbo"):
    """Load CLIP text encoders and VAE from diffusers."""
    import torch
    from diffusers import AutoencoderKL
    from transformers import CLIPTextModel, CLIPTextModelWithProjection
    from transformers import CLIPTokenizer

    hf_dir = os.path.join(_SCRIPT_DIR, "weights", "hf_cache")

    print("Loading SDXL pipeline components...")
    t0 = time.perf_counter()

    tokenizer = CLIPTokenizer.from_pretrained(
        hf_dir, subfolder="tokenizer")
    tokenizer_2 = CLIPTokenizer.from_pretrained(
        hf_dir, subfolder="tokenizer_2")
    text_encoder = CLIPTextModel.from_pretrained(
        hf_dir, subfolder="text_encoder", torch_dtype=torch.float16)
    text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(
        hf_dir, subfolder="text_encoder_2", torch_dtype=torch.float16)
    vae = AutoencoderKL.from_pretrained(
        hf_dir, subfolder="vae", torch_dtype=torch.float16)

    t1 = time.perf_counter()
    print(f"  Loaded in {t1-t0:.1f}s")
    return (tokenizer, tokenizer_2, text_encoder, text_encoder_2, vae)


def encode_prompt(tokenizer, tokenizer_2, text_encoder, text_encoder_2,
                  prompt, max_length=77):
    """Encode prompt with dual CLIP, return pooled + hidden states."""
    import torch

    # CLIP 1
    inputs = tokenizer(prompt, return_tensors="pt", padding="max_length",
                       max_length=max_length, truncation=True)
    with torch.no_grad():
        out1 = text_encoder(input_ids=inputs["input_ids"],
                            output_hidden_states=True)
        hidden1 = out1.hidden_states[-2]  # penultimate layer

    # CLIP 2
    inputs2 = tokenizer_2(prompt, return_tensors="pt", padding="max_length",
                          max_length=max_length, truncation=True)
    with torch.no_grad():
        out2 = text_encoder_2(input_ids=inputs2["input_ids"],
                              output_hidden_states=True)
        hidden2 = out2.hidden_states[-2]
        pooled = out2.text_embeds

    # Concatenate hidden states from both encoders
    context = torch.cat([hidden1, hidden2], dim=-1)
    return context.float().cpu().numpy()[0], pooled.float().cpu().numpy()[0]


def vae_decode(vae, latents):
    """Decode latents to image via VAE."""
    import torch
    from PIL import Image

    with torch.no_grad():
        lt = torch.from_numpy(latents).unsqueeze(0) if latents.ndim == 3 else \
             torch.from_numpy(latents)
        # Use float32 for CPU VAE (fp16 on CPU causes silent failures)
        lt = lt.to(dtype=torch.float32, device=vae.device)
        vae_f32 = vae.float()
        lt = lt / vae_f32.config.scaling_factor
        img = vae_f32.decode(lt).sample
    img = img.float().cpu().numpy()[0].transpose(1, 2, 0)
    img = np.clip((img + 1) / 2 * 255, 0, 255).astype(np.uint8)
    return Image.fromarray(img)


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------
def generate_image(model, tokenizer, tokenizer_2, text_encoder,
                   text_encoder_2, vae, prompt, height=512, width=512,
                   num_steps=4, guidance_scale=0.0, seed=42):
    """Generate image using SDXL UNet on WebGPU.

    When guidance_scale > 1, uses classifier-free guidance (CFG):
    runs UNet twice per step (unconditional + conditional) and blends.
    """
    np.random.seed(seed)
    lat_h, lat_w = height // VAE_SCALE_FACTOR, width // VAE_SCALE_FACTOR
    print(f"Image: {width}×{height} → latent: {lat_w}×{lat_h}")

    # Encode text
    print("Encoding prompt...")
    context, pooled = encode_prompt(
        tokenizer, tokenizer_2, text_encoder, text_encoder_2, prompt)
    print(f"  Context: {context.shape}, pooled: {pooled.shape}")

    use_cfg = guidance_scale > 1.0
    if use_cfg:
        # Unconditional context (empty prompt)
        context_uc, pooled_uc = encode_prompt(
            tokenizer, tokenizer_2, text_encoder, text_encoder_2, "")
        print(f"  CFG scale: {guidance_scale}")

    # Time IDs: [orig_h, orig_w, crop_top, crop_left, target_h, target_w]
    time_ids = np.array([height, width, 0, 0, height, width],
                        dtype=np.float32)

    # Initialize noise
    latent = np.random.randn(4, lat_h, lat_w).astype(np.float32)

    # Scheduler
    scheduler = EulerDiscreteScheduler()
    timesteps = scheduler.get_timesteps(num_steps)
    sigmas = scheduler.get_sigmas(timesteps)

    # Scale initial noise
    latent = latent * sigmas[0]

    print(f"Denoising ({num_steps} steps)...")
    t_start = time.perf_counter()

    for step in range(num_steps):
        sigma = sigmas[step]
        sigma_next = sigmas[step + 1]
        ts = timesteps[step]

        # Scale input
        latent_input = scheduler.scale_input(latent, sigma)

        if use_cfg:
            # Classifier-free guidance: 2 forward passes
            noise_cond = model.forward(latent_input, timestep=ts,
                                       context=context,
                                       pooled_text_embeds=pooled,
                                       time_ids=time_ids)
            noise_uncond = model.forward(latent_input, timestep=ts,
                                         context=context_uc,
                                         pooled_text_embeds=pooled_uc,
                                         time_ids=time_ids)
            noise_pred = noise_uncond + guidance_scale * (noise_cond - noise_uncond)
        else:
            noise_pred = model.forward(latent_input, timestep=ts,
                                       context=context,
                                       pooled_text_embeds=pooled,
                                       time_ids=time_ids)

        # Euler step
        latent = scheduler.step(noise_pred, sigma, sigma_next, latent)

        elapsed = time.perf_counter() - t_start
        print(f"  Step {step+1}/{num_steps} t={ts:.0f} "
              f"({elapsed:.1f}s total, {elapsed/(step+1):.2f}s/step)")

    print(f"  Done in {time.perf_counter()-t_start:.1f}s")

    # VAE decode
    print("VAE decoding...")
    image = vae_decode(vae, latent)
    return image


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------
def verify_with_random_weights():
    """Verify SDXL UNet pipeline with small random weights."""
    print("=" * 60)
    print("SDXL UNet WebGPU Pipeline Verification (random weights)")
    print("=" * 60)

    config = SDXL_CONFIGS["tiny"]
    ic = config["in_channels"]
    oc = config["out_channels"]
    mc = config["model_channels"]
    cm = config["channel_mult"]
    nh = config["n_head"]
    ss = config["spatial_size"]
    ted = config["time_emb_dim"]
    ng = config["num_groups"]
    np.random.seed(42)
    s = 0.02

    W = {}
    # Time embedding MLP
    W["time_embed.0.weight"] = np.random.randn(ted, mc).astype(np.float32) * s
    W["time_embed.0.bias"] = np.zeros(ted, dtype=np.float32)
    W["time_embed.2.weight"] = np.random.randn(ted, ted).astype(np.float32) * s
    W["time_embed.2.bias"] = np.zeros(ted, dtype=np.float32)

    # Input conv
    W["input_conv.weight"] = np.random.randn(mc, ic).astype(np.float32) * s
    W["input_conv.bias"] = np.zeros(mc, dtype=np.float32)

    # Down blocks
    for level, mult in enumerate(cm):
        ch = mc * mult
        ch_in = mc * (cm[level - 1] if level > 0 else 1)
        for rb in range(1):
            pfx = f"down.{level}.res{rb}."
            cin = ch_in if rb == 0 else ch
            W[pfx + "norm1.weight"] = np.ones(cin, dtype=np.float32)
            W[pfx + "norm1.bias"] = np.zeros(cin, dtype=np.float32)
            W[pfx + "conv1.weight"] = np.random.randn(
                ch, cin).astype(np.float32) * s
            W[pfx + "conv1.bias"] = np.zeros(ch, dtype=np.float32)
            W[pfx + "norm2.weight"] = np.ones(ch, dtype=np.float32)
            W[pfx + "norm2.bias"] = np.zeros(ch, dtype=np.float32)
            W[pfx + "conv2.weight"] = np.random.randn(
                ch, ch).astype(np.float32) * s
            W[pfx + "conv2.bias"] = np.zeros(ch, dtype=np.float32)
            W[pfx + "temb.weight"] = np.random.randn(
                ch, ted).astype(np.float32) * s
            W[pfx + "temb.bias"] = np.zeros(ch, dtype=np.float32)
            if cin != ch:
                W[pfx + "proj.weight"] = np.random.randn(
                    ch, cin).astype(np.float32) * s
                W[pfx + "proj.bias"] = np.zeros(ch, dtype=np.float32)

    # Mid block
    ch_mid = mc * cm[-1]
    for sub in ["res1", "res2"]:
        pfx = f"mid.{sub}."
        W[pfx + "norm1.weight"] = np.ones(ch_mid, dtype=np.float32)
        W[pfx + "norm1.bias"] = np.zeros(ch_mid, dtype=np.float32)
        W[pfx + "conv1.weight"] = np.random.randn(
            ch_mid, ch_mid).astype(np.float32) * s
        W[pfx + "conv1.bias"] = np.zeros(ch_mid, dtype=np.float32)
        W[pfx + "norm2.weight"] = np.ones(ch_mid, dtype=np.float32)
        W[pfx + "norm2.bias"] = np.zeros(ch_mid, dtype=np.float32)
        W[pfx + "conv2.weight"] = np.random.randn(
            ch_mid, ch_mid).astype(np.float32) * s
        W[pfx + "conv2.bias"] = np.zeros(ch_mid, dtype=np.float32)
        W[pfx + "temb.weight"] = np.random.randn(
            ch_mid, ted).astype(np.float32) * s
        W[pfx + "temb.bias"] = np.zeros(ch_mid, dtype=np.float32)

    W["mid.attn.norm.weight"] = np.ones(ch_mid, dtype=np.float32)
    W["mid.attn.norm.bias"] = np.zeros(ch_mid, dtype=np.float32)
    for proj in ["q", "k", "v", "proj"]:
        W[f"mid.attn.{proj}.weight"] = np.random.randn(
            ch_mid, ch_mid).astype(np.float32) * s
        W[f"mid.attn.{proj}.bias"] = np.zeros(ch_mid, dtype=np.float32)

    # Up blocks
    for level, mult in enumerate(reversed(cm)):
        ch = mc * mult
        if level == 0:
            ch_prev = ch_mid
        else:
            ch_prev = mc * list(reversed(cm))[level - 1]
        for rb in range(1):
            pfx = f"up.{level}.res{rb}."

            # Skip channels
            skip_idx = len(cm) - 1 - level
            ch_skip = mc * (cm[skip_idx - 1] if skip_idx > 0 else 1)
            if ch_skip != ch:
                W[pfx + "skip_proj.weight"] = np.random.randn(
                    ch, ch_skip).astype(np.float32) * s
                W[pfx + "skip_proj.bias"] = np.zeros(ch, dtype=np.float32)
            cin = ch_prev if rb == 0 else ch
            if cin != ch:
                W[pfx + "in_proj.weight"] = np.random.randn(
                    ch, cin).astype(np.float32) * s
                W[pfx + "in_proj.bias"] = np.zeros(ch, dtype=np.float32)

            W[pfx + "norm1.weight"] = np.ones(ch, dtype=np.float32)
            W[pfx + "norm1.bias"] = np.zeros(ch, dtype=np.float32)
            W[pfx + "conv1.weight"] = np.random.randn(
                ch, ch).astype(np.float32) * s
            W[pfx + "conv1.bias"] = np.zeros(ch, dtype=np.float32)
            W[pfx + "norm2.weight"] = np.ones(ch, dtype=np.float32)
            W[pfx + "norm2.bias"] = np.zeros(ch, dtype=np.float32)
            W[pfx + "conv2.weight"] = np.random.randn(
                ch, ch).astype(np.float32) * s
            W[pfx + "conv2.bias"] = np.zeros(ch, dtype=np.float32)
            W[pfx + "temb.weight"] = np.random.randn(
                ch, ted).astype(np.float32) * s
            W[pfx + "temb.bias"] = np.zeros(ch, dtype=np.float32)

    # Output conv
    W["output_conv.weight"] = np.random.randn(oc, mc).astype(np.float32) * s
    W["output_conv.bias"] = np.zeros(oc, dtype=np.float32)

    print(f"\nConfig: channels={mc}, mult={cm}, heads={nh}, "
          f"spatial={ss}×{ss}")
    print(f"  time_emb_dim={ted}, groups={ng}")

    model = SDXLWebGPU(
        W, in_channels=ic, out_channels=oc,
        model_channels=mc, channel_mult=cm,
        n_head=nh, n_res_blocks=1,
        spatial_size=ss, time_emb_dim=ted,
        num_groups=ng)

    latent = np.random.randn(ic, ss, ss).astype(np.float32) * 0.1
    timestep = 500

    t0 = time.time()
    out = model.forward(latent, timestep=timestep)
    t1 = time.time()

    print(f"\nForward pass: ({ic},{ss},{ss}) → {out.shape} "
          f"in {(t1-t0)*1000:.0f}ms")
    print(f"Output range: [{out.min():.4f}, {out.max():.4f}]")
    print(f"Output mean: {out.mean():.6f}")

    is_finite = np.all(np.isfinite(out))
    has_signal = out.std() > 1e-6
    correct_shape = out.shape == (oc, ss, ss)

    print(f"\nAll finite: {is_finite}")
    print(f"Has signal (std > 1e-6): {has_signal}")
    print(f"Correct shape: {correct_shape}")

    success = is_finite and has_signal and correct_shape
    print(f"\n{'PASS' if success else 'FAIL'}")
    return success


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Stable Diffusion XL on WebGPU")
    parser.add_argument("--verify", action="store_true",
                        help="Verify pipeline with random weights")
    parser.add_argument("--prompt", type=str,
                        default="a beautiful landscape painting")
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--steps", type=int, default=1,
                        help="Denoising steps (1-4 for Turbo)")
    parser.add_argument("--cfg", type=float, default=0.0,
                        help="CFG scale (0 for Turbo, no guidance needed)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default="output.png")
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--unet-only", action="store_true",
                        help="Run UNet with random inputs (no diffusers)")
    args = parser.parse_args()

    if args.verify:
        success = verify_with_random_weights()
        sys.exit(0 if success else 1)

    if args.unet_only:
        # Run UNet with random weights/inputs
        print("--- UNet-only mode (random weights) ---")
        config = SDXL_CONFIGS["tiny"]
        W = {}
        mc = config["model_channels"]
        cm = config["channel_mult"]
        ted = config["time_emb_dim"]
        ic = config["in_channels"]
        oc = config["out_channels"]
        ss = 16
        s = 0.02
        np.random.seed(42)

        # Generate weights (same as verify but with larger spatial)
        W["time_embed.0.weight"] = np.random.randn(
            ted, mc).astype(np.float32) * s
        W["time_embed.0.bias"] = np.zeros(ted, dtype=np.float32)
        W["time_embed.2.weight"] = np.random.randn(
            ted, ted).astype(np.float32) * s
        W["time_embed.2.bias"] = np.zeros(ted, dtype=np.float32)
        W["input_conv.weight"] = np.random.randn(
            mc, ic).astype(np.float32) * s
        W["input_conv.bias"] = np.zeros(mc, dtype=np.float32)
        for level, mult in enumerate(cm):
            ch = mc * mult
            ch_in = mc * (cm[level - 1] if level > 0 else 1)
            pfx = f"down.{level}.res0."
            W[pfx + "norm1.weight"] = np.ones(ch_in, dtype=np.float32)
            W[pfx + "norm1.bias"] = np.zeros(ch_in, dtype=np.float32)
            W[pfx + "conv1.weight"] = np.random.randn(
                ch, ch_in).astype(np.float32) * s
            W[pfx + "conv1.bias"] = np.zeros(ch, dtype=np.float32)
            W[pfx + "norm2.weight"] = np.ones(ch, dtype=np.float32)
            W[pfx + "norm2.bias"] = np.zeros(ch, dtype=np.float32)
            W[pfx + "conv2.weight"] = np.random.randn(
                ch, ch).astype(np.float32) * s
            W[pfx + "conv2.bias"] = np.zeros(ch, dtype=np.float32)
            W[pfx + "temb.weight"] = np.random.randn(
                ch, ted).astype(np.float32) * s
            W[pfx + "temb.bias"] = np.zeros(ch, dtype=np.float32)
            if ch_in != ch:
                W[pfx + "proj.weight"] = np.random.randn(
                    ch, ch_in).astype(np.float32) * s
                W[pfx + "proj.bias"] = np.zeros(ch, dtype=np.float32)
        ch_mid = mc * cm[-1]
        for sub in ["res1", "res2"]:
            pfx = f"mid.{sub}."
            W[pfx + "norm1.weight"] = np.ones(ch_mid, dtype=np.float32)
            W[pfx + "norm1.bias"] = np.zeros(ch_mid, dtype=np.float32)
            W[pfx + "conv1.weight"] = np.random.randn(
                ch_mid, ch_mid).astype(np.float32) * s
            W[pfx + "conv1.bias"] = np.zeros(ch_mid, dtype=np.float32)
            W[pfx + "norm2.weight"] = np.ones(ch_mid, dtype=np.float32)
            W[pfx + "norm2.bias"] = np.zeros(ch_mid, dtype=np.float32)
            W[pfx + "conv2.weight"] = np.random.randn(
                ch_mid, ch_mid).astype(np.float32) * s
            W[pfx + "conv2.bias"] = np.zeros(ch_mid, dtype=np.float32)
            W[pfx + "temb.weight"] = np.random.randn(
                ch_mid, ted).astype(np.float32) * s
            W[pfx + "temb.bias"] = np.zeros(ch_mid, dtype=np.float32)
        W["mid.attn.norm.weight"] = np.ones(ch_mid, dtype=np.float32)
        W["mid.attn.norm.bias"] = np.zeros(ch_mid, dtype=np.float32)
        for proj in ["q", "k", "v", "proj"]:
            W[f"mid.attn.{proj}.weight"] = np.random.randn(
                ch_mid, ch_mid).astype(np.float32) * s
            W[f"mid.attn.{proj}.bias"] = np.zeros(
                ch_mid, dtype=np.float32)
        for level, mult in enumerate(reversed(cm)):
            ch = mc * mult
            ch_prev = ch_mid if level == 0 else mc * list(
                reversed(cm))[level - 1]
            pfx = f"up.{level}.res0."
            skip_idx = len(cm) - 1 - level
            ch_skip = mc * (cm[skip_idx - 1] if skip_idx > 0 else 1)
            if ch_skip != ch:
                W[pfx + "skip_proj.weight"] = np.random.randn(
                    ch, ch_skip).astype(np.float32) * s
                W[pfx + "skip_proj.bias"] = np.zeros(
                    ch, dtype=np.float32)
            if ch_prev != ch:
                W[pfx + "in_proj.weight"] = np.random.randn(
                    ch, ch_prev).astype(np.float32) * s
                W[pfx + "in_proj.bias"] = np.zeros(
                    ch, dtype=np.float32)
            W[pfx + "norm1.weight"] = np.ones(ch, dtype=np.float32)
            W[pfx + "norm1.bias"] = np.zeros(ch, dtype=np.float32)
            W[pfx + "conv1.weight"] = np.random.randn(
                ch, ch).astype(np.float32) * s
            W[pfx + "conv1.bias"] = np.zeros(ch, dtype=np.float32)
            W[pfx + "norm2.weight"] = np.ones(ch, dtype=np.float32)
            W[pfx + "norm2.bias"] = np.zeros(ch, dtype=np.float32)
            W[pfx + "conv2.weight"] = np.random.randn(
                ch, ch).astype(np.float32) * s
            W[pfx + "conv2.bias"] = np.zeros(ch, dtype=np.float32)
            W[pfx + "temb.weight"] = np.random.randn(
                ch, ted).astype(np.float32) * s
            W[pfx + "temb.bias"] = np.zeros(ch, dtype=np.float32)
        W["output_conv.weight"] = np.random.randn(
            oc, mc).astype(np.float32) * s
        W["output_conv.bias"] = np.zeros(oc, dtype=np.float32)

        model = SDXLWebGPU(W, in_channels=ic, out_channels=oc,
                           model_channels=mc, channel_mult=cm,
                           n_head=config["n_head"], n_res_blocks=1,
                           spatial_size=ss, time_emb_dim=ted,
                           num_groups=config["num_groups"])

        latent = np.random.randn(ic, ss, ss).astype(np.float32) * 0.1

        # Warm up
        print("  Warm-up...")
        _ = model.forward(latent, timestep=500)

        # Timed
        n_steps = 5
        print(f"  Running {n_steps} timed steps...")
        t0 = time.perf_counter()
        for _ in range(n_steps):
            _ = model.forward(latent, timestep=500)
        t1 = time.perf_counter()
        print(f"  {n_steps} steps in {t1-t0:.1f}s "
              f"({(t1-t0)/n_steps*1000:.0f}ms/step)")
        return

    # Full pipeline with diffusers
    components = load_pipeline_components()
    tokenizer, tokenizer_2, text_encoder, text_encoder_2, vae = components

    # Load UNet weights
    wp = os.path.join(_SCRIPT_DIR, "weights", "unet_fp16.npz")
    if not os.path.exists(wp):
        print(f"UNet weights not found: {wp}")
        print("Run: python python/examples/webgpu/sdxl/convert_weights.py")
        sys.exit(1)

    print("Loading UNet weights...")
    t0 = time.perf_counter()
    data = np.load(wp)
    weights = {k: data[k] for k in data.files}
    print(f"  {len(weights)} tensors in {time.perf_counter()-t0:.1f}s")

    config = SDXL_CONFIGS["sdxl-turbo"]
    model = SDXLWebGPU(weights, **{k: v for k, v in config.items()
                                   if k != "transformer_depth"},
                       transformer_depth=config.get("transformer_depth"))

    if args.profile:
        model.enable_profiling()

    image = generate_image(
        model, tokenizer, tokenizer_2, text_encoder, text_encoder_2, vae,
        prompt=args.prompt, height=args.height, width=args.width,
        num_steps=args.steps, guidance_scale=args.cfg, seed=args.seed)

    out = os.path.join(_SCRIPT_DIR, args.output)
    image.save(out)
    print(f"\nSaved to {out}")

    if args.profile and model.profiler:
        report = model.profiler.report()
        print(f"\n{report}")
        html = os.path.join(_SCRIPT_DIR, "profile.html")
        model.profiler.save_html(html)
        print(f"Profile saved to {html}")


if __name__ == "__main__":
    main()
