"""
FLUX.2 Klein 4B inference on WebGPU via Triton.

Architecture: Dual-stream + single-stream DiT transformer
  - 5 double (joint txt+img) blocks
  - 20 single (concatenated) blocks
  - hidden_dim=3072, 24 heads, head_dim=128
  - Qwen3 text encoder (3 hidden layers → 7680 joint_attention_dim)
  - AutoencoderKLFlux2 VAE (32 latent channels, patch_size=[2,2])
  - FlowMatchEulerDiscreteScheduler

Usage:
    python python/examples/webgpu/flux-klein/model.py --verify
    python python/examples/webgpu/flux-klein/model.py --prompt "a cat" --steps 20

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
from triton.backends.webgpu.dawn_runner import GPUBuffer

# ---------------------------------------------------------------------------
# FLUX.2 Klein 4B architecture constants
# ---------------------------------------------------------------------------
HIDDEN_DIM = 3072
NUM_HEADS = 24
HEAD_DIM = 128
IN_CHANNELS = 128       # VAE latent channels after patchify (32 * 2 * 2)
JOINT_ATTN_DIM = 7680   # 3 Qwen3 layers * 2560
FF_DIM = 9216           # HIDDEN_DIM * mlp_ratio (3.0)
NUM_DOUBLE_BLOCKS = 5
NUM_SINGLE_BLOCKS = 20
PATCH_SIZE = 1
AXES_DIMS_ROPE = [32, 32, 32, 32]  # 4 axes, total 128 = HEAD_DIM
ROPE_THETA = 2000
TIMESTEP_CHANNELS = 256
EPS = 1e-6
VAE_SCALE_FACTOR = 8    # 2^(num_downsamples)
VAE_PATCH_SIZE = 2

# Text encoder
QWEN3_HIDDEN_SIZE = 2560
QWEN3_OUT_LAYERS = (10, 20, 30)
SYSTEM_MESSAGE = (
    "You are an AI that reasons about image descriptions. "
    "You give structured responses focusing on object relationships, "
    "object\nattribution and actions without speculation."
)


# ---------------------------------------------------------------------------
# CPU helper functions
# ---------------------------------------------------------------------------

def get_timestep_embedding(timesteps: np.ndarray, dim: int = 256) -> np.ndarray:
    """Sinusoidal timestep embedding (flip_sin_to_cos=True, downscale_freq_shift=0)."""
    half = dim // 2
    freqs = np.exp(-math.log(10000.0) / half * np.arange(half, dtype=np.float64))
    args = timesteps[:, None].astype(np.float64) * freqs[None, :]
    emb = np.concatenate([np.cos(args), np.sin(args)], axis=-1)
    return emb.astype(np.float32)


def compute_rope_freqs(ids: np.ndarray, axes_dim: List[int],
                       theta: float = 2000.0) -> Tuple[np.ndarray, np.ndarray]:
    """Compute RoPE cos/sin for 4D position IDs.

    ids: (S, 4) position IDs
    Returns: (cos, sin) each (S, HEAD_DIM)
    """
    cos_parts, sin_parts = [], []
    for i, dim in enumerate(axes_dim):
        pos = ids[:, i].astype(np.float64)
        half_dim = dim // 2
        freq = 1.0 / (theta ** (np.arange(half_dim, dtype=np.float64) / half_dim))
        angles = pos[:, None] * freq[None, :]  # (S, half_dim)
        # repeat_interleave_real=True: each angle appears twice
        cos_val = np.cos(angles)
        sin_val = np.sin(angles)
        cos_interleaved = np.repeat(cos_val, 2, axis=-1)  # (S, dim)
        sin_interleaved = np.repeat(sin_val, 2, axis=-1)
        cos_parts.append(cos_interleaved)
        sin_parts.append(sin_interleaved)
    return (np.concatenate(cos_parts, axis=-1).astype(np.float32),
            np.concatenate(sin_parts, axis=-1).astype(np.float32))


def apply_rotary_emb(x: np.ndarray, cos: np.ndarray, sin: np.ndarray) -> np.ndarray:
    """Apply rotary embedding to (S, H, D) using real-valued rotation.

    cos, sin: (S, D) broadcastable to (S, 1, D).
    Uses use_real_unbind_dim=-1 convention (interleaved pairs).
    """
    cos_ = cos[:, None, :]  # (S, 1, D)
    sin_ = sin[:, None, :]
    # Split into real/imag pairs: (S, H, D) -> (S, H, D//2, 2) -> unbind last
    x_pairs = x.reshape(*x.shape[:-1], -1, 2)
    x_real = x_pairs[..., 0]
    x_imag = x_pairs[..., 1]
    # Rotated: [-imag, real] interleaved
    x_rotated = np.stack([-x_imag, x_real], axis=-1).reshape(x.shape)
    return (x.astype(np.float32) * cos_ + x_rotated.astype(np.float32) * sin_).astype(np.float32)


def rms_norm_cpu(x: np.ndarray, w: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """RMSNorm on last dimension. x: (..., D), w: (D,)."""
    x_f = x.astype(np.float32)
    rms = np.sqrt(np.mean(x_f ** 2, axis=-1, keepdims=True) + eps)
    return (x_f / rms * w.astype(np.float32)).astype(np.float32)


def layer_norm_cpu(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """LayerNorm (no affine) on last dimension."""
    x_f = x.astype(np.float32)
    mean = np.mean(x_f, axis=-1, keepdims=True)
    var = np.var(x_f, axis=-1, keepdims=True)
    return ((x_f - mean) / np.sqrt(var + eps)).astype(np.float32)


def prepare_latent_ids(height: int, width: int) -> np.ndarray:
    """4D position IDs for image latents: (T=0, H, W, L=0)."""
    ids = np.zeros((height * width, 4), dtype=np.float32)
    h_coords = np.arange(height)
    w_coords = np.arange(width)
    hh, ww = np.meshgrid(h_coords, w_coords, indexing='ij')
    ids[:, 1] = hh.ravel()
    ids[:, 2] = ww.ravel()
    return ids


def prepare_text_ids(seq_len: int) -> np.ndarray:
    """4D position IDs for text tokens: (T=0, H=0, W=0, L=l)."""
    ids = np.zeros((seq_len, 4), dtype=np.float32)
    ids[:, 3] = np.arange(seq_len)
    return ids


def compute_empirical_mu(image_seq_len: int, num_steps: int) -> float:
    """Dynamic shift mu for FlowMatch scheduler."""
    a1, b1 = 8.73809524e-05, 1.89833333
    a2, b2 = 0.00016927, 0.45666666
    if image_seq_len > 4300:
        return float(a2 * image_seq_len + b2)
    m_200 = a2 * image_seq_len + b2
    m_10 = a1 * image_seq_len + b1
    a = (m_200 - m_10) / 190.0
    b = m_200 - 200.0 * a
    return float(a * num_steps + b)


# ---------------------------------------------------------------------------
# FLUX.2 Klein WebGPU model
# ---------------------------------------------------------------------------

class FluxKleinWebGPU(WebGPUModel):
    """FLUX.2 Klein 4B transformer on WebGPU.

    Architecture:
    1. Time + guidance embedding (sinusoidal → MLP)
    2. Modulation (SiLU → Linear) for all block types
    3. Input projections (x_embedder, context_embedder)
    4. 5 double-stream blocks (joint txt+img attention + separate FF)
    5. 20 single-stream blocks (parallel attention + FF on concatenated stream)
    6. AdaLN output → projection
    """

    def __init__(self, weights: Dict[str, np.ndarray], fp16_act: bool = False):
        # Collect all linear K dimensions for kernel compilation
        k_dims = {
            IN_CHANNELS, TIMESTEP_CHANNELS, HIDDEN_DIM,
            JOINT_ATTN_DIM, FF_DIM,
            FF_DIM * 2,      # 18432: FF gate+up, modulation out
            3 * HIDDEN_DIM + 2 * FF_DIM,  # 27648: single block fused QKV+MLP
            HIDDEN_DIM * 2,  # 6144: norm_out linear
            HIDDEN_DIM + FF_DIM,  # 12288: single block to_out
        }

        super().__init__(
            weights,
            n_layer=NUM_DOUBLE_BLOCKS + NUM_SINGLE_BLOCKS,
            n_head=NUM_HEADS,
            n_embd=HIDDEN_DIM,
            n_vocab=1,
            head_dim=HEAD_DIM,
            intermediate_size=FF_DIM,
            k_dimensions=k_dims,
            norm_eps=EPS,
            fp16_act=fp16_act,
        )

        # Pre-allocate dummy LayerNorm params (elementwise_affine=False)
        self._ln_ones = self.cache.runner.upload_to_gpu(
            np.ones(HIDDEN_DIM, dtype=np.float32), "ln_ones")
        self._ln_zeros = self.cache.runner.upload_to_gpu(
            np.zeros(HIDDEN_DIM, dtype=np.float32), "ln_zeros")

        self._upload_weights_to_gpu()
        self._split_single_block_weights()
        self._split_double_block_weights()
        self._free_large_cpu_weights()

    def _compile_model_kernels(self):
        """Compile FLUX-specific kernels."""
        self._compile_layer_norm()
        self._compile_silu_mul()
        self._compile_full_attn()
        self._compile_qk_norm_rope()

    def _upload_weights_to_gpu(self):
        """Upload all transformer weights to GPU as fp16.

        Skips weights that will be split/fused by _split_single_block_weights
        and _split_double_block_weights to avoid wasting ~6 GB of GPU memory
        on redundant buffers.
        """
        # Build skip set: weights that will be re-split/fused later
        skip = set()
        for i in range(NUM_SINGLE_BLOCKS):
            pfx = f"single_transformer_blocks.{i}"
            skip.add(f"{pfx}.attn.to_qkv_mlp_proj.weight")  # split → qkv, gate, up
            skip.add(f"{pfx}.attn.to_out.weight")            # split → out_attn, out_mlp
        for i in range(NUM_DOUBLE_BLOCKS):
            pfx = f"transformer_blocks.{i}"
            skip.add(f"{pfx}.ff.linear_in.weight")            # split → gate, up
            skip.add(f"{pfx}.ff_context.linear_in.weight")    # split → gate, up
            for k in ("to_q", "to_k", "to_v"):
                skip.add(f"{pfx}.attn.{k}.weight")            # fused → img_qkv
            for k in ("add_q_proj", "add_k_proj", "add_v_proj"):
                skip.add(f"{pfx}.attn.{k}.weight")            # fused → txt_qkv

        runner = self.cache.runner
        count = 0
        for name, w in self.weights.items():
            if name in skip:
                continue
            if w.ndim == 2 and w.size >= 256:
                # Large 2D weight → fp16 linear
                fp16 = w if w.dtype == np.float16 else w.astype(np.float16)
                self._gpu_weights[name] = runner.upload_to_gpu(fp16, name)
                count += 1
            elif w.ndim == 1:
                # 1D weight (norm, etc.) → fp32
                w32 = w.astype(np.float32) if w.dtype != np.float32 else w
                self._gpu_weights[name] = runner.upload_to_gpu(w32, name)
                count += 1
        print(f"  Uploaded {count} weight tensors to GPU (skipped {len(skip)} to-be-split)")

    def _split_single_block_weights(self):
        """Split fused single-block weights for GPU-resident chains.

        Splits QKV+MLP fused projection into 5 separate weights and
        splits output projection into attn+mlp parts, enabling the
        MLP path to stay entirely on GPU without CPU round-trips.
        """
        runner = self.cache.runner
        to_fp16 = lambda w: w.astype(np.float16) if w.dtype != np.float16 else w
        count = 0
        for i in range(NUM_SINGLE_BLOCKS):
            pfx = f"single_transformer_blocks.{i}"

            # Split fused QKV+MLP input projection (27648, 3072)
            fused_w = self.weights[f"{pfx}.attn.to_qkv_mlp_proj.weight"]

            # Fused QKV weight (9216, 3072) — one dispatch + readback
            qkv_w = fused_w[:3*HIDDEN_DIM, :].copy()
            self._gpu_weights[f"{pfx}.qkv_proj.weight"] = runner.upload_to_gpu(
                to_fp16(qkv_w), f"{pfx}.qkv_proj.weight")

            # Separate MLP gate/up weights — stay on GPU (gpu_out=True)
            gate_w = fused_w[3*HIDDEN_DIM:3*HIDDEN_DIM+FF_DIM, :].copy()
            up_w = fused_w[3*HIDDEN_DIM+FF_DIM:, :].copy()
            self._gpu_weights[f"{pfx}.gate_proj.weight"] = runner.upload_to_gpu(
                to_fp16(gate_w), f"{pfx}.gate_proj.weight")
            self._gpu_weights[f"{pfx}.up_proj.weight"] = runner.upload_to_gpu(
                to_fp16(up_w), f"{pfx}.up_proj.weight")

            # Split output projection (3072, 12288) into attn (3072, 3072) + mlp (3072, 9216)
            out_w = self.weights[f"{pfx}.attn.to_out.weight"]
            out_attn_w = out_w[:, :HIDDEN_DIM].copy()
            out_mlp_w = out_w[:, HIDDEN_DIM:].copy()

            self._gpu_weights[f"{pfx}.to_out_attn.weight"] = runner.upload_to_gpu(
                to_fp16(out_attn_w), f"{pfx}.to_out_attn.weight")
            self._gpu_weights[f"{pfx}.to_out_mlp.weight"] = runner.upload_to_gpu(
                to_fp16(out_mlp_w), f"{pfx}.to_out_mlp.weight")
            count += 5

            # Free original numpy arrays to reduce CPU memory pressure
            del self.weights[f"{pfx}.attn.to_qkv_mlp_proj.weight"]
            del self.weights[f"{pfx}.attn.to_out.weight"]
        print(f"  Split & uploaded {count} single-block weight tensors")

    def _split_double_block_weights(self):
        """Split double-block FF weights for GPU-resident chains.

        Splits ff.linear_in.weight (18432, 3072) into gate (9216, 3072)
        and up (9216, 3072) for both img and txt FF paths.  Also fuses
        per-stream QKV weights into a single (9216, 3072) tensor.
        This lets gate/up projections stay on GPU (gpu_out=True),
        eliminating the massive (T, 18432) readback + re-upload.
        """
        runner = self.cache.runner
        to_fp16 = lambda w: w.astype(np.float16) if w.dtype != np.float16 else w
        count = 0
        for i in range(NUM_DOUBLE_BLOCKS):
            pfx = f"transformer_blocks.{i}"

            # --- Split FF linear_in weights ---
            for ff_name in ("ff", "ff_context"):
                w = self.weights[f"{pfx}.{ff_name}.linear_in.weight"]  # (18432, 3072)
                gate_w = w[:FF_DIM, :].copy()   # (9216, 3072)
                up_w = w[FF_DIM:, :].copy()     # (9216, 3072)
                gn = f"{pfx}.{ff_name}.gate_weight"
                un = f"{pfx}.{ff_name}.up_weight"
                self._gpu_weights[gn] = runner.upload_to_gpu(to_fp16(gate_w), gn)
                self._gpu_weights[un] = runner.upload_to_gpu(to_fp16(up_w), un)
                count += 2

            # --- Fuse per-stream QKV weights ---
            for stream, keys in [
                ("img", ["attn.to_q.weight", "attn.to_k.weight", "attn.to_v.weight"]),
                ("txt", ["attn.add_q_proj.weight", "attn.add_k_proj.weight", "attn.add_v_proj.weight"]),
            ]:
                parts = [self.weights[f"{pfx}.{k}"] for k in keys]
                fused = np.concatenate(parts, axis=0).copy()  # (9216, 3072)
                fn = f"{pfx}.{stream}_qkv.weight"
                self._gpu_weights[fn] = runner.upload_to_gpu(to_fp16(fused), fn)
                count += 1

            # Free original numpy arrays
            for ff_name in ("ff", "ff_context"):
                del self.weights[f"{pfx}.{ff_name}.linear_in.weight"]
            for k in ("to_q", "to_k", "to_v"):
                del self.weights[f"{pfx}.attn.{k}.weight"]
            for k in ("add_q_proj", "add_k_proj", "add_v_proj"):
                del self.weights[f"{pfx}.attn.{k}.weight"]

        print(f"  Split & uploaded {count} double-block weight tensors")

    def _free_large_cpu_weights(self):
        """Free large 2D weight arrays from self.weights.

        After uploading + splitting, only small 1D norm weights are
        needed on CPU (for _qk_norm_rope calls).  Free everything else
        to reclaim ~7 GB of CPU memory.
        """
        large = [k for k, v in self.weights.items()
                 if isinstance(v, np.ndarray) and v.ndim >= 2]
        for k in large:
            del self.weights[k]
        import gc; gc.collect()
        print(f"  Freed {len(large)} large CPU weight arrays")

    # ------------------------------------------------------------------
    # Primitive wrappers
    # ------------------------------------------------------------------

    def _w(self, name: str):
        """Get weight as GPUBuffer (already uploaded)."""
        return self._gpu_weights[name]

    def _linear(self, x, weight_name: str, out_features: int,
                gpu_out: bool = False):
        """Linear projection using fp16 weight from GPU."""
        w = self._w(weight_name)
        # Bias-free model (all bias=False)
        T = x.shape[0] if not isinstance(x, GPUBuffer) else x.shape[0]
        K = x.shape[1] if not isinstance(x, GPUBuffer) else x.shape[1]
        zero_bias = np.zeros(out_features, dtype=np.float32)
        return self._linear_fp16w(x, w, zero_bias, out_features, K=K,
                                  gpu_out=gpu_out)

    def _layer_norm_no_affine(self, x, gpu_out: bool = False):
        """LayerNorm without learned affine (elementwise_affine=False)."""
        return super()._layer_norm(x, self._ln_ones, self._ln_zeros,
                                   eps=EPS, gpu_out=gpu_out)

    # ------------------------------------------------------------------
    # Timestep + guidance embedding
    # ------------------------------------------------------------------

    def _compute_temb(self, timestep: float) -> np.ndarray:
        """Compute timestep embedding on CPU (no guidance embedder for Klein).

        Returns: (1, HIDDEN_DIM) temb array.
        """
        t_scaled = np.array([timestep * 1000.0], dtype=np.float32)

        # Sinusoidal projection
        t_proj = get_timestep_embedding(t_scaled, TIMESTEP_CHANNELS)  # (1, 256)

        # Timestep MLP: Linear → SiLU → Linear
        t_emb = self._linear(t_proj, "time_guidance_embed.timestep_embedder.linear_1.weight",
                             HIDDEN_DIM)  # (1, 3072)
        t_emb = t_emb * (1.0 / (1.0 + np.exp(-t_emb)))  # SiLU on CPU
        t_emb = self._linear(t_emb, "time_guidance_embed.timestep_embedder.linear_2.weight",
                             HIDDEN_DIM)

        return t_emb  # (1, 3072)

    # ------------------------------------------------------------------
    # Modulation
    # ------------------------------------------------------------------

    def _compute_modulation(self, temb: np.ndarray, weight_name: str,
                            num_param_sets: int):
        """Compute modulation params: SiLU(temb) → Linear → chunk.

        Returns list of (shift, scale, gate) tuples.
        """
        mod = temb * (1.0 / (1.0 + np.exp(-temb)))  # SiLU
        out_dim = HIDDEN_DIM * 3 * num_param_sets
        mod = self._linear(mod, weight_name, out_dim)  # (1, out_dim)

        # Chunk into num_param_sets groups of 3, each (1, D)
        chunks = np.split(mod, 3 * num_param_sets, axis=-1)
        result = []
        for i in range(num_param_sets):
            shift = chunks[3 * i]
            scale = chunks[3 * i + 1]
            gate = chunks[3 * i + 2]
            result.append((shift, scale, gate))
        return result

    # ------------------------------------------------------------------
    # Double stream block
    # ------------------------------------------------------------------

    def _double_block(self, hidden_states,
                      encoder_hidden_states,
                      mod_img, mod_txt,
                      concat_cos: np.ndarray, concat_sin: np.ndarray,
                      block_idx: int,
                      gpu_state: bool = False,
                      _profile: bool = False):
        """One double-stream transformer block.

        hidden_states: (T_img, D) numpy or GPUBuffer
        encoder_hidden_states: (T_txt, D) numpy or GPUBuffer
        When gpu_state=True, both inputs/outputs are GPUBuffer.
        """
        pfx = f"transformer_blocks.{block_idx}"
        runner = self.cache.runner
        if _profile:
            _t = [time.perf_counter()]
            def _m():
                _t.append(time.perf_counter())

        is_gpu = isinstance(hidden_states, GPUBuffer)

        if is_gpu and gpu_state:
            # ---- Fully GPU-resident path ----
            (shift_msa_gpu, scale_msa_gpu, gate_msa_gpu), \
                (shift_mlp_gpu, scale_mlp_gpu, gate_mlp_gpu) = mod_img
            (c_shift_msa_gpu, c_scale_msa_gpu, c_gate_msa_gpu), \
                (c_shift_mlp_gpu, c_scale_mlp_gpu, c_gate_mlp_gpu) = mod_txt

            T_img = hidden_states.shape[0]
            T_txt = encoder_hidden_states.shape[0]
            T_total = T_img + T_txt
            HD = HEAD_DIM

            # Norm + modulation on GPU for both streams
            norm_x_gpu = self._layer_norm_no_affine(hidden_states,
                                                     gpu_out=True)
            norm_x_gpu = self._mod_scale_shift(
                norm_x_gpu, scale_msa_gpu, shift_msa_gpu,
                HIDDEN_DIM, gpu_out=True)
            norm_x_gpu.shape = (T_img, HIDDEN_DIM)

            norm_ctx_gpu = self._layer_norm_no_affine(
                encoder_hidden_states, gpu_out=True)
            norm_ctx_gpu = self._mod_scale_shift(
                norm_ctx_gpu, c_scale_msa_gpu, c_shift_msa_gpu,
                HIDDEN_DIM, gpu_out=True)
            norm_ctx_gpu.shape = (T_txt, HIDDEN_DIM)
            if _profile: _m()  # 0: norm+mod (GPU)
            if _profile: _m()  # 1: (skip upload)

            # QKV projections on GPU
            img_qkv_gpu = self._linear(norm_x_gpu,
                                        f"{pfx}.img_qkv.weight",
                                        3 * HIDDEN_DIM, gpu_out=True)
            img_qkv_gpu.shape = (T_img, 3 * HIDDEN_DIM)
            if _profile: _m()  # 2: img_QKV

            txt_qkv_gpu = self._linear(norm_ctx_gpu,
                                        f"{pfx}.txt_qkv.weight",
                                        3 * HIDDEN_DIM, gpu_out=True)
            txt_qkv_gpu.shape = (T_txt, 3 * HIDDEN_DIM)
            if _profile: _m()  # 3: txt_QKV

            # QK-RMSNorm + RoPE on GPU
            img_cos = concat_cos[T_txt:]
            img_sin = concat_sin[T_txt:]
            q_img, k_img, v_img = self._qk_norm_rope(
                img_qkv_gpu,
                self.weights[f"{pfx}.attn.norm_q.weight"].astype(np.float32),
                self.weights[f"{pfx}.attn.norm_k.weight"].astype(np.float32),
                img_cos, img_sin, NUM_HEADS, T_img, eps=EPS, gpu_out=True)

            txt_cos = concat_cos[:T_txt]
            txt_sin = concat_sin[:T_txt]
            add_q, add_k, add_v = self._qk_norm_rope(
                txt_qkv_gpu,
                self.weights[f"{pfx}.attn.norm_added_q.weight"].astype(np.float32),
                self.weights[f"{pfx}.attn.norm_added_k.weight"].astype(np.float32),
                txt_cos, txt_sin, NUM_HEADS, T_txt, eps=EPS, gpu_out=True)
            if _profile: _m()  # 4: norm+RoPE (GPU)

            # Concat Q/K/V on GPU: [txt; img] for each
            q_cat = self._concat_gpu(add_q, q_img)
            q_cat.shape = (T_total, NUM_HEADS, HD)
            k_cat = self._concat_gpu(add_k, k_img)
            k_cat.shape = (T_total, NUM_HEADS, HD)
            v_cat = self._concat_gpu(add_v, v_img)
            v_cat.shape = (T_total, NUM_HEADS, HD)
            if _profile: _m()  # 5: concat (GPU)

            # Full attention on GPU
            attn_gpu = self._full_attention_multihead(
                q_cat, k_cat, v_cat, NUM_HEADS, gpu_out=True)
            if _profile: _m()  # 6: attention (GPU)

            # Split attn into txt/img on GPU, then reshape to (T, D)
            N_txt_elems = T_txt * NUM_HEADS * HD
            N_img_elems = T_img * NUM_HEADS * HD
            ctx_attn_gpu = self._split_gpu(attn_gpu, 0, N_txt_elems)
            ctx_attn_gpu.shape = (T_txt, HIDDEN_DIM)
            img_attn_gpu = self._split_gpu(attn_gpu, N_txt_elems, N_img_elems)
            img_attn_gpu.shape = (T_img, HIDDEN_DIM)

            # Output projections + gated residual on GPU
            img_proj_gpu = self._linear(img_attn_gpu,
                                         f"{pfx}.attn.to_out.0.weight",
                                         HIDDEN_DIM, gpu_out=True)
            self._gate_residual_add(hidden_states, gate_msa_gpu,
                                     img_proj_gpu, HIDDEN_DIM)

            ctx_proj_gpu = self._linear(ctx_attn_gpu,
                                         f"{pfx}.attn.to_add_out.weight",
                                         HIDDEN_DIM, gpu_out=True)
            self._gate_residual_add(encoder_hidden_states, c_gate_msa_gpu,
                                     ctx_proj_gpu, HIDDEN_DIM)
            if _profile: _m()  # 7: out_proj+res (GPU)

            # --- Image FF on GPU ---
            ff_norm_x = self._layer_norm_no_affine(hidden_states,
                                                    gpu_out=True)
            ff_norm_x = self._mod_scale_shift(
                ff_norm_x, scale_mlp_gpu, shift_mlp_gpu,
                HIDDEN_DIM, gpu_out=True)
            ff_norm_x.shape = (T_img, HIDDEN_DIM)
            ff_gate_gpu = self._linear(ff_norm_x,
                                        f"{pfx}.ff.gate_weight",
                                        FF_DIM, gpu_out=True)
            ff_up_gpu = self._linear(ff_norm_x,
                                      f"{pfx}.ff.up_weight",
                                      FF_DIM, gpu_out=True)
            ff_act = self._silu_mul(ff_gate_gpu, ff_up_gpu, gpu_out=True)
            ff_out_gpu = self._linear(ff_act,
                                       f"{pfx}.ff.linear_out.weight",
                                       HIDDEN_DIM, gpu_out=True)
            self._gate_residual_add(hidden_states, gate_mlp_gpu,
                                     ff_out_gpu, HIDDEN_DIM)
            if _profile: _m()  # 8: img_FF (GPU)

            # --- Text FF on GPU ---
            ff_norm_ctx = self._layer_norm_no_affine(
                encoder_hidden_states, gpu_out=True)
            ff_norm_ctx = self._mod_scale_shift(
                ff_norm_ctx, c_scale_mlp_gpu, c_shift_mlp_gpu,
                HIDDEN_DIM, gpu_out=True)
            ff_norm_ctx.shape = (T_txt, HIDDEN_DIM)
            ff_gate_ctx = self._linear(ff_norm_ctx,
                                        f"{pfx}.ff_context.gate_weight",
                                        FF_DIM, gpu_out=True)
            ff_up_ctx = self._linear(ff_norm_ctx,
                                      f"{pfx}.ff_context.up_weight",
                                      FF_DIM, gpu_out=True)
            ff_ctx_act = self._silu_mul(ff_gate_ctx, ff_up_ctx,
                                         gpu_out=True)
            ff_ctx_out_gpu = self._linear(
                ff_ctx_act,
                f"{pfx}.ff_context.linear_out.weight",
                HIDDEN_DIM, gpu_out=True)
            self._gate_residual_add(encoder_hidden_states,
                                     c_gate_mlp_gpu,
                                     ff_ctx_out_gpu, HIDDEN_DIM)
            if _profile: _m()  # 9: txt_FF (GPU)

            if _profile:
                labels = ["norm+mod", "(skip)", "img_QKV", "txt_QKV",
                           "norm+RoPE", "concat", "attention",
                           "out_proj+res", "img_FF", "txt_FF"]
                parts = [(_t[j+1] - _t[j]) * 1000
                         for j in range(len(_t) - 1)]
                print(f"  [double {block_idx}] " + " | ".join(
                    f"{l}:{v:.1f}" for l, v in zip(labels, parts))
                    + f"  TOTAL:{sum(parts):.1f}ms")

            return encoder_hidden_states, hidden_states

        # ---- CPU fallback path (original) ----
        (shift_msa, scale_msa, gate_msa), (shift_mlp, scale_mlp, gate_mlp) = mod_img
        (c_shift_msa, c_scale_msa, c_gate_msa), (c_shift_mlp, c_scale_mlp, c_gate_mlp) = mod_txt

        if is_gpu:
            hidden_states = runner.readback(hidden_states).reshape(
                hidden_states.shape)
            encoder_hidden_states = runner.readback(
                encoder_hidden_states).reshape(encoder_hidden_states.shape)

        T_txt = encoder_hidden_states.shape[0]

        # --- Image stream attention ---
        norm_x = self._layer_norm_no_affine(hidden_states)
        norm_x = (1.0 + scale_msa) * norm_x + shift_msa

        # --- Text stream attention ---
        norm_ctx = self._layer_norm_no_affine(encoder_hidden_states)
        norm_ctx = (1.0 + c_scale_msa) * norm_ctx + c_shift_msa
        if _profile: _m()  # 0: norm+mod

        # Pre-upload norm inputs for QKV reuse (avoids re-uploads)
        upload_dt = np.float16 if self.fp16_act else np.float32
        norm_x_gpu = runner.upload_to_gpu(
            norm_x.astype(upload_dt), f"tmp_norm_img_{block_idx}")
        norm_x_gpu.shape = norm_x.shape
        norm_ctx_gpu = runner.upload_to_gpu(
            norm_ctx.astype(upload_dt), f"tmp_norm_txt_{block_idx}")
        norm_ctx_gpu.shape = norm_ctx.shape
        if _profile: _m()  # 1: upload

        # Fused QKV for image → stays on GPU
        img_qkv_gpu = self._linear(norm_x_gpu, f"{pfx}.img_qkv.weight",
                                    3 * HIDDEN_DIM, gpu_out=True)
        img_qkv_gpu.shape = (hidden_states.shape[0], 3 * HIDDEN_DIM)
        if _profile: _m()  # 2: img_QKV

        # Fused QKV for text → stays on GPU
        txt_qkv_gpu = self._linear(norm_ctx_gpu, f"{pfx}.txt_qkv.weight",
                                    3 * HIDDEN_DIM, gpu_out=True)
        txt_qkv_gpu.shape = (T_txt, 3 * HIDDEN_DIM)
        if _profile: _m()  # 3: txt_QKV

        # Fused QK-RMSNorm + RoPE on GPU for image stream
        T_img = hidden_states.shape[0]
        img_cos = concat_cos[T_txt:]  # image portion of RoPE
        img_sin = concat_sin[T_txt:]
        q_img, k_img, v_img = self._qk_norm_rope(
            img_qkv_gpu,
            self.weights[f"{pfx}.attn.norm_q.weight"].astype(np.float32),
            self.weights[f"{pfx}.attn.norm_k.weight"].astype(np.float32),
            img_cos, img_sin, NUM_HEADS, T_img, eps=EPS, gpu_out=False)

        # Fused QK-RMSNorm + RoPE on GPU for text stream
        txt_cos = concat_cos[:T_txt]
        txt_sin = concat_sin[:T_txt]
        add_q, add_k, add_v = self._qk_norm_rope(
            txt_qkv_gpu,
            self.weights[f"{pfx}.attn.norm_added_q.weight"].astype(np.float32),
            self.weights[f"{pfx}.attn.norm_added_k.weight"].astype(np.float32),
            txt_cos, txt_sin, NUM_HEADS, T_txt, eps=EPS, gpu_out=False)
        if _profile: _m()  # 4: norm+RoPE (GPU)

        # Concatenate text + image
        q = np.concatenate([add_q, q_img], axis=0)
        k = np.concatenate([add_k, k_img], axis=0)
        v = np.concatenate([add_v, v_img], axis=0)
        if _profile: _m()  # 5: concat

        # Full attention
        attn_out = self._full_attention_multihead(q, k, v, NUM_HEADS)  # (T_total, H, D)
        attn_out = attn_out.reshape(-1, HIDDEN_DIM)  # (T_total, D)
        if _profile: _m()  # 6: attention

        # Split back
        ctx_attn = attn_out[:T_txt]
        img_attn = attn_out[T_txt:]

        # Image attention output projection
        img_attn = self._linear(img_attn, f"{pfx}.attn.to_out.0.weight", HIDDEN_DIM)
        img_attn = gate_msa * img_attn
        hidden_states = hidden_states + img_attn

        # Text attention output projection
        ctx_attn = self._linear(ctx_attn, f"{pfx}.attn.to_add_out.weight", HIDDEN_DIM)
        ctx_attn = c_gate_msa * ctx_attn
        encoder_hidden_states = encoder_hidden_states + ctx_attn
        if _profile: _m()  # 7: out_proj+residual

        # --- Image FF (full GPU chain: gate/up stay on GPU) ---
        norm_x = self._layer_norm_no_affine(hidden_states)
        norm_x = norm_x * (1.0 + scale_mlp) + shift_mlp
        norm_x_ff_gpu = runner.upload_to_gpu(
            norm_x.astype(upload_dt), f"tmp_ff_img_{block_idx}")
        norm_x_ff_gpu.shape = norm_x.shape
        gate_gpu = self._linear(norm_x_ff_gpu, f"{pfx}.ff.gate_weight", FF_DIM, gpu_out=True)
        up_gpu = self._linear(norm_x_ff_gpu, f"{pfx}.ff.up_weight", FF_DIM, gpu_out=True)
        ff_act = self._silu_mul(gate_gpu, up_gpu, gpu_out=True)
        ff_out = self._linear(ff_act, f"{pfx}.ff.linear_out.weight", HIDDEN_DIM)
        hidden_states = hidden_states + gate_mlp * ff_out
        if _profile: _m()  # 8: img_FF

        # --- Text FF (full GPU chain: gate/up stay on GPU) ---
        norm_ctx = self._layer_norm_no_affine(encoder_hidden_states)
        norm_ctx = norm_ctx * (1.0 + c_scale_mlp) + c_shift_mlp
        norm_ctx_ff_gpu = runner.upload_to_gpu(
            norm_ctx.astype(upload_dt), f"tmp_ff_txt_{block_idx}")
        norm_ctx_ff_gpu.shape = norm_ctx.shape
        gate_ctx_gpu = self._linear(norm_ctx_ff_gpu, f"{pfx}.ff_context.gate_weight", FF_DIM, gpu_out=True)
        up_ctx_gpu = self._linear(norm_ctx_ff_gpu, f"{pfx}.ff_context.up_weight", FF_DIM, gpu_out=True)
        ff_ctx_act = self._silu_mul(gate_ctx_gpu, up_ctx_gpu, gpu_out=True)
        ff_ctx_out = self._linear(ff_ctx_act, f"{pfx}.ff_context.linear_out.weight", HIDDEN_DIM)
        encoder_hidden_states = encoder_hidden_states + c_gate_mlp * ff_ctx_out
        if _profile: _m()  # 9: txt_FF

        # Clip for fp16 safety
        encoder_hidden_states = np.clip(encoder_hidden_states, -65504, 65504)

        if _profile:
            labels = ["norm+mod", "upload", "img_QKV", "txt_QKV", "norm+RoPE",
                       "concat", "attention", "out_proj+res", "img_FF", "txt_FF"]
            parts = [(_t[j+1] - _t[j]) * 1000 for j in range(len(_t) - 1)]
            print(f"  [double {block_idx}] " + " | ".join(
                f"{l}:{v:.1f}" for l, v in zip(labels, parts))
                + f"  TOTAL:{sum(parts):.1f}ms")

        return encoder_hidden_states, hidden_states

    # ------------------------------------------------------------------
    # Single stream block
    # ------------------------------------------------------------------

    def _single_block(self, hidden_states: np.ndarray,
                      mod_params,
                      concat_cos: np.ndarray, concat_sin: np.ndarray,
                      block_idx: int,
                      gpu_state: bool = False,
                      _profile: bool = False):
        """One single-stream transformer block (fully GPU-resident).

        hidden_states: (T_total, D) — numpy or GPUBuffer.
        When gpu_state=True, hidden_states is GPUBuffer, and the
        result is also returned as GPUBuffer (no CPU readbacks).

        Zero readbacks when gpu_state=True:
        - LayerNorm on GPU
        - Modulation on GPU
        - QKV + MLP projections on GPU
        - Fused QK-RMSNorm + RoPE on GPU
        - Attention on GPU
        - Output projections + add on GPU
        - Gated residual on GPU
        """
        pfx = f"single_transformer_blocks.{block_idx}"
        shift, scale, gate = mod_params
        runner = self.cache.runner
        if _profile:
            _t = [time.perf_counter()]
            def _m():
                _t.append(time.perf_counter())

        is_gpu = isinstance(hidden_states, GPUBuffer)

        if is_gpu and gpu_state:
            # Fully GPU-resident path: norm + modulation on GPU
            norm_x_gpu = self._layer_norm_no_affine(hidden_states,
                                                     gpu_out=True)
            T = hidden_states.shape[0]
            # Use pre-uploaded GPUBuffers or upload numpy
            if isinstance(scale, GPUBuffer):
                scale_gpu, shift_gpu = scale, shift
            else:
                scale_gpu = runner.upload_to_gpu(
                    scale.ravel().astype(np.float32),
                    f"tmp_sb_scale_{block_idx}")
                shift_gpu = runner.upload_to_gpu(
                    shift.ravel().astype(np.float32),
                    f"tmp_sb_shift_{block_idx}")
            norm_x_gpu = self._mod_scale_shift(
                norm_x_gpu, scale_gpu, shift_gpu, HIDDEN_DIM, gpu_out=True)
            norm_x_gpu.shape = (T, HIDDEN_DIM)
            if _profile: _m()  # 0: norm+mod (GPU)
            if _profile: _m()  # 1: upload (skip)
        else:
            # CPU path (original)
            if is_gpu:
                hidden_np = runner.readback(hidden_states).reshape(
                    hidden_states.shape)
            else:
                hidden_np = hidden_states
            norm_x = self._layer_norm_no_affine(hidden_np)
            norm_x = (1.0 + scale) * norm_x + shift
            if _profile: _m()  # 0: norm+mod

            upload_dt = np.float16 if self.fp16_act else np.float32
            norm_x_gpu = runner.upload_to_gpu(
                norm_x.astype(upload_dt), f"tmp_sb_norm_{block_idx}")
            norm_x_gpu.shape = norm_x.shape
            T = norm_x.shape[0]
            if _profile: _m()  # 1: upload

        # Fused QKV projection → stays on GPU
        qkv_gpu = self._linear(norm_x_gpu, f"{pfx}.qkv_proj.weight",
                                3 * HIDDEN_DIM, gpu_out=True)
        qkv_gpu.shape = (T, 3 * HIDDEN_DIM)
        if _profile: _m()  # 2: QKV proj (no readback)

        # MLP projections → stay on GPU
        gate_proj_gpu = self._linear(norm_x_gpu, f"{pfx}.gate_proj.weight",
                                FF_DIM, gpu_out=True)
        up_gpu = self._linear(norm_x_gpu, f"{pfx}.up_proj.weight",
                              FF_DIM, gpu_out=True)
        if _profile: _m()  # 3: MLP proj

        # Fused QK-RMSNorm + RoPE on GPU → Q, K, V stay on GPU
        q_gpu, k_gpu, v_gpu = self._qk_norm_rope(
            qkv_gpu,
            self.weights[f"{pfx}.attn.norm_q.weight"].astype(np.float32),
            self.weights[f"{pfx}.attn.norm_k.weight"].astype(np.float32),
            concat_cos, concat_sin, NUM_HEADS, T, eps=EPS, gpu_out=True)
        if _profile: _m()  # 4: QK_norm+RoPE (GPU)

        # Attention on GPU (GPUBuffer input)
        attn_out_gpu = self._full_attention_multihead(
            q_gpu, k_gpu, v_gpu, NUM_HEADS, gpu_out=True)
        attn_out_gpu.shape = (T, HIDDEN_DIM)
        if _profile: _m()  # 5: attn

        # SiLU·mul stays on GPU
        mlp_out_gpu = self._silu_mul(gate_proj_gpu, up_gpu, gpu_out=True)
        if _profile: _m()  # 6: silu_mul

        # Output projections on GPU
        attn_proj_gpu = self._linear(
            attn_out_gpu, f"{pfx}.to_out_attn.weight",
            HIDDEN_DIM, gpu_out=True)
        mlp_proj_gpu = self._linear(
            mlp_out_gpu, f"{pfx}.to_out_mlp.weight",
            HIDDEN_DIM, gpu_out=True)
        if _profile: _m()  # 7: out proj

        # Add attn + mlp on GPU
        out_gpu = self._add(attn_proj_gpu, mlp_proj_gpu, gpu_out=True)
        out_gpu.shape = (T, HIDDEN_DIM)

        if gpu_state and is_gpu:
            # Gated residual on GPU: hidden_states += gate * out
            if isinstance(gate, GPUBuffer):
                gate_gpu = gate
            else:
                gate_gpu = runner.upload_to_gpu(
                    gate.ravel().astype(np.float32),
                    f"tmp_sb_gate_{block_idx}")
            self._gate_residual_add(hidden_states, gate_gpu, out_gpu,
                                     HIDDEN_DIM)
            if _profile: _m()  # 8: gate_residual (GPU, no readback)
            if _profile:
                _m()  # 9: (no clip needed - GPU values stay bounded)
                labels = ["norm+mod", "(skip)", "QKV_proj", "MLP_proj",
                           "norm+RoPE", "attn", "silu_mul",
                           "out_proj", "gate_res", "done"]
                parts = "  |  ".join(f"{l}={(_t[i+1]-_t[i])*1000:.0f}" for i, l in enumerate(labels))
                print(f"    SB{block_idx}: {parts}")
            return hidden_states
        else:
            # CPU path: readback and residual on CPU
            out = runner.readback(out_gpu).reshape(T, HIDDEN_DIM)
            if _profile: _m()  # 8: add+readback
            if is_gpu:
                hidden_np = runner.readback(hidden_states).reshape(
                    hidden_states.shape)
            else:
                hidden_np = hidden_states
            hidden_np = hidden_np + gate * out
            hidden_np = np.clip(hidden_np, -65504, 65504)
            if _profile:
                _m()  # 9: residual+clip
                labels = ["norm+mod", "upload", "QKV_proj", "MLP_proj",
                           "norm+RoPE", "attn", "silu_mul",
                           "out_proj", "add+read", "residual"]
                parts = "  |  ".join(f"{l}={(_t[i+1]-_t[i])*1000:.0f}" for i, l in enumerate(labels))
                print(f"    SB{block_idx}: {parts}")
            return hidden_np

    # ------------------------------------------------------------------
    # Output layer
    # ------------------------------------------------------------------

    def _output_layer(self, hidden_states: np.ndarray,
                      temb: np.ndarray) -> np.ndarray:
        """AdaLayerNormContinuous → Linear projection.

        hidden_states: (T_img, D)
        Returns: (T_img, IN_CHANNELS)
        """
        # AdaLayerNormContinuous: LayerNorm → SiLU(temb) → Linear → scale + shift
        norm_x = layer_norm_cpu(hidden_states, EPS)
        emb = temb * (1.0 / (1.0 + np.exp(-temb)))  # SiLU
        emb = self._linear(emb, "norm_out.linear.weight", HIDDEN_DIM * 2)  # (1, 6144)
        scale, shift = np.split(emb, 2, axis=-1)  # Each (1, 3072)
        norm_x = norm_x * (1.0 + scale) + shift

        # Final projection
        output = self._linear(norm_x, "proj_out.weight", IN_CHANNELS)
        return output

    # ------------------------------------------------------------------
    # Full forward pass
    # ------------------------------------------------------------------

    def forward(self, latents: np.ndarray,
                encoder_hidden_states: np.ndarray,
                timestep: float,
                img_ids: np.ndarray,
                txt_ids: np.ndarray,
                _profile: bool = False) -> np.ndarray:
        """Run one denoising step.

        latents: (T_img, IN_CHANNELS) packed image latents
        encoder_hidden_states: (T_txt, JOINT_ATTN_DIM) text features
        timestep: fractional timestep in [0, 1]
        img_ids: (T_img, 4) position IDs
        txt_ids: (T_txt, 4) position IDs
        Returns: (T_img, IN_CHANNELS) noise prediction
        """
        if _profile:
            _t = [time.perf_counter()]
            def _mark(label):
                _t.append(time.perf_counter())
                return label
            _labels = []
        else:
            _mark = None

        T_txt = encoder_hidden_states.shape[0]

        # 1. Timestep embedding (no guidance for Klein)
        temb = self._compute_temb(timestep)  # (1, D)
        if _profile: _labels.append(_mark("temb"))

        # 2. Compute modulation parameters
        mod_img = self._compute_modulation(
            temb, "double_stream_modulation_img.linear.weight", num_param_sets=2)
        mod_txt = self._compute_modulation(
            temb, "double_stream_modulation_txt.linear.weight", num_param_sets=2)
        mod_single = self._compute_modulation(
            temb, "single_stream_modulation.linear.weight", num_param_sets=1)
        if _profile: _labels.append(_mark("modulation"))

        # 3. Input projections
        hidden_states = self._linear(latents, "x_embedder.weight", HIDDEN_DIM)
        ctx = self._linear(encoder_hidden_states, "context_embedder.weight", HIDDEN_DIM)
        if _profile: _labels.append(_mark("input_proj"))

        # 4. Compute RoPE frequencies
        image_rope = compute_rope_freqs(img_ids, AXES_DIMS_ROPE, ROPE_THETA)
        text_rope = compute_rope_freqs(txt_ids, AXES_DIMS_ROPE, ROPE_THETA)
        concat_cos = np.concatenate([text_rope[0], image_rope[0]], axis=0)
        concat_sin = np.concatenate([text_rope[1], image_rope[1]], axis=0)
        if _profile: _labels.append(_mark("rope_freqs"))

        # 5. Double stream blocks (GPU-resident: upload once, readback after all)
        runner = self.cache.runner
        upload_dt = np.float16 if self.fp16_act else np.float32

        # Reset concat/split buffer counters to reuse GPU buffers across forward passes
        type(self)._concat_counter = 0
        type(self)._split_counter = 0

        # Upload hidden_states and ctx to GPU
        hs_gpu = runner.upload_to_gpu(
            hidden_states.astype(upload_dt), "tmp_dbl_hs")
        hs_gpu.shape = hidden_states.shape
        ctx_gpu = runner.upload_to_gpu(
            ctx.astype(upload_dt), "tmp_dbl_ctx")
        ctx_gpu.shape = ctx.shape

        # Pre-upload double block modulation params (same for all 5 blocks)
        def _upload_mod_set(mod_set, prefix):
            """Upload (shift, scale, gate) tuples to GPU."""
            result = []
            for j, (sh, sc, gt) in enumerate(mod_set):
                sh_g = runner.upload_to_gpu(
                    sh.ravel().astype(np.float32), f"{prefix}_shift_{j}")
                sc_g = runner.upload_to_gpu(
                    sc.ravel().astype(np.float32), f"{prefix}_scale_{j}")
                gt_g = runner.upload_to_gpu(
                    gt.ravel().astype(np.float32), f"{prefix}_gate_{j}")
                result.append((sh_g, sc_g, gt_g))
            return result

        mod_img_gpu = _upload_mod_set(mod_img, "db_img")
        mod_txt_gpu = _upload_mod_set(mod_txt, "db_txt")

        # --- GPU-resident double blocks (no batching, sequential submit) ---
        for i in range(NUM_DOUBLE_BLOCKS):
            ctx_gpu, hs_gpu = self._double_block(
                hs_gpu, ctx_gpu, mod_img_gpu, mod_txt_gpu,
                concat_cos, concat_sin, i,
                gpu_state=True, _profile=(_profile and i < 2))
            if _profile: _labels.append(_mark(f"double_{i}"))

        # Concatenate ctx + hs on GPU for single stream
        T_img = hs_gpu.shape[0]
        hs_gpu = self._concat_gpu(ctx_gpu, hs_gpu)
        T_total = T_txt + T_img
        hs_gpu.shape = (T_total, HIDDEN_DIM)
        if _profile: _labels.append(_mark("concat"))

        # 7. Single stream blocks (separate batch)
        runner.begin_batch()
        # Pre-upload modulation params (same for all 20 blocks)
        shift, scale, gate = mod_single[0]
        sb_scale_gpu = runner.upload_to_gpu(
            scale.ravel().astype(np.float32), "sb_mod_scale")
        sb_shift_gpu = runner.upload_to_gpu(
            shift.ravel().astype(np.float32), "sb_mod_shift")
        sb_gate_gpu = runner.upload_to_gpu(
            gate.ravel().astype(np.float32), "sb_mod_gate")

        for i in range(NUM_SINGLE_BLOCKS):
            hs_gpu = self._single_block(
                hs_gpu, (sb_shift_gpu, sb_scale_gpu, sb_gate_gpu),
                concat_cos, concat_sin, i,
                gpu_state=True,
                _profile=(_profile and i < 3))
            if _profile: _labels.append(_mark(f"single_{i}"))
        # Submit batch and readback in one shot
        rb_results = runner.end_batch(readback_buffers=[hs_gpu])
        hidden_states = rb_results[id(hs_gpu)].reshape(hs_gpu.shape)

        # 8. Remove text tokens
        hidden_states = hidden_states[T_txt:]

        # 9. Output layer
        output = self._output_layer(hidden_states, temb)

        if _profile:
            _labels.append(_mark("output"))
            print("\n  --- Forward pass breakdown ---")
            total = _t[-1] - _t[0]
            for j, label in enumerate(_labels):
                dt = (_t[j+1] - _t[j]) * 1000
                pct = dt / (total * 1000) * 100
                print(f"  {label:20s}  {dt:8.1f} ms  ({pct:5.1f}%)")
            # Aggregate summaries
            dbl_ms = sum((_t[j+1] - _t[j]) * 1000 for j, l in enumerate(_labels) if l.startswith("double_"))
            sgl_ms = sum((_t[j+1] - _t[j]) * 1000 for j, l in enumerate(_labels) if l.startswith("single_"))
            print(f"  {'--- double total':20s}  {dbl_ms:8.1f} ms  ({dbl_ms/(total*1000)*100:5.1f}%)")
            print(f"  {'--- single total':20s}  {sgl_ms:8.1f} ms  ({sgl_ms/(total*1000)*100:5.1f}%)")
            print(f"  {'--- TOTAL':20s}  {total*1000:8.1f} ms")
            print(f"  avg double block: {dbl_ms/NUM_DOUBLE_BLOCKS:.1f} ms")
            print(f"  avg single block: {sgl_ms/NUM_SINGLE_BLOCKS:.1f} ms")

        return output


# ---------------------------------------------------------------------------
# Text encoding (Qwen3)
# ---------------------------------------------------------------------------

def encode_prompt(prompt: str, tokenizer, text_encoder, device="cpu",
                  max_seq_len: int = 512) -> np.ndarray:
    """Encode prompt using Qwen3 text encoder.

    Returns: (1, seq_len, JOINT_ATTN_DIM) prompt embeddings.
    """
    import torch

    messages = [
        {"role": "system", "content": SYSTEM_MESSAGE},
        {"role": "user", "content": prompt},
    ]

    inputs = tokenizer.apply_chat_template(
        [messages],
        add_generation_prompt=False,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=max_seq_len,
    )

    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    with torch.no_grad():
        output = text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False,
        )

    # Stack outputs from 3 intermediate layers
    stacked = torch.stack(
        [output.hidden_states[k] for k in QWEN3_OUT_LAYERS], dim=1)
    # stacked: (1, 3, seq_len, 2560)

    B, C, S, D = stacked.shape
    prompt_embeds = stacked.permute(0, 2, 1, 3).reshape(B, S, C * D)
    # prompt_embeds: (1, seq_len, 7680)

    return prompt_embeds.float().cpu().numpy()


# ---------------------------------------------------------------------------
# VAE decode
# ---------------------------------------------------------------------------

def vae_decode(latents: np.ndarray, latent_ids: np.ndarray,
               vae, vae_bn_mean: np.ndarray, vae_bn_std: np.ndarray,
               height: int, width: int) -> np.ndarray:
    """Decode latents to image using FLUX2 VAE.

    latents: (T_img, IN_CHANNELS) packed
    Returns: (H, W, 3) uint8 image
    """
    import torch

    lt = torch.from_numpy(latents).unsqueeze(0)  # (1, T, C)
    ids = torch.from_numpy(latent_ids).unsqueeze(0)  # (1, T, 4)

    # Unpack with position IDs → (1, C, H_lat, W_lat)
    data = lt[0]
    pos = ids[0]
    ch = data.shape[1]
    h_ids = pos[:, 1].to(torch.int64)
    w_ids = pos[:, 2].to(torch.int64)
    h = torch.max(h_ids) + 1
    w = torch.max(w_ids) + 1
    flat_ids = h_ids * w + w_ids
    out = torch.zeros((h * w, ch), dtype=data.dtype)
    out.scatter_(0, flat_ids.unsqueeze(1).expand(-1, ch), data)
    unpacked = out.view(h, w, ch).permute(2, 0, 1).unsqueeze(0)  # (1, C, H, W)

    # BatchNorm denormalization
    bn_mean = torch.from_numpy(vae_bn_mean).view(1, -1, 1, 1)
    bn_std = torch.from_numpy(vae_bn_std).view(1, -1, 1, 1)
    unpacked = unpacked * bn_std + bn_mean

    # Unpatchify: (1, C, H, W) where C=128 → (1, 32, 2H, 2W)
    B, C_total, H_lat, W_lat = unpacked.shape
    unpacked = unpacked.reshape(B, C_total // 4, 2, 2, H_lat, W_lat)
    unpacked = unpacked.permute(0, 1, 4, 2, 5, 3)
    unpacked = unpacked.reshape(B, C_total // 4, H_lat * 2, W_lat * 2)

    # VAE decode
    with torch.no_grad():
        vae_out = vae.decode(unpacked.to(vae.dtype), return_dict=False)[0]

    # Post-process: [-1, 1] → [0, 255]
    img = vae_out.float().squeeze(0).permute(1, 2, 0).cpu().numpy()
    img = ((img + 1.0) / 2.0 * 255.0).clip(0, 255).astype(np.uint8)
    return img


# ---------------------------------------------------------------------------
# Denoising loop
# ---------------------------------------------------------------------------

def generate_image(model: FluxKleinWebGPU,
                   prompt_embeds: np.ndarray,
                   height: int, width: int,
                   num_steps: int = 20,
                   seed: int = 42,
                   profile: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """Run the full denoising loop.

    Returns: (latents, latent_ids) as numpy arrays.
    """
    # Compute latent dimensions
    lat_h = 2 * (height // (VAE_SCALE_FACTOR * 2))
    lat_w = 2 * (width // (VAE_SCALE_FACTOR * 2))
    num_channels = IN_CHANNELS // 4  # 32

    # Generate noise: (1, C*4, H//2, W//2) then pack
    rng = np.random.RandomState(seed)
    noise = rng.randn(1, num_channels * 4, lat_h // 2, lat_w // 2).astype(np.float32)

    # Pack: (1, C, H, W) → (T_img, C)
    B, C, H, W = noise.shape
    latents = noise.reshape(B, C, H * W).transpose(0, 2, 1).squeeze(0)  # (T_img, C)

    # Position IDs
    img_ids = prepare_latent_ids(H, W)
    txt_ids = prepare_text_ids(prompt_embeds.shape[1])

    # Prompt embeds: (1, seq, D) → (seq, D)
    pe = prompt_embeds.squeeze(0)

    # Scheduler: compute sigmas with dynamic shifting
    image_seq_len = latents.shape[0]
    mu = compute_empirical_mu(image_seq_len, num_steps)
    sigmas = np.linspace(1.0, 1.0 / num_steps, num_steps)

    # Apply exponential shift
    from diffusers.schedulers.scheduling_flow_match_euler_discrete import (
        FlowMatchEulerDiscreteScheduler,
    )
    scheduler = FlowMatchEulerDiscreteScheduler(shift=3.0, use_dynamic_shifting=True)
    scheduler.set_timesteps(num_steps, mu=mu)
    timesteps = scheduler.timesteps.numpy()
    sigmas_sched = scheduler.sigmas.numpy()

    print(f"  Latent shape: ({H}, {W}) = {image_seq_len} tokens")
    print(f"  Text shape: {pe.shape}")
    print(f"  Steps: {num_steps}, mu={mu:.3f}")
    print(f"  Sigmas: {sigmas_sched[:4]}...")

    for i, t in enumerate(timesteps):
        t0 = time.time()

        sigma = sigmas_sched[i]
        timestep_frac = t.item() / 1000.0

        noise_pred = model.forward(
            latents, pe,
            timestep=timestep_frac,
            img_ids=img_ids,
            txt_ids=txt_ids,
            _profile=(profile and i == 0),
        )

        # Euler step: latents = latents + (sigma_next - sigma) * vel
        sigma_next = sigmas_sched[i + 1] if i + 1 < len(sigmas_sched) else 0.0
        dt = sigma_next - sigma
        latents = latents + dt * noise_pred

        elapsed = time.time() - t0
        print(f"  Step {i+1}/{num_steps}: t={t.item():.1f}, "
              f"sigma={sigma:.4f}, dt={dt:.4f}, "
              f"pred std={noise_pred.std():.4f}, "
              f"latent std={latents.std():.4f}, "
              f"time={elapsed:.2f}s")

    return latents, img_ids


# ---------------------------------------------------------------------------
# Verification with random weights
# ---------------------------------------------------------------------------

def verify_with_random_weights():
    """Quick verification with tiny random weights."""
    print("=== Verification with random weights ===")
    rng = np.random.RandomState(42)
    s = 0.02

    W = {}
    D = HIDDEN_DIM

    # Timestep embedding (no guidance embedder for Klein)
    W["time_guidance_embed.timestep_embedder.linear_1.weight"] = rng.randn(D, TIMESTEP_CHANNELS).astype(np.float32) * s
    W["time_guidance_embed.timestep_embedder.linear_2.weight"] = rng.randn(D, D).astype(np.float32) * s

    # Modulations
    W["double_stream_modulation_img.linear.weight"] = rng.randn(D * 6, D).astype(np.float32) * s
    W["double_stream_modulation_txt.linear.weight"] = rng.randn(D * 6, D).astype(np.float32) * s
    W["single_stream_modulation.linear.weight"] = rng.randn(D * 3, D).astype(np.float32) * s

    # Input projections
    W["x_embedder.weight"] = rng.randn(D, IN_CHANNELS).astype(np.float32) * s
    W["context_embedder.weight"] = rng.randn(D, JOINT_ATTN_DIM).astype(np.float32) * s

    # Double blocks
    for i in range(NUM_DOUBLE_BLOCKS):
        pfx = f"transformer_blocks.{i}"
        for proj in ["to_q", "to_k", "to_v"]:
            W[f"{pfx}.attn.{proj}.weight"] = rng.randn(D, D).astype(np.float32) * s
        W[f"{pfx}.attn.to_out.0.weight"] = rng.randn(D, D).astype(np.float32) * s
        for proj in ["add_q_proj", "add_k_proj", "add_v_proj"]:
            W[f"{pfx}.attn.{proj}.weight"] = rng.randn(D, D).astype(np.float32) * s
        W[f"{pfx}.attn.to_add_out.weight"] = rng.randn(D, D).astype(np.float32) * s
        for norm in ["norm_q", "norm_k", "norm_added_q", "norm_added_k"]:
            W[f"{pfx}.attn.{norm}.weight"] = np.ones(HEAD_DIM, dtype=np.float32)
        W[f"{pfx}.ff.linear_in.weight"] = rng.randn(FF_DIM * 2, D).astype(np.float32) * s
        W[f"{pfx}.ff.linear_out.weight"] = rng.randn(D, FF_DIM).astype(np.float32) * s
        W[f"{pfx}.ff_context.linear_in.weight"] = rng.randn(FF_DIM * 2, D).astype(np.float32) * s
        W[f"{pfx}.ff_context.linear_out.weight"] = rng.randn(D, FF_DIM).astype(np.float32) * s

    # Single blocks
    for i in range(NUM_SINGLE_BLOCKS):
        pfx = f"single_transformer_blocks.{i}"
        fused = 3 * D + 2 * FF_DIM
        W[f"{pfx}.attn.to_qkv_mlp_proj.weight"] = rng.randn(fused, D).astype(np.float32) * s
        W[f"{pfx}.attn.norm_q.weight"] = np.ones(HEAD_DIM, dtype=np.float32)
        W[f"{pfx}.attn.norm_k.weight"] = np.ones(HEAD_DIM, dtype=np.float32)
        W[f"{pfx}.attn.to_out.weight"] = rng.randn(D, D + FF_DIM).astype(np.float32) * s

    # Output
    W["norm_out.linear.weight"] = rng.randn(D * 2, D).astype(np.float32) * s
    W["proj_out.weight"] = rng.randn(IN_CHANNELS, D).astype(np.float32) * s

    print(f"  Created {len(W)} weight tensors")

    model = FluxKleinWebGPU(W)

    # Fake inputs
    T_img = 64  # 8x8 latent grid
    T_txt = 8
    latents = rng.randn(T_img, IN_CHANNELS).astype(np.float32) * 0.1
    ctx = rng.randn(T_txt, JOINT_ATTN_DIM).astype(np.float32) * 0.1
    img_ids = prepare_latent_ids(8, 8)
    txt_ids = prepare_text_ids(T_txt)

    t0 = time.time()
    out = model.forward(latents, ctx, timestep=0.5,
                        img_ids=img_ids, txt_ids=txt_ids)
    t1 = time.time()

    print(f"\nForward pass: ({T_img}, {IN_CHANNELS}) → {out.shape} "
          f"in {(t1-t0)*1000:.0f}ms")
    print(f"Output range: [{out.min():.4f}, {out.max():.4f}]")
    print(f"Output mean: {out.mean():.6f}, std: {out.std():.6f}")

    is_finite = np.all(np.isfinite(out))
    has_signal = out.std() > 1e-6
    correct_shape = out.shape == (T_img, IN_CHANNELS)

    print(f"\nAll finite: {is_finite}")
    print(f"Has signal: {has_signal}")
    print(f"Correct shape: {correct_shape}")

    success = is_finite and has_signal and correct_shape
    print(f"\n{'PASS' if success else 'FAIL'}")
    return success


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="FLUX.2 Klein 4B on WebGPU")
    parser.add_argument("--verify", action="store_true",
                        help="Verify with random weights")
    parser.add_argument("--prompt", type=str,
                        default="a fluffy white cat sitting on a windowsill")
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--steps", type=int, default=20)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default="output.png")
    parser.add_argument("--profile", action="store_true")
    args = parser.parse_args()

    if args.verify:
        success = verify_with_random_weights()
        sys.exit(0 if success else 1)

    # --- Full inference ---
    import torch
    weights_dir = os.path.join(_SCRIPT_DIR, "weights")
    hf_cache = os.path.join(weights_dir, "hf_cache")

    # Step 1: Load text encoder and encode prompt
    print("=== Loading text encoder (Qwen3) ===")
    from transformers import AutoTokenizer, AutoModelForCausalLM

    tokenizer = AutoTokenizer.from_pretrained(
        os.path.join(hf_cache, "tokenizer"), local_files_only=True)
    text_encoder = AutoModelForCausalLM.from_pretrained(
        os.path.join(hf_cache, "text_encoder"),
        torch_dtype=torch.bfloat16, local_files_only=True)

    print(f"  Text encoder loaded: {sum(p.numel() for p in text_encoder.parameters()) / 1e9:.2f}B params")

    print(f"\n=== Encoding prompt: '{args.prompt}' ===")
    prompt_embeds = encode_prompt(args.prompt, tokenizer, text_encoder)
    print(f"  Prompt embeddings: {prompt_embeds.shape}")

    # Free text encoder
    del text_encoder, tokenizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    import gc; gc.collect()
    print("  Text encoder freed")

    # Step 2: Load VAE for later use (keep on CPU)
    print("\n=== Loading VAE ===")
    from diffusers.models import AutoencoderKLFlux2
    import json

    vae_path = os.path.join(hf_cache, "vae")
    vae = AutoencoderKLFlux2.from_pretrained(vae_path, local_files_only=True)
    vae.eval()

    # Extract BatchNorm stats
    vae_bn_mean = vae.bn.running_mean.float().numpy()
    vae_bn_var = vae.bn.running_var.float().numpy()
    vae_config = json.load(open(os.path.join(vae_path, "config.json")))
    bn_eps = vae_config.get("batch_norm_eps", 1e-4)
    vae_bn_std = np.sqrt(vae_bn_var + bn_eps)
    print(f"  VAE loaded, BN mean range: [{vae_bn_mean.min():.3f}, {vae_bn_mean.max():.3f}]")

    # Step 3: Load transformer weights and init WebGPU model
    print("\n=== Loading transformer weights ===")
    npz_path = os.path.join(weights_dir, "transformer_fp16.npz")
    if not os.path.exists(npz_path):
        print(f"  NPZ not found at {npz_path}, running conversion...")
        from convert_weights import convert
        convert()

    t0 = time.time()
    weights = dict(np.load(npz_path))
    print(f"  Loaded {len(weights)} tensors in {time.time()-t0:.1f}s")

    print("\n=== Initializing WebGPU model ===")
    model = FluxKleinWebGPU(weights)

    if args.profile:
        model.enable_profiling()

    # Step 4: Run denoising loop
    print(f"\n=== Generating {args.height}x{args.width} image ({args.steps} steps) ===")
    latents, img_ids = generate_image(
        model, prompt_embeds,
        args.height, args.width,
        num_steps=args.steps,
        seed=args.seed,
        profile=args.profile,
    )

    # Step 5: VAE decode
    print("\n=== VAE decoding ===")
    image = vae_decode(latents, img_ids, vae, vae_bn_mean, vae_bn_std,
                       args.height, args.width)
    print(f"  Image shape: {image.shape}")
    print(f"  Pixel range: [{image.min()}, {image.max()}]")

    # Save
    from PIL import Image
    Image.fromarray(image).save(args.output)
    print(f"\n  Saved to {args.output}")

    if args.profile:
        pass  # profiling output printed during forward pass


if __name__ == "__main__":
    main()
