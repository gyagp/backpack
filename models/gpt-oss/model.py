"""
GPT-OSS-20B inference on WebGPU via Triton.

Architecture: 24-layer MoE transformer (32 experts, top-4 routing)
with MXFP4 quantized expert weights, GQA attention with sinks,
alternating full/sliding-window attention, and YaRN RoPE.

Usage:
    python python/examples/webgpu/gpt-oss/model.py --prompt "Hello" --max-tokens 50
    python python/examples/webgpu/gpt-oss/model.py --verify

Requirements:
    Dawn WebGPU library built at third_party/webgpu/dawn/build/
"""
import os
import sys
import time
from typing import Dict, Optional

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_SCRIPT_DIR))

import numpy as np

from common.model_base import WebGPUModel
from common.utils import _parse_safetensors, load_tokenizer, generate
from triton.backends.webgpu.dawn_runner import GPUBuffer

# ---------------------------------------------------------------------------
# MXFP4 utilities
# ---------------------------------------------------------------------------

# FP4 E2M1 lookup table: nibble → float value
FP4_LUT = np.array([
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,      # positive
    0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,  # negative
], dtype=np.float32)


def dequant_mxfp4_block(blocks, scales, out_K):
    """Dequantize one expert's MXFP4 weight matrix to fp32.

    Args:
        blocks: (N, n_blocks, 16) uint8 — packed FP4 data
        scales: (N, n_blocks) uint8 — E8M0 scale per 32-element block
        out_K: original K dimension (may need trimming after unpack)

    Returns:
        (N, out_K) float32 weight matrix
    """
    N, n_blocks, _ = blocks.shape
    # Unpack nibbles: each byte → 2 FP4 values
    low = (blocks & 0x0F).astype(np.int32)
    high = ((blocks >> 4) & 0x0F).astype(np.int32)
    # Interleave: [byte0_low, byte0_high, byte1_low, byte1_high, ...]
    unpacked = np.stack([low, high], axis=-1).reshape(N, n_blocks, 32)

    # Lookup FP4 values
    fp_values = FP4_LUT[unpacked]

    # Compute E8M0 scales: 2^(byte - 127)
    scale_factors = np.power(2.0, scales.astype(np.float32) - 127.0)
    # Broadcast: (N, n_blocks, 1) * (N, n_blocks, 32)
    fp_values *= scale_factors[:, :, np.newaxis]

    # Flatten blocks → K and trim to original size
    result = fp_values.reshape(N, n_blocks * 32)
    return result[:, :out_K]


def prepare_mxfp4_gpu(blocks_u8, scales_u8):
    """Prepare MXFP4 weight data for GPU upload.

    Args:
        blocks_u8: (N, n_blocks, 16) uint8 — raw from safetensors
        scales_u8: (N, n_blocks) uint8 — E8M0 scales

    Returns:
        (blocks_i32, scales_i32):
            blocks_i32: (N, n_blocks*4) int32 — reinterpreted bytes as i32
            scales_i32: (N, ceil(n_blocks/4)) int32 — packed 4 scale bytes per i32
    """
    N, n_blocks, _ = blocks_u8.shape

    # Blocks: reshape (N, n_blocks, 16) → (N, n_blocks*16) u8 → view as i32
    blocks_flat = blocks_u8.reshape(N, n_blocks * 16)
    # Pad to multiple of 4 bytes (should already be since 16 is multiple of 4)
    blocks_i32 = np.ascontiguousarray(blocks_flat).view(np.int32)
    # Shape: (N, n_blocks * 4) i32

    # Scales: pack 4 u8 values into each i32
    n_scale_words = (n_blocks + 3) // 4
    scales_padded = np.zeros((N, n_scale_words * 4), dtype=np.uint8)
    scales_padded[:, :n_blocks] = scales_u8
    scales_i32 = np.ascontiguousarray(scales_padded).view(np.int32)
    # Shape: (N, n_scale_words) i32

    return blocks_i32, scales_i32


# ---------------------------------------------------------------------------
# Model configuration
# ---------------------------------------------------------------------------

LAYER_TYPES = [
    "sliding_attention", "full_attention",
    "sliding_attention", "full_attention",
    "sliding_attention", "full_attention",
    "sliding_attention", "full_attention",
    "sliding_attention", "full_attention",
    "sliding_attention", "full_attention",
    "sliding_attention", "full_attention",
    "sliding_attention", "full_attention",
    "sliding_attention", "full_attention",
    "sliding_attention", "full_attention",
    "sliding_attention", "full_attention",
    "sliding_attention", "full_attention",
]

GPTOSS_CONFIG = {
    "hf_repo": "openai/gpt-oss-20b",
    "safetensors_files": [
        "model-00000-of-00002.safetensors",
        "model-00001-of-00002.safetensors",
        "model-00002-of-00002.safetensors",
    ],
}


# ---------------------------------------------------------------------------
# GPT-OSS-20B Model
# ---------------------------------------------------------------------------

class GptOssWebGPU(WebGPUModel):
    """GPT-OSS-20B inference on WebGPU via Triton kernels.

    24 layers, 32 MoE experts (top-4), 64 Q-heads, 8 KV-heads,
    hidden_size=2880, head_dim=64, MXFP4 expert weights.
    """

    MAX_SEQ_LEN = 2048
    SLIDING_WINDOW = 128
    NUM_EXPERTS = 32
    TOP_K_EXPERTS = 4

    def __init__(self, weights: Dict[str, np.ndarray],
                 n_layer: int = 24, n_head: int = 64, n_kv_heads: int = 8,
                 n_embd: int = 2880, intermediate_size: int = 2880,
                 n_vocab: int = 201088, head_dim: int = 64,
                 rope_theta: float = 150000.0, rms_norm_eps: float = 1e-5,
                 max_seq_len: int = 2048,
                 num_experts: int = 32, top_k_experts: int = 4):
        self.MAX_SEQ_LEN = max_seq_len
        self.NUM_EXPERTS = num_experts
        self.TOP_K_EXPERTS = top_k_experts

        # Q, K, V output dimensions
        self.q_dim = n_head * head_dim       # 4096
        # (kv_dim and n_rep are set by base class)

        super().__init__(
            weights, n_layer=n_layer, n_head=n_head, n_embd=n_embd,
            n_vocab=n_vocab, n_kv_heads=n_kv_heads,
            intermediate_size=intermediate_size, head_dim=head_dim,
            rope_theta=rope_theta, rms_norm_eps=rms_norm_eps,
            k_dimensions={n_embd, self.q_dim},
        )

        # Fused QKV output dimension
        self.qkv_out = self.q_dim + 2 * self.kv_dim  # 5120

        self._fuse_qkv_weights()
        self._precompute_rope_tables()
        self._upload_weights_to_gpu()
        self._init_kv_cache()

    def _fuse_qkv_weights(self):
        """Fuse Q, K, V projection weights/biases into single matrices."""
        E = self.n_embd
        for i in range(self.n_layer):
            pfx = f"layers.{i}.self_attn."
            # Fuse weights: [Q; K; V] along output dim
            q_w = self.weights[pfx + "q_proj.weight"].reshape(self.q_dim, E)
            k_w = self.weights[pfx + "k_proj.weight"].reshape(self.kv_dim, E)
            v_w = self.weights[pfx + "v_proj.weight"].reshape(self.kv_dim, E)
            self.weights[pfx + "qkv_proj.weight"] = np.concatenate(
                [q_w, k_w, v_w], axis=0)  # (5120, 2880)
            del self.weights[pfx + "q_proj.weight"]
            del self.weights[pfx + "k_proj.weight"]
            del self.weights[pfx + "v_proj.weight"]
            # Fuse biases
            q_b = self.weights[pfx + "q_proj.bias"]
            k_b = self.weights[pfx + "k_proj.bias"]
            v_b = self.weights[pfx + "v_proj.bias"]
            self.weights[pfx + "qkv_proj.bias"] = np.concatenate(
                [q_b.ravel(), k_b.ravel(), v_b.ravel()])
            del self.weights[pfx + "q_proj.bias"]
            del self.weights[pfx + "k_proj.bias"]
            del self.weights[pfx + "v_proj.bias"]

    def _compile_model_kernels(self):
        """Compile MXFP4, gated activation, and RMSNorm kernels."""
        self._compile_rms_norm()
        self._compile_mxfp4_kernels()

    def _precompute_rope_tables(self):
        """Precompute YaRN RoPE cos/sin tables and upload to GPU."""
        dim = self.head_dim  # 64
        inv_freq = 1.0 / (self.rope_theta ** (
            np.arange(0, dim, 2, dtype=np.float64) / dim))
        positions = np.arange(self.MAX_SEQ_LEN, dtype=np.float64)
        freqs = np.outer(positions, inv_freq)  # (MAX_SEQ_LEN, dim//2)
        self._cos_table = np.cos(freqs).astype(np.float32)
        self._sin_table = np.sin(freqs).astype(np.float32)
        # Upload to GPU for GPU-side RoPE
        runner = self.cache.runner
        self._rope_cos_gpu = runner.upload_to_gpu(
            self._cos_table.ravel(), "rope_cos_table")
        self._rope_sin_gpu = runner.upload_to_gpu(
            self._sin_table.ravel(), "rope_sin_table")

    def _init_kv_cache(self):
        """Pre-allocate GPU KV cache buffers for all layers."""
        runner = self.cache.runner
        n_kv = self.n_kv_heads
        HD = self.head_dim
        buf_size = self.MAX_SEQ_LEN * n_kv * HD
        self._gpu_kv_cache = {}  # layer -> (K_gpu, V_gpu, cur_len)
        for i in range(self.n_layer):
            k_buf = runner.upload_to_gpu(
                np.zeros(buf_size, dtype=np.float32), f"kv_cache_K_{i}")
            v_buf = runner.upload_to_gpu(
                np.zeros(buf_size, dtype=np.float32), f"kv_cache_V_{i}")
            self._gpu_kv_cache[i] = (k_buf, v_buf, 0)
        # Set kv_cache so forward() knows it's initialized
        self.kv_cache = self._gpu_kv_cache

        # Pre-allocate MoE working buffers (reused across layers)
        E = self.n_embd
        self._moe_x_gpu = runner.upload_to_gpu(
            np.zeros(E, dtype=np.float32), "moe_x_buf")
        self._moe_x_gpu.shape = (1, E)
        self._moe_acc_gpu = runner.upload_to_gpu(
            np.zeros(E, dtype=np.float32), "moe_acc_buf")
        self._moe_acc_gpu.shape = (1, E)
        self._moe_zeros = np.zeros(E, dtype=np.float32).tobytes()

    def _upload_weights_to_gpu(self):
        """Upload all weights to GPU."""
        E = self.n_embd
        runner = self.cache.runner
        t0 = time.time()

        for i in range(self.n_layer):
            pfx = f"layers.{i}."

            # RMSNorm weights
            self._upload_norm_weight(pfx + "input_layernorm.weight")
            self._upload_norm_weight(pfx + "post_attention_layernorm.weight")

            # Fused QKV weight (fp16) + bias
            self._upload_linear_weight_fp16(
                pfx + "self_attn.qkv_proj.weight", self.qkv_out, E)
            self._upload_bias(pfx + "self_attn.qkv_proj.bias")

            # O projection (fp16) + bias
            self._upload_linear_weight_fp16(
                pfx + "self_attn.o_proj.weight", E, self.q_dim)
            self._upload_bias(pfx + "self_attn.o_proj.bias")

            # Attention sinks — upload to GPU
            sinks = self.weights[pfx + "self_attn.sinks"]
            self._gpu_weights[pfx + "self_attn.sinks"] = \
                runner.upload_to_gpu(sinks.ravel().astype(np.float32),
                                     pfx + "self_attn.sinks")

            # Router weights (keep as CPU for routing)
            # self.weights[pfx + "mlp.router.weight"] and .bias in memory

            # MoE expert weights (MXFP4)
            self._upload_expert_weights(i)

        # Final norm
        self._upload_norm_weight("norm.weight")

        # Embed tokens: keep fp32 in CPU for token ID lookup
        # (not uploaded to GPU)

        # LM head (separate weight) — upload as fp16
        self._upload_linear_weight_fp16(
            "lm_head.weight", self.n_vocab, E)
        self._upload_zero_bias("zero_bias_V", self.n_vocab)
        self._upload_zero_bias("zero_bias_QKV", self.qkv_out)
        self._upload_zero_bias("zero_bias_E", E)

        elapsed = time.time() - t0
        self._print_gpu_weight_stats()
        print(f"  Weight upload took {elapsed:.1f}s")

    def _upload_expert_weights(self, layer_idx: int):
        """Upload MXFP4 expert weights for one layer."""
        runner = self.cache.runner
        pfx = f"layers.{layer_idx}.mlp.experts."

        for proj_name, N_out in [("gate_up_proj", 2 * self.intermediate_size),
                                  ("down_proj", self.n_embd)]:
            blocks_key = pfx + proj_name + "_blocks"
            scales_key = pfx + proj_name + "_scales"
            bias_key = pfx + proj_name + "_bias"

            blocks_all = self.weights[blocks_key]   # (32, N, n_blocks, 16) u8
            scales_all = self.weights[scales_key]   # (32, N, n_blocks) u8
            bias_all = self.weights[bias_key]       # (32, N) fp32

            for e in range(self.NUM_EXPERTS):
                # Prepare GPU data for this expert
                blocks_i32, scales_i32 = prepare_mxfp4_gpu(
                    blocks_all[e], scales_all[e])

                gpu_name = f"layers.{layer_idx}.expert.{e}.{proj_name}"
                self._gpu_weights[gpu_name + ".blocks"] = \
                    runner.upload_to_gpu(blocks_i32, gpu_name + ".blocks")
                self._gpu_weights[gpu_name + ".scales"] = \
                    runner.upload_to_gpu(scales_i32, gpu_name + ".scales")
                self._gpu_weights[gpu_name + ".bias"] = \
                    runner.upload_to_gpu(
                        bias_all[e].astype(np.float32), gpu_name + ".bias")

            # Free CPU copies to save RAM
            del self.weights[blocks_key]
            del self.weights[scales_key]
            del self.weights[bias_key]

    # -- MoE --

    def _moe_block(self, x_cpu, layer: int):
        """Mixture of Experts block — GPU-resident accumulation.

        x_cpu: numpy array of norm output for routing + GPU upload
        Pre-allocated buffers _moe_x_gpu and _moe_acc_gpu must be
        written before calling (done in _decode_gpu).

        Returns: GPUBuffer (1, n_embd) — weighted expert output
        """
        pfx = f"layers.{layer}.mlp."
        E = self.n_embd
        N_gate_up = 2 * self.intermediate_size

        # Router (CPU — small 32×2880 matmul + top-k)
        router_w = self.weights[pfx + "router.weight"]  # (32, 2880)
        router_b = self.weights[pfx + "router.bias"]    # (32,)
        logits = x_cpu @ router_w.T + router_b           # (1, 32)
        logits_flat = logits.ravel()

        # Top-k selection
        top_k_idx = np.argpartition(logits_flat, -self.TOP_K_EXPERTS
                                     )[-self.TOP_K_EXPERTS:]
        top_k_vals = logits_flat[top_k_idx]
        # Softmax over selected
        top_k_vals -= top_k_vals.max()
        exp_vals = np.exp(top_k_vals)
        routing_weights = exp_vals / exp_vals.sum()

        x_moe = self._moe_x_gpu
        moe_acc = self._moe_acc_gpu

        # Run active experts — all on GPU, accumulate with weighted add
        for i in range(self.TOP_K_EXPERTS):
            e = int(top_k_idx[i])
            w = float(routing_weights[i])

            gate_up_name = f"layers.{layer}.expert.{e}.gate_up_proj"
            down_name = f"layers.{layer}.expert.{e}.down_proj"

            # gate_up matmul on GPU
            gate_up = self._linear_mxfp4(
                x_moe,
                self._gpu_weights[gate_up_name + ".blocks"],
                self._gpu_weights[gate_up_name + ".scales"],
                self._gpu_weights[gate_up_name + ".bias"],
                N_gate_up, E, gpu_out=True)

            # Gated activation on GPU
            activated = self._gptoss_gate(gate_up, self.intermediate_size,
                                          gpu_out=True)

            # down_proj matmul on GPU — stays on GPU
            down_gpu = self._linear_mxfp4(
                activated,
                self._gpu_weights[down_name + ".blocks"],
                self._gpu_weights[down_name + ".scales"],
                self._gpu_weights[down_name + ".bias"],
                E, self.intermediate_size, gpu_out=True)

            # Weighted accumulation on GPU: moe_acc += w * down_gpu
            self._add_scaled(moe_acc, down_gpu, w)

        return moe_acc

    # -- GPU decode phases --

    def _attn_phase(self, x_gpu, layer: int, pos: int):
        """Attention phase — dispatches into current batch (caller manages
        begin_batch/end_batch). Returns (x_gpu, rn2_gpu).

        Dispatches: norm1, qkv, rope_q, rope_kv, attn, o_proj, res1, norm2
        """
        pfx = f"layers.{layer}."
        E = self.n_embd
        HD = self.head_dim
        n_head = self.n_head
        n_kv = self.n_kv_heads
        n_rep = self.n_rep
        half_rot = HD // 2
        q_size = n_head * HD
        kv_size = n_kv * HD

        if self._profiling: self._set_gpu_op(f"L{layer}/norm1")
        rn1 = self._rms_norm(
            x_gpu, self._gpu_weights[pfx + "input_layernorm.weight"],
            gpu_out=True)

        if self._profiling: self._set_gpu_op(f"L{layer}/qkv")
        qkv_gpu = self._linear_fp16w(
            rn1,
            self._gpu_weights[pfx + "self_attn.qkv_proj.weight.fp16"],
            self._gpu_weights[pfx + "self_attn.qkv_proj.bias"],
            self.qkv_out, K=E, gpu_out=True)

        # Fused RoPE Q + RoPE K + scatter KV (saves 1 dispatch)
        if self._profiling: self._set_gpu_op(f"L{layer}/rope_qkv")
        K_cache_gpu, V_cache_gpu, cur_len = self._gpu_kv_cache[layer]
        cache_offset = cur_len * n_kv * HD
        rope_out = self.cache.run(
            self._fused_rope_result, grid=(n_head + n_kv,),
            buffers={
                'QKV': qkv_gpu,
                'Q_out': np.zeros(q_size, dtype=np.float32),
                'K_cache': K_cache_gpu, 'V_cache': V_cache_gpu,
                'CosTable': self._rope_cos_gpu,
                'SinTable': self._rope_sin_gpu,
            },
            scalars={
                'n_head': n_head, 'q_size': q_size,
                'kv_size': kv_size, 'pos': pos,
                'half_rot': half_rot, 'cache_offset': cache_offset,
            },
            gpu_outputs={'Q_out', 'K_cache', 'V_cache'})
        q_rot_gpu = rope_out['Q_out']

        T_total = cur_len + 1
        self._gpu_kv_cache[layer] = (K_cache_gpu, V_cache_gpu, T_total)

        if self._profiling: self._set_gpu_op(f"L{layer}/attn")
        is_sliding = (LAYER_TYPES[layer] == "sliding_attention")
        kv_start = max(0, T_total - self.SLIDING_WINDOW) if is_sliding else 0
        T_win = T_total - kv_start
        scale = np.float32(1.0 / np.sqrt(HD))
        kv_stride = n_kv * HD

        attn_out = self.cache.run(
            self._gqa_attn_sink_result, grid=(n_head,),
            buffers={
                'Q': q_rot_gpu,
                'K_cache': K_cache_gpu, 'V_cache': V_cache_gpu,
                'Sinks': self._gpu_weights[pfx + "self_attn.sinks"],
                'Out': np.zeros(n_head * HD, dtype=np.float32),
            },
            scalars={
                'kv_stride': kv_stride, 'kv_start': kv_start,
                'n_rep': n_rep, 'T_win': T_win,
                'scale': scale, 'neg_inf': np.float32(-1e9),
            },
            gpu_outputs={'Out'})
        attn_gpu = attn_out['Out']
        attn_gpu.shape = (1, n_head * HD)

        if self._profiling: self._set_gpu_op(f"L{layer}/o_proj")
        o_gpu = self._linear_fp16w(
            attn_gpu,
            self._gpu_weights[pfx + "self_attn.o_proj.weight.fp16"],
            self._gpu_weights[pfx + "self_attn.o_proj.bias"],
            E, K=self.q_dim, gpu_out=True)

        # Fused residual add + RMSNorm (saves 1 dispatch vs separate ops)
        if self._profiling: self._set_gpu_op(f"L{layer}/res1+norm2")
        rn2 = self._add_rms_norm(
            x_gpu, o_gpu,
            self._gpu_weights[pfx + "post_attention_layernorm.weight"],
            gpu_out=True)

        return x_gpu, rn2

    def _moe_phase(self, x_gpu, norm_cpu, layer: int):
        """MoE phase — dispatches into current batch (caller manages
        begin_batch/end_batch). Returns x_gpu with residual added.

        Dispatches: 4×(gate_up, gate, down, add_scaled) + res2
        """
        if self._profiling: self._set_gpu_op(f"L{layer}/moe")
        moe_out_gpu = self._moe_block(norm_cpu, layer)

        if self._profiling: self._set_gpu_op(f"L{layer}/res2")
        self._add_inplace(x_gpu, moe_out_gpu)

        return x_gpu

    def _decode_gpu(self, x, layer: int, pos: int):
        """Fully GPU-resident decode — single layer (fallback for prefill)."""
        runner = self.cache.runner
        E = self.n_embd

        if not isinstance(x, GPUBuffer):
            x_np = x.ravel().astype(np.float32)
            x_gpu = runner.upload_to_gpu(x_np, f"__decode_x_{layer}")
            x_gpu.shape = (1, E)
        else:
            x_gpu = x

        runner.begin_batch()
        x_gpu, rn2 = self._attn_phase(x_gpu, layer, pos)
        runner.end_batch()

        norm_cpu = runner.readback(rn2).reshape(1, E)
        runner.write_buffer(self._moe_x_gpu.handle,
                            np.ascontiguousarray(norm_cpu.ravel()).tobytes())
        runner.write_buffer(self._moe_acc_gpu.handle, self._moe_zeros)

        runner.begin_batch()
        x_gpu = self._moe_phase(x_gpu, norm_cpu, layer)
        runner.end_batch()

        if self._profiling:
            self._clear_gpu_op()
        x_gpu.shape = (1, E)
        return x_gpu

    # -- Transformer block --

    def _transformer_block(self, x, layer: int, pos: int):
        """One transformer layer — fully GPU-resident decode."""
        return self._decode_gpu(x, layer, pos)

    # -- Forward pass --

    def forward(self, token_ids: np.ndarray,
                use_cache: bool = False,
                pos_offset: int = 0) -> np.ndarray:
        """Run GPT-OSS forward pass.

        Args:
            token_ids: (T,) int32 token IDs
            use_cache: whether to use KV cache (for decode)
            pos_offset: position offset for embeddings

        Returns:
            logits: (T, n_vocab) float32
        """
        T = len(token_ids)

        # Re-init KV cache if reset by generate()
        if self.kv_cache is None:
            self._init_kv_cache()

        # Token embeddings (CPU lookup)
        wte = self.weights["embed_tokens.weight"]
        x = wte[token_ids].reshape(T, self.n_embd)

        if T == 1 and use_cache:
            # Decode mode: two-phase batched dispatch per layer
            pos = pos_offset
            for layer in range(self.n_layer):
                x = self._transformer_block(x, layer, pos)
        else:
            # Prefill: process all tokens (simplified, no batching)
            # Reset GPU KV cache lengths
            for layer in range(self.n_layer):
                k, v, _ = self._gpu_kv_cache[layer]
                self._gpu_kv_cache[layer] = (k, v, 0)

            for t in range(T):
                x_t = x[t:t+1]
                pos = pos_offset + t
                for layer in range(self.n_layer):
                    x_t = self._transformer_block(x_t, layer, pos)
                if t < T - 1:
                    if isinstance(x_t, GPUBuffer):
                        x_t = self.cache.runner.readback(x_t).reshape(
                            1, self.n_embd)
                    x[t:t+1] = x_t  # save for later (not needed for decode)
            x = x_t  # only keep last token's output

        # Final RMSNorm
        if self._profiling: self._set_gpu_op("final_norm")
        x = self._rms_norm(x, self._gpu_weights["norm.weight"])

        # LM head (GPU fp16)
        if self._profiling: self._set_gpu_op("lm_head")
        logits = self._linear_fp16w(
            x, self._gpu_weights["lm_head.weight.fp16"],
            self._gpu_weights["zero_bias_V"],
            self.n_vocab, K=self.n_embd)

        if self._profiling: self._clear_gpu_op()
        return logits


# ---------------------------------------------------------------------------
# Weight downloading and loading
# ---------------------------------------------------------------------------

def download_gptoss_weights(model_dir: str = None):
    """Download GPT-OSS-20B weights from HuggingFace."""
    if model_dir is None:
        model_dir = os.path.join(_SCRIPT_DIR, "..", "..", "gitignore", "models", os.path.basename(_SCRIPT_DIR), "weights")
    import requests

    os.makedirs(model_dir, exist_ok=True)

    hf_repo = GPTOSS_CONFIG["hf_repo"]
    base_url = f"https://huggingface.co/{hf_repo}/resolve/main"

    # Auth headers
    hf_token = os.environ.get("HF_TOKEN") or os.environ.get(
        "HUGGING_FACE_HUB_TOKEN")
    if not hf_token:
        token_file = os.path.join(
            os.path.expanduser("~"), ".cache", "huggingface", "token")
        if os.path.exists(token_file):
            with open(token_file) as f:
                hf_token = f.read().strip()
    headers = {"Authorization": f"Bearer {hf_token}"} if hf_token else {}

    # Download tokenizer
    tokenizer_path = os.path.join(model_dir, "tokenizer.json")
    if not os.path.exists(tokenizer_path):
        print(f"Downloading tokenizer from {hf_repo}...")
        resp = requests.get(f"{base_url}/tokenizer.json", headers=headers)
        resp.raise_for_status()
        with open(tokenizer_path, 'wb') as f:
            f.write(resp.content)
        print(f"  Saved to {tokenizer_path}")
    else:
        print(f"Tokenizer cached at {tokenizer_path}")

    # Download safetensors files
    st_files = GPTOSS_CONFIG["safetensors_files"]
    for st_file in st_files:
        st_path = os.path.join(model_dir, st_file)
        if os.path.exists(st_path):
            # Validate file size
            try:
                head_resp = requests.head(
                    f"{base_url}/{st_file}", headers=headers,
                    allow_redirects=True, timeout=10)
                expected_size = int(head_resp.headers.get('content-length', 0))
                actual_size = os.path.getsize(st_path)
                if expected_size and actual_size < expected_size:
                    print(f"  {st_file} incomplete, re-downloading...")
                    os.remove(st_path)
                else:
                    print(f"  {st_file} already cached "
                          f"({actual_size // (1024**2)}MB)")
                    continue
            except Exception:
                print(f"  {st_file} already cached (size check skipped)")
                continue

        print(f"  Downloading {st_file}...")
        resp = requests.get(f"{base_url}/{st_file}",
                            headers=headers, stream=True)
        resp.raise_for_status()
        total = int(resp.headers.get('content-length', 0))
        downloaded = 0
        with open(st_path, 'wb') as f:
            for chunk in resp.iter_content(chunk_size=8192 * 1024):
                f.write(chunk)
                downloaded += len(chunk)
                if total:
                    pct = downloaded * 100 // total
                    mb = downloaded // (1024 * 1024)
                    total_mb = total // (1024 * 1024)
                    print(f"\r  Progress: {pct}% ({mb}MB / {total_mb}MB)",
                          end="", flush=True)
        print()

    return model_dir, tokenizer_path


def load_gptoss_weights(model_dir: str) -> Dict[str, np.ndarray]:
    """Load GPT-OSS weights from safetensors files.

    Returns dict with cleaned key names (strips 'model.' prefix).
    """
    all_weights = {}
    st_files = GPTOSS_CONFIG["safetensors_files"]

    for st_file in st_files:
        st_path = os.path.join(model_dir, st_file)
        print(f"  Loading {st_file}...")
        tensors = _parse_safetensors(st_path)

        for key, arr in tensors.items():
            # Strip 'model.' prefix for cleaner naming
            clean_key = key.replace("model.", "") if key.startswith(
                "model.") else key
            # Convert fp16 to fp32 for non-U8 tensors
            if arr.dtype == np.float16:
                arr = arr.astype(np.float32)
            all_weights[clean_key] = arr

        del tensors  # free memory

    print(f"  Loaded {len(all_weights)} tensors")
    return all_weights


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def verify_with_random_weights():
    """Verify GPT-OSS pipeline with small random weights (no download)."""
    print("=" * 60)
    print("GPT-OSS WebGPU Pipeline Verification (random weights)")
    print("=" * 60)

    # Scaled-down config: 2 layers, 4 experts, top-2 routing
    # n_embd=384 triggers loop-mode kernels (>256) for fp16w support
    n_layer = 2
    n_head, n_kv_heads = 6, 2
    n_embd, intermediate_size, n_vocab = 384, 384, 256
    head_dim = n_embd // n_head  # 64
    num_experts = 4
    top_k = 2

    kv_dim = n_kv_heads * head_dim
    q_dim = n_head * head_dim
    qkv_out = q_dim + 2 * kv_dim

    np.random.seed(42)
    weights = {}
    weights["embed_tokens.weight"] = np.random.randn(
        n_vocab, n_embd).astype(np.float32) * 0.02
    weights["lm_head.weight"] = np.random.randn(
        n_vocab, n_embd).astype(np.float32) * 0.02
    weights["norm.weight"] = np.ones(n_embd, dtype=np.float32)

    # Use alternating sliding/full as in the real model
    layer_types = ["sliding_attention", "full_attention"]

    for i in range(n_layer):
        pfx = f"layers.{i}."
        weights[pfx + "input_layernorm.weight"] = np.ones(
            n_embd, dtype=np.float32)
        weights[pfx + "post_attention_layernorm.weight"] = np.ones(
            n_embd, dtype=np.float32)

        # QKV (will be fused by constructor)
        weights[pfx + "self_attn.q_proj.weight"] = np.random.randn(
            q_dim, n_embd).astype(np.float32) * 0.02
        weights[pfx + "self_attn.q_proj.bias"] = np.zeros(
            q_dim, dtype=np.float32)
        weights[pfx + "self_attn.k_proj.weight"] = np.random.randn(
            kv_dim, n_embd).astype(np.float32) * 0.02
        weights[pfx + "self_attn.k_proj.bias"] = np.zeros(
            kv_dim, dtype=np.float32)
        weights[pfx + "self_attn.v_proj.weight"] = np.random.randn(
            kv_dim, n_embd).astype(np.float32) * 0.02
        weights[pfx + "self_attn.v_proj.bias"] = np.zeros(
            kv_dim, dtype=np.float32)
        weights[pfx + "self_attn.o_proj.weight"] = np.random.randn(
            n_embd, q_dim).astype(np.float32) * 0.02
        weights[pfx + "self_attn.o_proj.bias"] = np.zeros(
            n_embd, dtype=np.float32)

        # Sinks (attention sink positions): zeros for test
        weights[pfx + "self_attn.sinks"] = np.zeros(
            n_head * head_dim, dtype=np.float32)

        # Router
        weights[pfx + "mlp.router.weight"] = np.random.randn(
            num_experts, n_embd).astype(np.float32) * 0.1
        weights[pfx + "mlp.router.bias"] = np.zeros(
            num_experts, dtype=np.float32)

        # Expert weights as MXFP4 format
        N_gate_up = 2 * intermediate_size
        block_size = 32
        for proj_name, N_out in [("gate_up_proj", N_gate_up),
                                  ("down_proj", n_embd)]:
            K_in = n_embd if proj_name == "gate_up_proj" else intermediate_size
            n_blocks = (K_in + block_size - 1) // block_size
            # Random MXFP4 blocks: (num_experts, N_out, n_blocks, 16) u8
            blocks = np.random.randint(
                0, 256, (num_experts, N_out, n_blocks, 16), dtype=np.uint8)
            # Random scales: (num_experts, N_out, n_blocks) u8
            scales = np.full(
                (num_experts, N_out, n_blocks), 127, dtype=np.uint8)
            # Zero bias: (num_experts, N_out) fp32
            bias = np.zeros((num_experts, N_out), dtype=np.float32)
            epfx = pfx + "mlp.experts."
            weights[epfx + proj_name + "_blocks"] = blocks
            weights[epfx + proj_name + "_scales"] = scales
            weights[epfx + proj_name + "_bias"] = bias

    print(f"\nModel: {n_layer} layers, {n_head} Q-heads, {n_kv_heads} KV-heads, "
          f"{n_embd} embd, {num_experts} experts (top-{top_k}), {n_vocab} vocab")

    # Temporarily override LAYER_TYPES for small model
    orig_layer_types = globals()['LAYER_TYPES']
    globals()['LAYER_TYPES'] = layer_types[:n_layer]

    try:
        model = GptOssWebGPU(
            weights, n_layer=n_layer, n_head=n_head, n_kv_heads=n_kv_heads,
            n_embd=n_embd, intermediate_size=intermediate_size,
            n_vocab=n_vocab, head_dim=head_dim,
            num_experts=num_experts, top_k_experts=top_k)

        # Forward pass
        token_ids = np.array([1, 42, 100, 200], dtype=np.int32)
        T = len(token_ids)
        t0 = time.time()
        logits = model.forward(token_ids)
        t1 = time.time()

        print(f"\nForward pass: {token_ids} → shape {logits.shape} "
              f"in {(t1-t0)*1000:.0f}ms")
        print(f"Logits range: [{logits.min():.4f}, {logits.max():.4f}]")
        print(f"Predictions: {logits.argmax(axis=1)}")

        ok = True
        if logits.shape != (1, n_vocab):
            # Forward returns last-token logits for prefill
            if logits.shape[-1] != n_vocab:
                print(f"FAIL: unexpected vocab dim {logits.shape[-1]}")
                ok = False
        if np.any(np.isnan(logits)):
            print("FAIL: NaN in logits")
            ok = False
        if np.any(np.isinf(logits)):
            print("FAIL: Inf in logits")
            ok = False

        if ok:
            print("\nAll checks PASSED!")
        else:
            print("\nSome checks FAILED!")
        return ok
    finally:
        globals()['LAYER_TYPES'] = orig_layer_types


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="GPT-OSS-20B on WebGPU via Triton")
    parser.add_argument("--prompt", type=str,
                        default="The future of AI is",
                        help="Prompt for text generation")
    parser.add_argument("--max-tokens", type=int, default=50,
                        help="Maximum tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="Sampling temperature")
    parser.add_argument("--weights-dir", type=str,
                        default=None,
                        help="Directory for cached weights")
    parser.add_argument("--verify", action="store_true",
                        help="Verify with random weights (no download)")
    parser.add_argument("--verify-real", action="store_true",
                        help="Quick verification with real weights")
    parser.add_argument("--profile", action="store_true",
                        help="Profile a short generation and save HTML report")
    args = parser.parse_args()

    if args.verify:
        success = verify_with_random_weights()
        sys.exit(0 if success else 1)

    # Download weights if needed
    print("=" * 60)
    print("GPT-OSS-20B on WebGPU")
    print("=" * 60)

    model_dir, tokenizer_path = download_gptoss_weights(args.weights_dir)

    # Load weights
    print("\nLoading weights...")
    t0 = time.time()
    weights = load_gptoss_weights(model_dir)
    print(f"  Weight loading took {time.time() - t0:.1f}s")

    # Create model
    print("\nCreating model...")
    model = GptOssWebGPU(weights)

    # Print GPU info
    runner = model.cache.runner
    print(f"\nGPU: {runner.adapter_info.get('device', 'unknown')}")
    print(f"Backend: {runner.adapter_info.get('backend', 'unknown')}")
    print(f"GPU memory used: {runner._total_gpu_bytes / (1024**3):.2f} GB")

    if args.verify_real:
        # Quick verification: encode prompt and run one forward pass
        tokenizer = load_tokenizer(tokenizer_path)
        token_ids = np.array(tokenizer.encode(args.prompt), dtype=np.int32)
        print(f"\nVerification: prompt='{args.prompt}' ({len(token_ids)} tokens)")
        t0 = time.time()
        logits = model.forward(token_ids)
        elapsed = time.time() - t0
        next_token = logits[-1].argmax()
        print(f"  Forward pass: {elapsed*1000:.0f}ms")
        print(f"  Next token: {next_token} = '{tokenizer.decode([int(next_token)])}'")
        print(f"  Logits shape: {logits.shape}")
        print(f"  Logits range: [{logits.min():.3f}, {logits.max():.3f}]")
        return

    if args.profile:
        model.enable_profiling()
        print(f"Profiling enabled (GPU timestamps: "
              f"{model.profiler.gpu_enabled})")

        # Profile a short generation
        from common.profiler import InferenceProfiler
        tokenizer = load_tokenizer(tokenizer_path)
        generate(model, args.prompt, tokenizer=tokenizer,
                 max_tokens=min(args.max_tokens, 10),
                 temperature=args.temperature)

        model.save_profile(_SCRIPT_DIR, "GPT-OSS-20B")
        return

    # Generate text
    tokenizer = load_tokenizer(tokenizer_path)
    generate(model, args.prompt, tokenizer=tokenizer,
             max_tokens=args.max_tokens,
             temperature=args.temperature)


if __name__ == "__main__":
    main()
