"""
Shared base class for WebGPU model inference.

Provides:
  - KernelCache: compile-once WGSL kernel cache
  - WebGPUModel: base class with all shared primitive ops (linear, norm,
    activation, attention, add) and kernel compilation/weight upload helpers

Subclasses implement architecture-specific blocks (_attention_block,
_mlp_block, _transformer_block, forward) and weight upload logic.
"""
import os
import sys
import time
import json
import struct
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional

import numpy as np

import triton
import triton.language as tl
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource
from triton.backends.webgpu.llvm_to_wgsl import translate_llvm_to_wgsl
from triton.backends.webgpu.dawn_runner import DawnRunner, GPUBuffer

from common.kernels import (
    layer_norm_kernel, layer_norm_loop_kernel,
    rms_norm_kernel, rms_norm_loop_kernel,
    add_rms_norm_loop_kernel,
    linear_kernel, linear_loop_kernel, linear_loop_fp16w_kernel,
    linear_loop_fp16w_wide_kernel,
    linear_Wt_fp16_kernel,
    linear_q4_kernel, linear_q4_add_kernel, linear_q4_wide_kernel,
    add_kernel, add_inplace_kernel,
    mod_scale_shift_kernel, gate_residual_add_kernel,
    concat_2d_kernel, split_copy_kernel,
    gelu_kernel, silu_mul_kernel, silu_mul_fused_kernel,
    silu_mul_fused_rows_kernel,
    gelu_mul_kernel,
    silu_kernel, sigmoid_kernel, mul_kernel,
    causal_attn_kernel, causal_attn_multihead_kernel,
    full_attn_kernel, full_attn_multihead_kernel, gqa_decode_attn_kernel,
    partial_rope_decode_kernel, rope_kv_scatter_kernel,
    fused_rope_qkv_kernel,
    partial_rope_prefill_kernel, rope_kv_scatter_prefill_kernel,
    group_norm_kernel,
    linear_mxfp4_kernel, gptoss_gate_kernel,
    gqa_decode_attn_sink_kernel, add_scaled_kernel,
    qk_norm_rope_kernel,
    embed_gather_kernel, argmax_kernel,
)

WEBGPU_TARGET = GPUTarget("webgpu", 0, 32)


# ---------------------------------------------------------------------------
# Size-class buffer pool — patched onto DawnRunner for GPU buffer reuse
# ---------------------------------------------------------------------------

def _round_to_size_class(size):
    """Round buffer size up to the next size class.

    - ≤256B: 256 (minimum WebGPU buffer)
    - 256B–4KB: next multiple of 256
    - >4KB: next power of 2
    """
    if size <= 256:
        return 256
    if size <= 4096:
        return ((size + 255) // 256) * 256
    p = 1
    while p < size:
        p <<= 1
    return p


def _install_buffer_pool(runner):
    """Install size-class memory pool on a DawnRunner instance.

    Wraps _get_or_create_buffer so transient buffers (names starting with
    '__') are allocated at rounded size classes.  When a buffer is freed
    it returns to a per-class free list for zero-cost reuse.
    """
    if getattr(runner, '_has_buffer_pool', False):
        return
    runner._has_buffer_pool = True
    runner._pool_free = {}
    runner._pool_alloc = 0
    runner._pool_reuse = 0
    runner._pool_bytes = 0

    original_get_or_create = runner._get_or_create_buffer

    def _pooled_get_or_create(name, size, usage):
        key = (name, size, usage)
        if key in runner._buffer_cache:
            return runner._buffer_cache[key]

        # Transient buffers use the pool
        if name.startswith("__"):
            rounded = _round_to_size_class(size)
            free_list = runner._pool_free.get(rounded)
            if free_list:
                buf = free_list.pop()
                runner._pool_reuse += 1
                runner._buffer_cache[key] = buf
                return buf
            # Create at rounded size via original method (bypasses pool)
            pool_name = f"__pool_{rounded}_{runner._pool_alloc}"
            buf = original_get_or_create(pool_name, rounded, usage)
            runner._pool_alloc += 1
            runner._pool_bytes += rounded
            runner._buffer_cache[key] = buf
            return buf

        return original_get_or_create(name, size, usage)

    runner._get_or_create_buffer = _pooled_get_or_create

    # Expose pool stats in gpu_memory_stats
    original_stats = runner.gpu_memory_stats

    def _pool_stats():
        s = original_stats()
        pool_free_count = sum(len(v) for v in runner._pool_free.values())
        s['pool_alloc'] = runner._pool_alloc
        s['pool_reuse'] = runner._pool_reuse
        s['pool_free'] = pool_free_count
        s['pool_bytes_mb'] = runner._pool_bytes / 1024 / 1024
        return s

    runner.gpu_memory_stats = _pool_stats


# ---------------------------------------------------------------------------
# Kernel cache — compile once, reuse
# ---------------------------------------------------------------------------

class KernelCache:
    """Caches compiled WGSL kernels keyed by (kernel_fn, constexprs)."""

    def __init__(self):
        self._cache: Dict[str, object] = {}
        self.runner = DawnRunner()
        self.profiler = None  # Set by WebGPUModel.enable_profiling()
        self._gpu_op_name = None  # Current op name for GPU timestamps
        _install_buffer_pool(self.runner)

    def _key(self, fn, signature: dict, constexprs: dict, num_warps: int) -> str:
        # Include pointer types in key so fp16/fp32 variants are distinct
        ptr_types = '_'.join(v for v in signature.values()
                             if isinstance(v, str) and v.startswith('*'))
        return f"{fn.fn.__name__}_{ptr_types}_{'_'.join(f'{k}={v}' for k,v in sorted(constexprs.items()))}_{num_warps}w"

    def get_or_compile(self, fn, signature: dict, constexprs: dict,
                       num_warps: int = 4):
        key = self._key(fn, signature, constexprs, num_warps)
        if key not in self._cache:
            sig_no_ce = {k: v for k, v in signature.items()
                         if v != 'constexpr'}
            src = ASTSource(fn=fn, signature=signature,
                            constexprs=constexprs)
            compiled = triton.compile(src, target=WEBGPU_TARGET,
                                      options={'num_warps': num_warps})
            result = translate_llvm_to_wgsl(
                compiled.asm['llir'], sig_no_ce,
                num_warps=num_warps, warp_size=32,
                use_native_subgroups=self.runner.has_subgroups)
            self._cache[key] = result
        return self._cache[key]

    def run(self, result, grid, buffers, scalars=None,
             gpu_outputs=None, timestamp_writes_ptr=None):
        # Record CPU-timed GPU dispatch + allocate GPU timestamps when profiler is active
        if self.profiler and self.profiler.enabled:
            import time
            name = self._gpu_op_name or "gpu_dispatch"
            # Allocate GPU timestamp pair for this dispatch
            if timestamp_writes_ptr is None and self.profiler.gpu_enabled:
                b_idx, e_idx = self.profiler.allocate_gpu_timestamps(name)
                if b_idx >= 0:
                    timestamp_writes_ptr = self.profiler.get_timestamp_writes_ptr(b_idx, e_idx)
            begin_ns = time.perf_counter_ns()
            out = self.runner.run_kernel(
                wgsl_code=result.wgsl,
                buffer_bindings=result.buffer_bindings,
                param_fields=result.param_fields,
                workgroup_size=result.workgroup_size,
                grid=grid,
                buffers=buffers,
                scalars=scalars or {},
                gpu_outputs=gpu_outputs,
                timestamp_writes_ptr=timestamp_writes_ptr)
            end_ns = time.perf_counter_ns()
            self.profiler.record_dispatch(name, begin_ns, end_ns)
            return out

        return self.runner.run_kernel(
            wgsl_code=result.wgsl,
            buffer_bindings=result.buffer_bindings,
            param_fields=result.param_fields,
            workgroup_size=result.workgroup_size,
            grid=grid,
            buffers=buffers,
            scalars=scalars or {},
            gpu_outputs=gpu_outputs,
            timestamp_writes_ptr=timestamp_writes_ptr)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _next_pow2(n):
    """Return the smallest power of 2 >= n."""
    p = 1
    while p < n:
        p *= 2
    return p


# ---------------------------------------------------------------------------
# WebGPU model base class
# ---------------------------------------------------------------------------

class WebGPUModel:
    """Base class for transformer model inference on WebGPU.

    Provides shared:
      - Kernel compilation for linear, add, attention, normalization, activations
      - Primitive ops: _linear, _add, _causal_attention, _layer_norm,
        _rms_norm, _gelu, _silu_mul, _apply_rope
      - Weight upload helpers: _upload_linear_weight, _upload_bias

    Subclasses must implement:
      - _compile_model_kernels(): compile architecture-specific kernels
      - _upload_weights_to_gpu(): upload model weights
      - _attention_block(): attention with architecture-specific projections
      - _mlp_block(): MLP with architecture-specific activation
      - _transformer_block(): pre-norm transformer block
      - forward(): full forward pass
    """

    # Max threads per workgroup (WebGPU spec minimum = 256)
    MAX_WG_THREADS = 256
    # Fixed block size for loop-based kernels
    LOOP_BLOCK = 128
    # Max workgroups per dispatch dimension (D3D12/WebGPU limit)
    MAX_DISPATCH_DIM = 65535

    def __init__(self, weights: Dict[str, np.ndarray],
                 n_layer: int, n_head: int, n_embd: int,
                 n_vocab: int,
                 # Optional architecture params
                 n_kv_heads: int = None,
                 intermediate_size: int = None,
                 head_dim: int = None,
                 rope_theta: float = 10000.0,
                 rms_norm_eps: float = 1e-5,
                 norm_eps: float = 1e-5,
                 # K dimensions to pre-compile single-pass linear kernels for
                 # (e.g. {768, 3072} for GPT-2 or {576, 1536} for SmolLM2)
                 k_dimensions: Set[int] = None,
                 # Use fp16 for activation buffers (halves GPU bandwidth)
                 fp16_act: bool = False):
        self.fp16_act = fp16_act
        self.n_layer = n_layer
        self.n_head = n_head
        self.n_embd = n_embd
        self.n_vocab = n_vocab
        self.n_kv_heads = n_kv_heads if n_kv_heads is not None else n_head
        self.intermediate_size = intermediate_size if intermediate_size is not None else 4 * n_embd
        self.head_dim = head_dim if head_dim is not None else n_embd // n_head
        self.kv_dim = self.n_kv_heads * self.head_dim
        self.n_rep = n_head // self.n_kv_heads
        self.rope_theta = rope_theta
        self.rms_norm_eps = rms_norm_eps
        self.norm_eps = norm_eps
        self.weights = weights
        self.kv_cache = None
        self.cache = KernelCache()
        self._gpu_weights: Dict[str, GPUBuffer] = {}
        self.profiler = None  # Set via enable_profiling()

        # Collect all K dimensions this model needs for single-pass linear
        if k_dimensions is not None:
            self._k_dimensions = k_dimensions
        else:
            self._k_dimensions = {n_embd}

        self._compile_kernels()

    @staticmethod
    def _nw(block_size):
        """Compute num_warps for a given block size."""
        return max(1, block_size // 32)

    def _needs_loop(self, block_size):
        """True if block_size would exceed workgroup thread limit."""
        return self._nw(block_size) * 32 > self.MAX_WG_THREADS

    def enable_profiling(self):
        """Enable CPU+GPU profiling for this model.

        Creates an InferenceProfiler that uses WebGPU timestamp queries
        for GPU timing and perf_counter_ns for CPU timing.
        Wires the profiler into KernelCache so GPU timestamps are
        automatically injected into every dispatch.
        """
        from common.profiler import InferenceProfiler
        self.profiler = InferenceProfiler(self.cache.runner)
        self.profiler.enable()
        self.cache.profiler = self.profiler

    def disable_profiling(self):
        """Disable profiling and release GPU resources."""
        if self.profiler:
            self.profiler.destroy()
            self.cache.profiler = None
            self.profiler = None

    # ------------------------------------------------------------------
    # Profiling helpers — zero overhead when profiling is disabled
    # ------------------------------------------------------------------

    @property
    def _profiling(self) -> bool:
        """True when profiling is active. Use in model code as:
            if self._profiling: ...
        """
        p = self.profiler
        return p is not None and p.enabled

    def _set_gpu_op(self, name: str):
        """Set the GPU operation name for the next dispatch(es).

        Must be called before KernelCache.run() so the profiler records
        the dispatch under this name.  Call _clear_gpu_op() after the
        last dispatch in a block.

        Usage:
            self._set_gpu_op(f"L{layer}/qkv")
            result = self._proj(x, ...)
            self._clear_gpu_op()
        """
        self.cache._gpu_op_name = name

    def _clear_gpu_op(self):
        """Clear the GPU operation name (reset to default 'gpu_dispatch')."""
        self.cache._gpu_op_name = None

    def _begin_cpu(self, name: str):
        """Begin a CPU profiling scope.

        Usage:
            self._begin_cpu(f"L{layer}/rms_norm")
            result = self._rms_norm_cpu(x, ...)
            self._end_cpu(f"L{layer}/rms_norm")
        """
        self.profiler._cpu.begin(name)

    def _end_cpu(self, name: str):
        """End a CPU profiling scope."""
        self.profiler._cpu.end(name)

    def save_profile(self, script_dir: str, model_name: str):
        """Generate profiling report (console + HTML) and disable profiling.

        Call after inference is complete:
            if args.profile:
                model.save_profile(_SCRIPT_DIR, "Phi-4 mini")

        Generates:
          1. Console profiling report
          2. HTML timeline at {script_dir}/profile.html
        """
        if not self.profiler:
            return
        if hasattr(self.profiler, 'finish'):
            self.profiler.finish()
        self.profiler.report()

        from common.profiler_html import generate_html_report
        profile_path = os.path.join(script_dir, "profile.html")
        runner = self.cache.runner
        adapter = runner.adapter_info
        generate_html_report(
            self.profiler,
            output_path=profile_path,
            title=f"{model_name} — {adapter.get('description', 'GPU')}",
            adapter_info=adapter,
            memory_info=self.get_memory_info())
        print(f"HTML profile: {profile_path}")
        self.disable_profiling()

    def get_memory_info(self) -> dict:
        """Collect GPU and CPU memory usage statistics for profiling."""
        runner = self.cache.runner
        gpu_stats = runner.gpu_memory_stats()

        # Weight memory
        weight_bytes = sum(g.size for g in self._gpu_weights.values())
        if hasattr(self, '_gpu_weight_chunks'):
            for name, chunks in self._gpu_weight_chunks.items():
                weight_bytes += sum(buf.size for _, _, buf in chunks[1:])

        # CPU weight memory
        cpu_weight_bytes = sum(w.nbytes for w in self.weights.values())

        return {
            'gpu_total_mb': round(gpu_stats['total_allocated_mb'], 1),
            'gpu_weight_mb': round(weight_bytes / 1024 / 1024, 1),
            'gpu_buffer_cache_mb': round(gpu_stats['buffer_cache_mb'], 1),
            'gpu_alloc_count': gpu_stats['alloc_count'],
            'gpu_weight_count': len(self._gpu_weights),
            'gpu_pipeline_count': gpu_stats['pipeline_cache_entries'],
            'cpu_weight_mb': round(cpu_weight_bytes / 1024 / 1024, 1),
        }

    def _compile_kernels(self):
        """Compile all required shared kernels.

        Subclasses can override _compile_model_kernels() to add more.
        """
        # Disable fp16_act if GPU lacks f16 shader support
        if self.fp16_act and not self.cache.runner.has_f16:
            print("Warning: fp16_act requested but GPU lacks f16 shader support, falling back to fp32")
            self.fp16_act = False

        E = self.n_embd
        HD = self.head_dim
        LB = self.LOOP_BLOCK
        _nw = self._nw

        # --- Linear projection ---
        # Check if ANY K dimension needs loop mode
        any_loop = any(self._needs_loop(_next_pow2(k))
                       for k in self._k_dimensions)
        if any_loop:
            self._linear_loop = True
            self._linear_sig = {
                'X': '*fp32', 'W': '*fp32', 'Bias': '*fp32', 'Y': '*fp32',
                'K': 'i32', 'stride_x': 'i32', 'stride_w': 'i32',
                'N': 'i32', 'BLOCK_K': 'constexpr',
            }
            self._linear_result = self.cache.get_or_compile(
                linear_loop_kernel, self._linear_sig,
                {'BLOCK_K': LB}, num_warps=_nw(LB))

            # fp16 weight variant — requires DXC (Shader Model 6+) for
            # native array<f16> storage buffers on D3D12.
            # Enabled when the adapter reports ShaderF16 support.
            if self.cache.runner.has_f16:
                self._has_fp16_linear = True
                self._linear_fp16w_sig = {
                    'X': '*fp32', 'W': '*fp16', 'Bias': '*fp32', 'Y': '*fp32',
                    'K': 'i32', 'stride_x': 'i32', 'stride_w': 'i32',
                    'N': 'i32', 'BLOCK_K': 'constexpr',
                }
                self._linear_fp16w_result = self.cache.get_or_compile(
                    linear_loop_fp16w_kernel, self._linear_fp16w_sig,
                    {'BLOCK_K': LB}, num_warps=_nw(LB))

                # Wide kernel for N > MAX_DISPATCH_DIM (e.g. lm_head)
                self._linear_fp16w_wide_sig = {
                    'X': '*fp32', 'W': '*fp16', 'Bias': '*fp32', 'Y': '*fp32',
                    'K': 'i32', 'stride_x': 'i32', 'stride_w': 'i32',
                    'N': 'i32', 'grid_y': 'i32', 'BLOCK_K': 'constexpr',
                }
                self._linear_fp16w_wide_result = self.cache.get_or_compile(
                    linear_loop_fp16w_wide_kernel,
                    self._linear_fp16w_wide_sig,
                    {'BLOCK_K': LB}, num_warps=_nw(LB))

                # Barrier-free transposed-weight kernel
                self._linear_Wt_fp16_sig = {
                    'X': '*fp32', 'W_T': '*fp16', 'Bias': '*fp32',
                    'Y': '*fp32', 'K': 'i32', 'N': 'i32',
                    'stride_x': 'i32', 'BLOCK_N': 'constexpr',
                }
                self._linear_Wt_fp16_result = self.cache.get_or_compile(
                    linear_Wt_fp16_kernel, self._linear_Wt_fp16_sig,
                    {'BLOCK_N': LB}, num_warps=_nw(LB))

                # INT4 fused dequantize+matmul kernel
                self._linear_q4_sig = {
                    'X': '*fp32', 'W_Q4': '*i32',
                    'Scales': '*fp16', 'Zeros': '*fp16',
                    'Bias': '*fp32', 'Y': '*fp32',
                    'K': 'i32', 'stride_x': 'i32',
                    'stride_w_q4': 'i32', 'n_groups': 'i32',
                    'N': 'i32', 'BLOCK_K': 'constexpr',
                }
                self._linear_q4_result = self.cache.get_or_compile(
                    linear_q4_kernel, self._linear_q4_sig,
                    {'BLOCK_K': LB}, num_warps=_nw(LB))

                # INT4 matmul + residual add (Y += dot + bias)
                self._linear_q4_add_result = self.cache.get_or_compile(
                    linear_q4_add_kernel, self._linear_q4_sig,
                    {'BLOCK_K': LB}, num_warps=_nw(LB))

                # INT4 wide kernel for N > 65535 (lm_head)
                self._linear_q4_wide_sig = {
                    'X': '*fp32', 'W_Q4': '*i32',
                    'Scales': '*fp16', 'Zeros': '*fp16',
                    'Bias': '*fp32', 'Y': '*fp32',
                    'K': 'i32', 'stride_x': 'i32',
                    'stride_w_q4': 'i32', 'n_groups': 'i32',
                    'N': 'i32', 'grid_y': 'i32',
                    'BLOCK_K': 'constexpr',
                }
                self._linear_q4_wide_result = self.cache.get_or_compile(
                    linear_q4_wide_kernel, self._linear_q4_wide_sig,
                    {'BLOCK_K': LB}, num_warps=_nw(LB))
            else:
                self._has_fp16_linear = False
        else:
            self._linear_loop = False
            self._has_fp16_linear = False
            self._linear_sig = {
                'X': '*fp32', 'W': '*fp32', 'Bias': '*fp32', 'Y': '*fp32',
                'stride_x': 'i32', 'stride_w': 'i32', 'N': 'i32',
                'BLOCK_K': 'constexpr',
            }
            # Pre-compile one kernel per unique K dimension
            self._linear_results = {}  # {K_dim: (result, BK)}
            for k_dim in sorted(self._k_dimensions):
                bk = _next_pow2(k_dim)
                result = self.cache.get_or_compile(
                    linear_kernel, self._linear_sig,
                    {'BLOCK_K': bk}, num_warps=_nw(bk))
                self._linear_results[k_dim] = (result, bk)

        # --- Element-wise add ---
        ADD_BLOCK = LB
        self._add_sig = {
            'X': '*fp32', 'Y': '*fp32', 'Out': '*fp32',
            'N': 'i32', 'BLOCK': 'constexpr',
        }
        self._add_result = self.cache.get_or_compile(
            add_kernel, self._add_sig,
            {'BLOCK': ADD_BLOCK}, num_warps=_nw(ADD_BLOCK))
        self._add_block = ADD_BLOCK

        # --- fp16 activation variants (same kernels, fp16 I/O signatures) ---
        if self.fp16_act and self.cache.runner.has_f16:
            self._compile_fp16_act_kernels()

        # --- In-place add (for GPU-resident residual connections) ---
        self._add_ip_sig = {
            'X': '*fp32', 'Y': '*fp32',
            'N': 'i32', 'BLOCK': 'constexpr',
        }
        self._add_ip_result = self.cache.get_or_compile(
            add_inplace_kernel, self._add_ip_sig,
            {'BLOCK': ADD_BLOCK}, num_warps=_nw(ADD_BLOCK))

        # --- Modulation: Out = (1 + Scale) * X + Shift ---
        self._mod_ss_sig = {
            'X': '*fp32', 'Scale': '*fp32', 'Shift': '*fp32',
            'Out': '*fp32', 'D': 'i32', 'N': 'i32',
            'BLOCK': 'constexpr',
        }
        self._mod_ss_result = self.cache.get_or_compile(
            mod_scale_shift_kernel, self._mod_ss_sig,
            {'BLOCK': ADD_BLOCK}, num_warps=_nw(ADD_BLOCK))

        # --- Gated residual: Residual += Gate * X ---
        self._gate_res_sig = {
            'Residual': '*fp32', 'Gate': '*fp32', 'X': '*fp32',
            'D': 'i32', 'N': 'i32', 'BLOCK': 'constexpr',
        }
        self._gate_res_result = self.cache.get_or_compile(
            gate_residual_add_kernel, self._gate_res_sig,
            {'BLOCK': ADD_BLOCK}, num_warps=_nw(ADD_BLOCK))

        # --- Concat 2D: Out = [A; B] along axis 0 ---
        self._concat_sig = {
            'A': '*fp32', 'B': '*fp32', 'Out': '*fp32',
            'N_a': 'i32', 'N_total': 'i32', 'BLOCK': 'constexpr',
        }
        self._concat_result = self.cache.get_or_compile(
            concat_2d_kernel, self._concat_sig,
            {'BLOCK': ADD_BLOCK}, num_warps=_nw(ADD_BLOCK))

        # --- Split copy: Dst = Src[src_offset:src_offset+N] ---
        self._split_sig = {
            'Src': '*fp32', 'Dst': '*fp32',
            'src_offset': 'i32', 'N': 'i32', 'BLOCK': 'constexpr',
        }
        self._split_result = self.cache.get_or_compile(
            split_copy_kernel, self._split_sig,
            {'BLOCK': ADD_BLOCK}, num_warps=_nw(ADD_BLOCK))

        # --- Causal attention ---
        BS_hd = _next_pow2(HD)
        self._attn_sig = {
            'Q': '*fp32', 'K': '*fp32', 'V': '*fp32', 'Out': '*fp32',
            'stride_q': 'i32', 'stride_k': 'i32',
            'stride_v': 'i32', 'stride_o': 'i32',
            'seq_len': 'i32', 'scale': 'fp32',
            'neg_inf': 'fp32',
            'BLOCK_HD': 'constexpr',
        }
        self._attn_result = self.cache.get_or_compile(
            causal_attn_kernel, self._attn_sig,
            {'BLOCK_HD': BS_hd}, num_warps=_nw(BS_hd))
        self._attn_bhd = BS_hd

        # --- Multi-head causal attention ---
        self._mh_attn_sig = {
            'Q': '*fp32', 'K': '*fp32', 'V': '*fp32', 'Out': '*fp32',
            'stride_q_t': 'i32', 'stride_q_h': 'i32',
            'stride_k_t': 'i32', 'stride_k_h': 'i32',
            'stride_v_t': 'i32', 'stride_v_h': 'i32',
            'stride_o_t': 'i32', 'stride_o_h': 'i32',
            'n_rep': 'i32', 'scale': 'fp32', 'neg_inf': 'fp32',
            'BLOCK_HD': 'constexpr',
        }
        self._mh_attn_result = self.cache.get_or_compile(
            causal_attn_multihead_kernel, self._mh_attn_sig,
            {'BLOCK_HD': BS_hd}, num_warps=_nw(BS_hd))

        # --- GQA decode attention ---
        self._gqa_attn_sig = {
            'Q': '*fp32', 'K_cache': '*fp32', 'V_cache': '*fp32',
            'Out': '*fp32',
            'kv_stride': 'i32', 'n_rep': 'i32',
            'T_total': 'i32', 'scale': 'fp32', 'neg_inf': 'fp32',
            'BLOCK_HD': 'constexpr',
        }
        self._gqa_attn_result = self.cache.get_or_compile(
            gqa_decode_attn_kernel, self._gqa_attn_sig,
            {'BLOCK_HD': BS_hd}, num_warps=_nw(BS_hd))

        # --- Partial RoPE (decode, T=1) ---
        self._rope_sig = {
            'X': '*fp32', 'Y': '*fp32', 'CosTable': '*fp32',
            'SinTable': '*fp32',
            'src_offset': 'i32', 'pos': 'i32', 'half_rot': 'i32',
            'BLOCK_HD': 'constexpr',
        }
        self._rope_result = self.cache.get_or_compile(
            partial_rope_decode_kernel, self._rope_sig,
            {'BLOCK_HD': BS_hd}, num_warps=_nw(BS_hd))

        # --- RoPE K + V scatter to KV cache ---
        self._rope_kv_sig = {
            'QKV': '*fp32', 'K_cache': '*fp32', 'V_cache': '*fp32',
            'CosTable': '*fp32', 'SinTable': '*fp32',
            'q_size': 'i32', 'kv_size': 'i32', 'pos': 'i32',
            'half_rot': 'i32', 'cache_offset': 'i32',
            'BLOCK_HD': 'constexpr',
        }
        self._rope_kv_result = self.cache.get_or_compile(
            rope_kv_scatter_kernel, self._rope_kv_sig,
            {'BLOCK_HD': BS_hd}, num_warps=_nw(BS_hd))

        # --- Fused RoPE Q + K scatter + V copy (decode, T=1) ---
        self._fused_rope_sig = {
            'QKV': '*fp32', 'Q_out': '*fp32',
            'K_cache': '*fp32', 'V_cache': '*fp32',
            'CosTable': '*fp32', 'SinTable': '*fp32',
            'n_head': 'i32', 'q_size': 'i32', 'kv_size': 'i32',
            'pos': 'i32', 'half_rot': 'i32', 'cache_offset': 'i32',
            'BLOCK_HD': 'constexpr',
        }
        self._fused_rope_result = self.cache.get_or_compile(
            fused_rope_qkv_kernel, self._fused_rope_sig,
            {'BLOCK_HD': BS_hd}, num_warps=_nw(BS_hd))

        # --- Prefill RoPE Q (multi-token) ---
        self._rope_prefill_sig = {
            'X': '*fp32', 'Y': '*fp32',
            'Cos': '*fp32', 'Sin': '*fp32',
            'x_offset': 'i32', 'x_stride_t': 'i32',
            'y_stride_t': 'i32', 'half_rot': 'i32',
            'BLOCK_HD': 'constexpr',
        }
        self._rope_prefill_result = self.cache.get_or_compile(
            partial_rope_prefill_kernel, self._rope_prefill_sig,
            {'BLOCK_HD': BS_hd}, num_warps=_nw(BS_hd))

        # --- GQA decode attention with sinks + sliding window ---
        self._gqa_attn_sink_sig = {
            'Q': '*fp32', 'K_cache': '*fp32', 'V_cache': '*fp32',
            'Sinks': '*fp32', 'Out': '*fp32',
            'kv_stride': 'i32', 'kv_start': 'i32', 'n_rep': 'i32',
            'T_win': 'i32', 'scale': 'fp32', 'neg_inf': 'fp32',
            'BLOCK_HD': 'constexpr',
        }
        self._gqa_attn_sink_result = self.cache.get_or_compile(
            gqa_decode_attn_sink_kernel, self._gqa_attn_sink_sig,
            {'BLOCK_HD': BS_hd}, num_warps=_nw(BS_hd))

        # --- Weighted add: Acc += alpha * X (AXPY) ---
        AXPY_BLOCK = self.LOOP_BLOCK
        self._add_scaled_sig = {
            'Acc': '*fp32', 'X': '*fp32',
            'alpha': 'fp32', 'N': 'i32', 'BLOCK': 'constexpr',
        }
        self._add_scaled_result = self.cache.get_or_compile(
            add_scaled_kernel, self._add_scaled_sig,
            {'BLOCK': AXPY_BLOCK}, num_warps=_nw(AXPY_BLOCK))
        self._add_scaled_block = AXPY_BLOCK

        # --- Prefill RoPE K + V scatter (multi-token) ---
        self._rope_kv_prefill_sig = {
            'QKV': '*fp32', 'K_cache': '*fp32', 'V_cache': '*fp32',
            'Cos': '*fp32', 'Sin': '*fp32',
            'q_size': 'i32', 'kv_size': 'i32', 'qkv_stride_t': 'i32',
            'cache_stride_t': 'i32', 'half_rot': 'i32',
            'BLOCK_HD': 'constexpr',
        }
        self._rope_kv_prefill_result = self.cache.get_or_compile(
            rope_kv_scatter_prefill_kernel, self._rope_kv_prefill_sig,
            {'BLOCK_HD': BS_hd}, num_warps=_nw(BS_hd))

        # Let subclass compile additional kernels
        self._compile_model_kernels()

        # Warm up GPU pipelines: call get_pipeline_info() on every compiled
        # kernel to force synchronous shader compilation now (before prefill)
        # rather than lazily on first dispatch.
        self._warmup_gpu_pipelines()

    def _warmup_gpu_pipelines(self):
        """Pre-create GPU pipelines for all compiled Triton kernels.

        Uses parallel async compilation via prefetch_pipelines_async()
        to overlap shader compilations across threads.  This reduces
        warm-up time from sum(compile_times) to max(compile_times).

        Without this, GPU shader compilation happens lazily during the
        first forward() pass (prefill), adding ~100ms per unique kernel.
        """
        import time
        t0 = time.perf_counter()
        runner = self.cache.runner

        # Collect all compiled kernel specs
        specs = []
        for attr_name in dir(self):
            if not attr_name.endswith('_result'):
                continue
            result = getattr(self, attr_name, None)
            if result is None:
                continue
            if hasattr(result, 'wgsl') and hasattr(result, 'buffer_bindings'):
                specs.append((result.wgsl, result.buffer_bindings,
                              result.param_fields))

        # Use async parallel compilation if available, else serial fallback
        if hasattr(runner, 'prefetch_pipelines_async') and len(specs) > 1:
            count = runner.prefetch_pipelines_async(specs)
            # Some specs may already be cached; count only new compilations
            if count == 0:
                count = len(specs)  # all were cached
        else:
            count = 0
            for wgsl, bbs, pfs in specs:
                runner.get_pipeline_info(wgsl, bbs, pfs)
                count += 1

        t1 = time.perf_counter()
        print(f"  Warmed {len(specs)} GPU pipelines in {(t1-t0)*1000:.0f}ms")

    def _compile_fp16_act_kernels(self):
        """Compile fp16 I/O variants of hot-path kernels.

        Same kernel functions, but with *fp16 buffer signatures for
        activations.  Internal computation stays fp32 (all kernels use
        .to(tl.float32) on loads).  Bias and norm weights stay fp32.
        """
        LB = self.LOOP_BLOCK
        _nw = self._nw

        # Linear: fp16 in, fp16 weight, fp32 bias, fp16 out
        self._linear_fp16io_sig = {
            'X': '*fp16', 'W': '*fp16', 'Bias': '*fp32', 'Y': '*fp16',
            'K': 'i32', 'stride_x': 'i32', 'stride_w': 'i32',
            'N': 'i32', 'BLOCK_K': 'constexpr',
        }
        self._linear_fp16io_result = self.cache.get_or_compile(
            linear_loop_fp16w_kernel, self._linear_fp16io_sig,
            {'BLOCK_K': LB}, num_warps=_nw(LB))

        # Add: fp16 in, fp16 out
        self._add_fp16_sig = {
            'X': '*fp16', 'Y': '*fp16', 'Out': '*fp16',
            'N': 'i32', 'BLOCK': 'constexpr',
        }
        self._add_fp16_result = self.cache.get_or_compile(
            add_kernel, self._add_fp16_sig,
            {'BLOCK': LB}, num_warps=_nw(LB))

        # Add in-place: fp16
        self._add_ip_fp16_sig = {
            'X': '*fp16', 'Y': '*fp16',
            'N': 'i32', 'BLOCK': 'constexpr',
        }
        self._add_ip_fp16_result = self.cache.get_or_compile(
            add_inplace_kernel, self._add_ip_fp16_sig,
            {'BLOCK': LB}, num_warps=_nw(LB))

        # Silu·mul: fp16 in, fp16 out
        self._sm_fp16_sig = {
            'Gate': '*fp16', 'Up': '*fp16', 'Out': '*fp16',
            'N': 'i32', 'BLOCK': 'constexpr',
        }
        self._sm_fp16_result = self.cache.get_or_compile(
            silu_mul_kernel, self._sm_fp16_sig,
            {'BLOCK': LB}, num_warps=_nw(LB))

        print(f"  Compiled fp16 activation kernels (linear, add, silu_mul)")

    def _compile_model_kernels(self):
        """Override in subclass to compile architecture-specific kernels."""
        pass

    # --- Norm kernel compilation helpers (used by subclasses) ---

    def _compile_layer_norm(self):
        """Compile LayerNorm kernels and store results on self."""
        E = self.n_embd
        BS_ln = _next_pow2(E)
        LB = self.LOOP_BLOCK
        _nw = self._nw
        if self._needs_loop(BS_ln):
            self._ln_loop = True
            self._ln_sig = {
                'X': '*fp32', 'Y': '*fp32', 'W': '*fp32', 'B': '*fp32',
                'Mean': '*fp32', 'Rstd': '*fp32', 'stride': 'i32',
                'N': 'i32', 'eps': 'fp32', 'BLOCK': 'constexpr',
            }
            self._ln_result = self.cache.get_or_compile(
                layer_norm_loop_kernel, self._ln_sig,
                {'BLOCK': LB}, num_warps=_nw(LB))
        else:
            self._ln_loop = False
            self._ln_sig = {
                'X': '*fp32', 'Y': '*fp32', 'W': '*fp32', 'B': '*fp32',
                'Mean': '*fp32', 'Rstd': '*fp32', 'stride': 'i32',
                'N': 'i32', 'eps': 'fp32', 'BLOCK_SIZE': 'constexpr',
            }
            self._ln_result = self.cache.get_or_compile(
                layer_norm_kernel, self._ln_sig,
                {'BLOCK_SIZE': BS_ln}, num_warps=_nw(BS_ln))

    def _compile_rms_norm(self):
        """Compile RMSNorm kernels and store results on self."""
        E = self.n_embd
        BS_rn = _next_pow2(E)
        LB = self.LOOP_BLOCK
        _nw = self._nw
        if self._needs_loop(BS_rn):
            self._rn_loop = True
            self._rn_sig = {
                'X': '*fp32', 'Y': '*fp32', 'W': '*fp32',
                'Rstd': '*fp32', 'stride': 'i32',
                'N': 'i32', 'eps': 'fp32', 'BLOCK': 'constexpr',
            }
            self._rn_result = self.cache.get_or_compile(
                rms_norm_loop_kernel, self._rn_sig,
                {'BLOCK': LB}, num_warps=_nw(LB))

            # Fused residual add + RMSNorm
            self._add_rn_sig = {
                'X': '*fp32', 'Residual': '*fp32',
                'Y': '*fp32', 'W': '*fp32',
                'Rstd': '*fp32', 'stride': 'i32',
                'N': 'i32', 'eps': 'fp32', 'BLOCK': 'constexpr',
            }
            self._add_rn_result = self.cache.get_or_compile(
                add_rms_norm_loop_kernel, self._add_rn_sig,
                {'BLOCK': LB}, num_warps=_nw(LB))
        else:
            self._rn_loop = False
            self._rn_sig = {
                'X': '*fp32', 'Y': '*fp32', 'W': '*fp32',
                'Rstd': '*fp32', 'stride': 'i32',
                'N': 'i32', 'eps': 'fp32', 'BLOCK_SIZE': 'constexpr',
            }
            self._rn_result = self.cache.get_or_compile(
                rms_norm_kernel, self._rn_sig,
                {'BLOCK_SIZE': BS_rn}, num_warps=_nw(BS_rn))

    def _compile_gelu(self):
        """Compile GELU kernel and store results on self."""
        GELU_BLOCK = self.LOOP_BLOCK
        self._gelu_sig = {
            'X': '*fp32', 'Y': '*fp32',
            'N': 'i32', 'BLOCK': 'constexpr',
        }
        self._gelu_result = self.cache.get_or_compile(
            gelu_kernel, self._gelu_sig,
            {'BLOCK': GELU_BLOCK}, num_warps=self._nw(GELU_BLOCK))
        self._gelu_block = GELU_BLOCK

    def _compile_silu_mul(self):
        """Compile fused SiLU*mul (SwiGLU) kernel and store results on self."""
        SM_BLOCK = self.LOOP_BLOCK
        self._sm_sig = {
            'Gate': '*fp32', 'Up': '*fp32', 'Out': '*fp32',
            'N': 'i32', 'BLOCK': 'constexpr',
        }
        self._sm_result = self.cache.get_or_compile(
            silu_mul_kernel, self._sm_sig,
            {'BLOCK': SM_BLOCK}, num_warps=self._nw(SM_BLOCK))
        self._sm_block = SM_BLOCK

        # Fused variant: reads [gate|up] from a single concatenated buffer
        self._smf_sig = {
            'GateUp': '*fp32', 'Out': '*fp32',
            'N': 'i32', 'BLOCK': 'constexpr',
        }
        self._smf_result = self.cache.get_or_compile(
            silu_mul_fused_kernel, self._smf_sig,
            {'BLOCK': SM_BLOCK}, num_warps=self._nw(SM_BLOCK))
        self._smf_block = SM_BLOCK

        # Fused multi-row variant: (T, 2*N) → (T, N)
        self._smfr_sig = {
            'GateUp': '*fp32', 'Out': '*fp32',
            'N': 'i32', 'BLOCK': 'constexpr',
        }
        self._smfr_result = self.cache.get_or_compile(
            silu_mul_fused_rows_kernel, self._smfr_sig,
            {'BLOCK': SM_BLOCK}, num_warps=self._nw(SM_BLOCK))

    def _compile_gelu_mul(self):
        """Compile fused GELU*mul (GeGLU) kernel and store results on self."""
        GM_BLOCK = self.LOOP_BLOCK
        self._gm_sig = {
            'Gate': '*fp32', 'Up': '*fp32', 'Out': '*fp32',
            'N': 'i32', 'BLOCK': 'constexpr',
        }
        self._gm_result = self.cache.get_or_compile(
            gelu_mul_kernel, self._gm_sig,
            {'BLOCK': GM_BLOCK}, num_warps=self._nw(GM_BLOCK))
        self._gm_block = GM_BLOCK

    def _compile_silu(self):
        """Compile standalone SiLU kernel and store results on self."""
        SILU_BLOCK = self.LOOP_BLOCK
        self._silu_sig = {
            'X': '*fp32', 'Y': '*fp32',
            'N': 'i32', 'BLOCK': 'constexpr',
        }
        self._silu_result = self.cache.get_or_compile(
            silu_kernel, self._silu_sig,
            {'BLOCK': SILU_BLOCK}, num_warps=self._nw(SILU_BLOCK))
        self._silu_block = SILU_BLOCK

    def _compile_sigmoid(self):
        """Compile sigmoid kernel and store results on self."""
        SIG_BLOCK = self.LOOP_BLOCK
        self._sig_sig = {
            'X': '*fp32', 'Y': '*fp32',
            'N': 'i32', 'BLOCK': 'constexpr',
        }
        self._sig_result = self.cache.get_or_compile(
            sigmoid_kernel, self._sig_sig,
            {'BLOCK': SIG_BLOCK}, num_warps=self._nw(SIG_BLOCK))
        self._sig_block = SIG_BLOCK

    def _compile_mul(self):
        """Compile element-wise multiply kernel and store results on self."""
        MUL_BLOCK = self.LOOP_BLOCK
        self._mul_sig = {
            'X': '*fp32', 'Y': '*fp32', 'Out': '*fp32',
            'N': 'i32', 'BLOCK': 'constexpr',
        }
        self._mul_result = self.cache.get_or_compile(
            mul_kernel, self._mul_sig,
            {'BLOCK': MUL_BLOCK}, num_warps=self._nw(MUL_BLOCK))
        self._mul_block = MUL_BLOCK

    def _compile_full_attn(self):
        """Compile non-causal (full) attention kernel."""
        HD = self.head_dim
        BS_hd = _next_pow2(HD)
        self._full_attn_sig = {
            'Q': '*fp32', 'K': '*fp32', 'V': '*fp32', 'Out': '*fp32',
            'stride_q': 'i32', 'stride_k': 'i32',
            'stride_v': 'i32', 'stride_o': 'i32',
            'seq_len': 'i32', 'scale': 'fp32',
            'neg_inf': 'fp32',
            'BLOCK_HD': 'constexpr',
        }
        self._full_attn_result = self.cache.get_or_compile(
            full_attn_kernel, self._full_attn_sig,
            {'BLOCK_HD': BS_hd}, num_warps=self._nw(BS_hd))
        self._full_attn_bhd = BS_hd

        # Multi-head variant
        self._full_attn_mh_sig = {
            'Q': '*fp32', 'K': '*fp32', 'V': '*fp32', 'Out': '*fp32',
            'stride_q_t': 'i32', 'stride_q_h': 'i32',
            'stride_k_t': 'i32', 'stride_k_h': 'i32',
            'stride_v_t': 'i32', 'stride_v_h': 'i32',
            'stride_o_t': 'i32', 'stride_o_h': 'i32',
            'seq_len': 'i32', 'scale': 'fp32',
            'neg_inf': 'fp32',
            'BLOCK_HD': 'constexpr',
        }
        self._full_attn_mh_result = self.cache.get_or_compile(
            full_attn_multihead_kernel, self._full_attn_mh_sig,
            {'BLOCK_HD': BS_hd}, num_warps=self._nw(BS_hd))

        # fp16 activation variant
        if self.fp16_act:
            self._full_attn_mh_fp16_sig = {
                'Q': '*fp16', 'K': '*fp16', 'V': '*fp16', 'Out': '*fp16',
                'stride_q_t': 'i32', 'stride_q_h': 'i32',
                'stride_k_t': 'i32', 'stride_k_h': 'i32',
                'stride_v_t': 'i32', 'stride_v_h': 'i32',
                'stride_o_t': 'i32', 'stride_o_h': 'i32',
                'seq_len': 'i32', 'scale': 'fp32',
                'neg_inf': 'fp32',
                'BLOCK_HD': 'constexpr',
            }
            self._full_attn_mh_fp16_result = self.cache.get_or_compile(
                full_attn_multihead_kernel, self._full_attn_mh_fp16_sig,
                {'BLOCK_HD': BS_hd}, num_warps=self._nw(BS_hd))

    def _compile_qk_norm_rope(self):
        """Compile fused QK RMSNorm + RoPE kernel."""
        HD = self.head_dim
        BS_hd = _next_pow2(HD)
        self._qknr_sig = {
            'QKV': '*fp32', 'Q_out': '*fp32', 'K_out': '*fp32', 'V_out': '*fp32',
            'NormQ': '*fp32', 'NormK': '*fp32', 'Cos': '*fp32', 'Sin': '*fp32',
            'n_head': 'i32', 'stride_t': 'i32', 'eps': 'fp32',
            'BLOCK_HD': 'constexpr',
        }
        self._qknr_result = self.cache.get_or_compile(
            qk_norm_rope_kernel, self._qknr_sig,
            {'BLOCK_HD': BS_hd}, num_warps=self._nw(BS_hd))

        # fp16 activation variant: fp16 QKV in/out, fp32 norm weights & RoPE
        if self.fp16_act:
            self._qknr_fp16_sig = {
                'QKV': '*fp16',
                'Q_out': '*fp16', 'K_out': '*fp16', 'V_out': '*fp16',
                'NormQ': '*fp32', 'NormK': '*fp32',
                'Cos': '*fp32', 'Sin': '*fp32',
                'n_head': 'i32', 'stride_t': 'i32', 'eps': 'fp32',
                'BLOCK_HD': 'constexpr',
            }
            self._qknr_fp16_result = self.cache.get_or_compile(
                qk_norm_rope_kernel, self._qknr_fp16_sig,
                {'BLOCK_HD': BS_hd}, num_warps=self._nw(BS_hd))
        self._qknr_bhd = BS_hd

    def _compile_group_norm(self):
        """Compile GroupNorm kernel and store results on self."""
        GN_BLOCK = self.LOOP_BLOCK
        self._gn_sig = {
            'X': '*fp32', 'Y': '*fp32', 'W': '*fp32', 'B': '*fp32',
            'Rstd': '*fp32', 'stride': 'i32', 'N': 'i32',
            'num_groups': 'i32', 'eps': 'fp32', 'BLOCK': 'constexpr',
        }
        self._gn_result = self.cache.get_or_compile(
            group_norm_kernel, self._gn_sig,
            {'BLOCK': GN_BLOCK}, num_warps=self._nw(GN_BLOCK))
        self._gn_block = GN_BLOCK

    def _compile_embed_gather(self):
        """Compile GPU embedding gather kernel (loop-based for large E)."""
        LB = self.LOOP_BLOCK
        self._embed_gather_sig = {
            'TokenId': '*i32', 'Embedding': '*fp32', 'Out': '*fp32',
            'stride_e': 'i32',
            'BLOCK_E': 'constexpr',
        }
        self._embed_gather_result = self.cache.get_or_compile(
            embed_gather_kernel, self._embed_gather_sig,
            {'BLOCK_E': LB},
            num_warps=self._nw(LB))

    def _compile_argmax(self):
        """Compile GPU argmax kernel for greedy sampling."""
        ARGMAX_BLOCK = self.LOOP_BLOCK
        self._argmax_sig = {
            'Logits': '*fp32', 'TokenOut': '*i32', 'N': 'i32',
            'BLOCK': 'constexpr',
        }
        self._argmax_result = self.cache.get_or_compile(
            argmax_kernel, self._argmax_sig,
            {'BLOCK': ARGMAX_BLOCK},
            num_warps=self._nw(ARGMAX_BLOCK))

    def _embed_gather(self, token_id_gpu, embedding_gpu, gpu_out=True):
        """GPU embedding lookup: Out = Embedding[token_id, :].

        token_id_gpu: GPUBuffer with single i32 token ID
        embedding_gpu: GPUBuffer with (vocab, E) fp32 embedding table
        Returns: GPUBuffer (1, E) or numpy (1, E)
        """
        E = self.n_embd
        out = self.cache.run(
            self._embed_gather_result, grid=(1,),
            buffers={
                'TokenId': token_id_gpu,
                'Embedding': embedding_gpu,
                'Out': np.zeros(E, dtype=np.float32),
            },
            scalars={'stride_e': E},
            gpu_outputs={'Out'} if gpu_out else None)
        if gpu_out:
            gpu_buf = out['Out']
            gpu_buf.shape = (1, E)
            return gpu_buf
        return out['Out'][:E].reshape(1, E)

    def _argmax(self, logits_gpu, gpu_out=True):
        """GPU argmax: TokenOut = argmax(Logits).

        logits_gpu: GPUBuffer with (N,) fp32 logits
        Returns: GPUBuffer with single i32 token, or int
        """
        N = self.n_vocab
        out = self.cache.run(
            self._argmax_result, grid=(1,),
            buffers={
                'Logits': logits_gpu,
                'TokenOut': np.zeros(1, dtype=np.int32),
            },
            scalars={'N': N},
            gpu_outputs={'TokenOut'} if gpu_out else None)
        if gpu_out:
            return out['TokenOut']
        return int(out['TokenOut'][0])

    # --- Weight upload helpers ---

    def _upload_linear_weight(self, name: str, N: int, K: int) -> GPUBuffer:
        """Upload weight matrix, padding if needed for single-pass linear.

        Args:
            name: weight key in self.weights
            N: output features (number of rows)
            K: input features (number of columns)

        Returns:
            GPUBuffer with the uploaded (and possibly padded) weight.
        """
        runner = self.cache.runner
        w = self.weights[name].reshape(N, K)
        if self._linear_loop:
            flat = w.ravel()
        else:
            # Find the matching BK for this K dimension
            bk = self._get_bk(K)
            if K < bk:
                padded = np.zeros((N, bk), dtype=np.float32)
                padded[:, :K] = w
                flat = padded.ravel()
            else:
                flat = w.ravel()
        buf = runner.upload_to_gpu(flat, name)
        self._gpu_weights[name] = buf
        return buf

    def _upload_linear_weight_fp16(self, name: str, N: int, K: int) -> GPUBuffer:
        """Upload weight matrix as fp16 for fp16-weight linear kernel.

        If weights are already fp16 (e.g. dequantized directly to fp16),
        uploads as-is. Otherwise converts fp32 to fp16 first.
        """
        runner = self.cache.runner
        w = self.weights[name].reshape(N, K)
        if w.dtype != np.float16:
            w = w.astype(np.float16)
        flat = w.ravel()
        fp16_name = name + ".fp16"
        buf = runner.upload_to_gpu(flat, fp16_name)
        self._gpu_weights[fp16_name] = buf
        return buf

    def _upload_linear_weight_fp16_transposed(self, name: str,
                                               N: int, K: int) -> GPUBuffer:
        """Upload weight as fp16 in transposed (K, N) layout.

        Original W is (N, K) row-major.  Transposing to (K, N) makes
        the N dimension contiguous, enabling coalesced per-thread reads
        in the barrier-free matvec kernel.
        """
        runner = self.cache.runner
        w = self.weights[name].reshape(N, K)
        if w.dtype != np.float16:
            w = w.astype(np.float16)
        w_t = np.ascontiguousarray(w.T)  # (K, N) contiguous fp16
        fp16_name = name + ".fp16"
        buf = runner.upload_to_gpu(w_t.ravel(), fp16_name)
        self._gpu_weights[fp16_name] = buf
        return buf

    def _upload_q4_weight(self, name: str, N: int, K: int,
                          group_size: int = 128):
        """Upload INT4 packed weights + scales + zeros to GPU.

        Expects self.weights to contain:
          name + '.q4':     uint8 packed (N, K_pad/2)
          name + '.scales': fp16 (N, n_groups)
          name + '.zeros':  fp16 (N, n_groups)
        """
        runner = self.cache.runner
        q4 = self.weights[name + ".q4"]  # uint8 (N, K_pad/2)
        scales = self.weights[name + ".scales"]  # fp16 (N, n_groups)
        zeros = self.weights[name + ".zeros"]    # fp16 (N, n_groups)

        # Reinterpret uint8 packed data as i32 (4 bytes = 8 INT4 vals)
        q4_flat = q4.ravel()
        # Ensure length is divisible by 4 for i32 view
        assert q4_flat.nbytes % 4 == 0
        q4_i32 = q4_flat.view(np.int32)
        q4_buf = runner.upload_to_gpu(q4_i32, name + ".q4.gpu")
        self._gpu_weights[name + ".q4.gpu"] = q4_buf

        # Upload scales and zeros as fp16
        s_flat = scales.ravel()
        if s_flat.dtype != np.float16:
            s_flat = s_flat.astype(np.float16)
        s_buf = runner.upload_to_gpu(s_flat, name + ".scales.gpu")
        self._gpu_weights[name + ".scales.gpu"] = s_buf

        z_flat = zeros.ravel()
        if z_flat.dtype != np.float16:
            z_flat = z_flat.astype(np.float16)
        z_buf = runner.upload_to_gpu(z_flat, name + ".zeros.gpu")
        self._gpu_weights[name + ".zeros.gpu"] = z_buf

    def _upload_bias(self, name: str) -> GPUBuffer:
        """Upload a bias vector to GPU."""
        runner = self.cache.runner
        buf = runner.upload_to_gpu(self.weights[name], name)
        self._gpu_weights[name] = buf
        return buf

    def _upload_norm_weight(self, name: str) -> GPUBuffer:
        """Upload a normalization weight to GPU."""
        runner = self.cache.runner
        buf = runner.upload_to_gpu(self.weights[name], name)
        self._gpu_weights[name] = buf
        return buf

    def _upload_zero_bias(self, name: str, size: int) -> GPUBuffer:
        """Upload a zero bias buffer to GPU."""
        runner = self.cache.runner
        buf = runner.upload_to_gpu(
            np.zeros(size, dtype=np.float32), name)
        self._gpu_weights[name] = buf
        return buf

    # Maximum GPU buffer size (D3D12 limit: buffers >2GB are silently broken)
    MAX_GPU_BUFFER_SIZE = 2 * 1024 * 1024 * 1024  # 2 GiB

    def _upload_embedding_weight(self, name: str, n_vocab: int,
                                  n_embd: int) -> GPUBuffer:
        """Upload embedding/LM-head weight, padding if needed.

        For very large embeddings (>2GB), splits into multiple GPU buffers
        stored as a list in self._gpu_weight_chunks[name].
        """
        runner = self.cache.runner
        wte = self.weights[name]  # (n_vocab, n_embd)
        if self._linear_loop:
            flat = wte.ravel()
        else:
            bk = self._get_bk(n_embd)
            if n_embd < bk:
                wte_pad = np.zeros((n_vocab, bk), dtype=np.float32)
                wte_pad[:, :n_embd] = wte
                flat = wte_pad.ravel()
            else:
                flat = wte.ravel()

        if flat.nbytes > self.MAX_GPU_BUFFER_SIZE:
            # Split into chunks that fit in 2GB GPU buffers
            K = flat.size // n_vocab  # stride per row (padded or not)
            max_rows = self.MAX_GPU_BUFFER_SIZE // (K * 4)
            chunks = []
            for start in range(0, n_vocab, max_rows):
                end = min(start + max_rows, n_vocab)
                chunk_data = flat[start * K : end * K]
                buf = runner.upload_to_gpu(
                    chunk_data, f"{name}_chunk_{start}")
                buf.shape = (end - start, K)
                chunks.append((start, end, buf))
            # Store chunk list for _linear_chunked to use
            if not hasattr(self, '_gpu_weight_chunks'):
                self._gpu_weight_chunks = {}
            self._gpu_weight_chunks[name] = chunks
            # Also store a sentinel in _gpu_weights so callers know it exists
            # Use the first chunk as a stand-in (will be handled specially)
            self._gpu_weights[name] = chunks[0][2]
            total_bytes = sum(buf.size for _, _, buf in chunks)
            return chunks[0][2]
        else:
            buf = runner.upload_to_gpu(flat, name)
            self._gpu_weights[name] = buf
            return buf

    def _get_bk(self, K: int) -> int:
        """Get the block K (BK) for a given K dimension in single-pass mode.

        Finds the best matching pre-compiled kernel for this K.
        """
        if self._linear_loop:
            return K  # loop mode doesn't pad

        # Find exact match first
        if K in self._linear_results:
            return self._linear_results[K][1]

        # Find the smallest BK that fits K
        best_bk = None
        for k_dim, (_, bk) in sorted(self._linear_results.items()):
            if bk >= K:
                if best_bk is None or bk < best_bk:
                    best_bk = bk
        if best_bk is not None:
            return best_bk

        # Fallback: use n_embd BK
        return self._linear_results[self.n_embd][1]

    def _get_linear_result(self, K: int):
        """Get the compiled linear kernel result for a given K dimension."""
        if self._linear_loop:
            return self._linear_result

        # Find exact match
        if K in self._linear_results:
            return self._linear_results[K][0]

        # Find the smallest BK >= K
        best = None
        best_bk = None
        for k_dim, (result, bk) in sorted(self._linear_results.items()):
            if bk >= _next_pow2(K):
                if best_bk is None or bk < best_bk:
                    best = result
                    best_bk = bk
        if best is not None:
            return best

        # Fallback
        return self._linear_results[self.n_embd][0]

    def _print_gpu_weight_stats(self):
        """Print GPU weight upload statistics."""
        total_bytes = sum(g.size for g in self._gpu_weights.values())
        # Add bytes from chunked weight buffers (>2GB embeddings)
        if hasattr(self, '_gpu_weight_chunks'):
            for name, chunks in self._gpu_weight_chunks.items():
                # Subtract the first chunk already counted in _gpu_weights
                total_bytes += sum(buf.size for _, _, buf in chunks[1:])
        print(f"  Uploaded {len(self._gpu_weights)} weight tensors "
              f"({total_bytes / 1024 / 1024:.0f} MB) to GPU")

    # =====================================================================
    # Primitive ops
    # =====================================================================

    def _layer_norm(self, x, w, b, eps: float = None,
                    gpu_out: bool = False):
        """LayerNorm: y = (x - mean) / sqrt(var + eps) * w + b.

        x: (T, E). w, b may be numpy or GPUBuffer.
        """
        if eps is None:
            eps = self.norm_eps
        if isinstance(x, GPUBuffer):
            T = x.shape[0] if x.shape else 1
        else:
            T = x.shape[0]
        E = self.n_embd
        out = self.cache.run(
            self._ln_result, grid=(T,),
            buffers={
                'X': x if isinstance(x, GPUBuffer) else x.ravel(),
                'Y': np.zeros(T * E, dtype=np.float32),
                'W': w, 'B': b,
                'Mean': np.zeros(T, dtype=np.float32),
                'Rstd': np.zeros(T, dtype=np.float32),
            },
            scalars={'stride': E, 'N': E, 'eps': eps},
            gpu_outputs={'Y', 'Mean', 'Rstd'} if gpu_out else None)
        if gpu_out:
            gpu_buf = out['Y']
            gpu_buf.shape = (T, E)
            return gpu_buf
        return out['Y'].reshape(T, E)

    def _rms_norm(self, x, w, eps: float = None,
                  gpu_out: bool = False):
        """RMSNorm: x / sqrt(mean(x^2) + eps) * w.

        x: (T, E). w may be numpy or GPUBuffer.
        """
        if eps is None:
            eps = self.rms_norm_eps
        if isinstance(x, GPUBuffer):
            T = x.shape[0] if x.shape else 1
        else:
            T = x.shape[0]
        E = self.n_embd
        out = self.cache.run(
            self._rn_result, grid=(T,),
            buffers={
                'X': x if isinstance(x, GPUBuffer) else x.ravel(),
                'Y': np.zeros(T * E, dtype=np.float32),
                'W': w,
                'Rstd': np.zeros(T, dtype=np.float32),
            },
            scalars={'stride': E, 'N': E, 'eps': eps},
            gpu_outputs={'Y', 'Rstd'} if gpu_out else None)
        if gpu_out:
            gpu_buf = out['Y']
            gpu_buf.shape = (T, E)
            return gpu_buf
        return out['Y'].reshape(T, E)

    def _add_rms_norm(self, x_gpu, residual_gpu, w, eps: float = None,
                      gpu_out: bool = False):
        """Fused residual add + RMSNorm: x += residual; y = rms_norm(x) * w.

        Both x_gpu and residual_gpu must be GPUBuffer.
        x_gpu is modified in-place (residual added).
        Returns normalized output Y.
        """
        if eps is None:
            eps = self.rms_norm_eps
        T = x_gpu.shape[0] if x_gpu.shape else 1
        E = self.n_embd
        out = self.cache.run(
            self._add_rn_result, grid=(T,),
            buffers={
                'X': x_gpu,
                'Residual': residual_gpu,
                'Y': np.zeros(T * E, dtype=np.float32),
                'W': w,
                'Rstd': np.zeros(T, dtype=np.float32),
            },
            scalars={'stride': E, 'N': E, 'eps': eps},
            gpu_outputs={'X', 'Y', 'Rstd'} if gpu_out else None)
        if gpu_out:
            gpu_buf = out['Y']
            gpu_buf.shape = (T, E)
            return gpu_buf
        return out['Y'].reshape(T, E)

    def _linear(self, x, w, bias, out_features: int,
                gpu_out: bool = False):
        """x: (T, K) @ w^T: (N, K) + bias -> (T, N).

        w and bias may be numpy arrays or GPUBuffer objects.
        If gpu_out=True, returns a GPUBuffer instead of numpy.

        When N exceeds MAX_DISPATCH_DIM (65535), automatically chunks
        along the N dimension and concatenates results.
        """
        N = out_features

        # --- Chunked dispatch when N exceeds workgroup dispatch limit ---
        if N > self.MAX_DISPATCH_DIM:
            return self._linear_chunked(x, w, bias, N, gpu_out=gpu_out)

        x_is_gpu = isinstance(x, GPUBuffer)
        w_is_gpu = isinstance(w, GPUBuffer)

        if x_is_gpu:
            T = x.shape[0] if x.shape else 1
            K = x.shape[1] if (x.shape and len(x.shape) > 1) else self.n_embd
        else:
            T, K = x.shape[0], x.shape[1]
        N = out_features

        if self._linear_loop:
            out = self.cache.run(
                self._linear_result, grid=(T, N),
                buffers={
                    'X': x if x_is_gpu else x.ravel(),
                    'W': w if w_is_gpu else w.reshape(N, K).ravel(),
                    'Bias': bias,
                    'Y': np.zeros(T * N, dtype=np.float32),
                },
                scalars={'K': K, 'stride_x': K, 'stride_w': K, 'N': N},
                gpu_outputs={'Y'} if gpu_out else None)
        else:
            # Single-pass: select kernel based on K dimension
            result = self._get_linear_result(K)
            BK = self._get_bk(K)

            if w_is_gpu:
                w_val = w
                stride_w = BK
            else:
                if K < BK:
                    w_pad = np.zeros((N, BK), dtype=np.float32)
                    w_pad[:, :K] = w.reshape(N, -1)[:, :K]
                    w_val = w_pad.ravel()
                else:
                    w_val = w.reshape(N, K).ravel()
                stride_w = BK

            if x_is_gpu:
                # GPUBuffer data has stride K, but single-pass needs
                # stride BK with zero-padding. Readback and pad.
                x_np = self.cache.runner.readback(x).reshape(T, K)
                if K < BK:
                    x_pad = np.zeros((T, BK), dtype=np.float32)
                    x_pad[:, :K] = x_np
                    x_val = x_pad.ravel()
                else:
                    x_val = x_np.ravel()
            else:
                if K < BK:
                    x_pad = np.zeros((T, BK), dtype=np.float32)
                    x_pad[:, :K] = x
                    x_val = x_pad.ravel()
                else:
                    x_val = x.ravel()
            stride_x = BK

            out = self.cache.run(
                result, grid=(T, N),
                buffers={
                    'X': x_val, 'W': w_val, 'Bias': bias,
                    'Y': np.zeros(T * N, dtype=np.float32),
                },
                scalars={'stride_x': stride_x, 'stride_w': stride_w,
                         'N': N},
                gpu_outputs={'Y'} if gpu_out else None)

        if gpu_out:
            gpu_buf = out['Y']
            gpu_buf.shape = (T, N)
            return gpu_buf
        return out['Y'].reshape(T, N)

    def _linear_fp16w(self, x, w_fp16, bias, out_features: int,
                      K: int = None, gpu_out: bool = False):
        """Linear projection with fp16 weight buffer.

        x: (T, K) fp32 input
        w_fp16: GPUBuffer containing fp16 weights (N, K)
        bias: GPUBuffer fp32 bias (N,)
        Halves memory bandwidth vs fp32 weights.

        For large N (> MAX_DISPATCH_DIM), uses a wide kernel that maps
        a 2D grid to a 1D output index, avoiding chunked dispatch.
        """
        N = out_features
        x_is_gpu = isinstance(x, GPUBuffer)
        w_is_gpu = isinstance(w_fp16, GPUBuffer)

        if x_is_gpu:
            T = x.shape[0] if x.shape else 1
            if K is None:
                K = x.shape[1] if (x.shape and len(x.shape) > 1) else self.n_embd
        else:
            T = x.shape[0]
            if K is None:
                K = x.shape[1]

        # Use wide kernel for large N (avoids chunked dispatch)
        if N > self.MAX_DISPATCH_DIM and T == 1:
            grid_y = self.MAX_DISPATCH_DIM
            grid_x = (N + grid_y - 1) // grid_y
            out = self.cache.run(
                self._linear_fp16w_wide_result, grid=(grid_x, grid_y),
                buffers={
                    'X': x if x_is_gpu else x.ravel(),
                    'W': w_fp16 if w_is_gpu else w_fp16.ravel(),
                    'Bias': bias,
                    'Y': np.zeros(N, dtype=np.float32),
                },
                scalars={'K': K, 'stride_x': K, 'stride_w': K,
                         'N': N, 'grid_y': grid_y},
                gpu_outputs={'Y'} if gpu_out else None)
            if gpu_out:
                gpu_buf = out['Y']
                gpu_buf.shape = (1, N)
                return gpu_buf
            return out['Y'].reshape(1, N)

        # Chunked dispatch for T>1 with large N
        if N > self.MAX_DISPATCH_DIM:
            return self._linear_fp16w_chunked(x, w_fp16, bias, N,
                                              K=K, gpu_out=gpu_out)

        if self.fp16_act:
            kern = self._linear_fp16io_result
            out_dt = np.float16
            x_buf = x if x_is_gpu else x.astype(np.float16).ravel()
        else:
            kern = self._linear_fp16w_result
            out_dt = np.float32
            x_buf = x if x_is_gpu else x.ravel()

        out = self.cache.run(
                kern, grid=(T, N),
                buffers={
                    'X': x_buf,
                    'W': w_fp16 if w_is_gpu else w_fp16.ravel(),
                    'Bias': bias,
                    'Y': np.zeros(T * N, dtype=out_dt),
                },
                scalars={'K': K, 'stride_x': K, 'stride_w': K, 'N': N},
                gpu_outputs={'Y'} if gpu_out else None)

        if gpu_out:
            gpu_buf = out['Y']
            gpu_buf.shape = (T, N)
            return gpu_buf
        return out['Y'].reshape(T, N)

    def _linear_fp16w_chunked(self, x, w_fp16, bias, N: int,
                               K: int = None, gpu_out: bool = False):
        """Chunked fp16-weight linear for N > MAX_DISPATCH_DIM.

        Splits the output dimension into chunks, dispatching each
        within the WebGPU limit, then concatenates results.
        """
        x_is_gpu = isinstance(x, GPUBuffer)
        w_is_gpu = isinstance(w_fp16, GPUBuffer)

        if x_is_gpu:
            T = x.shape[0] if x.shape else 1
            if K is None:
                K = x.shape[1] if (x.shape and len(x.shape) > 1) else self.n_embd
        else:
            T = x.shape[0]
            if K is None:
                K = x.shape[1]

        chunk_size = self.MAX_DISPATCH_DIM
        out_chunks = []
        runner = self.cache.runner
        bias_is_gpu = isinstance(bias, GPUBuffer)

        for start in range(0, N, chunk_size):
            end = min(start + chunk_size, N)
            n_chunk = end - start

            # Slice fp16 weight: each row is K fp16 elements = K*2 bytes
            if w_is_gpu:
                w_buf = runner.gpu_slice(
                    w_fp16, start * K * 2, n_chunk * K * 2,
                    f"_fp16w_chunk_w_{start}")
            else:
                w_buf = w_fp16.reshape(N, K)[start:end].ravel()

            # Slice bias (fp32, 4 bytes per element)
            if bias_is_gpu:
                b_buf = runner.gpu_slice(
                    bias, start * 4, n_chunk * 4,
                    f"_fp16w_chunk_b_{start}")
            else:
                b_buf = bias.ravel()[start:end]

            chunk_out = self.cache.run(
                self._linear_fp16w_result, grid=(T, n_chunk),
                buffers={
                    'X': x if x_is_gpu else x.ravel(),
                    'W': w_buf,
                    'Bias': b_buf,
                    'Y': np.zeros(T * n_chunk, dtype=np.float32),
                },
                scalars={'K': K, 'stride_x': K, 'stride_w': K, 'N': n_chunk},
                gpu_outputs=None)
            out_chunks.append(chunk_out['Y'].reshape(T, n_chunk))

        result = np.concatenate(out_chunks, axis=1)
        if gpu_out:
            buf = runner.upload_to_gpu(result.ravel(), "_fp16w_chunked_out")
            buf.shape = (T, N)
            return buf
        return result

    def _linear_fp16w_transposed(self, x, w_t_fp16, bias,
                                  out_features: int,
                                  K: int = None, gpu_out: bool = False):
        """Linear projection using transposed fp16 weights (barrier-free).

        w_t_fp16: GPUBuffer with transposed fp16 weights in (K, N) layout.
        Each thread independently computes one output element — no
        inter-thread reduction and no workgroup barriers.

        Handles any N (no 65535 dispatch limit issue because the grid
        second dimension is ceil(N/BLOCK_N) which is always small).
        """
        N = out_features
        x_is_gpu = isinstance(x, GPUBuffer)
        w_is_gpu = isinstance(w_t_fp16, GPUBuffer)

        if x_is_gpu:
            T = x.shape[0] if x.shape else 1
            if K is None:
                K = x.shape[1] if (x.shape and len(x.shape) > 1) else self.n_embd
        else:
            T = x.shape[0]
            if K is None:
                K = x.shape[1]

        BLOCK_N = self.LOOP_BLOCK
        grid_y = (N + BLOCK_N - 1) // BLOCK_N
        out = self.cache.run(
            self._linear_Wt_fp16_result, grid=(T, grid_y),
            buffers={
                'X': x if x_is_gpu else x.ravel(),
                'W_T': w_t_fp16 if w_is_gpu else w_t_fp16.ravel(),
                'Bias': bias,
                'Y': np.zeros(T * N, dtype=np.float32),
            },
            scalars={'K': K, 'N': N, 'stride_x': K},
            gpu_outputs={'Y'} if gpu_out else None)

        if gpu_out:
            gpu_buf = out['Y']
            gpu_buf.shape = (T, N)
            return gpu_buf
        return out['Y'].reshape(T, N)

    def _linear_q4(self, x, w_q4, scales, zeros, bias,
                   out_features: int, K: int = None,
                   gpu_out: bool = False):
        """Linear projection with INT4 packed weights (fused dequant).

        w_q4: GPUBuffer of packed i32 INT4 data
        scales/zeros: GPUBuffer of fp16 per-group factors
        """
        N = out_features
        x_is_gpu = isinstance(x, GPUBuffer)

        if x_is_gpu:
            T = x.shape[0] if x.shape else 1
            if K is None:
                K = x.shape[1] if (x.shape and len(x.shape) > 1) else self.n_embd
        else:
            T = x.shape[0]
            if K is None:
                K = x.shape[1]

        n_groups = K // 128  # GROUP_SIZE = 128
        stride_w_q4 = K // 8  # i32 elements per row

        out = self.cache.run(
            self._linear_q4_result, grid=(T, N),
            buffers={
                'X': x if x_is_gpu else x.ravel(),
                'W_Q4': w_q4,
                'Scales': scales,
                'Zeros': zeros,
                'Bias': bias,
                'Y': np.zeros(T * N, dtype=np.float32),
            },
            scalars={'K': K, 'stride_x': K,
                     'stride_w_q4': stride_w_q4,
                     'n_groups': n_groups, 'N': N},
            gpu_outputs={'Y'} if gpu_out else None)

        if gpu_out:
            gpu_buf = out['Y']
            gpu_buf.shape = (T, N)
            return gpu_buf
        return out['Y'].reshape(T, N)

    def _linear_q4_dp4a(self, x, w_q4, scales, zeros, bias,
                         out_features: int, K: int = None,
                         gpu_out: bool = False):
        """INT4 matmul using hand-crafted WGSL with subgroupAdd.

        Same math as _linear_q4 (fp32 activations × fp32 dequantized INT4
        weights), but uses a direct WGSL kernel with subgroupAdd reduction
        instead of going through the Triton LLVM→WGSL pipeline.

        No int8 activation quantization — preserves full fp32 precision.
        Same interface as _linear_q4.
        """
        from common.wgsl_kernels import (
            WGSL_Q4_DP4A_KERNEL, Q4_DP4A_BINDINGS, pack_dp4a_params, TILE_N,
        )

        N = out_features
        x_is_gpu = isinstance(x, GPUBuffer)

        if x_is_gpu:
            T = x.shape[0] if x.shape else 1
            if K is None:
                K = x.shape[1] if (x.shape and len(x.shape) > 1) else self.n_embd
        else:
            T = x.shape[0]
            if K is None:
                K = x.shape[1]

        n_groups = K // 128
        stride_w_q4 = K // 8
        params = pack_dp4a_params(K, stride_w_q4, n_groups, N)

        # Cache params as GPUBuffer keyed by (K, N) to avoid overwrites
        # during batched dispatch (multiple matmuls per batch share the
        # same '_params_' buffer name otherwise).
        runner = self.cache.runner
        params_key = f"__dp4a_params_{K}_{N}"
        if not hasattr(self, '_dp4a_params_cache'):
            self._dp4a_params_cache = {}
        if params_key not in self._dp4a_params_cache:
            self._dp4a_params_cache[params_key] = \
                runner.upload_to_gpu(params, params_key)
        params_gpu = self._dp4a_params_cache[params_key]

        out = runner.run_kernel(
            wgsl_code=WGSL_Q4_DP4A_KERNEL,
            buffer_bindings=Q4_DP4A_BINDINGS,
            param_fields=[],
            workgroup_size=256,
            grid=(T, (N + TILE_N - 1) // TILE_N),  # TILE_N outputs per workgroup
            buffers={
                'X': x if x_is_gpu else x.ravel().astype(np.float32),
                'W_Q4': w_q4,
                'Scales': scales,
                'Zeros': zeros,
                'Bias': bias,
                'Y': np.zeros(T * N, dtype=np.float32),
                '_params_': params_gpu,
            },
            scalars={},
            gpu_outputs={'Y'} if gpu_out else None)

        if gpu_out:
            gpu_buf = out['Y']
            gpu_buf.shape = (T, N)
            return gpu_buf
        return out['Y'].reshape(T, N)

    def _linear_fp16w_wgsl(self, x, w_fp16_u32, bias, out_features: int,
                            K: int = None, gpu_out: bool = False):
        """FP16 GEMM using hand-crafted WGSL with subgroupAdd + vec4 dot.

        w_fp16_u32: GPUBuffer storing fp16 weights as u32 (2 fp16 per u32).
        Uses unpack2x16float for fp16→fp32 conversion and dot(vec4) for
        4 FMAs per instruction. subgroupAdd for warp-level reduction.
        K must be divisible by 4.
        """
        from common.wgsl_kernels import (
            WGSL_FP16_GEMM_KERNEL, FP16_GEMM_BINDINGS,
            pack_fp16_gemm_params, FP16_GEMM_TILE_N,
        )

        N = out_features
        x_is_gpu = isinstance(x, GPUBuffer)

        if x_is_gpu:
            T = x.shape[0] if x.shape else 1
            if K is None:
                K = x.shape[1] if (x.shape and len(x.shape) > 1) else self.n_embd
        else:
            T = x.shape[0]
            if K is None:
                K = x.shape[1]

        params = pack_fp16_gemm_params(K, N)

        runner = self.cache.runner
        params_key = f"__fp16gemm_params_{K}_{N}"
        if not hasattr(self, '_fp16gemm_params_cache'):
            self._fp16gemm_params_cache = {}
        if params_key not in self._fp16gemm_params_cache:
            self._fp16gemm_params_cache[params_key] = \
                runner.upload_to_gpu(params, params_key)
        params_gpu = self._fp16gemm_params_cache[params_key]

        out = runner.run_kernel(
            wgsl_code=WGSL_FP16_GEMM_KERNEL,
            buffer_bindings=FP16_GEMM_BINDINGS,
            param_fields=[],
            workgroup_size=256,
            grid=(T, (N + FP16_GEMM_TILE_N - 1) // FP16_GEMM_TILE_N),
            buffers={
                'X': x if x_is_gpu else x.ravel().astype(np.float32),
                'W': w_fp16_u32,
                'Bias': bias,
                'Y': np.zeros(T * N, dtype=np.float32),
                '_params_': params_gpu,
            },
            scalars={},
            gpu_outputs={'Y'} if gpu_out else None)

        if gpu_out:
            gpu_buf = out['Y']
            gpu_buf.shape = (T, N)
            return gpu_buf
        return out['Y'].reshape(T, N)

    def _linear_chunked(self, x, w, bias, N: int,
                        gpu_out: bool = False):
        """Chunked linear projection for N > MAX_DISPATCH_DIM.

        Splits the output dimension N into chunks that fit within the
        WebGPU dispatch limit, runs each chunk separately, and
        concatenates the results.

        Handles three weight storage scenarios:
        1. Pre-chunked GPU buffers (from _upload_embedding_weight for >2GB)
        2. Single GPU buffer (uses gpu_slice for sub-2GB weights)
        3. NumPy array (sliced on CPU, uploaded per chunk)
        """
        x_is_gpu = isinstance(x, GPUBuffer)
        w_is_gpu = isinstance(w, GPUBuffer)

        if x_is_gpu:
            T = x.shape[0] if x.shape else 1
            K = x.shape[1] if (x.shape and len(x.shape) > 1) else self.n_embd
        else:
            T, K = x.shape[0], x.shape[1]

        chunk_size = self.MAX_DISPATCH_DIM
        out_chunks = []
        runner = self.cache.runner

        # Check if we have pre-chunked weight buffers
        weight_chunks = None
        if hasattr(self, '_gpu_weight_chunks'):
            for name, chunks in self._gpu_weight_chunks.items():
                # Match by GPU buffer handle
                if any(buf.handle == w.handle for _, _, buf in chunks):
                    weight_chunks = chunks
                    break

        bias_is_gpu = isinstance(bias, GPUBuffer)
        if not bias_is_gpu:
            bias_np = bias.ravel() if hasattr(bias, 'ravel') else bias

        if weight_chunks is not None:
            # Pre-chunked path: iterate over pre-split GPU buffers
            # Each chunk has (row_start, row_end, gpu_buf)
            # We need to further split each chunk if it exceeds dispatch limit
            for chunk_start, chunk_end, w_buf in weight_chunks:
                chunk_N = chunk_end - chunk_start
                for sub_start in range(0, chunk_N, chunk_size):
                    sub_end = min(sub_start + chunk_size, chunk_N)
                    n_sub = sub_end - sub_start
                    global_start = chunk_start + sub_start

                    # Slice weight from this pre-chunked buffer
                    w_offset = sub_start * K * 4
                    w_size = n_sub * K * 4
                    w_sub = runner.gpu_slice(
                        w_buf, w_offset, w_size,
                        f"_lm_sub_w_{global_start}")

                    # Slice or extract bias
                    if bias_is_gpu:
                        b_sub = runner.gpu_slice(
                            bias, global_start * 4, n_sub * 4,
                            f"_lm_sub_b_{global_start}")
                    else:
                        b_chunk = bias_np[global_start:global_start + n_sub]
                        b_sub = runner.upload_to_gpu(
                            b_chunk.ravel(), f"_lm_sub_b_{global_start}")

                    chunk_out = self._linear(x, w_sub, b_sub, n_sub)
                    out_chunks.append(chunk_out)
        else:
            # Standard path: single weight source
            if w_is_gpu:
                w_source = w
                w_on_gpu = True
            else:
                w_np = w.reshape(N, K)
                w_on_gpu = False

            for start in range(0, N, chunk_size):
                end = min(start + chunk_size, N)
                n_chunk = end - start

                if w_on_gpu:
                    w_offset = start * K * 4
                    w_size = n_chunk * K * 4
                    w_buf = runner.gpu_slice(
                        w_source, w_offset, w_size,
                        f"_lm_chunk_w_{start}")
                else:
                    w_chunk = w_np[start:end]
                    w_buf = runner.upload_to_gpu(
                        w_chunk.ravel(), f"_lm_chunk_w_{start}")

                if bias_is_gpu:
                    b_buf = runner.gpu_slice(
                        bias, start * 4, n_chunk * 4,
                        f"_lm_chunk_b_{start}")
                else:
                    b_chunk = bias_np[start:end]
                    b_buf = runner.upload_to_gpu(
                        b_chunk.ravel(), f"_lm_chunk_b_{start}")

                chunk_out = self._linear(x, w_buf, b_buf, n_chunk)
                out_chunks.append(chunk_out)

        result = np.concatenate(out_chunks, axis=1)
        if gpu_out:
            buf = runner.upload_to_gpu(result.ravel(), "_lm_chunked_out")
            buf.shape = (T, N)
            return buf
        return result

    def _gelu(self, x, gpu_out: bool = False):
        """GELU activation.

        Accepts numpy or GPUBuffer input.
        """
        x_is_gpu = isinstance(x, GPUBuffer)
        if x_is_gpu:
            N = x.size // 4  # f32 = 4 bytes
            shape = x.shape
        else:
            N = x.size
            shape = x.shape

        grid_size = (N + self._gelu_block - 1) // self._gelu_block
        out = self.cache.run(
            self._gelu_result, grid=(grid_size,),
            buffers={
                'X': x if x_is_gpu else x.ravel(),
                'Y': np.zeros(N, dtype=np.float32),
            },
            scalars={'N': N},
            gpu_outputs={'Y'} if gpu_out else None)
        if gpu_out:
            gpu_buf = out['Y']
            gpu_buf.shape = shape
            return gpu_buf
        return out['Y'].reshape(shape)

    def _silu_mul(self, gate, up, gpu_out: bool = False):
        """Fused SwiGLU: SiLU(gate) * up.

        Accepts numpy or GPUBuffer inputs.
        """
        gate_is_gpu = isinstance(gate, GPUBuffer)
        up_is_gpu = isinstance(up, GPUBuffer)
        if self.fp16_act:
            kern = self._sm_fp16_result
            out_dt = np.float16
            elem_bytes = 2
        else:
            kern = self._sm_result
            out_dt = np.float32
            elem_bytes = 4
        if gate_is_gpu:
            N = gate.size // elem_bytes
            shape = gate.shape
        else:
            N = gate.size
            shape = gate.shape

        grid_size = (N + self._sm_block - 1) // self._sm_block
        out = self.cache.run(
            kern, grid=(grid_size,),
            buffers={
                'Gate': gate if gate_is_gpu else gate.ravel(),
                'Up': up if up_is_gpu else up.ravel(),
                'Out': np.zeros(N, dtype=out_dt),
            },
            scalars={'N': N},
            gpu_outputs={'Out'} if gpu_out else None)
        if gpu_out:
            gpu_buf = out['Out']
            gpu_buf.shape = shape
            return gpu_buf
        return out['Out'].reshape(shape)

    def _silu_mul_fused(self, gate_up, N, gpu_out: bool = False):
        """Fused SwiGLU from concatenated [gate|up] buffer.

        gate_up: (T, 2*N) — first N columns are gate, last N are up.
        Output: (T, N) = SiLU(gate) * up.
        Avoids CPU-side split of gate and up tensors.
        """
        is_gpu = isinstance(gate_up, GPUBuffer)
        if is_gpu:
            T = gate_up.shape[0] if gate_up.shape and len(gate_up.shape) > 1 else 1
        else:
            T = gate_up.shape[0]

        grid_n = (N + self._smf_block - 1) // self._smf_block
        if T > 1:
            # Multi-row: 2D grid (T, ceil(N/BLOCK))
            result = self._smfr_result
            grid = (T, grid_n)
        else:
            # Single-row: 1D grid
            result = self._smf_result
            grid = (grid_n,)

        out = self.cache.run(
            result, grid=grid,
            buffers={
                'GateUp': gate_up if is_gpu else gate_up.ravel(),
                'Out': np.zeros(T * N, dtype=np.float32),
            },
            scalars={'N': N},
            gpu_outputs={'Out'} if gpu_out else None)
        if gpu_out:
            gpu_buf = out['Out']
            gpu_buf.shape = (T, N)
            return gpu_buf
        return out['Out'].reshape(T, N)

    def _gelu_mul(self, gate, up, gpu_out: bool = False):
        """Fused GeGLU: GELU(gate) * up.

        Accepts numpy or GPUBuffer inputs.
        """
        gate_is_gpu = isinstance(gate, GPUBuffer)
        up_is_gpu = isinstance(up, GPUBuffer)
        if gate_is_gpu:
            N = gate.size // 4
            shape = gate.shape
        else:
            N = gate.size
            shape = gate.shape

        grid_size = (N + self._gm_block - 1) // self._gm_block
        out = self.cache.run(
            self._gm_result, grid=(grid_size,),
            buffers={
                'Gate': gate if gate_is_gpu else gate.ravel(),
                'Up': up if up_is_gpu else up.ravel(),
                'Out': np.zeros(N, dtype=np.float32),
            },
            scalars={'N': N},
            gpu_outputs={'Out'} if gpu_out else None)
        if gpu_out:
            gpu_buf = out['Out']
            gpu_buf.shape = shape
            return gpu_buf
        return out['Out'].reshape(shape)

    def _silu(self, x, gpu_out: bool = False):
        """SiLU (Swish) activation.

        Accepts numpy or GPUBuffer input.
        """
        x_is_gpu = isinstance(x, GPUBuffer)
        if x_is_gpu:
            N = x.size // 4
            shape = x.shape
        else:
            N = x.size
            shape = x.shape

        grid_size = (N + self._silu_block - 1) // self._silu_block
        out = self.cache.run(
            self._silu_result, grid=(grid_size,),
            buffers={
                'X': x if x_is_gpu else x.ravel(),
                'Y': np.zeros(N, dtype=np.float32),
            },
            scalars={'N': N},
            gpu_outputs={'Y'} if gpu_out else None)
        if gpu_out:
            gpu_buf = out['Y']
            gpu_buf.shape = shape
            return gpu_buf
        return out['Y'].reshape(shape)

    def _sigmoid(self, x, gpu_out: bool = False):
        """Sigmoid activation.

        Accepts numpy or GPUBuffer input.
        """
        x_is_gpu = isinstance(x, GPUBuffer)
        if x_is_gpu:
            N = x.size // 4
            shape = x.shape
        else:
            N = x.size
            shape = x.shape

        grid_size = (N + self._sig_block - 1) // self._sig_block
        out = self.cache.run(
            self._sig_result, grid=(grid_size,),
            buffers={
                'X': x if x_is_gpu else x.ravel(),
                'Y': np.zeros(N, dtype=np.float32),
            },
            scalars={'N': N},
            gpu_outputs={'Y'} if gpu_out else None)
        if gpu_out:
            gpu_buf = out['Y']
            gpu_buf.shape = shape
            return gpu_buf
        return out['Y'].reshape(shape)

    def _mul(self, a, b, gpu_out: bool = False):
        """Element-wise multiply: Out = a * b.

        Accepts numpy or GPUBuffer inputs.
        """
        a_is_gpu = isinstance(a, GPUBuffer)
        b_is_gpu = isinstance(b, GPUBuffer)
        if a_is_gpu:
            N = a.size // 4
            shape = a.shape
        elif b_is_gpu:
            N = b.size // 4
            shape = b.shape
        else:
            N = a.size
            shape = a.shape

        grid_size = (N + self._mul_block - 1) // self._mul_block
        out = self.cache.run(
            self._mul_result, grid=(grid_size,),
            buffers={
                'X': a if a_is_gpu else a.ravel(),
                'Y': b if b_is_gpu else b.ravel(),
                'Out': np.zeros(N, dtype=np.float32),
            },
            scalars={'N': N},
            gpu_outputs={'Out'} if gpu_out else None)
        if gpu_out:
            gpu_buf = out['Out']
            gpu_buf.shape = shape
            return gpu_buf
        return out['Out'].reshape(shape)

    def _add_inplace(self, x_gpu, y_gpu):
        """In-place add: x += y on GPU. Both must be GPUBuffer.

        Modifies x_gpu in-place, avoiding buffer allocation conflicts
        that arise with the toggle-pool in residual connections.
        """
        elem_bytes = 2 if self.fp16_act else 4
        N = x_gpu.size // elem_bytes
        kern = self._add_ip_fp16_result if self.fp16_act else self._add_ip_result
        grid_size = (N + self._add_block - 1) // self._add_block
        self.cache.run(
            kern, grid=(grid_size,),
            buffers={'X': x_gpu, 'Y': y_gpu},
            scalars={'N': N},
            gpu_outputs={'X'})
        return x_gpu

    def _mod_scale_shift(self, x_gpu, scale_gpu, shift_gpu, D: int,
                         gpu_out: bool = False):
        """Modulation: Out = (1 + scale) * x + shift on GPU.

        x_gpu: (T, D) GPUBuffer. scale_gpu, shift_gpu: (D,) GPUBuffer.
        Broadcasts scale/shift over rows.
        """
        N = x_gpu.size // 4  # fp32 elements
        grid_size = (N + self._add_block - 1) // self._add_block
        out = self.cache.run(
            self._mod_ss_result, grid=(grid_size,),
            buffers={
                'X': x_gpu,
                'Scale': scale_gpu,
                'Shift': shift_gpu,
                'Out': np.zeros(N, dtype=np.float32),
            },
            scalars={'D': D, 'N': N},
            gpu_outputs={'Out'} if gpu_out else None)
        if gpu_out:
            gpu_buf = out['Out']
            gpu_buf.shape = x_gpu.shape
            return gpu_buf
        return out['Out'].reshape(x_gpu.shape)

    def _gate_residual_add(self, residual_gpu, gate_gpu, x_gpu, D: int):
        """Fused gated residual: residual += gate * x on GPU.

        residual_gpu: (T, D) GPUBuffer (modified in-place).
        gate_gpu: (D,) GPUBuffer (broadcast over rows).
        x_gpu: (T, D) GPUBuffer.
        """
        N = residual_gpu.size // 4  # fp32 elements
        grid_size = (N + self._add_block - 1) // self._add_block
        self.cache.run(
            self._gate_res_result, grid=(grid_size,),
            buffers={
                'Residual': residual_gpu,
                'Gate': gate_gpu,
                'X': x_gpu,
            },
            scalars={'D': D, 'N': N},
            gpu_outputs={'Residual'})
        return residual_gpu

    _concat_counter = 0

    def _concat_gpu(self, a_gpu, b_gpu, gpu_out: bool = True):
        """Concatenate two GPUBuffers along axis 0: Out = [A; B].

        a_gpu, b_gpu: GPUBuffer (flattened).
        Returns GPUBuffer with combined elements.

        Uses uniquely-named output buffers to avoid the toggle pool's
        2-slot aliasing when >2 concat outputs coexist (e.g., Q, K, V).
        """
        N_a = a_gpu.size // 4
        N_b = b_gpu.size // 4
        N_total = N_a + N_b
        buf_size = N_total * 4  # bytes
        grid_size = (N_total + self._add_block - 1) // self._add_block

        # Allocate uniquely-named buffer via internal cache (no data write)
        storage_usage = 0x00000080 | 0x00000004 | 0x00000008  # STORAGE|COPY_SRC|COPY_DST
        WebGPUModel._concat_counter += 1
        buf_name = f"__concat_{WebGPUModel._concat_counter}_{buf_size}"
        out_handle = self.cache.runner._get_or_create_buffer(
            buf_name, buf_size, storage_usage)
        out_buf = GPUBuffer(self.cache.runner, out_handle, buf_size,
                           np.float32, owned=False)

        out = self.cache.run(
            self._concat_result, grid=(grid_size,),
            buffers={
                'A': a_gpu, 'B': b_gpu,
                'Out': out_buf,
            },
            scalars={'N_a': N_a, 'N_total': N_total},
            gpu_outputs={'Out'} if gpu_out else None)
        if gpu_out:
            return out['Out']
        return out['Out']

    _split_counter = 0

    def _split_gpu(self, src_gpu, offset_elems: int, n_elems: int):
        """Extract a slice from a GPUBuffer: Dst = Src[offset:offset+N].

        Returns a new GPUBuffer of size n_elems.
        """
        buf_size = n_elems * 4
        grid_size = (n_elems + self._add_block - 1) // self._add_block

        storage_usage = 0x00000080 | 0x00000004 | 0x00000008
        WebGPUModel._split_counter += 1
        buf_name = f"__split_{WebGPUModel._split_counter}_{buf_size}"
        dst_handle = self.cache.runner._get_or_create_buffer(
            buf_name, buf_size, storage_usage)
        dst_buf = GPUBuffer(self.cache.runner, dst_handle, buf_size,
                           np.float32, owned=False)

        out = self.cache.run(
            self._split_result, grid=(grid_size,),
            buffers={
                'Src': src_gpu,
                'Dst': dst_buf,
            },
            scalars={'src_offset': offset_elems, 'N': n_elems},
            gpu_outputs={'Dst'})
        return out['Dst']

    def _add_scaled(self, acc_gpu, x_gpu, alpha: float):
        """Acc += alpha * X on GPU (AXPY). Both must be GPUBuffer."""
        N = acc_gpu.size // 4  # f32 elements
        grid_size = (N + self._add_scaled_block - 1) // self._add_scaled_block
        self.cache.run(
            self._add_scaled_result, grid=(grid_size,),
            buffers={'Acc': acc_gpu, 'X': x_gpu},
            scalars={'alpha': np.float32(alpha), 'N': N},
            gpu_outputs={'Acc'})
        return acc_gpu

    def _add(self, a, b, gpu_out: bool = False):
        """Element-wise add: Out = a + b.

        Accepts numpy or GPUBuffer inputs.
        """
        a_is_gpu = isinstance(a, GPUBuffer)
        b_is_gpu = isinstance(b, GPUBuffer)
        if self.fp16_act:
            kern = self._add_fp16_result
            out_dt = np.float16
            elem_bytes = 2
        else:
            kern = self._add_result
            out_dt = np.float32
            elem_bytes = 4
        if a_is_gpu:
            N = a.size // elem_bytes
            shape = a.shape
        elif b_is_gpu:
            N = b.size // elem_bytes
            shape = b.shape
        else:
            N = a.size
            shape = a.shape

        grid_size = (N + self._add_block - 1) // self._add_block
        out = self.cache.run(
            kern, grid=(grid_size,),
            buffers={
                'X': a if a_is_gpu else a.ravel(),
                'Y': b if b_is_gpu else b.ravel(),
                'Out': np.zeros(N, dtype=out_dt),
            },
            scalars={'N': N},
            gpu_outputs={'Out'} if gpu_out else None)
        if gpu_out:
            gpu_buf = out['Out']
            gpu_buf.shape = shape
            return gpu_buf
        return out['Out'].reshape(shape)

    def _apply_rope(self, x: np.ndarray, positions: np.ndarray) -> np.ndarray:
        """Apply Rotary Position Embeddings (RoPE).

        x: (T, n_heads, head_dim)
        positions: (T,) integer position indices

        Uses non-interleaved rotation: first half / second half of head_dim.
        """
        T, n_heads, HD = x.shape
        half = HD // 2

        inv_freq = 1.0 / (self.rope_theta ** (
            np.arange(0, HD, 2, dtype=np.float32) / HD))
        angles = positions[:, None].astype(np.float32) * inv_freq[None, :]
        cos_vals = np.cos(angles)[:, None, :]  # (T, 1, half)
        sin_vals = np.sin(angles)[:, None, :]  # (T, 1, half)

        x1 = x[..., :half]
        x2 = x[..., half:]

        out = np.empty_like(x)
        out[..., :half] = x1 * cos_vals - x2 * sin_vals
        out[..., half:] = x2 * cos_vals + x1 * sin_vals
        return out

    def _causal_attention(self, q: np.ndarray, k: np.ndarray,
                          v: np.ndarray) -> np.ndarray:
        """Single-head causal attention: (T, HD) -> (T, HD)."""
        T, HD = q.shape
        BHD = self._attn_bhd
        scale = float(1.0 / np.sqrt(HD))

        if HD < BHD:
            q_pad = np.zeros((T, BHD), dtype=np.float32)
            q_pad[:, :HD] = q
            k_pad = np.zeros((T, BHD), dtype=np.float32)
            k_pad[:, :HD] = k
            v_pad = np.zeros((T, BHD), dtype=np.float32)
            v_pad[:, :HD] = v
        else:
            q_pad, k_pad, v_pad = q, k, v

        out = self.cache.run(
            self._attn_result, grid=(T,),
            buffers={
                'Q': q_pad.ravel(), 'K': k_pad.ravel(), 'V': v_pad.ravel(),
                'Out': np.zeros(T * BHD, dtype=np.float32),
            },
            scalars={
                'stride_q': BHD, 'stride_k': BHD,
                'stride_v': BHD, 'stride_o': BHD,
                'seq_len': T, 'scale': scale,
                'neg_inf': float(-1e9),
            })
        return out['Out'].reshape(T, BHD)[:, :HD]

    def _causal_attention_multihead(self, Q: np.ndarray, K: np.ndarray,
                                    V: np.ndarray, n_rep: int) -> np.ndarray:
        """Multi-head causal attention with GQA: all heads in one dispatch.

        Q: (T, n_head, HD), K: (T, n_kv, HD), V: (T, n_kv, HD)
        Returns: (T, n_head, HD)
        """
        T, n_head, HD = Q.shape
        n_kv = K.shape[1]
        BHD = self._attn_bhd
        scale = float(1.0 / np.sqrt(HD))

        Q_c = np.ascontiguousarray(Q, dtype=np.float32)
        K_c = np.ascontiguousarray(K, dtype=np.float32)
        V_c = np.ascontiguousarray(V, dtype=np.float32)

        out = self.cache.run(
            self._mh_attn_result, grid=(T, n_head),
            buffers={
                'Q': Q_c.ravel(), 'K': K_c.ravel(), 'V': V_c.ravel(),
                'Out': np.zeros(T * n_head * BHD, dtype=np.float32),
            },
            scalars={
                'stride_q_t': n_head * BHD, 'stride_q_h': BHD,
                'stride_k_t': n_kv * BHD, 'stride_k_h': BHD,
                'stride_v_t': n_kv * BHD, 'stride_v_h': BHD,
                'stride_o_t': n_head * BHD, 'stride_o_h': BHD,
                'n_rep': n_rep, 'scale': scale,
                'neg_inf': float(-1e9),
            })
        return out['Out'].reshape(T, n_head, BHD)[:, :, :HD]

    def _full_attention(self, q: np.ndarray, k: np.ndarray,
                        v: np.ndarray) -> np.ndarray:
        """Single-head non-causal (full) attention: (T, HD) -> (T, HD).

        Attends to ALL positions with no causal mask.
        Used by ViT/SAM/SDXL encoder blocks.
        """
        T, HD = q.shape
        BHD = self._full_attn_bhd
        scale = float(1.0 / np.sqrt(HD))

        if HD < BHD:
            q_pad = np.zeros((T, BHD), dtype=np.float32)
            q_pad[:, :HD] = q
            k_pad = np.zeros((T, BHD), dtype=np.float32)
            k_pad[:, :HD] = k
            v_pad = np.zeros((T, BHD), dtype=np.float32)
            v_pad[:, :HD] = v
        else:
            q_pad, k_pad, v_pad = q, k, v

        out = self.cache.run(
            self._full_attn_result, grid=(T,),
            buffers={
                'Q': q_pad.ravel(), 'K': k_pad.ravel(), 'V': v_pad.ravel(),
                'Out': np.zeros(T * BHD, dtype=np.float32),
            },
            scalars={
                'stride_q': BHD, 'stride_k': BHD,
                'stride_v': BHD, 'stride_o': BHD,
                'seq_len': T, 'scale': scale,
                'neg_inf': float(-1e9),
            })
        return out['Out'].reshape(T, BHD)[:, :HD]

    def _full_attention_multihead(self, q, k, v, n_head: int,
                                  gpu_out: bool = False):
        """Multi-head non-causal (full) attention.

        q: (T, n_head, HD) numpy or GPUBuffer
        k: (T, n_head, HD) numpy or GPUBuffer
        v: (T, n_head, HD) numpy or GPUBuffer
        Returns: (T, n_head, HD)

        Used by DiT blocks that need full bidirectional attention.
        """
        q_is_gpu = isinstance(q, GPUBuffer)

        if q_is_gpu:
            T = q.shape[0]
            HD = q.shape[2] if len(q.shape) == 3 else q.shape[-1] // n_head
        elif q.ndim == 2:
            HD = q.shape[-1] // n_head
            T = q.shape[0]
        else:
            T, _, HD = q.shape
        BHD = self._full_attn_bhd

        if q_is_gpu:
            # GPUBuffer path: already (T, n_head, HD) on GPU, no padding needed
            q_buf, k_buf, v_buf = q, k, v
        else:
            q_c = q.reshape(T, n_head, HD)
            k_c = k.reshape(T, n_head, HD)
            v_c = v.reshape(T, n_head, HD)

            pad_dt = np.float16 if self.fp16_act else np.float32
            if HD < BHD:
                q_pad = np.zeros((T, n_head, BHD), dtype=pad_dt)
                q_pad[:, :, :HD] = q_c
                k_pad = np.zeros((T, n_head, BHD), dtype=pad_dt)
                k_pad[:, :, :HD] = k_c
                v_pad = np.zeros((T, n_head, BHD), dtype=pad_dt)
                v_pad[:, :, :HD] = v_c
            else:
                if self.fp16_act:
                    q_pad = q_c.astype(np.float16)
                    k_pad = k_c.astype(np.float16)
                    v_pad = v_c.astype(np.float16)
                else:
                    q_pad, k_pad, v_pad = q_c, k_c, v_c
            q_buf, k_buf, v_buf = q_pad.ravel(), k_pad.ravel(), v_pad.ravel()

        if self.fp16_act:
            kern = self._full_attn_mh_fp16_result
            out_dt = np.float16
        else:
            kern = self._full_attn_mh_result
            out_dt = np.float32

        scale = float(1.0 / np.sqrt(HD))
        out = self.cache.run(
            kern, grid=(T, n_head),
            buffers={
                'Q': q_buf, 'K': k_buf, 'V': v_buf,
                'Out': np.zeros(T * n_head * BHD, dtype=out_dt),
            },
            scalars={
                'stride_q_t': n_head * BHD, 'stride_q_h': BHD,
                'stride_k_t': n_head * BHD, 'stride_k_h': BHD,
                'stride_v_t': n_head * BHD, 'stride_v_h': BHD,
                'stride_o_t': n_head * BHD, 'stride_o_h': BHD,
                'seq_len': T, 'scale': scale,
                'neg_inf': float(-1e9),
            },
            gpu_outputs={'Out'} if gpu_out else None)
        if gpu_out:
            gpu_buf = out['Out']
            gpu_buf.shape = (T, n_head, BHD)
            return gpu_buf
        return out['Out'].reshape(T, n_head, BHD)[:, :, :HD]

    def _qk_norm_rope(self, qkv_gpu, norm_q_w, norm_k_w,
                       cos, sin, n_head: int, T: int,
                       eps: float = 1e-6,
                       gpu_out: bool = False):
        """Fused per-head QK RMSNorm + FLUX-style RoPE on GPU.

        qkv_gpu: GPUBuffer of shape (T, 3*n_head*HD) — fused QKV output.
        norm_q_w, norm_k_w: (HD,) norm weights (numpy or GPUBuffer).
        cos, sin: (T, HD) numpy arrays, pre-computed RoPE tables.
        Returns: (q, k, v) as GPUBuffers or numpy arrays of shape (T, n_head, HD).
        """
        HD = self.head_dim
        stride_t = 3 * n_head * HD
        total = T * n_head * HD
        if self.fp16_act:
            kern = self._qknr_fp16_result
            out_dt = np.float16
        else:
            kern = self._qknr_result
            out_dt = np.float32
        out = self.cache.run(
            kern, grid=(T, n_head),
            buffers={
                'QKV': qkv_gpu,
                'Q_out': np.zeros(total, dtype=out_dt),
                'K_out': np.zeros(total, dtype=out_dt),
                'V_out': np.zeros(total, dtype=out_dt),
                'NormQ': norm_q_w if isinstance(norm_q_w, GPUBuffer) else norm_q_w.astype(np.float32),
                'NormK': norm_k_w if isinstance(norm_k_w, GPUBuffer) else norm_k_w.astype(np.float32),
                'Cos': cos.astype(np.float32).ravel(),
                'Sin': sin.astype(np.float32).ravel(),
            },
            scalars={
                'n_head': n_head, 'stride_t': stride_t, 'eps': float(eps),
            },
            gpu_outputs={'Q_out', 'K_out', 'V_out'} if gpu_out else None)
        if gpu_out:
            for k in ('Q_out', 'K_out', 'V_out'):
                out[k].shape = (T, n_head, HD)
            return out['Q_out'], out['K_out'], out['V_out']
        return (out['Q_out'].reshape(T, n_head, HD),
                out['K_out'].reshape(T, n_head, HD),
                out['V_out'].reshape(T, n_head, HD))

    def _apply_partial_rope(self, x: np.ndarray, positions: np.ndarray,
                            rotary_dim: int) -> np.ndarray:
        """Apply Rotary Position Embeddings to only part of head_dim.

        x: (T, n_heads, head_dim)
        positions: (T,) integer position indices
        rotary_dim: number of dimensions to rotate (must be even)

        The first rotary_dim dimensions get RoPE, the rest pass through.
        Used by Phi-3/Phi-4 models with partial_rotary_factor.
        """
        T, n_heads, HD = x.shape
        half_rot = rotary_dim // 2

        inv_freq = 1.0 / (self.rope_theta ** (
            np.arange(0, rotary_dim, 2, dtype=np.float32) / rotary_dim))
        angles = positions[:, None].astype(np.float32) * inv_freq[None, :]
        cos_vals = np.cos(angles)[:, None, :]  # (T, 1, half_rot)
        sin_vals = np.sin(angles)[:, None, :]  # (T, 1, half_rot)

        x_rot = x[..., :rotary_dim]
        x_pass = x[..., rotary_dim:]

        x1 = x_rot[..., :half_rot]
        x2 = x_rot[..., half_rot:]

        out = np.empty_like(x)
        out[..., :half_rot] = x1 * cos_vals - x2 * sin_vals
        out[..., half_rot:rotary_dim] = x2 * cos_vals + x1 * sin_vals
        out[..., rotary_dim:] = x_pass
        return out

    def _group_norm(self, x, w, b, num_groups: int, num_channels: int,
                    eps: float = 1e-5, gpu_out: bool = False):
        """GroupNorm: normalize within each group of channels.

        x: (N_batch * spatial, num_channels)
        w, b: (num_channels,)
        """
        if isinstance(x, GPUBuffer):
            total_elems = x.size // 4
            num_rows = total_elems // num_channels
        else:
            num_rows = x.shape[0]

        channels_per_group = num_channels // num_groups
        total_groups = num_rows * num_groups

        out = self.cache.run(
            self._gn_result, grid=(total_groups,),
            buffers={
                'X': x if isinstance(x, GPUBuffer) else x.ravel(),
                'Y': np.zeros(num_rows * num_channels, dtype=np.float32),
                'W': w, 'B': b,
                'Rstd': np.zeros(total_groups, dtype=np.float32),
            },
            scalars={
                'stride': num_channels,
                'N': channels_per_group,
                'num_groups': num_groups,
                'eps': eps,
            },
            gpu_outputs={'Y'} if gpu_out else None)
        if gpu_out:
            gpu_buf = out['Y']
            gpu_buf.shape = (num_rows, num_channels)
            return gpu_buf
        return out['Y'].reshape(num_rows, num_channels)

    # --- MXFP4 matmul (for GPT-OSS MoE experts) ---

    def _compile_mxfp4_kernels(self):
        """Compile MXFP4 matmul and GPT-OSS gated activation kernels."""
        MXFP4_BLOCK_K = 32
        _nw = lambda bk: max(1, min(bk // 32, self.MAX_WG_THREADS // 32))

        self._linear_mxfp4_sig = {
            'X': '*fp32', 'W_blocks': '*i32',
            'W_scales': '*i32', 'Bias': '*fp32', 'Y': '*fp32',
            'K': 'i32', 'stride_x': 'i32',
            'stride_blocks': 'i32', 'stride_scales': 'i32',
            'N': 'i32', 'BLOCK_K': 'constexpr',
        }
        self._linear_mxfp4_result = self.cache.get_or_compile(
            linear_mxfp4_kernel, self._linear_mxfp4_sig,
            {'BLOCK_K': MXFP4_BLOCK_K}, num_warps=_nw(MXFP4_BLOCK_K))

        GATE_BLOCK = self.LOOP_BLOCK
        self._gptoss_gate_sig = {
            'X': '*fp32', 'Y': '*fp32',
            'N': 'i32', 'BLOCK': 'constexpr',
        }
        self._gptoss_gate_result = self.cache.get_or_compile(
            gptoss_gate_kernel, self._gptoss_gate_sig,
            {'BLOCK': GATE_BLOCK}, num_warps=_nw(GATE_BLOCK))
        self._gptoss_gate_block = GATE_BLOCK

    def _linear_mxfp4(self, x, w_blocks, w_scales, bias,
                       out_features: int, K: int,
                       gpu_out: bool = False):
        """Linear projection with MXFP4 packed weights (fused dequant).

        w_blocks: GPUBuffer of packed i32 FP4 data, shape (N, K//8)
        w_scales: GPUBuffer of packed i32 E8M0 scales, shape (N, ceil(K_blocks/4))
        """
        N = out_features
        x_is_gpu = isinstance(x, GPUBuffer)
        T = (x.shape[0] if x.shape else 1) if x_is_gpu else x.shape[0]
        stride_blocks = K // 8
        n_mxblocks = K // 32
        stride_scales = (n_mxblocks + 3) // 4

        out = self.cache.run(
            self._linear_mxfp4_result, grid=(T, N),
            buffers={
                'X': x if x_is_gpu else x.ravel(),
                'W_blocks': w_blocks,
                'W_scales': w_scales,
                'Bias': bias,
                'Y': np.zeros(T * N, dtype=np.float32),
            },
            scalars={'K': K, 'stride_x': K,
                     'stride_blocks': stride_blocks,
                     'stride_scales': stride_scales, 'N': N},
            gpu_outputs={'Y'} if gpu_out else None)

        if gpu_out:
            gpu_buf = out['Y']
            gpu_buf.shape = (T, N)
            return gpu_buf
        return out['Y'].reshape(T, N)

    def _gptoss_gate(self, gate_up, N, gpu_out: bool = False):
        """GPT-OSS gated activation (interleaved gate/up).

        gate_up: (T, 2*N) with gate=X[::2], up=X[1::2]
        Returns: (T, N)
        """
        is_gpu = isinstance(gate_up, GPUBuffer)
        T = (gate_up.shape[0] if gate_up.shape and len(gate_up.shape) > 1
             else 1) if is_gpu else gate_up.shape[0]

        grid_n = (N + self._gptoss_gate_block - 1) // self._gptoss_gate_block

        out = self.cache.run(
            self._gptoss_gate_result, grid=(grid_n,),
            buffers={
                'X': gate_up if is_gpu else gate_up.ravel(),
                'Y': np.zeros(T * N, dtype=np.float32),
            },
            scalars={'N': N},
            gpu_outputs={'Y'} if gpu_out else None)

        if gpu_out:
            gpu_buf = out['Y']
            gpu_buf.shape = (T, N)
            return gpu_buf
        return out['Y'].reshape(T, N)
