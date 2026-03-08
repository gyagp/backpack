"""Stage 1: Model Compiler — Generates WGSL shaders + model manifest.

This script compiles all Triton kernels to WGSL and exports them alongside
hand-written WGSL kernels into a portable bundle. The bundle can be consumed
by any Stage 2 runtime (Python, C++, JavaScript).

Usage:
    python models/compile_model.py --model qwen3-1.7B --gguf-file path/to/model.gguf --output-dir build/qwen3-1.7B

Output:
    build/qwen3-1.7B/
        manifest.json          — model config, kernel list, dispatch plans
        kernels/
            rms_norm.wgsl
            q8_matmul.wgsl
            q8_matmul_add.wgsl
            fused_qknorm_rope.wgsl
            gqa_chunked_pass1.wgsl
            gqa_chunked_pass2.wgsl
            silu_mul_fused.wgsl
            add_rms_norm.wgsl
            add_inplace.wgsl
            fp16_gemm.wgsl
            causal_attn_multihead.wgsl
            subgroup_matrix_q8.wgsl
            qknorm_rope_prefill.wgsl
            argmax.wgsl
"""
import os
import sys
import json
import time
import argparse
import hashlib

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _SCRIPT_DIR)

import numpy as np


def compile_triton_kernel(kernel_fn, signature, constexprs, num_warps,
                          has_subgroups=True):
    """Compile a Triton kernel to WGSL and return (wgsl_code, buffer_bindings, param_fields, workgroup_size)."""
    import triton
    from triton.compiler import ASTSource
    from triton.backends.compiler import GPUTarget
    from triton.backends.webgpu.llvm_to_wgsl import translate_llvm_to_wgsl

    target = GPUTarget("webgpu", 0, 32)
    src = ASTSource(fn=kernel_fn, signature=signature, constexprs=constexprs)
    compiled = triton.compile(src, target=target, options={'num_warps': num_warps})
    sig_no_ce = {k: v for k, v in signature.items() if v != 'constexpr'}
    result = translate_llvm_to_wgsl(
        compiled.asm['llir'], sig_no_ce,
        num_warps=num_warps, warp_size=32,
        use_native_subgroups=has_subgroups)
    return result


def export_kernel(name, wgsl_code, buffer_bindings, param_fields,
                  workgroup_size, output_dir):
    """Save a WGSL kernel to disk and return its manifest entry."""
    kernels_dir = os.path.join(output_dir, "kernels")
    os.makedirs(kernels_dir, exist_ok=True)

    filename = f"{name}.wgsl"
    filepath = os.path.join(kernels_dir, filename)
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(wgsl_code)

    # Serialize bindings
    bindings = []
    for bb in buffer_bindings:
        bindings.append({
            "name": bb.name,
            "binding": bb.binding,
            "access": bb.access,
            "elem_type": bb.elem_type,
        })

    # Serialize param fields
    params = []
    for pf in (param_fields or []):
        params.append({
            "name": pf.name,
            "wgsl_type": pf.wgsl_type,
            "byte_size": pf.byte_size,
        })

    return {
        "name": name,
        "file": f"kernels/{filename}",
        "workgroup_size": workgroup_size,
        "bindings": bindings,
        "params": params,
        "wgsl_hash": hashlib.sha256(wgsl_code.encode()).hexdigest()[:16],
    }


def compile_all_kernels(output_dir, has_subgroups=True):
    """Compile all Triton kernels and export hand-written WGSL kernels."""
    from common.kernels import (
        rms_norm_loop_kernel, add_rms_norm_loop_kernel,
        silu_mul_fused_kernel, argmax_kernel,
        causal_attn_multihead_kernel,
        fused_qknorm_rope_qkv_kernel,
        qknorm_rope_prefill_kernel,
        gqa_decode_attn_kernel,
        add_inplace_kernel,
    )
    from common.wgsl_kernels import (
        WSGL_Q8_0_KERNEL, Q8_DP4A_BINDINGS,
        WSGL_Q8_0_ADD_KERNEL, Q8_ADD_BINDINGS,
        WGSL_FP16_GEMM_KERNEL, FP16_GEMM_BINDINGS,
        WGSL_GQA_CHUNKED_PASS1, GQA_CHUNKED_PASS1_BINDINGS,
        WGSL_GQA_CHUNKED_PASS2, GQA_CHUNKED_PASS2_BINDINGS,
        WGSL_SUBGROUP_MATRIX_Q8_KERNEL, SUBGROUP_MATRIX_Q8_BINDINGS,
        WSGL_SUBGROUP_MATRIX_Q8_ADD_KERNEL, SUBGROUP_MATRIX_Q8_ADD_BINDINGS,
    )

    BLOCK = 128
    HD = 128
    nw = lambda bs: max(1, bs // 32)

    kernel_entries = []

    # --- Triton-compiled kernels ---
    print("  Compiling Triton kernels to WGSL...")

    triton_kernels = [
        ("rms_norm", rms_norm_loop_kernel,
         {'X': '*fp32', 'Y': '*fp32', 'W': '*fp32', 'Rstd': '*fp32',
          'stride': 'i32', 'N': 'i32', 'eps': 'fp32', 'BLOCK': 'constexpr'},
         {'BLOCK': BLOCK}, nw(BLOCK)),
        ("add_rms_norm", add_rms_norm_loop_kernel,
         {'X': '*fp32', 'Residual': '*fp32', 'Y': '*fp32', 'W': '*fp32',
          'Rstd': '*fp32', 'stride': 'i32', 'N': 'i32', 'eps': 'fp32',
          'BLOCK': 'constexpr'},
         {'BLOCK': BLOCK}, nw(BLOCK)),
        ("silu_mul_fused", silu_mul_fused_kernel,
         {'GateUp': '*fp32', 'Out': '*fp32', 'N': 'i32', 'BLOCK': 'constexpr'},
         {'BLOCK': BLOCK}, nw(BLOCK)),
        ("argmax", argmax_kernel,
         {'Logits': '*fp32', 'TokenOut': '*i32', 'N': 'i32',
          'BLOCK': 'constexpr'},
         {'BLOCK': BLOCK}, nw(BLOCK)),
        ("causal_attn_multihead", causal_attn_multihead_kernel,
         {'Q': '*fp32', 'K': '*fp32', 'V': '*fp32', 'Out': '*fp32',
          'stride_q_t': 'i32', 'stride_q_h': 'i32',
          'stride_k_t': 'i32', 'stride_k_h': 'i32',
          'stride_v_t': 'i32', 'stride_v_h': 'i32',
          'stride_o_t': 'i32', 'stride_o_h': 'i32',
          'n_rep': 'i32', 'scale': 'fp32', 'neg_inf': 'fp32',
          'BLOCK_HD': 'constexpr'},
         {'BLOCK_HD': HD}, nw(HD)),
        ("fused_qknorm_rope", fused_qknorm_rope_qkv_kernel,
         {'QKV': '*fp32', 'Q_out': '*fp32', 'K_cache': '*fp32',
          'V_cache': '*fp32', 'CosTable': '*fp32', 'SinTable': '*fp32',
          'NormQ': '*fp32', 'NormK': '*fp32',
          'n_head': 'i32', 'q_size': 'i32', 'kv_size': 'i32',
          'pos': 'i32', 'half_rot': 'i32', 'cache_offset': 'i32',
          'eps': 'fp32', 'BLOCK_HD': 'constexpr'},
         {'BLOCK_HD': HD}, nw(HD)),
        ("qknorm_rope_prefill", qknorm_rope_prefill_kernel,
         {'QKV': '*fp32', 'Q_out': '*fp32', 'K_out': '*fp32',
          'V_out': '*fp32', 'CosTable': '*fp32', 'SinTable': '*fp32',
          'NormQ': '*fp32', 'NormK': '*fp32',
          'n_head': 'i32', 'q_size': 'i32', 'kv_size': 'i32',
          'qkv_stride_t': 'i32', 'q_stride_t': 'i32',
          'kv_stride_t': 'i32', 'half_rot': 'i32', 'eps': 'fp32',
          'BLOCK_HD': 'constexpr'},
         {'BLOCK_HD': HD}, nw(HD)),
        ("add_inplace", add_inplace_kernel,
         {'X': '*fp32', 'Y': '*fp32', 'N': 'i32', 'BLOCK': 'constexpr'},
         {'BLOCK': BLOCK}, nw(BLOCK)),
    ]

    for name, fn, sig, ce, nwarps in triton_kernels:
        t0 = time.perf_counter()
        result = compile_triton_kernel(fn, sig, ce, nwarps, has_subgroups)
        t1 = time.perf_counter()
        entry = export_kernel(
            name, result.wgsl, result.buffer_bindings,
            result.param_fields, result.workgroup_size, output_dir)
        kernel_entries.append(entry)
        print(f"    {name}: {(t1-t0)*1000:.0f}ms, wg={result.workgroup_size}")

    # --- Hand-written WGSL kernels ---
    print("  Exporting hand-written WGSL kernels...")

    wgsl_kernels = [
        ("q8_matmul", WSGL_Q8_0_KERNEL, Q8_DP4A_BINDINGS, [], 256),
        ("q8_matmul_add", WSGL_Q8_0_ADD_KERNEL, Q8_ADD_BINDINGS, [], 256),
        ("fp16_gemm", WGSL_FP16_GEMM_KERNEL, FP16_GEMM_BINDINGS, [], 256),
        ("gqa_chunked_pass1", WGSL_GQA_CHUNKED_PASS1,
         GQA_CHUNKED_PASS1_BINDINGS, [], 32),
        ("gqa_chunked_pass2", WGSL_GQA_CHUNKED_PASS2,
         GQA_CHUNKED_PASS2_BINDINGS, [], 32),
        ("subgroup_matrix_q8", WGSL_SUBGROUP_MATRIX_Q8_KERNEL,
         SUBGROUP_MATRIX_Q8_BINDINGS, [], 128),
        ("subgroup_matrix_q8_add", WSGL_SUBGROUP_MATRIX_Q8_ADD_KERNEL,
         SUBGROUP_MATRIX_Q8_ADD_BINDINGS, [], 128),
    ]

    for name, wgsl, bindings, params, wg_size in wgsl_kernels:
        entry = export_kernel(name, wgsl, bindings, params, wg_size, output_dir)
        kernel_entries.append(entry)
        print(f"    {name}: wg={wg_size}")

    return kernel_entries


def build_decode_plan(model_config):
    """Build the per-layer dispatch plan for fast decode.

    Returns a list of dispatch descriptors that the C++ runtime can execute
    without any Python-side logic.
    """
    E = model_config["n_embd"]
    n_head = model_config["n_head"]
    n_kv = model_config["n_kv_heads"]
    HD = model_config["head_dim"]
    IM = model_config["intermediate_size"]
    n_layer = model_config["n_layer"]
    n_vocab = model_config["n_vocab"]

    q_dim = n_head * HD
    kv_dim = n_kv * HD
    qkv_out = q_dim + 2 * kv_dim
    Q8_TILE_N = 8
    GQA_CHUNK_SIZE = 64
    max_seq = 2048

    max_n_chunks = (max_seq + GQA_CHUNK_SIZE - 1) // GQA_CHUNK_SIZE

    # Buffer layout: each buffer has a name and byte size
    buffers = {
        "x": E * 4,
        "norm_out": E * 4,
        "qkv_out": qkv_out * 4,
        "q_rot": q_dim * 4,
        "attn_out": q_dim * 4,
        "proj_out": E * 4,
        "gate_up": 2 * IM * 4,
        "silu_out": IM * 4,
        "rstd": 16,
        "logits": n_vocab * 4,
        "attn_partials": n_head * max_n_chunks * (HD + 2) * 4,
    }

    # Per-layer dispatches (for layer i, not first or last)
    layer_dispatches = [
        {"kernel": "q8_matmul", "grid": [1, (qkv_out + Q8_TILE_N - 1) // Q8_TILE_N],
         "inputs": ["norm_out"], "weights": ["qkv_proj"], "output": "qkv_out",
         "params_type": "q8", "K": E, "N": qkv_out},
        {"kernel": "fused_qknorm_rope", "grid_expr": "n_head + n_kv",
         "inputs": ["qkv_out"], "output": "q_rot",
         "kv_cache": True},
        {"kernel": "gqa_chunked_pass1", "grid": ["n_head", "max_n_chunks"],
         "inputs": ["q_rot"], "kv_cache": True, "output": "attn_partials"},
        {"kernel": "gqa_chunked_pass2", "grid": ["n_head"],
         "inputs": ["attn_partials"], "output": "attn_out"},
        {"kernel": "q8_matmul", "grid": [1, (E + Q8_TILE_N - 1) // Q8_TILE_N],
         "inputs": ["attn_out"], "weights": ["o_proj"], "output": "proj_out",
         "params_type": "q8", "K": q_dim, "N": E},
        {"kernel": "add_rms_norm", "grid": [1],
         "inputs": ["x", "proj_out"], "output": "norm_out",
         "weights": ["post_attn_norm"]},
        {"kernel": "q8_matmul", "grid": [1, (2 * IM + Q8_TILE_N - 1) // Q8_TILE_N],
         "inputs": ["norm_out"], "weights": ["gate_up_proj"], "output": "gate_up",
         "params_type": "q8", "K": E, "N": 2 * IM},
        {"kernel": "silu_mul_fused", "grid": [(IM + 128 - 1) // 128],
         "inputs": ["gate_up"], "output": "silu_out"},
        {"kernel": "q8_matmul_add", "grid": [1, (E + Q8_TILE_N - 1) // Q8_TILE_N],
         "inputs": ["silu_out"], "weights": ["down_proj"], "output": "x",
         "params_type": "q8", "K": IM, "N": E},
    ]

    return {
        "buffers": buffers,
        "n_layers": n_layer,
        "layer_dispatches": layer_dispatches,
        "first_layer_prefix": [
            {"kernel": "rms_norm", "grid": [1],
             "inputs": ["x"], "output": "norm_out",
             "weights": ["input_norm"]},
        ],
        "inter_layer_suffix": [
            {"kernel": "rms_norm", "grid": [1],
             "inputs": ["x"], "output": "norm_out",
             "weights": ["next_input_norm"]},
        ],
        "final_dispatches": [
            {"kernel": "rms_norm", "grid": [1],
             "inputs": ["x"], "output": "norm_out",
             "weights": ["final_norm"]},
            {"kernel": "fp16_gemm",
             "grid_expr": "lm_head_grid",
             "inputs": ["norm_out"], "weights": ["lm_head"],
             "output": "logits"},
        ],
        "gqa_chunk_size": GQA_CHUNK_SIZE,
        "max_seq_len": max_seq,
        "max_n_chunks": max_n_chunks,
    }


def compile_model(model_name, gguf_path, output_dir):
    """Main entry: compile a model and export the kernel bundle."""
    print(f"Compiling model: {model_name}")

    if model_name.startswith("qwen3"):
        from importlib.util import spec_from_file_location, module_from_spec
        spec = spec_from_file_location("qm",
            os.path.join(_SCRIPT_DIR, "qwen-3-1.7B", "model.py"))
        qmodel = module_from_spec(spec)
        spec.loader.exec_module(qmodel)
        model_config = dict(qmodel.QWEN_CONFIGS["1.7B"])
        model_config.pop("hf_repo", None)
        model_config["model_type"] = "qwen3"
        model_config["tie_word_embeddings"] = True
        model_config["attention_bias"] = False
    else:
        raise ValueError(f"Unknown model: {model_name}")

    os.makedirs(output_dir, exist_ok=True)

    # Compile kernels
    t0 = time.perf_counter()
    kernel_entries = compile_all_kernels(output_dir)
    t1 = time.perf_counter()
    print(f"  Compiled {len(kernel_entries)} kernels in {(t1-t0)*1000:.0f}ms")

    # Build decode plan
    decode_plan = build_decode_plan(model_config)

    # Build manifest
    manifest = {
        "version": 1,
        "model": model_config,
        "kernels": kernel_entries,
        "decode_plan": decode_plan,
        "gguf_file": os.path.basename(gguf_path) if gguf_path else None,
    }

    manifest_path = os.path.join(output_dir, "manifest.json")
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)
    print(f"  Manifest: {manifest_path}")

    return manifest


def main():
    parser = argparse.ArgumentParser(description="Stage 1: Compile model to WGSL bundle")
    parser.add_argument("--model", type=str, default="qwen3-1.7B",
                        help="Model name (e.g., qwen3-1.7B)")
    parser.add_argument("--gguf-file", type=str, default=None,
                        help="Path to GGUF weights file")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory for compiled bundle")
    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = os.path.join(
            os.path.dirname(_SCRIPT_DIR), "build", args.model)

    compile_model(args.model, args.gguf_file, args.output_dir)
    print("Done.")


if __name__ == "__main__":
    main()
