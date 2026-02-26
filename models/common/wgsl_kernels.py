"""WGSL INT4 matmul kernel — multi-output tiled, no shared memory.

ORT-inspired: each workgroup produces TILE_N=8 output elements,
with 256 threads (8 warps × 32). Each warp handles one output via
subgroupAdd. No shared memory for X — relies on L1/L2 cache for
X data reuse across warps (all warps read same X addresses).

The benefit over single-output (wg=128) is 8× fewer dispatches,
which reduces GPU scheduler overhead.

fp32 precision throughout.
"""

import struct
import numpy as np
from triton.backends.webgpu.dawn_runner import BufferBinding

TILE_N = 8

Q4_DP4A_BINDINGS = [
    BufferBinding(name='X', binding=0, access='read_write', elem_type='f32'),
    BufferBinding(name='W_Q4', binding=1, access='read_write', elem_type='u32'),
    BufferBinding(name='Scales', binding=2, access='read_write', elem_type='u32'),
    BufferBinding(name='Zeros', binding=3, access='read_write', elem_type='u32'),
    BufferBinding(name='Bias', binding=4, access='read_write', elem_type='f32'),
    BufferBinding(name='Y', binding=5, access='read_write', elem_type='f32'),
    BufferBinding(name='_params_', binding=6, access='read_write', elem_type='u32'),
]


def pack_dp4a_params(K, stride_w_q4, n_groups, N):
    data = struct.pack('<IIII', K, stride_w_q4, n_groups, N)
    return np.frombuffer(data, dtype=np.uint32).copy()


WGSL_Q4_DP4A_KERNEL = """
enable subgroups;

@group(0) @binding(0) var<storage, read_write> X: array<f32>;
@group(0) @binding(1) var<storage, read_write> W_Q4: array<u32>;
@group(0) @binding(2) var<storage, read_write> Scales: array<u32>;
@group(0) @binding(3) var<storage, read_write> Zeros: array<u32>;
@group(0) @binding(4) var<storage, read_write> Bias: array<f32>;
@group(0) @binding(5) var<storage, read_write> Y: array<f32>;
@group(0) @binding(6) var<storage, read_write> _params_: array<u32>;

const TILE_N: u32 = 8u;

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) lid: vec3<u32>,
        @builtin(workgroup_id) wid: vec3<u32>) {
    let row = wid.x;
    let tile_col = wid.y;
    let tid = lid.x;

    let K = _params_[0];
    let stride_w_q4 = _params_[1];
    let n_groups = _params_[2];
    let N = _params_[3];
    let x_base = row * K;

    // 8 warps × 32 threads: each warp computes one output
    let warp_id = tid / 32u;
    let lane = tid % 32u;
    let col = tile_col * TILE_N + warp_id;

    // Each lane handles 4 elements per 128-element group
    let word_in_group = lane / 2u;
    let nib_shift = (lane % 2u) * 16u;

    var acc: f32 = 0.0;

    if (col < N) {
        let w_base = col * stride_w_q4;
        let s_base = col * n_groups;

        for (var g = 0u; g < n_groups; g = g + 1u) {
            let si = s_base + g;
            let sp = unpack2x16float(Scales[si / 2u]);
            let w_scale = select(sp.x, sp.y, (si & 1u) != 0u);
            let zp = unpack2x16float(Zeros[si / 2u]);
            let w_zero = select(zp.x, zp.y, (si & 1u) != 0u);

            let k_base = g * 128u + lane * 4u;
            let x0 = X[x_base + k_base];
            let x1 = X[x_base + k_base + 1u];
            let x2 = X[x_base + k_base + 2u];
            let x3 = X[x_base + k_base + 3u];

            let packed_w = W_Q4[w_base + g * 16u + word_in_group];
            let nibbles4 = (packed_w >> nib_shift) & 0xFFFFu;

            let q0 = f32(nibbles4 & 0xFu) * w_scale + w_zero;
            let q1 = f32((nibbles4 >> 4u) & 0xFu) * w_scale + w_zero;
            let q2 = f32((nibbles4 >> 8u) & 0xFu) * w_scale + w_zero;
            let q3 = f32((nibbles4 >> 12u) & 0xFu) * w_scale + w_zero;

            acc += x0 * q0 + x1 * q1 + x2 * q2 + x3 * q3;
        }
    }

    let warp_sum = subgroupAdd(acc);

    if (lane == 0u && col < N) {
        Y[row * N + col] = warp_sum + Bias[col];
    }
}
"""


# ---------------------------------------------------------------------------
# FP16-weight GEMM kernel — multi-output tiled, subgroupAdd reduction
# ---------------------------------------------------------------------------
#
# Same structure as Q4 kernel: 256 threads (8 warps × 32), TILE_N=8
# outputs per workgroup. Each warp computes one output column via
# subgroupAdd.
#
# Weights stored as u32 (two fp16 packed per u32). Each thread
# processes 4 fp16 weight values per iteration using
# unpack2x16float() — extracting two vec2<f32> per u32 load.
#
# Key advantages over Triton-compiled linear_loop_fp16w_kernel:
# - subgroupAdd for warp reduction (no workgroup barriers)
# - 8× fewer dispatches (TILE_N=8)
# - vec4 dot product for 4 FMAs per instruction
# - unpack2x16float avoids D3D12 f16 typed buffer issue

FP16_GEMM_TILE_N = 8

FP16_GEMM_BINDINGS = [
    BufferBinding(name='X', binding=0, access='read_write', elem_type='f32'),
    BufferBinding(name='W', binding=1, access='read_write', elem_type='u32'),
    BufferBinding(name='Bias', binding=2, access='read_write', elem_type='f32'),
    BufferBinding(name='Y', binding=3, access='read_write', elem_type='f32'),
    BufferBinding(name='_params_', binding=4, access='read_write', elem_type='u32'),
]


def pack_fp16_gemm_params(K, N):
    """Pack scalar params into u32 array for the fp16 GEMM kernel."""
    data = struct.pack('<II', K, N)
    return np.frombuffer(data, dtype=np.uint32).copy()


WGSL_FP16_GEMM_KERNEL = """
enable subgroups;

@group(0) @binding(0) var<storage, read_write> X: array<f32>;
@group(0) @binding(1) var<storage, read_write> W: array<u32>;
@group(0) @binding(2) var<storage, read_write> Bias: array<f32>;
@group(0) @binding(3) var<storage, read_write> Y: array<f32>;
@group(0) @binding(4) var<storage, read_write> _params_: array<u32>;

const TILE_N: u32 = 8u;

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) lid: vec3<u32>,
        @builtin(workgroup_id) wid: vec3<u32>) {
    let row = wid.x;
    let tile_col = wid.y;
    let tid = lid.x;

    let K = _params_[0];
    let N = _params_[1];
    let x_base = row * K;

    // 8 warps × 32 threads: each warp computes one output
    let warp_id = tid / 32u;
    let lane = tid % 32u;
    let col = tile_col * TILE_N + warp_id;

    // Each lane handles K/32 elements, strided by 4 (vec4 processing)
    // lane processes elements: lane*4, lane*4 + 32*4, lane*4 + 64*4, ...
    var acc: f32 = 0.0;

    if (col < N) {
        // W layout: (N, K) row-major, stored as u32 (2 fp16 per u32)
        // w_base = col * (K/2) — index into u32 array
        let w_base = col * (K / 2u);

        // Each lane processes 4 consecutive elements starting at lane*4
        // Stride between lane iterations: 32 lanes * 4 elements = 128
        let stride = 128u;
        var k = lane * 4u;

        for (; k + 3u < K; k = k + stride) {
            // Load 4 fp32 activations
            let x0 = X[x_base + k];
            let x1 = X[x_base + k + 1u];
            let x2 = X[x_base + k + 2u];
            let x3 = X[x_base + k + 3u];

            // Load 2 u32 = 4 fp16 weights, unpack to fp32
            let w01 = unpack2x16float(W[w_base + k / 2u]);
            let w23 = unpack2x16float(W[w_base + k / 2u + 1u]);

            // dot(vec4<f32>, vec4<f32>) — 4 FMAs
            acc += dot(vec4<f32>(x0, x1, x2, x3),
                       vec4<f32>(w01.x, w01.y, w23.x, w23.y));
        }
    }

    // Reduce across 32 lanes in warp
    let warp_sum = subgroupAdd(acc);

    if (lane == 0u && col < N) {
        Y[row * N + col] = warp_sum + Bias[col];
    }
}
"""
