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

TILE_N = 16

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
enable f16;

@group(0) @binding(0) var<storage, read_write> X: array<f32>;
@group(0) @binding(1) var<storage, read_write> W_Q4: array<u32>;
@group(0) @binding(2) var<storage, read_write> Scales: array<u32>;
@group(0) @binding(3) var<storage, read_write> Zeros: array<u32>;
@group(0) @binding(4) var<storage, read_write> Bias: array<f32>;
@group(0) @binding(5) var<storage, read_write> Y: array<f32>;
@group(0) @binding(6) var<storage, read_write> _params_: array<u32>;

const TILE_N: u32 = 16u;

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

    // 8 warps × 32 threads: each warp computes TWO outputs
    let warp_id = tid / 32u;
    let lane = tid % 32u;
    let col0 = tile_col * TILE_N + warp_id * 2u;
    let col1 = col0 + 1u;

    // Each lane handles 4 elements per 128-element group
    let word_in_group = lane / 2u;
    let nib_shift = (lane % 2u) * 16u;

    var acc0: f16 = 0.0h;
    var acc1: f16 = 0.0h;

    if (col0 < N) {
        let w_base0 = col0 * stride_w_q4;
        let s_base0 = col0 * n_groups;
        let w_base1 = col1 * stride_w_q4;
        let s_base1 = col1 * n_groups;

        for (var g = 0u; g < n_groups; g = g + 1u) {
            let k_base = g * 128u + lane * 4u;
            let x0 = f16(X[x_base + k_base]);
            let x1 = f16(X[x_base + k_base + 1u]);
            let x2 = f16(X[x_base + k_base + 2u]);
            let x3 = f16(X[x_base + k_base + 3u]);
            let x_vec = vec4<f16>(x0, x1, x2, x3);
            let dot_1 = x0 + x1 + x2 + x3;

            // Col 0
            let si0 = s_base0 + g;
            let sp0 = unpack2x16float(Scales[si0 / 2u]);
            let w_scale0 = f16(select(sp0.x, sp0.y, (si0 & 1u) != 0u));
            let zp0 = unpack2x16float(Zeros[si0 / 2u]);
            let w_zero0 = f16(select(zp0.x, zp0.y, (si0 & 1u) != 0u));

            let packed_w0 = W_Q4[w_base0 + g * 16u + word_in_group];
            let nibbles4_0 = (packed_w0 >> nib_shift) & 0xFFFFu;
            let q_vec0 = vec4<f16>(
                f16(nibbles4_0 & 0xFu),
                f16((nibbles4_0 >> 4u) & 0xFu),
                f16((nibbles4_0 >> 8u) & 0xFu),
                f16((nibbles4_0 >> 12u) & 0xFu)
            );
            let dot_q0 = dot(x_vec, q_vec0);
            acc0 += dot_q0 * w_scale0 + dot_1 * w_zero0;

            // Col 1
            if (col1 < N) {
                let si1 = s_base1 + g;
                let sp1 = unpack2x16float(Scales[si1 / 2u]);
                let w_scale1 = f16(select(sp1.x, sp1.y, (si1 & 1u) != 0u));
                let zp1 = unpack2x16float(Zeros[si1 / 2u]);
                let w_zero1 = f16(select(zp1.x, zp1.y, (si1 & 1u) != 0u));

                let packed_w1 = W_Q4[w_base1 + g * 16u + word_in_group];
                let nibbles4_1 = (packed_w1 >> nib_shift) & 0xFFFFu;
                let q_vec1 = vec4<f16>(
                    f16(nibbles4_1 & 0xFu),
                    f16((nibbles4_1 >> 4u) & 0xFu),
                    f16((nibbles4_1 >> 8u) & 0xFu),
                    f16((nibbles4_1 >> 12u) & 0xFu)
                );
                let dot_q1 = dot(x_vec, q_vec1);
                acc1 += dot_q1 * w_scale1 + dot_1 * w_zero1;
            }
        }
    }

    let warp_sum0 = subgroupAdd(acc0);
    let warp_sum1 = subgroupAdd(acc1);

    if (lane == 0u) {
        if (col0 < N) {
            Y[row * N + col0] = f32(warp_sum0) + Bias[col0];
        }
        if (col1 < N) {
            Y[row * N + col1] = f32(warp_sum1) + Bias[col1];
        }
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

FP16_GEMM_TILE_N = 16

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

const TILE_N: u32 = 16u;

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


# ---------------------------------------------------------------------------
# Token embedding gather (GPU)
# ---------------------------------------------------------------------------

EMBED_GATHER_BINDINGS = [
    BufferBinding(name='TokenId', binding=0, access='read_write', elem_type='u32'),
    BufferBinding(name='W', binding=1, access='read_write', elem_type='u32'),
    BufferBinding(name='X', binding=2, access='read_write', elem_type='f32'),
    BufferBinding(name='_params_', binding=3, access='read_write', elem_type='u32'),
]


def pack_embed_gather_params(E, vocab_size):
    data = struct.pack('<II', int(E), int(vocab_size))
    return np.frombuffer(data, dtype=np.uint32).copy()


WGSL_EMBED_GATHER_KERNEL = """
@group(0) @binding(0) var<storage, read_write> TokenId: array<u32>;
@group(0) @binding(1) var<storage, read_write> W: array<u32>;
@group(0) @binding(2) var<storage, read_write> X: array<f32>;
@group(0) @binding(3) var<storage, read_write> _params_: array<u32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    let E = _params_[0];
    let V = _params_[1];
    if (i >= E) {
        return;
    }

    let tid = min(TokenId[0], V - 1u);
    let flat_idx = tid * E + i;
    let pair = unpack2x16float(W[flat_idx >> 1u]);
    X[i] = select(pair.x, pair.y, (flat_idx & 1u) != 0u);
}
"""


# ---------------------------------------------------------------------------
# GPU greedy sampling (argmax) kernels
# ---------------------------------------------------------------------------

ARGMAX_STAGE1_BINDINGS = [
    BufferBinding(name='Logits', binding=0, access='read_write', elem_type='f32'),
    BufferBinding(name='GroupMaxVals', binding=1, access='read_write', elem_type='f32'),
    BufferBinding(name='GroupMaxIdx', binding=2, access='read_write', elem_type='u32'),
    BufferBinding(name='_params_', binding=3, access='read_write', elem_type='u32'),
]

ARGMAX_STAGE2_BINDINGS = [
    BufferBinding(name='GroupMaxVals', binding=0, access='read_write', elem_type='f32'),
    BufferBinding(name='GroupMaxIdx', binding=1, access='read_write', elem_type='u32'),
    BufferBinding(name='OutToken', binding=2, access='read_write', elem_type='u32'),
    BufferBinding(name='_params_', binding=3, access='read_write', elem_type='u32'),
]

TOPK_STAGE2_BINDINGS = [
    BufferBinding(name='Logits', binding=0, access='read_write', elem_type='f32'),
    BufferBinding(name='TopKIdx', binding=1, access='read_write', elem_type='u32'),
    BufferBinding(name='TopKVal', binding=2, access='read_write', elem_type='f32'),
    BufferBinding(name='GroupMaxVals', binding=3, access='read_write', elem_type='f32'),
    BufferBinding(name='GroupMaxIdx', binding=4, access='read_write', elem_type='u32'),
    BufferBinding(name='_params_', binding=5, access='read_write', elem_type='u32'),
]

TOPK_UPDATE_GROUP_BINDINGS = [
    BufferBinding(name='Logits', binding=0, access='read_write', elem_type='f32'),
    BufferBinding(name='GroupMaxVals', binding=1, access='read_write', elem_type='f32'),
    BufferBinding(name='GroupMaxIdx', binding=2, access='read_write', elem_type='u32'),
    BufferBinding(name='TopKIdx', binding=3, access='read_write', elem_type='u32'),
    BufferBinding(name='_params_', binding=4, access='read_write', elem_type='u32'),
]

TOPK_FUSED_BINDINGS = [
    BufferBinding(name='Logits', binding=0, access='read_write', elem_type='f32'),
    BufferBinding(name='GroupMaxVals', binding=1, access='read_write', elem_type='f32'),
    BufferBinding(name='GroupMaxIdx', binding=2, access='read_write', elem_type='u32'),
    BufferBinding(name='TopKIdx', binding=3, access='read_write', elem_type='u32'),
    BufferBinding(name='TopKVal', binding=4, access='read_write', elem_type='f32'),
    BufferBinding(name='_params_', binding=5, access='read_write', elem_type='u32'),
]

TOPK_SAMPLE_BINDINGS = [
    BufferBinding(name='TopKIdx', binding=0, access='read_write', elem_type='u32'),
    BufferBinding(name='TopKVal', binding=1, access='read_write', elem_type='f32'),
    BufferBinding(name='OutToken', binding=2, access='read_write', elem_type='u32'),
    BufferBinding(name='_params_', binding=3, access='read_write', elem_type='u32'),
]


def pack_argmax_stage1_params(N):
    data = struct.pack('<I', N)
    return np.frombuffer(data, dtype=np.uint32).copy()


def pack_argmax_stage2_params(M):
    data = struct.pack('<I', M)
    return np.frombuffer(data, dtype=np.uint32).copy()


def pack_topk_sample_params(K, temperature, rng_state):
    data = struct.pack('<IfI', int(K), float(temperature), int(rng_state) & 0xFFFFFFFF)
    return np.frombuffer(data, dtype=np.uint32).copy()


WGSL_ARGMAX_STAGE1_KERNEL = """
@group(0) @binding(0) var<storage, read_write> Logits: array<f32>;
@group(0) @binding(1) var<storage, read_write> GroupMaxVals: array<f32>;
@group(0) @binding(2) var<storage, read_write> GroupMaxIdx: array<u32>;
@group(0) @binding(3) var<storage, read_write> _params_: array<u32>;

var<workgroup> s_val: array<f32, 256>;
var<workgroup> s_idx: array<u32, 256>;

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) lid: vec3<u32>,
        @builtin(workgroup_id) wid: vec3<u32>) {
    let lane = lid.x;
    let gid = wid.x * 256u + lane;
    let N = _params_[0];

    var v: f32 = -1.0e30;
    var i: u32 = 0u;
    if (gid < N) {
        v = Logits[gid];
        i = gid;
    }

    s_val[lane] = v;
    s_idx[lane] = i;
    workgroupBarrier();

    var stride = 128u;
    loop {
        if (stride == 0u) {
            break;
        }
        if (lane < stride) {
            let ov = s_val[lane + stride];
            let oi = s_idx[lane + stride];
            if (ov > s_val[lane] || (ov == s_val[lane] && oi < s_idx[lane])) {
                s_val[lane] = ov;
                s_idx[lane] = oi;
            }
        }
        workgroupBarrier();
        stride = stride >> 1u;
    }

    if (lane == 0u) {
        GroupMaxVals[wid.x] = s_val[0];
        GroupMaxIdx[wid.x] = s_idx[0];
    }
}
"""


WGSL_ARGMAX_STAGE2_KERNEL = """
@group(0) @binding(0) var<storage, read_write> GroupMaxVals: array<f32>;
@group(0) @binding(1) var<storage, read_write> GroupMaxIdx: array<u32>;
@group(0) @binding(2) var<storage, read_write> OutToken: array<u32>;
@group(0) @binding(3) var<storage, read_write> _params_: array<u32>;

var<workgroup> s_val: array<f32, 256>;
var<workgroup> s_idx: array<u32, 256>;

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) lid: vec3<u32>) {
    let lane = lid.x;
    let M = _params_[0];

    var best_v: f32 = -1.0e30;
    var best_i: u32 = 0u;

    var j = lane;
    loop {
        if (j >= M) {
            break;
        }
        let v = GroupMaxVals[j];
        let i = GroupMaxIdx[j];
        if (v > best_v || (v == best_v && i < best_i)) {
            best_v = v;
            best_i = i;
        }
        j = j + 256u;
    }

    s_val[lane] = best_v;
    s_idx[lane] = best_i;
    workgroupBarrier();

    var stride = 128u;
    loop {
        if (stride == 0u) {
            break;
        }
        if (lane < stride) {
            let ov = s_val[lane + stride];
            let oi = s_idx[lane + stride];
            if (ov > s_val[lane] || (ov == s_val[lane] && oi < s_idx[lane])) {
                s_val[lane] = ov;
                s_idx[lane] = oi;
            }
        }
        workgroupBarrier();
        stride = stride >> 1u;
    }

    if (lane == 0u) {
        OutToken[0] = s_idx[0];
    }
}
"""


WGSL_TOPK_STAGE2_KERNEL = """
@group(0) @binding(0) var<storage, read_write> Logits: array<f32>;
@group(0) @binding(1) var<storage, read_write> TopKIdx: array<u32>;
@group(0) @binding(2) var<storage, read_write> TopKVal: array<f32>;
@group(0) @binding(3) var<storage, read_write> GroupMaxVals: array<f32>;
@group(0) @binding(4) var<storage, read_write> GroupMaxIdx: array<u32>;
@group(0) @binding(5) var<storage, read_write> _params_: array<u32>;

var<workgroup> s_val: array<f32, 256>;
var<workgroup> s_idx: array<u32, 256>;

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) lid: vec3<u32>) {
    let lane = lid.x;
    let M = _params_[0];
    let step = _params_[1];

    var best_v: f32 = -1.0e30;
    var best_i: u32 = 0u;

    var j = lane;
    loop {
        if (j >= M) {
            break;
        }
        let v = GroupMaxVals[j];
        let i = GroupMaxIdx[j];
        if (v > best_v || (v == best_v && i < best_i)) {
            best_v = v;
            best_i = i;
        }
        j = j + 256u;
    }

    s_val[lane] = best_v;
    s_idx[lane] = best_i;
    workgroupBarrier();

    var stride = 128u;
    loop {
        if (stride == 0u) {
            break;
        }
        if (lane < stride) {
            let ov = s_val[lane + stride];
            let oi = s_idx[lane + stride];
            if (ov > s_val[lane] || (ov == s_val[lane] && oi < s_idx[lane])) {
                s_val[lane] = ov;
                s_idx[lane] = oi;
            }
        }
        workgroupBarrier();
        stride = stride >> 1u;
    }

    if (lane == 0u) {
        let idx = s_idx[0];
        let val = s_val[0];
        TopKIdx[step] = idx;
        TopKVal[step] = val;
        Logits[idx] = -1.0e30;
    }
}
"""


WGSL_TOPK_UPDATE_GROUP_KERNEL = """
@group(0) @binding(0) var<storage, read_write> Logits: array<f32>;
@group(0) @binding(1) var<storage, read_write> GroupMaxVals: array<f32>;
@group(0) @binding(2) var<storage, read_write> GroupMaxIdx: array<u32>;
@group(0) @binding(3) var<storage, read_write> TopKIdx: array<u32>;
@group(0) @binding(4) var<storage, read_write> _params_: array<u32>;

var<workgroup> s_val: array<f32, 256>;
var<workgroup> s_idx: array<u32, 256>;

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) lid: vec3<u32>) {
    let lane = lid.x;
    let N = _params_[0];
    let step = _params_[1];

    let sel_idx = TopKIdx[step];
    let group_id = sel_idx >> 8u;
    let base = group_id * 256u;
    let gid = base + lane;

    var v: f32 = -1.0e30;
    var i: u32 = base;
    if (gid < N) {
        v = Logits[gid];
        i = gid;
    }

    s_val[lane] = v;
    s_idx[lane] = i;
    workgroupBarrier();

    var stride = 128u;
    loop {
        if (stride == 0u) {
            break;
        }
        if (lane < stride) {
            let ov = s_val[lane + stride];
            let oi = s_idx[lane + stride];
            if (ov > s_val[lane] || (ov == s_val[lane] && oi < s_idx[lane])) {
                s_val[lane] = ov;
                s_idx[lane] = oi;
            }
        }
        workgroupBarrier();
        stride = stride >> 1u;
    }

    if (lane == 0u) {
        GroupMaxVals[group_id] = s_val[0];
        GroupMaxIdx[group_id] = s_idx[0];
    }
}
"""


WGSL_TOPK_FUSED_KERNEL = """
@group(0) @binding(0) var<storage, read_write> Logits: array<f32>;
@group(0) @binding(1) var<storage, read_write> GroupMaxVals: array<f32>;
@group(0) @binding(2) var<storage, read_write> GroupMaxIdx: array<u32>;
@group(0) @binding(3) var<storage, read_write> TopKIdx: array<u32>;
@group(0) @binding(4) var<storage, read_write> TopKVal: array<f32>;
@group(0) @binding(5) var<storage, read_write> _params_: array<u32>;

var<workgroup> s_val: array<f32, 256>;
var<workgroup> s_idx: array<u32, 256>;
var<workgroup> s_gid: array<u32, 256>;

fn better(v: f32, i: u32, best_v: f32, best_i: u32) -> bool {
    return (v > best_v) || (v == best_v && i < best_i);
}

const TOPK_FUSED_MAX_K: u32 = 64u;

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) lid: vec3<u32>) {
    let lane = lid.x;
    let N = _params_[0];
    let M = _params_[1];
    let K = _params_[2];

    var step: u32 = 0u;
    loop {
        if (step >= TOPK_FUSED_MAX_K) {
            break;
        }
        let enabled = step < K;

        var best_v: f32 = -1.0e30;
        var best_i: u32 = 0u;
        var best_g: u32 = 0u;

        var g: u32 = lane;
        loop {
            if (g >= M) {
                break;
            }
            let v = select(-1.0e30, GroupMaxVals[g], enabled);
            let i = GroupMaxIdx[g];
            if (better(v, i, best_v, best_i)) {
                best_v = v;
                best_i = i;
                best_g = g;
            }
            g = g + 256u;
        }

        s_val[lane] = best_v;
        s_idx[lane] = best_i;
        s_gid[lane] = best_g;
        workgroupBarrier();

        var stride = 128u;
        loop {
            if (stride == 0u) {
                break;
            }
            if (lane < stride) {
                let ov = s_val[lane + stride];
                let oi = s_idx[lane + stride];
                let og = s_gid[lane + stride];
                if (better(ov, oi, s_val[lane], s_idx[lane])) {
                    s_val[lane] = ov;
                    s_idx[lane] = oi;
                    s_gid[lane] = og;
                }
            }
            workgroupBarrier();
            stride = stride >> 1u;
        }

        if (lane == 0u) {
            best_v = s_val[0];
            best_i = s_idx[0];
            best_g = s_gid[0];

            if (enabled) {
                TopKIdx[step] = best_i;
                TopKVal[step] = best_v;
            }

            if (enabled && best_i < N) {
                Logits[best_i] = -1.0e30;
            }

            s_gid[0] = best_g;
        }
        storageBarrier();
        workgroupBarrier();

        let best_g_bcast = s_gid[0];

        let base = best_g_bcast * 256u;
        var grp_best_v: f32 = -1.0e30;
        var grp_best_i: u32 = base;
        let idx = base + lane;
        if (enabled && idx < N) {
            let v = Logits[idx];
            grp_best_v = v;
            grp_best_i = idx;
        }

        s_val[lane] = grp_best_v;
        s_idx[lane] = grp_best_i;
        workgroupBarrier();

        stride = 128u;
        loop {
            if (stride == 0u) {
                break;
            }
            if (lane < stride) {
                let ov = s_val[lane + stride];
                let oi = s_idx[lane + stride];
                if (better(ov, oi, s_val[lane], s_idx[lane])) {
                    s_val[lane] = ov;
                    s_idx[lane] = oi;
                }
            }
            workgroupBarrier();
            stride = stride >> 1u;
        }

        if (lane == 0u) {
            if (enabled) {
                GroupMaxVals[best_g_bcast] = s_val[0];
                GroupMaxIdx[best_g_bcast] = s_idx[0];
            }
        }
        storageBarrier();
        workgroupBarrier();

        step = step + 1u;
    }
}
"""


WGSL_TOPK_SAMPLE_KERNEL = """
@group(0) @binding(0) var<storage, read_write> TopKIdx: array<u32>;
@group(0) @binding(1) var<storage, read_write> TopKVal: array<f32>;
@group(0) @binding(2) var<storage, read_write> OutToken: array<u32>;
@group(0) @binding(3) var<storage, read_write> _params_: array<u32>;

fn xorshift32(x: u32) -> u32 {
    var s = x;
    s = s ^ (s << 13u);
    s = s ^ (s >> 17u);
    s = s ^ (s << 5u);
    return s;
}

@compute @workgroup_size(1)
fn main() {
    let K = _params_[0];
    let temperature = bitcast<f32>(_params_[1]);
    let rng_in = _params_[2];

    if (K == 0u) {
        OutToken[0] = 0u;
        return;
    }

    if (temperature <= 0.0) {
        OutToken[0] = TopKIdx[0];
        return;
    }

    // Gumbel-max sampling over top-k logits:
    //   sample = argmax_i (logit_i / T + Gumbel(0,1)_i)
    // This is equivalent to categorical softmax sampling but avoids
    // cumulative-CDF numerical issues.
    let inv_t = 1.0 / max(temperature, 1.0e-6);
    let prev_token = OutToken[0];
    let repeat_penalty = 1.15;
    var best_score: f32 = -1.0e30;
    var chosen: u32 = TopKIdx[0];

    var i: u32 = 0u;
    loop {
        if (i >= K) {
            break;
        }

        var logit_raw = TopKVal[i];
        // Light anti-repeat penalty for immediate token loops.
        if (TopKIdx[i] == prev_token) {
            if (logit_raw > 0.0) {
                logit_raw = logit_raw / repeat_penalty;
            } else {
                logit_raw = logit_raw * repeat_penalty;
            }
        }

        var logit = logit_raw * inv_t;
        if (!(logit == logit)) {
            logit = -1.0e30;
        }
        logit = clamp(logit, -80.0, 80.0);

        // Per-candidate random value derived from base RNG state.
        var s = rng_in ^ (i * 747796405u + 2891336453u);
        s = xorshift32(s);
        s = xorshift32(s ^ 0x9e3779b9u);
        var u = (f32(s) + 0.5) * (1.0 / 4294967296.0);
        u = clamp(u, 1.0e-7, 1.0 - 1.0e-7);

        let gumbel = -log(-log(u));
        let score = logit + gumbel;

        if (score > best_score) {
            best_score = score;
            chosen = TopKIdx[i];
        }
        i = i + 1u;
    }

    OutToken[0] = chosen;
}
"""
