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
# Q8_0-weight matvec kernel — W8A32 (int8 weights, fp32 activations)
# ---------------------------------------------------------------------------
#
# Same 256-thread, TILE_N=8 structure as Q4 kernel.
# Weights stored as u32 (4 packed int8 per u32).
# Scales stored as fp16 packed in u32 (one scale per 32-element block).
# No zero-point (Q8_0 is symmetric around 0).
#
# Dequant: extractBits(i32, offset, 8) gives sign-extended int8.
# Each 128-element iteration covers 4 Q8_0 blocks (4 scales).
# Lane's block_in_group = lane / 8.

Q8_TILE_N = 8

Q8_DP4A_BINDINGS = [
    BufferBinding(name='X', binding=0, access='read_write', elem_type='f32'),
    BufferBinding(name='W_Q8', binding=1, access='read_write', elem_type='u32'),
    BufferBinding(name='Scales', binding=2, access='read_write', elem_type='u32'),
    BufferBinding(name='Bias', binding=3, access='read_write', elem_type='f32'),
    BufferBinding(name='Y', binding=4, access='read_write', elem_type='f32'),
    BufferBinding(name='_params_', binding=5, access='read_write', elem_type='u32'),
]


def pack_q8_params(K, N):
    """Pack K and N into u32 params for the Q8_0 kernel."""
    data = struct.pack('<II', K, N)
    return np.frombuffer(data, dtype=np.uint32).copy()


WSGL_Q8_0_KERNEL = """
enable subgroups;

@group(0) @binding(0) var<storage, read_write> X: array<f32>;
@group(0) @binding(1) var<storage, read_write> W_Q8: array<u32>;
@group(0) @binding(2) var<storage, read_write> Scales: array<u32>;
@group(0) @binding(3) var<storage, read_write> Bias: array<f32>;
@group(0) @binding(4) var<storage, read_write> Y: array<f32>;
@group(0) @binding(5) var<storage, read_write> _params_: array<u32>;

const TILE_N: u32 = 8u;
const BLOCK_SIZE: u32 = 32u;

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

    // Each lane processes 4 elements per 128-element stride
    // 128 elements = 4 Q8_0 blocks (32 elements each)
    // lane 0-7 → block 0, lane 8-15 → block 1, etc.
    let block_in_stride = lane / 8u;
    let n_strides = K / 128u;
    let stride_w = K / 4u;  // u32 elements per weight row

    var acc: f32 = 0.0;

    if (col < N) {
        let w_base = col * stride_w;
        let n_blocks = K / BLOCK_SIZE;
        let s_base = col * n_blocks;

        for (var g = 0u; g < n_strides; g = g + 1u) {
            // Read 4 fp32 activations
            let k_base = g * 128u + lane * 4u;
            let x0 = X[x_base + k_base];
            let x1 = X[x_base + k_base + 1u];
            let x2 = X[x_base + k_base + 2u];
            let x3 = X[x_base + k_base + 3u];

            // Read packed int8 weights (4 int8 values as one u32)
            let packed_w = W_Q8[w_base + g * 32u + lane];

            // Extract 4 signed int8 values via sign-extending extractBits
            let w0 = f32(extractBits(i32(packed_w), 0u, 8u));
            let w1 = f32(extractBits(i32(packed_w), 8u, 8u));
            let w2 = f32(extractBits(i32(packed_w), 16u, 8u));
            let w3 = f32(extractBits(i32(packed_w), 24u, 8u));

            // Read per-block scale (fp16 packed in u32)
            let si = s_base + g * 4u + block_in_stride;
            let sp = unpack2x16float(Scales[si / 2u]);
            let scale = select(sp.x, sp.y, (si & 1u) != 0u);

            // Dequant + dot product: w_fp32 = w_int8 * scale
            acc += (x0 * w0 + x1 * w1 + x2 * w2 + x3 * w3) * scale;
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


# ---------------------------------------------------------------------------
# Experimental Q4_K GGUF kernel (direct block decode in WGSL)
# ---------------------------------------------------------------------------

Q4K_TILE_N = 8

Q4K_BINDINGS = [
    BufferBinding(name='X', binding=0, access='read_write', elem_type='f32'),
    BufferBinding(name='W_Q4K', binding=1, access='read_write', elem_type='u32'),
    BufferBinding(name='Bias', binding=2, access='read_write', elem_type='f32'),
    BufferBinding(name='Y', binding=3, access='read_write', elem_type='f32'),
    BufferBinding(name='_params_', binding=4, access='read_write', elem_type='u32'),
]


def pack_q4k_params(K, N, n_blocks, row_stride_words):
    """Pack scalar params into u32 array for Q4_K kernel."""
    data = struct.pack('<IIII', K, N, n_blocks, row_stride_words)
    return np.frombuffer(data, dtype=np.uint32).copy()


WGSL_Q4K_KERNEL = """
enable subgroups;

@group(0) @binding(0) var<storage, read_write> X: array<f32>;
@group(0) @binding(1) var<storage, read_write> W_Q4K: array<u32>;
@group(0) @binding(2) var<storage, read_write> Bias: array<f32>;
@group(0) @binding(3) var<storage, read_write> Y: array<f32>;
@group(0) @binding(4) var<storage, read_write> _params_: array<u32>;

const TILE_N: u32 = 8u;
const QK_K: u32 = 256u;
const BLOCK_WORDS: u32 = 36u; // 144 bytes / 4

fn get_u8(base_word: u32, byte_off: u32) -> u32 {
    let wi = base_word + byte_off / 4u;
    let sh = (byte_off % 4u) * 8u;
    return (W_Q4K[wi] >> sh) & 0xFFu;
}

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) lid: vec3<u32>,
        @builtin(workgroup_id) wid: vec3<u32>) {
    let row = wid.x;
    let tid = lid.x;
    let warp_id = tid / 32u;
    let lane = tid % 32u;
    let col = wid.y * TILE_N + warp_id;

    let K = _params_[0];
    let N = _params_[1];
    let n_blocks = _params_[2];
    let row_stride_words = _params_[3];

    var acc: f32 = 0.0;
    if (col < N) {
        let x_base = row * K;

        for (var b = 0u; b < n_blocks; b = b + 1u) {
            let block_base = col * row_stride_words + b * BLOCK_WORDS;

            let dd = unpack2x16float(W_Q4K[block_base]);
            let d = dd.x;
            let dmin = dd.y;

            let d0 = get_u8(block_base, 4u);
            let d1 = get_u8(block_base, 5u);
            let d2 = get_u8(block_base, 6u);
            let d3 = get_u8(block_base, 7u);
            let m0 = get_u8(block_base, 8u);
            let m1 = get_u8(block_base, 9u);
            let m2 = get_u8(block_base, 10u);
            let m3 = get_u8(block_base, 11u);
            let md0 = get_u8(block_base, 12u);
            let md1 = get_u8(block_base, 13u);
            let md2 = get_u8(block_base, 14u);
            let md3 = get_u8(block_base, 15u);

            for (var sb = 0u; sb < 8u; sb = sb + 1u) {
                var sc_u: u32;
                var mn_u: u32;
                if (sb < 4u) {
                    let dv = select(select(select(d0, d1, sb == 1u), d2, sb == 2u), d3, sb == 3u);
                    let mv = select(select(select(m0, m1, sb == 1u), m2, sb == 2u), m3, sb == 3u);
                    sc_u = dv & 0x3Fu;
                    mn_u = mv & 0x3Fu;
                } else {
                    let j = sb - 4u;
                    let dv = select(select(select(d0, d1, j == 1u), d2, j == 2u), d3, j == 3u);
                    let mv = select(select(select(m0, m1, j == 1u), m2, j == 2u), m3, j == 3u);
                    let mdv = select(select(select(md0, md1, j == 1u), md2, j == 2u), md3, j == 3u);
                    sc_u = (mdv & 0x0Fu) | ((dv >> 2u) & 0x30u);
                    mn_u = (mdv >> 4u) | ((mv >> 2u) & 0x30u);
                }

                let sc = d * f32(sc_u);
                let mn = dmin * f32(mn_u);
                let g = sb / 2u;
                let hi = (sb & 1u) == 1u;

                let i = lane;
                let kidx = b * QK_K + sb * 32u + i;
                if (kidx < K) {
                    let qb = get_u8(block_base, 16u + g * 32u + i);
                    let q = select(qb & 0x0Fu, (qb >> 4u) & 0x0Fu, hi);
                    let w = sc * f32(q) - mn;
                    acc = acc + X[x_base + kidx] * w;
                }
            }
        }
    }

    let sum = subgroupAdd(acc);
    if (lane == 0u && col < N) {
        Y[row * N + col] = sum + Bias[col];
    }
}
"""


# ---------------------------------------------------------------------------
# Experimental Q5_K GGUF kernel (direct block decode in WGSL)
# ---------------------------------------------------------------------------

Q5K_TILE_N = 8

Q5K_BINDINGS = [
    BufferBinding(name='X', binding=0, access='read_write', elem_type='f32'),
    BufferBinding(name='W_Q5K', binding=1, access='read_write', elem_type='u32'),
    BufferBinding(name='Bias', binding=2, access='read_write', elem_type='f32'),
    BufferBinding(name='Y', binding=3, access='read_write', elem_type='f32'),
    BufferBinding(name='_params_', binding=4, access='read_write', elem_type='u32'),
]


def pack_q5k_params(K, N, n_blocks, row_stride_words):
    data = struct.pack('<IIII', K, N, n_blocks, row_stride_words)
    return np.frombuffer(data, dtype=np.uint32).copy()


WGSL_Q5K_KERNEL = """
enable subgroups;

@group(0) @binding(0) var<storage, read_write> X: array<f32>;
@group(0) @binding(1) var<storage, read_write> W_Q5K: array<u32>;
@group(0) @binding(2) var<storage, read_write> Bias: array<f32>;
@group(0) @binding(3) var<storage, read_write> Y: array<f32>;
@group(0) @binding(4) var<storage, read_write> _params_: array<u32>;

const TILE_N: u32 = 8u;
const QK_K: u32 = 256u;
const BLOCK_WORDS: u32 = 44u; // 176 bytes / 4

fn get_u8(base_word: u32, byte_off: u32) -> u32 {
    let wi = base_word + byte_off / 4u;
    let sh = (byte_off % 4u) * 8u;
    return (W_Q5K[wi] >> sh) & 0xFFu;
}

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) lid: vec3<u32>,
        @builtin(workgroup_id) wid: vec3<u32>) {
    let row = wid.x;
    let tid = lid.x;
    let warp_id = tid / 32u;
    let lane = tid % 32u;
    let col = wid.y * TILE_N + warp_id;

    let K = _params_[0];
    let N = _params_[1];
    let n_blocks = _params_[2];
    let row_stride_words = _params_[3];

    var acc: f32 = 0.0;
    if (col < N) {
        let x_base = row * K;

        for (var b = 0u; b < n_blocks; b = b + 1u) {
            let block_base = col * row_stride_words + b * BLOCK_WORDS;

            let dd = unpack2x16float(W_Q5K[block_base]);
            let d = dd.x;
            let dmin = dd.y;

            let d0 = get_u8(block_base, 4u);
            let d1 = get_u8(block_base, 5u);
            let d2 = get_u8(block_base, 6u);
            let d3 = get_u8(block_base, 7u);
            let m0 = get_u8(block_base, 8u);
            let m1 = get_u8(block_base, 9u);
            let m2 = get_u8(block_base, 10u);
            let m3 = get_u8(block_base, 11u);
            let md0 = get_u8(block_base, 12u);
            let md1 = get_u8(block_base, 13u);
            let md2 = get_u8(block_base, 14u);
            let md3 = get_u8(block_base, 15u);

            for (var sb = 0u; sb < 8u; sb = sb + 1u) {
                var sc_u: u32;
                var mn_u: u32;
                if (sb < 4u) {
                    let dv = select(select(select(d0, d1, sb == 1u), d2, sb == 2u), d3, sb == 3u);
                    let mv = select(select(select(m0, m1, sb == 1u), m2, sb == 2u), m3, sb == 3u);
                    sc_u = dv & 0x3Fu;
                    mn_u = mv & 0x3Fu;
                } else {
                    let j = sb - 4u;
                    let dv = select(select(select(d0, d1, j == 1u), d2, j == 2u), d3, j == 3u);
                    let mv = select(select(select(m0, m1, j == 1u), m2, j == 2u), m3, j == 3u);
                    let mdv = select(select(select(md0, md1, j == 1u), md2, j == 2u), md3, j == 3u);
                    sc_u = (mdv & 0x0Fu) | ((dv >> 2u) & 0x30u);
                    mn_u = (mdv >> 4u) | ((mv >> 2u) & 0x30u);
                }

                let sc = d * f32(sc_u);
                let mn = dmin * f32(mn_u);
                let qs_group = sb / 2u;

                let i = lane;
                let kidx = b * QK_K + sb * 32u + i;
                if (kidx < K) {
                    let ql_byte = get_u8(block_base, 48u + qs_group * 32u + i);
                    let ql = select(ql_byte & 0x0Fu, (ql_byte >> 4u) & 0x0Fu, (sb & 1u) == 1u);
                    let qh_byte = get_u8(block_base, 16u + i);
                    let qh = (qh_byte >> sb) & 0x01u;
                    let q = ql | (qh << 4u);

                    let w = sc * f32(q) - mn;
                    acc = acc + X[x_base + kidx] * w;
                }
            }
        }
    }

    let sum = subgroupAdd(acc);
    if (lane == 0u && col < N) {
        Y[row * N + col] = sum + Bias[col];
    }
}
"""


# ---------------------------------------------------------------------------
# Experimental Q6_K GGUF kernel (direct block decode in WGSL)
# ---------------------------------------------------------------------------

Q6K_TILE_N = 8

Q6K_BINDINGS = [
    BufferBinding(name='X', binding=0, access='read_write', elem_type='f32'),
    BufferBinding(name='W_Q6K', binding=1, access='read_write', elem_type='u32'),
    BufferBinding(name='Bias', binding=2, access='read_write', elem_type='f32'),
    BufferBinding(name='Y', binding=3, access='read_write', elem_type='f32'),
    BufferBinding(name='_params_', binding=4, access='read_write', elem_type='u32'),
]


def pack_q6k_params(K, N, n_blocks, row_stride_bytes):
    data = struct.pack('<IIII', K, N, n_blocks, row_stride_bytes)
    return np.frombuffer(data, dtype=np.uint32).copy()


WGSL_Q6K_KERNEL = """
enable subgroups;

@group(0) @binding(0) var<storage, read_write> X: array<f32>;
@group(0) @binding(1) var<storage, read_write> W_Q6K: array<u32>;
@group(0) @binding(2) var<storage, read_write> Bias: array<f32>;
@group(0) @binding(3) var<storage, read_write> Y: array<f32>;
@group(0) @binding(4) var<storage, read_write> _params_: array<u32>;

const TILE_N: u32 = 8u;
const QK_K: u32 = 256u;
const BLOCK_BYTES: u32 = 210u;

fn get_u8(base_byte: u32, byte_off: u32) -> u32 {
    let abs_b = base_byte + byte_off;
    let wi = abs_b / 4u;
    let sh = (abs_b % 4u) * 8u;
    return (W_Q6K[wi] >> sh) & 0xFFu;
}

fn as_i8(v: u32) -> i32 {
    return select(i32(v), i32(v) - 256, v >= 128u);
}

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) lid: vec3<u32>,
        @builtin(workgroup_id) wid: vec3<u32>) {
    let row = wid.x;
    let tid = lid.x;
    let warp_id = tid / 32u;
    let lane = tid % 32u;
    let col = wid.y * TILE_N + warp_id;

    let K = _params_[0];
    let N = _params_[1];
    let n_blocks = _params_[2];
    let row_stride_bytes = _params_[3];

    var acc: f32 = 0.0;
    if (col < N) {
        let x_base = row * K;

        for (var b = 0u; b < n_blocks; b = b + 1u) {
            let block_base_b = col * row_stride_bytes + b * BLOCK_BYTES;

            // d at bytes [208:210] (fp16)
            let d_lo = get_u8(block_base_b, 208u);
            let d_hi = get_u8(block_base_b, 209u);
            let d_pack = d_lo | (d_hi << 8u);
            let d = unpack2x16float(d_pack).x;

            for (var sb = 0u; sb < 8u; sb = sb + 1u) {
                let idx = sb * 32u + lane;
                let kidx = b * QK_K + idx;
                if (kidx < K) {
                    // Q6_K uses 16 scale groups, each with 16 quantized values.
                    let g = idx / 16u;   // 0..15 (scale group)

                    // qh layout (64 bytes) follows 2 x 32-byte planes, each expanded
                    // by shifts [0,2,4,6] in the reference implementation.
                    let qg = idx / 32u;   // 0..7
                    let p32 = idx % 32u;  // 0..31
                    let qh_byte_off = 128u + (qg / 4u) * 32u + p32;
                    let qh_shift = (qg % 4u) * 2u;
                    let qh_b = get_u8(block_base_b, qh_byte_off);
                    let qh = (qh_b >> qh_shift) & 0x03u;

                    // ql layout (128 bytes) follows 2 x 64-byte planes expanded by
                    // shifts [0,4] in the reference implementation.
                    let ql_byte_off = (idx / 128u) * 64u + (idx % 64u);
                    let ql_shift = ((idx / 64u) % 2u) * 4u;
                    let ql_b = get_u8(block_base_b, ql_byte_off);
                    let ql = (ql_b >> ql_shift) & 0x0Fu;

                    let q_u = (qh << 4u) | ql; // 0..63
                    let q_i = i32(q_u) - 32;

                    // scales: 16 int8 at [192:208]
                    let s_i = as_i8(get_u8(block_base_b, 192u + g));

                    let w = d * f32(s_i) * f32(q_i);
                    acc = acc + X[x_base + kidx] * w;
                }
            }
        }
    }

    let sum = subgroupAdd(acc);
    if (lane == 0u && col < N) {
        Y[row * N + col] = sum + Bias[col];
    }
}
"""


