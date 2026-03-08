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

    // K_PER_ITER=8: each lane processes 8 elements per 256-element stride
    // (32 lanes × 8 elem/lane = 256 elements = 8 Q8_0 blocks per stride)
    let n_strides = K / 256u;
    let stride_w = K / 4u;  // u32 per weight row

    var acc: f32 = 0.0;

    if (col < N) {
        let w_base = col * stride_w;
        let n_blocks = K / 32u;
        let s_base = col * n_blocks;

        for (var g = 0u; g < n_strides; g = g + 1u) {
            // Read 8 fp32 activations (two vec4 loads)
            let k_base = g * 256u + lane * 8u;
            let xv0 = vec4<f32>(X[x_base + k_base],
                                X[x_base + k_base + 1u],
                                X[x_base + k_base + 2u],
                                X[x_base + k_base + 3u]);
            let xv1 = vec4<f32>(X[x_base + k_base + 4u],
                                X[x_base + k_base + 5u],
                                X[x_base + k_base + 6u],
                                X[x_base + k_base + 7u]);

            // Read 2 packed u32 weights (8 int8 values)
            let w_off = w_base + g * 64u + lane * 2u;
            let pw0 = W_Q8[w_off];
            let pw1 = W_Q8[w_off + 1u];

            // Extract int8 → f32 and form vec4 for dot product
            let wv0 = vec4<f32>(f32(extractBits(i32(pw0), 0u, 8u)),
                                f32(extractBits(i32(pw0), 8u, 8u)),
                                f32(extractBits(i32(pw0), 16u, 8u)),
                                f32(extractBits(i32(pw0), 24u, 8u)));
            let wv1 = vec4<f32>(f32(extractBits(i32(pw1), 0u, 8u)),
                                f32(extractBits(i32(pw1), 8u, 8u)),
                                f32(extractBits(i32(pw1), 16u, 8u)),
                                f32(extractBits(i32(pw1), 24u, 8u)));

            // Per-block scales: 2 blocks per 8 elements
            // block_idx = (g * 256 + lane * 8) / 32
            let bi = g * 8u + lane / 4u;
            let si0 = s_base + bi;
            let si1 = si0 + 1u;
            // Correction: lane * 8 spans TWO 32-element blocks when lane >= 4
            // Block for first 4 elements: (g*256 + lane*8) / 32 = g*8 + lane/4
            // Block for second 4 elements: (g*256 + lane*8 + 4) / 32
            let block0 = g * 8u + (lane * 8u) / 32u;
            let block1 = g * 8u + (lane * 8u + 4u) / 32u;
            let sp0 = unpack2x16float(Scales[(s_base + block0) / 2u]);
            let scale0 = select(sp0.x, sp0.y, ((s_base + block0) & 1u) != 0u);
            let sp1 = unpack2x16float(Scales[(s_base + block1) / 2u]);
            let scale1 = select(sp1.x, sp1.y, ((s_base + block1) & 1u) != 0u);

            // vec4 dot products with per-block scaling
            acc += dot(xv0, wv0) * scale0 + dot(xv1, wv1) * scale1;
        }
    }

    let warp_sum = subgroupAdd(acc);

    if (lane == 0u && col < N) {
        Y[row * N + col] = warp_sum + Bias[col];
    }
}
"""

# Q8_0 matmul + residual add: Y += matmul + bias (fuses down_proj + add)
Q8_ADD_BINDINGS = [
    BufferBinding(name='X', binding=0, access='read_write', elem_type='f32'),
    BufferBinding(name='W_Q8', binding=1, access='read_write', elem_type='u32'),
    BufferBinding(name='Scales', binding=2, access='read_write', elem_type='u32'),
    BufferBinding(name='Bias', binding=3, access='read_write', elem_type='f32'),
    BufferBinding(name='Y', binding=4, access='read_write', elem_type='f32'),
    BufferBinding(name='_params_', binding=5, access='read_write', elem_type='u32'),
]

WSGL_Q8_0_ADD_KERNEL = WSGL_Q8_0_KERNEL.replace(
    "Y[row * N + col] = warp_sum + Bias[col];",
    "Y[row * N + col] += warp_sum + Bias[col];"
)


# ---------------------------------------------------------------------------
# Cooperative matrix Q8_0 matmul — ORT/llama.cpp-style tiled prefill GEMM
# ---------------------------------------------------------------------------
#
# Uses chromium_experimental_subgroup_matrix for hardware cooperative matrix
# multiply-accumulate. This path is intended for multi-token prefill GEMMs
# (MxK @ KxN, M>1) where a tiled kernel can amortize dispatch overhead.
#
# The current implementation targets the Vulkan configs exposed by Dawn on
# NVIDIA today: f16 inputs with f32 accumulation at 16x16x16. Activations
# and dequantized Q8 weights are staged into f16 workgroup tiles, multiplied
# via subgroup-matrix MMA, then written back as fp32.
#
# Q8_0 layout matches the existing WGSL Q8 path:
# - W_Q8: u32 packed int8 weights, 4 weights per word, row-major by output row
# - Scales: fp16 per 32-element block, packed as two fp16 per u32
#
# Grid: (ceil(N/64), ceil(M/32))
# M = number of tokens (prefill), N = output features, K = input features

SUBGROUP_MATRIX_Q8_BINDINGS = [
    BufferBinding(name='X', binding=0, access='read_write', elem_type='f32'),
    BufferBinding(name='W_Q8', binding=1, access='read_write', elem_type='u32'),
    BufferBinding(name='Scales', binding=2, access='read_write', elem_type='u32'),
    BufferBinding(name='Bias', binding=3, access='read_write', elem_type='f32'),
    BufferBinding(name='Y', binding=4, access='read_write', elem_type='f32'),
    BufferBinding(name='_params_', binding=5, access='read_write', elem_type='u32'),
]


def pack_subgroup_matrix_params(M, N, K):
    """Pack M, N, K as 3 u32 params."""
    data = struct.pack('<III', M, N, K)
    while len(data) < 16:
        data += b'\x00'
    return np.frombuffer(data, dtype=np.uint32).copy()


WGSL_SUBGROUP_MATRIX_Q8_KERNEL = """
enable f16;
enable subgroups;
enable chromium_experimental_subgroup_matrix;

@group(0) @binding(0) var<storage, read_write> X: array<f32>;
@group(0) @binding(1) var<storage, read_write> W_Q8: array<u32>;
@group(0) @binding(2) var<storage, read_write> Scales: array<u32>;
@group(0) @binding(3) var<storage, read_write> Bias: array<f32>;
@group(0) @binding(4) var<storage, read_write> Y: array<f32>;
@group(0) @binding(5) var<storage, read_write> _params_: array<u32>;

// 32×32 output tile, 4 subgroups each computing a 16×16 sub-tile
// Subgroup layout: 2×2 grid, sg_row = sg_id & 1, sg_col = sg_id >> 1
const TILE_ROWS: u32 = 32u;
const TILE_COLS: u32 = 32u;
const TILE_K: u32 = 16u;
const SCALE_BLOCK: u32 = 32u;
const MAX_K_TILES: u32 = 384u;
const WG_SIZE: u32 = 128u;

var<workgroup> tile_A: array<f16, TILE_ROWS * TILE_K>;   // 32×16
var<workgroup> tile_B: array<f16, TILE_COLS * TILE_K>;   // 32×16
var<workgroup> tile_C: array<f32, TILE_ROWS * TILE_COLS>; // 32×32

@compute @workgroup_size(128)
fn main(@builtin(local_invocation_id) lid: vec3<u32>,
        @builtin(workgroup_id) wid: vec3<u32>,
        @builtin(subgroup_id) subgroup_id: u32) {
    let M = _params_[0];
    let N = _params_[1];
    let K = _params_[2];
    let local_idx = lid.x;

    let row_base = wid.y * TILE_ROWS;
    let col_base = wid.x * TILE_COLS;

    // Each subgroup owns a 16×16 sub-tile of the 32×32 output
    let sg_row = subgroup_id & 1u;
    let sg_col = subgroup_id >> 1u;

    let weight_stride = K / 4u;
    let scale_stride = K / SCALE_BLOCK;

    var matC: subgroup_matrix_result<f32, 16, 16>;
    for (var tile_idx = 0u; tile_idx < MAX_K_TILES; tile_idx += 1u) {
        let k_base = tile_idx * TILE_K;

        // Load tile_A: 32×16 (128 threads, 4 elements each)
        for (var i = local_idx; i < TILE_ROWS * TILE_K; i += WG_SIZE) {
            let local_row = i / TILE_K;
            let local_k = i % TILE_K;
            let global_row = row_base + local_row;
            let global_k = k_base + local_k;
            if (global_row < M && global_k < K) {
                tile_A[i] = f16(X[global_row * K + global_k]);
            } else {
                tile_A[i] = 0.0h;
            }
        }

        // Load tile_B: 32×16 dequantized Q8_0 weights
        for (var i = local_idx; i < TILE_COLS * TILE_K; i += WG_SIZE) {
            let local_col = i / TILE_K;
            let local_k = i % TILE_K;
            let global_col = col_base + local_col;
            let global_k = k_base + local_k;
            if (global_col < N && global_k < K) {
                let packed = W_Q8[global_col * weight_stride + global_k / 4u];
                let shift = (global_k & 3u) * 8u;
                let q = f32(extractBits(i32(packed), shift, 8u));
                let scale_idx = global_col * scale_stride + global_k / SCALE_BLOCK;
                let sp = unpack2x16float(Scales[scale_idx / 2u]);
                let scale = select(sp.x, sp.y, (scale_idx & 1u) != 0u);
                tile_B[i] = f16(q * scale);
            } else {
                tile_B[i] = 0.0h;
            }
        }

        workgroupBarrier();

        // Each subgroup loads its 16×16 sub-tile and does MMA
        let a_offset = sg_row * 16u * TILE_K;        // row offset in tile_A
        let b_offset = sg_col * 16u * TILE_K;        // col offset in tile_B

        let matA = subgroupMatrixLoad<subgroup_matrix_left<f16, 16, 16>>(
            &tile_A, a_offset, false, TILE_K);
        let matB = subgroupMatrixLoad<subgroup_matrix_right<f16, 16, 16>>(
            &tile_B, b_offset, true, TILE_K);
        matC = subgroupMatrixMultiplyAccumulate(matA, matB, matC);

        workgroupBarrier();
    }

    // Store each subgroup's 16×16 result into the 32×32 tile_C
    let c_offset = sg_row * 16u * TILE_COLS + sg_col * 16u;
    subgroupMatrixStore(&tile_C, c_offset, matC, false, TILE_COLS);
    workgroupBarrier();

    // Write tile_C to global Y with bias
    for (var i = local_idx; i < TILE_ROWS * TILE_COLS; i += WG_SIZE) {
        let local_row = i / TILE_COLS;
        let local_col = i % TILE_COLS;
        let global_row = row_base + local_row;
        let global_col = col_base + local_col;
        if (global_row < M && global_col < N) {
            Y[global_row * N + global_col] = tile_C[i] + Bias[global_col];
        }
    }
}
"""

# Cooperative matrix Q8_0 + residual add: Y[i] += matmul + bias
SUBGROUP_MATRIX_Q8_ADD_BINDINGS = list(SUBGROUP_MATRIX_Q8_BINDINGS)  # same layout

WSGL_SUBGROUP_MATRIX_Q8_ADD_KERNEL = WGSL_SUBGROUP_MATRIX_Q8_KERNEL.replace(
    "Y[global_row * N + global_col] = tile_C[i] + Bias[global_col];",
    "Y[global_row * N + global_col] += tile_C[i] + Bias[global_col];"
)


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


# ---------------------------------------------------------------------------
# Chunked GQA decode attention — two-pass for high GPU occupancy
# ---------------------------------------------------------------------------
#
# At seq_len=1024+, the single-workgroup-per-head attention kernel achieves
# only ~3% memory bandwidth utilization (16 WGs × 128 threads = 2048
# threads on a GPU with 172K capacity). Splitting T into chunks gives
# n_head × n_chunks workgroups, increasing occupancy proportionally.
#
# Pass 1: grid=(n_head, n_chunks). Each WG computes partial online softmax
#   (max_score, sum_exp, weighted_v_acc) over chunk_size T positions.
# Pass 2: grid=(n_head,). Merges partials across chunks.

GQA_CHUNK_SIZE = 64

GQA_CHUNKED_PASS1_BINDINGS = [
    BufferBinding(name='Q', binding=0, access='read_write', elem_type='f32'),
    BufferBinding(name='K_cache', binding=1, access='read_write', elem_type='f32'),
    BufferBinding(name='V_cache', binding=2, access='read_write', elem_type='f32'),
    BufferBinding(name='Partials', binding=3, access='read_write', elem_type='f32'),
    BufferBinding(name='_params_', binding=4, access='read_write', elem_type='u32'),
]

GQA_CHUNKED_PASS2_BINDINGS = [
    BufferBinding(name='Partials', binding=0, access='read_write', elem_type='f32'),
    BufferBinding(name='Out', binding=1, access='read_write', elem_type='f32'),
    BufferBinding(name='_params_', binding=2, access='read_write', elem_type='u32'),
]


def pack_gqa_chunked_params(kv_stride, n_rep, T_total, chunk_size,
                             n_chunks, scale_bits, neg_inf_bits,
                             max_chunks=32):
    data = struct.pack('<IIIIIiiI', kv_stride, n_rep, T_total, chunk_size,
                       n_chunks, scale_bits, neg_inf_bits, max_chunks)
    return np.frombuffer(data, dtype=np.uint32).copy()


WGSL_GQA_CHUNKED_PASS1 = """
enable subgroups;

@group(0) @binding(0) var<storage, read_write> Q: array<f32>;
@group(0) @binding(1) var<storage, read_write> K_cache: array<f32>;
@group(0) @binding(2) var<storage, read_write> V_cache: array<f32>;
@group(0) @binding(3) var<storage, read_write> Partials: array<f32>;
@group(0) @binding(4) var<storage, read_write> _params_: array<u32>;

const HD: u32 = 128u;
const CHUNK: u32 = 64u;
// Each thread processes HD_PER_THREAD elements, total 32 * 4 = 128
const HD_PER_THREAD: u32 = 4u;

@compute @workgroup_size(32)
fn main(@builtin(local_invocation_id) lid: vec3<u32>,
        @builtin(workgroup_id) wid: vec3<u32>) {
    let head = wid.x;
    let chunk_id = wid.y;
    let lane = lid.x;

    let kv_stride = _params_[0];
    let n_rep = _params_[1];
    let T_total = _params_[2];
    let n_chunks = _params_[4];
    let scale = bitcast<f32>(_params_[5]);
    let neg_inf = bitcast<f32>(_params_[6]);
    let max_chunks = _params_[7];

    let kv_head = head / n_rep;
    let kv_off = kv_head * HD;
    let q_base = head * HD;

    // Load Q into registers (4 elements per thread)
    let q0 = Q[q_base + lane * HD_PER_THREAD];
    let q1 = Q[q_base + lane * HD_PER_THREAD + 1u];
    let q2 = Q[q_base + lane * HD_PER_THREAD + 2u];
    let q3 = Q[q_base + lane * HD_PER_THREAD + 3u];

    let t_start = chunk_id * CHUNK;

    var acc0: f32 = 0.0; var acc1: f32 = 0.0;
    var acc2: f32 = 0.0; var acc3: f32 = 0.0;
    var m_prev: f32 = neg_inf;
    var l_prev: f32 = 0.0;

    for (var i = 0u; i < CHUNK; i = i + 1u) {
        let t = t_start + i;
        let valid = t < T_total;

        let k_base = select(0u, t * kv_stride + kv_off, valid);
        let k0 = select(0.0, K_cache[k_base + lane * HD_PER_THREAD], valid);
        let k1 = select(0.0, K_cache[k_base + lane * HD_PER_THREAD + 1u], valid);
        let k2 = select(0.0, K_cache[k_base + lane * HD_PER_THREAD + 2u], valid);
        let k3 = select(0.0, K_cache[k_base + lane * HD_PER_THREAD + 3u], valid);

        // Full 128-element dot product via 4-element partial + subgroupAdd
        let partial = q0 * k0 + q1 * k1 + q2 * k2 + q3 * k3;
        let dot = subgroupAdd(partial);
        let raw_score = dot * scale;
        let score = select(neg_inf, raw_score, valid);

        let m_new = max(m_prev, score);
        let exp_prev = exp(m_prev - m_new);
        let exp_score = exp(score - m_new);
        let l_new = l_prev * exp_prev + exp_score;
        let rescale = l_prev * exp_prev / max(l_new, 1e-10);
        let w = exp_score / max(l_new, 1e-10);

        let v_base = select(0u, t * kv_stride + kv_off, valid);
        let v0 = select(0.0, V_cache[v_base + lane * HD_PER_THREAD], valid);
        let v1 = select(0.0, V_cache[v_base + lane * HD_PER_THREAD + 1u], valid);
        let v2 = select(0.0, V_cache[v_base + lane * HD_PER_THREAD + 2u], valid);
        let v3 = select(0.0, V_cache[v_base + lane * HD_PER_THREAD + 3u], valid);

        acc0 = acc0 * rescale + v0 * w;
        acc1 = acc1 * rescale + v1 * w;
        acc2 = acc2 * rescale + v2 * w;
        acc3 = acc3 * rescale + v3 * w;

        m_prev = m_new;
        l_prev = l_new;
    }

    // Use max_chunks for the stride to prevent overlapping writes
    // across heads. Only write if this chunk is within n_chunks.
    let partial_stride = HD + 2u;
    let base = head * max_chunks * partial_stride + chunk_id * partial_stride;
    if (chunk_id < n_chunks) {
        if (lane == 0u) {
            Partials[base] = m_prev;
            Partials[base + 1u] = l_prev;
        }
        Partials[base + 2u + lane * HD_PER_THREAD] = acc0;
        Partials[base + 2u + lane * HD_PER_THREAD + 1u] = acc1;
        Partials[base + 2u + lane * HD_PER_THREAD + 2u] = acc2;
        Partials[base + 2u + lane * HD_PER_THREAD + 3u] = acc3;
    }
}
"""

WGSL_GQA_CHUNKED_PASS2 = """
enable subgroups;

@group(0) @binding(0) var<storage, read_write> Partials: array<f32>;
@group(0) @binding(1) var<storage, read_write> Out: array<f32>;
@group(0) @binding(2) var<storage, read_write> _params_: array<u32>;

const HD: u32 = 128u;
const HD_PER_THREAD: u32 = 4u;

@compute @workgroup_size(32)
fn main(@builtin(local_invocation_id) lid: vec3<u32>,
        @builtin(workgroup_id) wid: vec3<u32>) {
    let head = wid.x;
    let lane = lid.x;

    let n_chunks = _params_[4];
    let neg_inf = bitcast<f32>(_params_[6]);
    let max_chunks = _params_[7];

    let partial_stride = HD + 2u;
    let head_base = head * max_chunks * partial_stride;

    var acc0: f32 = 0.0; var acc1: f32 = 0.0;
    var acc2: f32 = 0.0; var acc3: f32 = 0.0;
    var m_prev: f32 = neg_inf;
    var l_prev: f32 = 0.0;

    for (var c = 0u; c < n_chunks; c = c + 1u) {
        let base = head_base + c * partial_stride;
        let m_chunk = Partials[base];
        let l_chunk = Partials[base + 1u];
        let v0 = Partials[base + 2u + lane * HD_PER_THREAD];
        let v1 = Partials[base + 2u + lane * HD_PER_THREAD + 1u];
        let v2 = Partials[base + 2u + lane * HD_PER_THREAD + 2u];
        let v3 = Partials[base + 2u + lane * HD_PER_THREAD + 3u];

        if (l_chunk > 0.0) {
            let m_new = max(m_prev, m_chunk);
            let exp_prev = exp(m_prev - m_new);
            let exp_chunk = exp(m_chunk - m_new);
            let l_new = l_prev * exp_prev + l_chunk * exp_chunk;
            let rescale = l_prev * exp_prev / max(l_new, 1e-10);
            let w = l_chunk * exp_chunk / max(l_new, 1e-10);

            acc0 = acc0 * rescale + v0 * w;
            acc1 = acc1 * rescale + v1 * w;
            acc2 = acc2 * rescale + v2 * w;
            acc3 = acc3 * rescale + v3 * w;

            m_prev = m_new;
            l_prev = l_new;
        }
    }

    Out[head * HD + lane * HD_PER_THREAD] = acc0;
    Out[head * HD + lane * HD_PER_THREAD + 1u] = acc1;
    Out[head * HD + lane * HD_PER_THREAD + 2u] = acc2;
    Out[head * HD + lane * HD_PER_THREAD + 3u] = acc3;
}
"""


