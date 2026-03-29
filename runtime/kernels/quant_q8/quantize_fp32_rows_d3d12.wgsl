// @meta bindings=4
requires packed_4x8_integer_dot_product;
enable subgroups;

// Quantize fp32 activations [M×K] into packed int8 + per-block scales.
// Grid: (ceil(K/256), M, 1)

@group(0) @binding(0) var<storage, read_write> X: array<f32>;
@group(0) @binding(1) var<storage, read_write> XQ: array<u32>;
@group(0) @binding(2) var<storage, read_write> XS: array<f32>;

struct Params { M: u32, K: u32, pad0: u32, pad1: u32, };
@group(0) @binding(3) var<uniform> params: Params;

const BK: u32 = 256u;

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) lid: vec3<u32>,
        @builtin(workgroup_id) wid: vec3<u32>) {
    let block_group = wid.x;
    let row = wid.y;
    let tid = lid.x;

    let M = params.M;
    let K = params.K;
    let row_valid = row < M;
    let gk = block_group * BK + tid;
    let in_range = row_valid && gk < K;

    let block_id = tid / 32u;
    let elem_in_block = tid % 32u;
    let pack_lane = elem_in_block % 4u;
    let pack_group = elem_in_block / 4u;

    let x_val = select(0.0, X[row * K + gk], in_range);
    var max_val = abs(x_val);
    max_val = max(max_val, subgroupShuffleXor(max_val, 16u));
    max_val = max(max_val, subgroupShuffleXor(max_val, 8u));
    max_val = max(max_val, subgroupShuffleXor(max_val, 4u));
    max_val = max(max_val, subgroupShuffleXor(max_val, 2u));
    max_val = max(max_val, subgroupShuffleXor(max_val, 1u));

    let x_scale = max_val / 127.0;
    let safe_scale = select(1.0, x_scale, x_scale != 0.0);
    let q_val = clamp(i32(round(x_val / safe_scale)), -127, 127);

    let byte_val = u32(q_val & 0xFF);
    let shifted = byte_val << (pack_lane * 8u);
    var packed = shifted;
    packed = packed | subgroupShuffleXor(packed, 1u);
    packed = packed | subgroupShuffleXor(packed, 2u);

    let stride_q = K / 4u;
    let stride_s = K / 32u;
    let block_base_q = block_group * 64u + block_id * 8u + pack_group;
    let block_base_s = block_group * 8u + block_id;

    if (row_valid && elem_in_block == 0u && block_base_s < stride_s) {
        XS[row * stride_s + block_base_s] = x_scale;
    }
    if (row_valid && pack_lane == 0u && block_base_q < stride_q) {
        XQ[row * stride_q + block_base_q] = packed;
    }
}
