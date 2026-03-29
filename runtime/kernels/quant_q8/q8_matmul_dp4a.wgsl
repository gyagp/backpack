// @meta bindings=6
requires packed_4x8_integer_dot_product;
enable subgroups;

// Q8_0 matmul using DP4A (dot4I8Packed) for integer dot products.
// W8A32: INT8 weights × FP32 activations.
//
// Strategy: quantize the fp32 activation vector to int8 per-block
// (same blocking as Q8_0 weights: blocks of 32), then use dp4a
// for the integer dot product, and rescale with both scales.
//
// Each lane processes 8 elements (2 dp4a calls per iteration).

@group(0) @binding(0) var<storage, read_write> X: array<f32>;
@group(0) @binding(1) var<storage, read_write> W_Q8: array<u32>;
@group(0) @binding(2) var<storage, read_write> Scales: array<u32>;
@group(0) @binding(3) var<storage, read_write> Bias: array<f32>;
@group(0) @binding(4) var<storage, read_write> Y: array<f32>;
@group(0) @binding(5) var<storage, read_write> _params_: array<u32>;

const TILE_N: u32 = 8u;
const MAX_STRIDES: u32 = 8u;   // ceil(2048 / 256) — supports up to K=2048

var<workgroup> smem_x_q: array<u32, 64>;   // quantized X: 256 int8 = 64 u32
var<workgroup> smem_x_s: array<f32, 8>;    // per-block scales for X: 256/32 = 8

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) lid: vec3<u32>,
        @builtin(workgroup_id) wid: vec3<u32>) {
    let row = wid.x;
    let tile_col = wid.y;
    let tid = lid.x;

    let K = _params_[0];
    let N = _params_[1];
    let x_base = row * K;

    let warp_id = tid / 32u;
    let lane = tid % 32u;
    let col = tile_col * TILE_N + warp_id;

    let stride_w = K / 4u;

    var acc: f32 = 0.0;
    let valid = col < N;
    let w_base = select(0u, col * stride_w, valid);
    let n_blocks = K / 32u;
    let s_base = select(0u, col * n_blocks, valid);

    for (var g = 0u; g < MAX_STRIDES; g = g + 1u) {
        let k_off = g * 256u;
        let in_range = k_off < K;

        // ── Phase 1: Cooperatively quantize X to int8 ──────────────────
        {
            let block_id = tid / 32u;  // 0..7
            let elem_in_block = tid % 32u;
            let x_val = select(0.0, X[x_base + k_off + tid], in_range);
            let abs_val = abs(x_val);

            // Reduce absmax within warp
            var max_val = abs_val;
            max_val = max(max_val, subgroupShuffleXor(max_val, 16u));
            max_val = max(max_val, subgroupShuffleXor(max_val, 8u));
            max_val = max(max_val, subgroupShuffleXor(max_val, 4u));
            max_val = max(max_val, subgroupShuffleXor(max_val, 2u));
            max_val = max(max_val, subgroupShuffleXor(max_val, 1u));

            let x_scale = max_val / 127.0;
            if (elem_in_block == 0u) {
                smem_x_s[block_id] = x_scale;
            }

            // Quantize this element to int8
            let safe_scale = select(1.0, x_scale, x_scale != 0.0);
            let q_val = clamp(i32(round(x_val / safe_scale)), -127, 127);

            // Pack 4 int8 values into u32 using subgroup shuffle
            let pack_lane = elem_in_block % 4u;
            let pack_group = elem_in_block / 4u;
            let byte_val = u32(q_val & 0xFF);
            let shifted = byte_val << (pack_lane * 8u);

            var packed = shifted;
            packed = packed | subgroupShuffleXor(packed, 1u);
            packed = packed | subgroupShuffleXor(packed, 2u);

            if (pack_lane == 0u) {
                smem_x_q[block_id * 8u + pack_group] = packed;
            }
        }
        workgroupBarrier();

        // ── Phase 2: DP4A matmul using quantized X ─────────────────────
        if (valid && in_range) {
            let k_base_in_stride = lane * 8u;
            let xq_off = k_base_in_stride / 4u;

            let xq0 = smem_x_q[xq_off];
            let xq1 = smem_x_q[xq_off + 1u];

            let w_off = w_base + g * 64u + lane * 2u;
            let wq0 = W_Q8[w_off];
            let wq1 = W_Q8[w_off + 1u];

            // DP4A: dot product of 4 packed int8 values
            let idot0 = dot4I8Packed(xq0, wq0);
            let idot1 = dot4I8Packed(xq1, wq1);

            // Rescale: result = int_sum * x_scale * w_scale
            let block0 = g * 8u + (lane * 8u) / 32u;
            let block1 = g * 8u + (lane * 8u + 4u) / 32u;

            let sp0 = unpack2x16float(Scales[(s_base + block0) / 2u]);
            let w_scale0 = select(sp0.x, sp0.y, ((s_base + block0) & 1u) != 0u);
            let sp1 = unpack2x16float(Scales[(s_base + block1) / 2u]);
            let w_scale1 = select(sp1.x, sp1.y, ((s_base + block1) & 1u) != 0u);

            let x_block0 = k_base_in_stride / 32u;
            let x_block1 = (k_base_in_stride + 4u) / 32u;
            let x_scale0 = smem_x_s[x_block0];
            let x_scale1 = smem_x_s[x_block1];

            acc += f32(idot0) * w_scale0 * x_scale0
                 + f32(idot1) * w_scale1 * x_scale1;
        }
        workgroupBarrier();
    }

    let warp_sum = subgroupAdd(acc);

    if (lane == 0u && valid) {
        Y[row * N + col] = warp_sum + Bias[col];
    }
}
