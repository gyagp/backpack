enable subgroups;

// Optimized Q8_0 matmul with vec4 loads for X (activation vector).
// Uses array<vec4<f32>> binding to enable 128-bit coalesced memory reads.
// K_PER_ITER=8: each lane processes 8 elements per 256-element stride.
// TILE_N=8: each workgroup computes 8 output elements (one per warp).

@group(0) @binding(0) var<storage, read_write> X_v4: array<vec4<f32>>;
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

    let warp_id = tid / 32u;
    let lane = tid % 32u;
    let col = tile_col * TILE_N + warp_id;

    let n_strides = K / 256u;
    let stride_w = K / 4u;

    var acc: f32 = 0.0;

    if (col < N) {
        let w_base = col * stride_w;
        let n_blocks = K / 32u;
        let s_base = col * n_blocks;

        // X base in vec4 units: row * K / 4
        let x_base_v4 = row * (K / 4u);

        for (var g = 0u; g < n_strides; g = g + 1u) {
            // Read 8 fp32 activations via 2 × vec4 loads (128-bit each)
            let k_v4_base = g * 64u + lane * 2u;  // 256 elements / 4 = 64 vec4s per stride
            let xv0 = X_v4[x_base_v4 + k_v4_base];
            let xv1 = X_v4[x_base_v4 + k_v4_base + 1u];

            // Read 2 packed u32 weights (8 int8 values)
            let w_off = w_base + g * 64u + lane * 2u;
            let pw0 = W_Q8[w_off];
            let pw1 = W_Q8[w_off + 1u];

            // Extract int8 → f32 via extractBits (sign-extended)
            let wv0 = vec4<f32>(f32(extractBits(i32(pw0), 0u, 8u)),
                                f32(extractBits(i32(pw0), 8u, 8u)),
                                f32(extractBits(i32(pw0), 16u, 8u)),
                                f32(extractBits(i32(pw0), 24u, 8u)));
            let wv1 = vec4<f32>(f32(extractBits(i32(pw1), 0u, 8u)),
                                f32(extractBits(i32(pw1), 8u, 8u)),
                                f32(extractBits(i32(pw1), 16u, 8u)),
                                f32(extractBits(i32(pw1), 24u, 8u)));

            // Per-block scales (2 blocks per 8 elements)
            let block0 = g * 8u + (lane * 8u) / 32u;
            let block1 = g * 8u + (lane * 8u + 4u) / 32u;
            let sp0 = unpack2x16float(Scales[(s_base + block0) / 2u]);
            let scale0 = select(sp0.x, sp0.y, ((s_base + block0) & 1u) != 0u);
            let sp1 = unpack2x16float(Scales[(s_base + block1) / 2u]);
            let scale1 = select(sp1.x, sp1.y, ((s_base + block1) & 1u) != 0u);

            acc += dot(xv0, wv0) * scale0 + dot(xv1, wv1) * scale1;
        }
    }

    let warp_sum = subgroupAdd(acc);

    if (lane == 0u && col < N) {
        Y[row * N + col] = warp_sum + Bias[col];
    }
}
