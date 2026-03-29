// @meta bindings=6
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
        Y[row * N + col] += warp_sum + Bias[col];
    }
}
