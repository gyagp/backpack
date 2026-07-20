// @meta bindings=8
// Fused qwen35 DeltaNet beta/alpha projection and scalar gates.
// Computes:
//   beta = sigmoid(X * W_beta^T)
//   gate = softplus(X * W_alpha^T + dt_bias) * ssm_a
//
// W_Q8 rows are packed as beta rows followed by alpha rows.
// Params: [K, rank, 0, 0]

@group(0) @binding(0) var<storage, read>       X_v4:      array<vec4<f32>>;
@group(0) @binding(1) var<storage, read>       W_Q8:      array<u32>;
@group(0) @binding(2) var<storage, read>       Scales:    array<u32>;
@group(0) @binding(3) var<storage, read>       DtBias:    array<f32>;
@group(0) @binding(4) var<storage, read>       SsmA:      array<f32>;
@group(0) @binding(5) var<storage, read_write> BetaOut:   array<f32>;
@group(0) @binding(6) var<storage, read_write> GateOut:   array<f32>;
@group(0) @binding(7) var<storage, read>       _params_:  array<u32>;

const TILE_N: u32 = 8u;
var<workgroup> reduce_scratch: array<f32, 256>;

fn softplus(x: f32) -> f32 {
    return max(x, 0.0) + log(1.0 + exp(-abs(x)));
}

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) lid: vec3<u32>,
        @builtin(workgroup_id) wid: vec3<u32>) {
    let tile_col = wid.y;
    let tid = lid.x;

    let K = _params_[0];
    let rank = _params_[1];
    let N = rank * 2u;

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

        for (var g = 0u; g < n_strides; g = g + 1u) {
            let k_v4_base = g * 64u + lane * 2u;
            let xv0 = X_v4[k_v4_base];
            let xv1 = X_v4[k_v4_base + 1u];

            let w_off = w_base + g * 64u + lane * 2u;
            let pw0 = W_Q8[w_off];
            let pw1 = W_Q8[w_off + 1u];

            let wv0 = vec4<f32>(f32(extractBits(i32(pw0), 0u, 8u)),
                                f32(extractBits(i32(pw0), 8u, 8u)),
                                f32(extractBits(i32(pw0), 16u, 8u)),
                                f32(extractBits(i32(pw0), 24u, 8u)));
            let wv1 = vec4<f32>(f32(extractBits(i32(pw1), 0u, 8u)),
                                f32(extractBits(i32(pw1), 8u, 8u)),
                                f32(extractBits(i32(pw1), 16u, 8u)),
                                f32(extractBits(i32(pw1), 24u, 8u)));

            let block0 = g * 8u + (lane * 8u) / 32u;
            let block1 = g * 8u + (lane * 8u + 4u) / 32u;
            let sp0 = unpack2x16float(Scales[(s_base + block0) / 2u]);
            let scale0 = select(sp0.x, sp0.y, ((s_base + block0) & 1u) != 0u);
            let sp1 = unpack2x16float(Scales[(s_base + block1) / 2u]);
            let scale1 = select(sp1.x, sp1.y, ((s_base + block1) & 1u) != 0u);

            acc += dot(xv0, wv0) * scale0 + dot(xv1, wv1) * scale1;
        }
    }

    reduce_scratch[tid] = acc;
    workgroupBarrier();
    for (var offset = 16u; offset > 0u; offset = offset / 2u) {
        if (lane < offset) { reduce_scratch[tid] += reduce_scratch[tid + offset]; }
        workgroupBarrier();
    }
    let sum = reduce_scratch[warp_id * 32u];
    if (lane == 0u && col < N) {
        if (col < rank) {
            BetaOut[col] = 1.0 / (1.0 + exp(-sum));
        } else {
            let j = col - rank;
            GateOut[j] = softplus(sum + DtBias[j]) * SsmA[j];
        }
    }
}
