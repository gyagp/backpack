// @meta bindings=6
enable subgroups;

// Batched fused: SiLU·mul + Q8_0 down-proj + residual add for prefill.
// Y[T×N] += (silu(gate) * up) × W_down^T
// GateUp layout: [T rows × 2*IM cols] where gate=[0..IM), up=[IM..2*IM)
//
// Grid: (ceil(T/TILE_M), ceil(N/TILE_N), 1)

@group(0) @binding(0) var<storage, read_write> GateUp: array<f32>;
@group(0) @binding(1) var<storage, read_write> W_Q8: array<u32>;
@group(0) @binding(2) var<storage, read_write> Scales: array<u32>;
@group(0) @binding(3) var<storage, read_write> Bias: array<f32>;
@group(0) @binding(4) var<storage, read_write> Y: array<f32>;
@group(0) @binding(5) var<storage, read_write> _params_: array<u32>;

const TILE_N: u32 = 8u;
const TILE_M: u32 = 8u;
const MAX_STRIDES: u32 = 64u;  // supports up to K=16384

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) lid: vec3<u32>,
        @builtin(workgroup_id) wid: vec3<u32>) {
    let tile_row = wid.x;
    let tile_col = wid.y;
    let tid = lid.x;

    let K = _params_[0];   // IM
    let N = _params_[1];   // E
    let IM = _params_[2];  // same as K
    let T = _params_[3];

    let warp_id = tid / 32u;
    let lane = tid % 32u;
    let col = tile_col * TILE_N + warp_id;

    let stride_w = K / 4u;
    let n_blocks = K / 32u;

    var acc: array<f32, 8>;
    for (var m = 0u; m < TILE_M; m++) { acc[m] = 0.0; }

    let valid = col < N;
    let w_base = select(0u, col * stride_w, valid);
    let s_base = select(0u, col * n_blocks, valid);

    for (var g = 0u; g < MAX_STRIDES; g = g + 1u) {
        let k_base = g * 256u + lane * 8u;
        let in_k = g * 256u < K;

        var wv0: vec4<f32>;
        var wv1: vec4<f32>;
        var scale0: f32 = 0.0;
        var scale1: f32 = 0.0;

        if (valid && in_k) {
            let w_off = w_base + g * 64u + lane * 2u;
            let pw0 = W_Q8[w_off];
            let pw1 = W_Q8[w_off + 1u];

            wv0 = vec4<f32>(f32(extractBits(i32(pw0), 0u, 8u)),
                            f32(extractBits(i32(pw0), 8u, 8u)),
                            f32(extractBits(i32(pw0), 16u, 8u)),
                            f32(extractBits(i32(pw0), 24u, 8u)));
            wv1 = vec4<f32>(f32(extractBits(i32(pw1), 0u, 8u)),
                            f32(extractBits(i32(pw1), 8u, 8u)),
                            f32(extractBits(i32(pw1), 16u, 8u)),
                            f32(extractBits(i32(pw1), 24u, 8u)));

            let block0 = g * 8u + (lane * 8u) / 32u;
            let block1 = g * 8u + (lane * 8u + 4u) / 32u;
            let sp0 = unpack2x16float(Scales[(s_base + block0) / 2u]);
            scale0 = select(sp0.x, sp0.y, ((s_base + block0) & 1u) != 0u);
            let sp1 = unpack2x16float(Scales[(s_base + block1) / 2u]);
            scale1 = select(sp1.x, sp1.y, ((s_base + block1) & 1u) != 0u);
        }

        for (var m = 0u; m < TILE_M; m++) {
            let row = tile_row * TILE_M + m;
            if (row < T && valid && in_k) {
                // Apply silu·mul on-the-fly: silu(gate[k]) * up[k]
                let gu_base = row * 2u * IM;
                var sv0: vec4<f32>;
                var sv1: vec4<f32>;
                for (var e = 0u; e < 4u; e++) {
                    let idx = k_base + e;
                    let gate = GateUp[gu_base + idx];
                    let up   = GateUp[gu_base + IM + idx];
                    sv0[e] = gate / (1.0 + exp(-gate)) * up;
                }
                for (var e = 0u; e < 4u; e++) {
                    let idx = k_base + 4u + e;
                    let gate = GateUp[gu_base + idx];
                    let up   = GateUp[gu_base + IM + idx];
                    sv1[e] = gate / (1.0 + exp(-gate)) * up;
                }
                acc[m] += dot(sv0, wv0) * scale0 + dot(sv1, wv1) * scale1;
            }
        }
    }

    for (var m = 0u; m < TILE_M; m++) {
        let warp_sum = subgroupAdd(acc[m]);
        let row = tile_row * TILE_M + m;
        if (lane == 0u && valid && row < T) {
            Y[row * N + col] += warp_sum + Bias[col];
        }
    }
}
