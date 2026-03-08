enable subgroups;

// Fused: RMSNorm + Q8_0 matmul.
// Reads raw X, computes RMSNorm inline (two-pass over X — second from L1 cache),
// then multiplies normalized X by Q8 weight matrix.
// Eliminates the separate rms_norm/rms_next dispatch.
//
// Pass 1: Load X, accumulate sum_sq per warp via subgroupAdd
// Pass 2: Re-load X (L1 cached, 8KB), multiply by rstd * NormW, dot with W_Q8
//
// Bindings:
//   0: X (read) — raw input vector (pre-norm), E floats
//   1: W_Q8 (read) — quantized weight matrix (N × K/4 u32)
//   2: Scales (read) — fp16 scales packed as u32
//   3: Bias (read) — per-output bias
//   4: Y (write) — output
//   5: _params_ — [K, N, 0, eps_as_u32]
//   6: NormW (read) — norm weight vector, E floats

@group(0) @binding(0) var<storage, read_write> X: array<f32>;
@group(0) @binding(1) var<storage, read_write> W_Q8: array<u32>;
@group(0) @binding(2) var<storage, read_write> Scales: array<u32>;
@group(0) @binding(3) var<storage, read_write> Bias: array<f32>;
@group(0) @binding(4) var<storage, read_write> Y: array<f32>;
@group(0) @binding(5) var<storage, read_write> _params_: array<u32>;
@group(0) @binding(6) var<storage, read_write> NormW: array<f32>;

const TILE_N: u32 = 8u;

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) lid: vec3<u32>,
        @builtin(workgroup_id) wid: vec3<u32>) {
    let row = wid.x;
    let tile_col = wid.y;
    let tid = lid.x;

    let K = _params_[0];
    let N = _params_[1];
    let eps = bitcast<f32>(_params_[3]);
    let x_base = row * K;

    let warp_id = tid / 32u;
    let lane = tid % 32u;
    let col = tile_col * TILE_N + warp_id;

    let n_strides = K / 256u;
    let stride_w = K / 4u;

    // ── Pass 1: Compute sum of squares for RMSNorm ──────────────────────
    // Each warp independently reads the full X and computes sum_sq.
    // X is 8KB (K=2048 × 4 bytes) — fits in L1 cache for pass 2.
    var sum_sq: f32 = 0.0;
    for (var g = 0u; g < n_strides; g = g + 1u) {
        let k_base = g * 256u + lane * 8u;
        let xv0 = vec4<f32>(X[x_base + k_base],
                            X[x_base + k_base + 1u],
                            X[x_base + k_base + 2u],
                            X[x_base + k_base + 3u]);
        let xv1 = vec4<f32>(X[x_base + k_base + 4u],
                            X[x_base + k_base + 5u],
                            X[x_base + k_base + 6u],
                            X[x_base + k_base + 7u]);
        sum_sq += dot(xv0, xv0) + dot(xv1, xv1);
    }
    let total_sq = subgroupAdd(sum_sq);
    let rstd = 1.0 / sqrt(total_sq / f32(K) + eps);

    // ── Pass 2: Normalized matmul ───────────────────────────────────────
    // Re-read X (L1 cached), apply RMSNorm weight, dot with Q8 weights.
    var acc: f32 = 0.0;

    if (col < N) {
        let w_base = col * stride_w;
        let n_blocks = K / 32u;
        let s_base = col * n_blocks;

        for (var g = 0u; g < n_strides; g = g + 1u) {
            let k_base = g * 256u + lane * 8u;

            // Re-read X from L1 cache + apply norm
            let raw0 = vec4<f32>(X[x_base + k_base],
                                 X[x_base + k_base + 1u],
                                 X[x_base + k_base + 2u],
                                 X[x_base + k_base + 3u]);
            let raw1 = vec4<f32>(X[x_base + k_base + 4u],
                                 X[x_base + k_base + 5u],
                                 X[x_base + k_base + 6u],
                                 X[x_base + k_base + 7u]);
            let nw0 = vec4<f32>(NormW[k_base],     NormW[k_base + 1u],
                                NormW[k_base + 2u], NormW[k_base + 3u]);
            let nw1 = vec4<f32>(NormW[k_base + 4u], NormW[k_base + 5u],
                                NormW[k_base + 6u], NormW[k_base + 7u]);
            let xv0 = raw0 * rstd * nw0;
            let xv1 = raw1 * rstd * nw1;

            // Read Q8 weights
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

    let warp_sum = subgroupAdd(acc);

    if (lane == 0u && col < N) {
        Y[row * N + col] = warp_sum + Bias[col];
    }
}
