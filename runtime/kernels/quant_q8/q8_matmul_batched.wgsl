// @meta bindings=6
enable subgroups;

// Batched Q8_0 matmul for prefill: Y[T×N] = X[T×K] × W[K×N]^T
// Each workgroup computes a TILE_M × TILE_N output tile.
// Weight row is Q8_0: read once, reused across all T (M) rows.
//
// Grid: (ceil(T/TILE_M), ceil(N/TILE_N), 1)
// WG: 256 threads = 8 warps
//
// Tile: TILE_M=8 rows × TILE_N=8 cols, K iterated in blocks of 32
// Each warp handles one output column, all TILE_M rows.
//
// Bindings:
//   0: X — fp32 activations [T × K]
//   1: W_Q8 — int8 weights packed as u32 [N × K/4]
//   2: Scales — fp16 weight scales packed as u32
//   3: Bias — per-output bias [N]
//   4: Y — output [T × N]
//   5: _params_ — [K, N, T, 0]

@group(0) @binding(0) var<storage, read_write> X: array<f32>;
@group(0) @binding(1) var<storage, read_write> W_Q8: array<u32>;
@group(0) @binding(2) var<storage, read_write> Scales: array<u32>;
@group(0) @binding(3) var<storage, read_write> Bias: array<f32>;
@group(0) @binding(4) var<storage, read_write> Y: array<f32>;
@group(0) @binding(5) var<storage, read_write> _params_: array<u32>;

const TILE_N: u32 = 8u;
const TILE_M: u32 = 8u;
const MAX_STRIDES: u32 = 24u;  // ceil(6144/256)

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) lid: vec3<u32>,
        @builtin(workgroup_id) wid: vec3<u32>) {
    let tile_row = wid.x;   // which TILE_M block of rows
    let tile_col = wid.y;   // which TILE_N block of columns
    let tid = lid.x;

    let K = _params_[0];
    let N = _params_[1];
    let T = _params_[2];

    let warp_id = tid / 32u;
    let lane = tid % 32u;
    let col = tile_col * TILE_N + warp_id;

    let stride_w = K / 4u;
    let n_blocks = K / 32u;

    // Each warp accumulates TILE_M rows for its output column
    var acc: array<f32, 8>;  // TILE_M accumulators
    for (var m = 0u; m < TILE_M; m++) { acc[m] = 0.0; }

    let valid = col < N;
    let w_base = select(0u, col * stride_w, valid);
    let s_base = select(0u, col * n_blocks, valid);

    for (var g = 0u; g < MAX_STRIDES; g = g + 1u) {
        let k_base = g * 256u + lane * 8u;
        let in_k = g * 256u < K;

        // Read 2 packed u32 weights (8 int8 values) — same for all rows
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

        // For each of TILE_M rows, read X and dot with same weights
        for (var m = 0u; m < TILE_M; m++) {
            let row = tile_row * TILE_M + m;
            if (row < T && valid && in_k) {
                let x_base = row * K;
                let xv0 = vec4<f32>(X[x_base + k_base],
                                    X[x_base + k_base + 1u],
                                    X[x_base + k_base + 2u],
                                    X[x_base + k_base + 3u]);
                let xv1 = vec4<f32>(X[x_base + k_base + 4u],
                                    X[x_base + k_base + 5u],
                                    X[x_base + k_base + 6u],
                                    X[x_base + k_base + 7u]);
                acc[m] += dot(xv0, wv0) * scale0 + dot(xv1, wv1) * scale1;
            }
        }
    }

    // Reduce across lanes within each warp
    for (var m = 0u; m < TILE_M; m++) {
        let warp_sum = subgroupAdd(acc[m]);
        let row = tile_row * TILE_M + m;
        if (lane == 0u && valid && row < T) {
            Y[row * N + col] = warp_sum + Bias[col];
        }
    }
}
