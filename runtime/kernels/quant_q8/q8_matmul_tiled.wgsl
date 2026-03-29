// @meta bindings=6
enable subgroups;

// Shared-memory tiled Q8_0 GEMM for batched prefill:
//   Y[T×N] = X[T×K] × W[N×K]^T + Bias[N]
//
// Tile: TILE_M(8) rows × TILE_N(32) columns per workgroup.
// 256 threads = 8 warps, each warp processes 4 output columns.
// X rows are loaded into shared memory and reused across all 32 columns.
//
// vs. q8_matmul_batched (TILE_N=8): 4× fewer workgroups, 4× more X reuse.
//
// Grid: (ceil(T/TILE_M), ceil(N/TILE_N), 1)
// Workgroup: 256 threads
//
// Shared memory: TILE_M × 256 = 2048 floats = 8 KB per K-block.

@group(0) @binding(0) var<storage, read_write> X: array<f32>;
@group(0) @binding(1) var<storage, read_write> W_Q8: array<u32>;
@group(0) @binding(2) var<storage, read_write> Scales: array<u32>;
@group(0) @binding(3) var<storage, read_write> Bias: array<f32>;
@group(0) @binding(4) var<storage, read_write> Y: array<f32>;
@group(0) @binding(5) var<storage, read_write> _params_: array<u32>;

const TILE_N: u32 = 32u;  // output columns per tile (4 per warp)
const TILE_M: u32 = 8u;   // output rows per tile
const COLS_PER_WARP: u32 = 4u;
const BK_LOAD: u32 = 256u;
const MAX_ITERS: u32 = 24u;

// Shared memory: TILE_M rows × BK_LOAD columns of X
var<workgroup> smem_x: array<f32, 2048>;  // 8 × 256

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) lid: vec3<u32>,
        @builtin(workgroup_id) wid: vec3<u32>) {
    let tile_row = wid.x;
    let tile_col = wid.y;
    let tid = lid.x;

    let K = _params_[0];
    let N = _params_[1];
    let T = _params_[2];

    let warp_id = tid / 32u;
    let lane = tid % 32u;

    let stride_w = K / 4u;
    let n_blocks = K / 32u;

    // Each warp handles COLS_PER_WARP=4 output columns
    var acc: array<array<f32, 4>, 8>;  // [TILE_M][COLS_PER_WARP]
    for (var m = 0u; m < TILE_M; m++) {
        for (var c = 0u; c < COLS_PER_WARP; c++) {
            acc[m][c] = 0.0;
        }
    }

    // Pre-compute column indices and validity
    var cols: array<u32, 4>;
    var col_valid: array<bool, 4>;
    var w_bases: array<u32, 4>;
    var s_bases: array<u32, 4>;
    for (var c = 0u; c < COLS_PER_WARP; c++) {
        cols[c] = tile_col * TILE_N + warp_id * COLS_PER_WARP + c;
        col_valid[c] = cols[c] < N;
        w_bases[c] = select(0u, cols[c] * stride_w, col_valid[c]);
        s_bases[c] = select(0u, cols[c] * n_blocks, col_valid[c]);
    }

    for (var g = 0u; g < MAX_ITERS; g = g + 1u) {
        let k_off = g * BK_LOAD;
        let in_k = k_off < K;

        // ── Cooperative load: X tile [TILE_M × BK_LOAD] into smem ───
        for (var r = 0u; r < TILE_M; r++) {
            let row = tile_row * TILE_M + r;
            let smem_idx = r * BK_LOAD + tid;
            if (in_k && row < T) {
                smem_x[smem_idx] = X[row * K + k_off + tid];
            } else {
                smem_x[smem_idx] = 0.0;
            }
        }
        workgroupBarrier();

        // ── Compute: each warp processes 4 columns from global W,
        //    dotting with X rows from shared memory ───────────────────
        if (in_k) {
            let k_base = lane * 8u;

            for (var c = 0u; c < COLS_PER_WARP; c++) {
                // Guard invalid columns but keep all threads active
                // (no divergent control flow before subgroupAdd)
                var wv0: vec4<f32> = vec4<f32>(0.0, 0.0, 0.0, 0.0);
                var wv1: vec4<f32> = vec4<f32>(0.0, 0.0, 0.0, 0.0);
                var scale0: f32 = 0.0;
                var scale1: f32 = 0.0;

                if (col_valid[c]) {
                    let w_off = w_bases[c] + g * 64u + lane * 2u;
                    let pw0 = W_Q8[w_off];
                    let pw1 = W_Q8[w_off + 1u];

                    wv0 = vec4<f32>(
                        f32(extractBits(i32(pw0), 0u, 8u)),
                        f32(extractBits(i32(pw0), 8u, 8u)),
                        f32(extractBits(i32(pw0), 16u, 8u)),
                        f32(extractBits(i32(pw0), 24u, 8u)));
                    wv1 = vec4<f32>(
                        f32(extractBits(i32(pw1), 0u, 8u)),
                        f32(extractBits(i32(pw1), 8u, 8u)),
                        f32(extractBits(i32(pw1), 16u, 8u)),
                        f32(extractBits(i32(pw1), 24u, 8u)));

                    let block0 = g * 8u + (lane * 8u) / 32u;
                    let block1 = g * 8u + (lane * 8u + 4u) / 32u;
                    let sp0 = unpack2x16float(Scales[(s_bases[c] + block0) / 2u]);
                    scale0 = select(sp0.x, sp0.y, ((s_bases[c] + block0) & 1u) != 0u);
                    let sp1 = unpack2x16float(Scales[(s_bases[c] + block1) / 2u]);
                    scale1 = select(sp1.x, sp1.y, ((s_bases[c] + block1) & 1u) != 0u);
                }

                for (var m = 0u; m < TILE_M; m++) {
                    let smem_base = m * BK_LOAD + k_base;
                    let xv0 = vec4<f32>(smem_x[smem_base],
                                        smem_x[smem_base + 1u],
                                        smem_x[smem_base + 2u],
                                        smem_x[smem_base + 3u]);
                    let xv1 = vec4<f32>(smem_x[smem_base + 4u],
                                        smem_x[smem_base + 5u],
                                        smem_x[smem_base + 6u],
                                        smem_x[smem_base + 7u]);
                    acc[m][c] += dot(xv0, wv0) * scale0 + dot(xv1, wv1) * scale1;
                }
            }
        }

        workgroupBarrier();
    }

    // Reduce across lanes within each warp, write results
    for (var c = 0u; c < COLS_PER_WARP; c++) {
        for (var m = 0u; m < TILE_M; m++) {
            let warp_sum = subgroupAdd(acc[m][c]);
            let row = tile_row * TILE_M + m;
            if (lane == 0u && col_valid[c] && row < T) {
                Y[row * N + cols[c]] = warp_sum + Bias[cols[c]];
            }
        }
    }
}
