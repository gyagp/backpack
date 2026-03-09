enable subgroups;

// D3D12-optimized Q8_0 GEMM: double-buffered smem, no MMA.
//   Y[T×N] = X[T×K] × W[N×K]^T + Bias[N]
//
// Same binding layout as q8_matmul_tiled but with double-buffered
// X caching to overlap next tile load with current compute.
//
// TILE_M=8, TILE_N=32 (4 cols per warp), 256 threads.
// params: [K, N, T, 0]
//
// Grid: (ceil(T/TILE_M), ceil(N/TILE_N), 1)

@group(0) @binding(0) var<storage, read_write> X: array<f32>;
@group(0) @binding(1) var<storage, read_write> W_Q8: array<u32>;
@group(0) @binding(2) var<storage, read_write> Scales: array<u32>;
@group(0) @binding(3) var<storage, read_write> Bias: array<f32>;
@group(0) @binding(4) var<storage, read_write> Y: array<f32>;
@group(0) @binding(5) var<storage, read_write> _params_: array<u32>;

const TILE_N: u32 = 32u;
const TILE_M: u32 = 8u;
const COLS_PER_WARP: u32 = 4u;
const BK: u32 = 256u;
const WG: u32 = 256u;
const MAX_ITERS: u32 = 24u;  // ceil(6144/256)

// Double-buffered shared memory for X tile
var<workgroup> smem_x: array<array<f32, 2048>, 2>;  // [2][8×256]

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

    var acc: array<array<f32, 4>, 8>;
    for (var m = 0u; m < TILE_M; m++) {
        for (var c = 0u; c < COLS_PER_WARP; c++) {
            acc[m][c] = 0.0;
        }
    }

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

    // Prefetch first tile into buffer 0
    for (var r = 0u; r < TILE_M; r++) {
        let row = tile_row * TILE_M + r;
        let idx = r * BK + tid;
        smem_x[0][idx] = select(0.0, X[row * K + tid], row < T && tid < K);
    }
    workgroupBarrier();

    for (var g = 0u; g < MAX_ITERS; g++) {
        let cur = g & 1u;
        let nxt = 1u - cur;
        let k_off = g * BK;
        let in_k = k_off < K;
        let k_next = (g + 1u) * BK;

        // Load next tile into nxt buffer (if not last)
        if (g + 1u < MAX_ITERS) {
            for (var r = 0u; r < TILE_M; r++) {
                let row = tile_row * TILE_M + r;
                let idx = r * BK + tid;
                smem_x[nxt][idx] = select(0.0, X[row * K + k_next + tid], row < T && k_next + tid < K);
            }
        }

        // Compute: each warp processes 4 columns from global W,
        // dotting with X rows from shared memory
        if (in_k) {
        let k_base = lane * 8u;

        for (var c = 0u; c < COLS_PER_WARP; c++) {
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
                let smem_base = m * BK + k_base;
                let xv0 = vec4<f32>(smem_x[cur][smem_base],
                                    smem_x[cur][smem_base + 1u],
                                    smem_x[cur][smem_base + 2u],
                                    smem_x[cur][smem_base + 3u]);
                let xv1 = vec4<f32>(smem_x[cur][smem_base + 4u],
                                    smem_x[cur][smem_base + 5u],
                                    smem_x[cur][smem_base + 6u],
                                    smem_x[cur][smem_base + 7u]);
                acc[m][c] += dot(xv0, wv0) * scale0 + dot(xv1, wv1) * scale1;
            }
        }
        }

        workgroupBarrier();
    }

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
