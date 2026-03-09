enable subgroups;

// D3D12-optimized fused SiLU+down+residual: double-buffered smem.
//   Y[T×N] += silu_mul(GateUp[T×2*IM]) × W_down[N×K]^T
//
// Same as q8_matmul_d3d12 but computes SiLU into smem instead of X.
// params: [K=IM, N=E, IM, T]
//
// Grid: (ceil(T/TILE_M), ceil(N/TILE_N), 1)

@group(0) @binding(0) var<storage, read_write> GateUp: array<f32>;
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
const MAX_ITERS: u32 = 24u;

var<workgroup> smem_s: array<array<f32, 2048>, 2>;

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) lid: vec3<u32>,
        @builtin(workgroup_id) wid: vec3<u32>) {
    let tile_row = wid.x;
    let tile_col = wid.y;
    let tid = lid.x;

    let K = _params_[0];
    let N = _params_[1];
    let IM = _params_[2];
    let T = _params_[3];

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

    // Prefetch first SiLU tile
    for (var r = 0u; r < TILE_M; r++) {
        let row = tile_row * TILE_M + r;
        let idx = r * BK + tid;
        if (row < T && tid < K) {
            let gu_base = row * 2u * IM;
            let gate = GateUp[gu_base + tid];
            let up = GateUp[gu_base + IM + tid];
            smem_s[0][idx] = gate / (1.0 + exp(-gate)) * up;
        } else {
            smem_s[0][idx] = 0.0;
        }
    }
    workgroupBarrier();

    for (var g = 0u; g < MAX_ITERS; g++) {
        let cur = g & 1u;
        let nxt = 1u - cur;
        let k_off = g * BK;
        let in_k = k_off < K;
        let k_next = (g + 1u) * BK;

        if (g + 1u < MAX_ITERS) {
            for (var r = 0u; r < TILE_M; r++) {
                let row = tile_row * TILE_M + r;
                let idx = r * BK + tid;
                if (row < T && k_next + tid < K) {
                    let gu_base = row * 2u * IM;
                    let gate = GateUp[gu_base + k_next + tid];
                    let up = GateUp[gu_base + IM + k_next + tid];
                    smem_s[nxt][idx] = gate / (1.0 + exp(-gate)) * up;
                } else {
                    smem_s[nxt][idx] = 0.0;
                }
            }
        }

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
                    f32(extractBits(i32(pw0), 0u, 8u)), f32(extractBits(i32(pw0), 8u, 8u)),
                    f32(extractBits(i32(pw0), 16u, 8u)), f32(extractBits(i32(pw0), 24u, 8u)));
                wv1 = vec4<f32>(
                    f32(extractBits(i32(pw1), 0u, 8u)), f32(extractBits(i32(pw1), 8u, 8u)),
                    f32(extractBits(i32(pw1), 16u, 8u)), f32(extractBits(i32(pw1), 24u, 8u)));
                let block0 = g * 8u + (lane * 8u) / 32u;
                let block1 = g * 8u + (lane * 8u + 4u) / 32u;
                let sp0 = unpack2x16float(Scales[(s_bases[c] + block0) / 2u]);
                scale0 = select(sp0.x, sp0.y, ((s_bases[c] + block0) & 1u) != 0u);
                let sp1 = unpack2x16float(Scales[(s_bases[c] + block1) / 2u]);
                scale1 = select(sp1.x, sp1.y, ((s_bases[c] + block1) & 1u) != 0u);
            }

            for (var m = 0u; m < TILE_M; m++) {
                let smem_base = m * BK + k_base;
                let sv0 = vec4<f32>(smem_s[cur][smem_base], smem_s[cur][smem_base + 1u],
                                    smem_s[cur][smem_base + 2u], smem_s[cur][smem_base + 3u]);
                let sv1 = vec4<f32>(smem_s[cur][smem_base + 4u], smem_s[cur][smem_base + 5u],
                                    smem_s[cur][smem_base + 6u], smem_s[cur][smem_base + 7u]);
                acc[m][c] += dot(sv0, wv0) * scale0 + dot(sv1, wv1) * scale1;
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
                Y[row * N + cols[c]] += warp_sum + Bias[cols[c]];
            }
        }
    }
}
