enable f16;
enable subgroups;
enable chromium_experimental_subgroup_matrix;

// Double-buffered subgroup-matrix Q8_0 GEMM:
//   Y[M×N] = X[M×K] × W[N×K]^T + Bias[N]
//
// Optimizations:
//   - Double-buffered smem: load next tile while computing current
//   - TILE_K=32: 2 MMA calls per K-block, halves barrier count
//   - Uniform params: enable early loop exit at actual K
//   - Vectorized 4-at-a-time Q8 dequant from packed u32
//
// 4 subgroups, 32×32 output tile. Grid: (ceil(N/32), ceil(M/32))

@group(0) @binding(0) var<storage, read_write> X: array<f32>;
@group(0) @binding(1) var<storage, read_write> W_Q8: array<u32>;
@group(0) @binding(2) var<storage, read_write> Scales: array<u32>;
@group(0) @binding(3) var<storage, read_write> Bias: array<f32>;
@group(0) @binding(4) var<storage, read_write> Y: array<f32>;

struct Params { M: u32, N: u32, K: u32, pad: u32, };
@group(0) @binding(5) var<uniform> params: Params;

const TM: u32 = 64u;
const TN: u32 = 64u;
const TK: u32 = 32u;
const MK: u32 = 16u;     // MMA tile K
const SB: u32 = 32u;     // Q8 scale block size
const WG: u32 = 512u;
const AB_A: u32 = 2048u; // TM * TK = 64*32
const AB_B: u32 = 2048u; // TN * TK = 64*32

var<workgroup> tA: array<array<f16, 2048>, 2>;  // double-buf A [64×32]
var<workgroup> tB: array<array<f16, 2048>, 2>;  // double-buf B [64×32]
var<workgroup> tC: array<f32, 4096>;             // output [64×64]

@compute @workgroup_size(512)
fn main(@builtin(local_invocation_id) lid: vec3<u32>,
        @builtin(workgroup_id) wid: vec3<u32>,
        @builtin(subgroup_id) sg_id: u32) {
    let M = params.M;  let N = params.N;  let K = params.K;
    let lx = lid.x;
    let rb = wid.y * TM;
    let cb = wid.x * TN;
    let sr = sg_id & 3u;   // 4 rows of 16 = 64
    let sc = sg_id >> 2u;  // 4 cols of 16 = 64
    let ws = K / 4u;
    let nkt = K / TK;

    var mC: subgroup_matrix_result<f32, 16, 16>;

    // ── Prefetch tile 0 into buffer 0 ────────────────────────────────
    for (var i = lx; i < AB_A; i += WG) {
        let r = i / TK;  let c = i % TK;
        let gr = rb + r;
        tA[0][i] = select(0.0h, f16(X[gr * K + c]), gr < M);
    }
    // Vectorized weight dequant: load 1 u32 (4 packed i8), write 4 f16
    // AB_B=2048 f16 = 512 u32 weights. 512 threads → 1 u32 per thread.
    {
        let wi = lx;  // 0..511 → one u32 per thread
        let row = wi / (TK / 4u);  // which of 64 weight rows
        let col4 = wi % (TK / 4u);  // which group of 4 i8 [0..7]
        let gc = cb + row;
        let base_k = col4 * 4u;
        if (gc < N) {
            let pk = W_Q8[gc * ws + col4];
            let si = gc * (K / SB) + base_k / SB;
            let sp = unpack2x16float(Scales[si / 2u]);
            let scale = select(sp.x, sp.y, (si & 1u) != 0u);
            let smem_base = row * TK + base_k;
            tB[0][smem_base]      = f16(f32(extractBits(i32(pk), 0u, 8u)) * scale);
            tB[0][smem_base + 1u] = f16(f32(extractBits(i32(pk), 8u, 8u)) * scale);
            tB[0][smem_base + 2u] = f16(f32(extractBits(i32(pk), 16u, 8u)) * scale);
            tB[0][smem_base + 3u] = f16(f32(extractBits(i32(pk), 24u, 8u)) * scale);
        } else {
            let smem_base = row * TK + base_k;
            tB[0][smem_base] = 0.0h; tB[0][smem_base + 1u] = 0.0h;
            tB[0][smem_base + 2u] = 0.0h; tB[0][smem_base + 3u] = 0.0h;
        }
    }
    workgroupBarrier();

    // ── Main loop: compute cur tile, load next ───────────────────────
    for (var ti = 0u; ti < nkt; ti++) {
        let cur = ti & 1u;
        let nxt = 1u - cur;
        let kn = (ti + 1u) * TK;

        // Load next tile (if not last iteration)
        if (ti + 1u < nkt) {
            for (var i = lx; i < AB_A; i += WG) {
                let r = i / TK;  let c = i % TK;
                let gr = rb + r;
                tA[nxt][i] = select(0.0h, f16(X[gr * K + kn + c]), gr < M);
            }
            // Vectorized weight dequant for next tile
            {
                let wi = lx;
                let row = wi / (TK / 4u);
                let col4 = wi % (TK / 4u);
                let gc = cb + row;
                let base_k = kn + col4 * 4u;
                if (gc < N && base_k < K) {
                    let pk = W_Q8[gc * ws + base_k / 4u];
                    let si = gc * (K / SB) + base_k / SB;
                    let sp = unpack2x16float(Scales[si / 2u]);
                    let scale = select(sp.x, sp.y, (si & 1u) != 0u);
                    let smem_base = row * TK + col4 * 4u;
                    tB[nxt][smem_base]      = f16(f32(extractBits(i32(pk), 0u, 8u)) * scale);
                    tB[nxt][smem_base + 1u] = f16(f32(extractBits(i32(pk), 8u, 8u)) * scale);
                    tB[nxt][smem_base + 2u] = f16(f32(extractBits(i32(pk), 16u, 8u)) * scale);
                    tB[nxt][smem_base + 3u] = f16(f32(extractBits(i32(pk), 24u, 8u)) * scale);
                } else {
                    let smem_base = row * TK + col4 * 4u;
                    tB[nxt][smem_base] = 0.0h; tB[nxt][smem_base + 1u] = 0.0h;
                    tB[nxt][smem_base + 2u] = 0.0h; tB[nxt][smem_base + 3u] = 0.0h;
                }
            }
        }

        // Compute: 2 MMA calls (K=32 = 2×16)
        let ao = sr * 16u * TK;
        let bo = sc * 16u * TK;

        let mA0 = subgroupMatrixLoad<subgroup_matrix_left<f16, 16, 16>>(
            &tA[cur], ao, false, TK);
        let mB0 = subgroupMatrixLoad<subgroup_matrix_right<f16, 16, 16>>(
            &tB[cur], bo, true, TK);
        mC = subgroupMatrixMultiplyAccumulate(mA0, mB0, mC);

        let mA1 = subgroupMatrixLoad<subgroup_matrix_left<f16, 16, 16>>(
            &tA[cur], ao + MK, false, TK);
        let mB1 = subgroupMatrixLoad<subgroup_matrix_right<f16, 16, 16>>(
            &tB[cur], bo + MK, true, TK);
        mC = subgroupMatrixMultiplyAccumulate(mA1, mB1, mC);

        workgroupBarrier();
    }

    // ── Store result ─────────────────────────────────────────────────
    subgroupMatrixStore(&tC, sr * 16u * TN + sc * 16u, mC, false, TN);
    workgroupBarrier();

    for (var i = lx; i < TM * TN; i += WG) {
        let r = i / TN;  let c = i % TN;
        let gr = rb + r;  let gc = cb + c;
        if (gr < M && gc < N) {
            Y[gr * N + gc] = tC[i] + Bias[gc];
        }
    }
}
