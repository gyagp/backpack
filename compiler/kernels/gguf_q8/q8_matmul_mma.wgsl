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

const TM: u32 = 32u;
const TN: u32 = 32u;
const TK: u32 = 32u;
const MK: u32 = 16u;     // MMA tile K
const SB: u32 = 32u;     // Q8 scale block size
const WG: u32 = 128u;
const AB: u32 = 1024u;   // TM * TK = 32*32

var<workgroup> tA: array<array<f16, 1024>, 2>;  // double-buf A [32×32]
var<workgroup> tB: array<array<f16, 1024>, 2>;  // double-buf B [32×32]
var<workgroup> tC: array<f32, 1024>;             // output [32×32]

@compute @workgroup_size(128)
fn main(@builtin(local_invocation_id) lid: vec3<u32>,
        @builtin(workgroup_id) wid: vec3<u32>,
        @builtin(subgroup_id) sg_id: u32) {
    let M = params.M;  let N = params.N;  let K = params.K;
    let lx = lid.x;
    let rb = wid.y * TM;
    let cb = wid.x * TN;
    let sr = sg_id & 1u;
    let sc = sg_id >> 1u;
    let ws = K / 4u;
    let nkt = K / TK;

    var mC: subgroup_matrix_result<f32, 16, 16>;

    // ── Prefetch tile 0 into buffer 0 ────────────────────────────────
    for (var i = lx; i < AB; i += WG) {
        let r = i / TK;  let c = i % TK;
        let gr = rb + r;
        tA[0][i] = select(0.0h, f16(X[gr * K + c]), gr < M);
    }
    for (var i = lx; i < AB; i += WG) {
        let r = i / TK;  let c = i % TK;
        let gc = cb + r;
        if (gc < N) {
            let pk = W_Q8[gc * ws + c / 4u];
            let q = f32(extractBits(i32(pk), (c & 3u) * 8u, 8u));
            let si = gc * (K / SB) + c / SB;
            let sp = unpack2x16float(Scales[si / 2u]);
            tB[0][i] = f16(q * select(sp.x, sp.y, (si & 1u) != 0u));
        } else { tB[0][i] = 0.0h; }
    }
    workgroupBarrier();

    // ── Main loop: compute cur tile, load next ───────────────────────
    for (var ti = 0u; ti < nkt; ti++) {
        let cur = ti & 1u;
        let nxt = 1u - cur;
        let kn = (ti + 1u) * TK;

        // Load next tile (if not last iteration)
        if (ti + 1u < nkt) {
            for (var i = lx; i < AB; i += WG) {
                let r = i / TK;  let c = i % TK;
                let gr = rb + r;
                tA[nxt][i] = select(0.0h, f16(X[gr * K + kn + c]), gr < M);
            }
            for (var i = lx; i < AB; i += WG) {
                let r = i / TK;  let c = i % TK;
                let gc = cb + r;  let gk = kn + c;
                if (gc < N) {
                    let pk = W_Q8[gc * ws + gk / 4u];
                    let q = f32(extractBits(i32(pk), (gk & 3u) * 8u, 8u));
                    let si = gc * (K / SB) + gk / SB;
                    let sp = unpack2x16float(Scales[si / 2u]);
                    tB[nxt][i] = f16(q * select(sp.x, sp.y, (si & 1u) != 0u));
                } else { tB[nxt][i] = 0.0h; }
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
