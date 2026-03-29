// @meta bindings=6
enable f16;
enable subgroups;
enable chromium_experimental_subgroup_matrix;

// Double-buffered MMA fused SiLU+down+residual:
//   Y[M×N] += silu_mul(GateUp[M×2*IM]) × W[N×K]^T + Bias[N]
//
// Same double-buffer + TILE_K=32 as q8_matmul_mma but with SiLU
// activation computed during tile_A staging.
//
// params: [K=IM, N=E, IM, M=T]
// Grid: (ceil(N/32), ceil(M/32))

@group(0) @binding(0) var<storage, read_write> GateUp: array<f32>;
@group(0) @binding(1) var<storage, read_write> W_Q8: array<u32>;
@group(0) @binding(2) var<storage, read_write> Scales: array<u32>;
@group(0) @binding(3) var<storage, read_write> Bias: array<f32>;
@group(0) @binding(4) var<storage, read_write> Y: array<f32>;

struct Params { K: u32, N: u32, IM: u32, M: u32, };
@group(0) @binding(5) var<uniform> params: Params;

const TM: u32 = 64u;
const TN: u32 = 64u;
const TK: u32 = 32u;
const MK: u32 = 16u;
const SB: u32 = 32u;
const WG: u32 = 512u;
const AB_A: u32 = 2048u;
const AB_B: u32 = 2048u;

var<workgroup> tA: array<array<f16, 2048>, 2>;
var<workgroup> tB: array<array<f16, 2048>, 2>;
var<workgroup> tC: array<f32, 4096>;

fn load_silu(gu: ptr<storage, array<f32>, read_write>, row: u32, k: u32, IM: u32) -> f16 {
    let base = row * 2u * IM;
    let gate = (*gu)[base + k];
    let up = (*gu)[base + IM + k];
    return f16(gate / (1.0 + exp(-gate)) * up);
}

@compute @workgroup_size(512)
fn main(@builtin(local_invocation_id) lid: vec3<u32>,
        @builtin(workgroup_id) wid: vec3<u32>,
        @builtin(subgroup_id) sg_id: u32) {
    let K = params.K;  let N = params.N;
    let IM = params.IM;  let M = params.M;
    let lx = lid.x;
    let rb = wid.y * TM;
    let cb = wid.x * TN;
    let sr = sg_id & 3u;
    let sc = sg_id >> 2u;
    let ws = K / 4u;
    let nkt = K / TK;

    var mC: subgroup_matrix_result<f32, 16, 16>;

    // Prefetch tile 0
    for (var i = lx; i < AB_A; i += WG) {
        let r = i / TK;  let c = i % TK;
        let gr = rb + r;
        tA[0][i] = select(0.0h, load_silu(&GateUp, gr, c, IM), gr < M);
    }
    for (var i = lx; i < AB_B; i += WG) {
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

    for (var ti = 0u; ti < nkt; ti++) {
        let cur = ti & 1u;
        let nxt = 1u - cur;
        let kn = (ti + 1u) * TK;

        if (ti + 1u < nkt) {
            for (var i = lx; i < AB_A; i += WG) {
                let r = i / TK;  let c = i % TK;
                let gr = rb + r;
                tA[nxt][i] = select(0.0h, load_silu(&GateUp, gr, kn + c, IM), gr < M);
            }
            for (var i = lx; i < AB_B; i += WG) {
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

        let ao = sr * 16u * TK;
        let bo = sc * 16u * TK;
        let mA0 = subgroupMatrixLoad<subgroup_matrix_left<f16, 16, 16>>(&tA[cur], ao, false, TK);
        let mB0 = subgroupMatrixLoad<subgroup_matrix_right<f16, 16, 16>>(&tB[cur], bo, true, TK);
        mC = subgroupMatrixMultiplyAccumulate(mA0, mB0, mC);

        let mA1 = subgroupMatrixLoad<subgroup_matrix_left<f16, 16, 16>>(&tA[cur], ao + MK, false, TK);
        let mB1 = subgroupMatrixLoad<subgroup_matrix_right<f16, 16, 16>>(&tB[cur], bo + MK, true, TK);
        mC = subgroupMatrixMultiplyAccumulate(mA1, mB1, mC);

        workgroupBarrier();
    }

    subgroupMatrixStore(&tC, sr * 16u * TN + sc * 16u, mC, false, TN);
    workgroupBarrier();

    for (var i = lx; i < TM * TN; i += WG) {
        let r = i / TN;  let c = i % TN;
        let gr = rb + r;  let gc = cb + c;
        if (gr < M && gc < N) {
            Y[gr * N + gc] += tC[i] + Bias[gc];
        }
    }
}
