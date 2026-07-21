requires packed_4x8_integer_dot_product;
enable subgroups;

// Q4_K matvec over an activation quantized once by q8_quantize_dp4a.
// Each workgroup produces eight output rows (one per subgroup). Keeping one
// accumulator per subgroup isolates activation reuse from register-heavy
// multi-column tiling.
@group(0) @binding(0) var<storage, read> XQ: array<u32>;
@group(0) @binding(1) var<storage, read> XS: array<f32>;
@group(0) @binding(2) var<storage, read> W: array<u32>;
@group(0) @binding(3) var<storage, read> Bias: array<f32>;
@group(0) @binding(4) var<storage, read_write> Y: array<f32>;
@group(0) @binding(5) var<storage, read> P: array<u32>;

const BLOCK_WORDS: u32 = 36u;
const COLS_PER_WARP: u32 = 1u;
var<workgroup> xq: array<u32, 64>;
var<workgroup> xs: array<f32, 8>;

fn u8_at(base: u32, off: u32) -> u32 {
    return (W[base + off / 4u] >> ((off & 3u) * 8u)) & 255u;
}

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) lid: vec3<u32>,
        @builtin(workgroup_id) wid: vec3<u32>) {
    let tid = lid.x;
    let warp = tid / 32u;
    let lane = tid & 31u;
    let K = P[0];
    let N = P[1];
    let nb = P[2];
    let rs = P[3];
    var cols: array<u32, 1>;
    var valid: array<bool, 1>;
    var acc: array<f32, 1>;
    for (var c = 0u; c < COLS_PER_WARP; c++) {
        cols[c] = wid.y * 8u + warp;
        valid[c] = cols[c] < N;
        acc[c] = 0.0;
    }

    for (var b = 0u; b < nb; b++) {
        if (tid < 64u) { xq[tid] = XQ[b * 64u + tid]; }
        if (tid < 8u) { xs[tid] = XS[b * 8u + tid]; }
        workgroupBarrier();

        let sb = lane / 4u;
        let j = sb & 3u;
        let qgroup = sb / 2u;
        let high = (sb & 1u) != 0u;
        let elem0 = (lane & 3u) * 8u;
        let aq0 = xq[lane * 2u];
        let aq1 = xq[lane * 2u + 1u];
        let asum = dot4I8Packed(aq0, 0x01010101u) +
                   dot4I8Packed(aq1, 0x01010101u);

        for (var c = 0u; c < COLS_PER_WARP; c++) {
            if (valid[c]) {
                let base = cols[c] * rs + b * BLOCK_WORDS;
                let dm = unpack2x16float(W[base]);
                let dv = u8_at(base, 4u + j);
                let mv = u8_at(base, 8u + j);
                var sc: u32;
                var mn: u32;
                if (sb < 4u) {
                    sc = dv & 63u;
                    mn = mv & 63u;
                } else {
                    let md = u8_at(base, 12u + j);
                    sc = (md & 15u) | ((dv >> 2u) & 48u);
                    mn = (md >> 4u) | ((mv >> 2u) & 48u);
                }
                var w0 = 0u;
                var w1 = 0u;
                for (var i = 0u; i < 4u; i++) {
                    let qb0 = u8_at(base, 16u + qgroup * 32u + elem0 + i);
                    let qb1 = u8_at(base, 20u + qgroup * 32u + elem0 + i);
                    let q0 = select(qb0 & 15u, qb0 >> 4u, high);
                    let q1 = select(qb1 & 15u, qb1 >> 4u, high);
                    w0 |= q0 << (i * 8u);
                    w1 |= q1 << (i * 8u);
                }
                let dot = dot4I8Packed(aq0, w0) + dot4I8Packed(aq1, w1);
                acc[c] += xs[sb] * (dm.x * f32(sc) * f32(dot) -
                                     dm.y * f32(mn) * f32(asum));
            }
        }
        workgroupBarrier();
    }

    for (var c = 0u; c < COLS_PER_WARP; c++) {
        let total = subgroupAdd(acc[c]);
        if (lane == 0u && valid[c]) {
            Y[cols[c]] = total + Bias[cols[c]];
        }
    }
}
