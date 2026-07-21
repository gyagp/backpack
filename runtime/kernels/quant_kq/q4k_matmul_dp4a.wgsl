requires packed_4x8_integer_dot_product;
enable subgroups;

@group(0) @binding(0) var<storage, read> X: array<f32>;
@group(0) @binding(1) var<storage, read> W: array<u32>;
@group(0) @binding(2) var<storage, read> Bias: array<f32>;
@group(0) @binding(3) var<storage, read_write> Y: array<f32>;
@group(0) @binding(4) var<storage, read> P: array<u32>;

const QK_K: u32 = 256u;
const BLOCK_WORDS: u32 = 36u;
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
    let col = wid.y * 8u + warp;
    let K = P[0];
    let N = P[1];
    let nb = P[2];
    let rs = P[3];
    let row = wid.x;
    let block32 = tid / 32u;
    let elem32 = tid & 31u;
    let pack_lane = elem32 & 3u;
    let pack_group = elem32 / 4u;
    var acc = 0.0;

    for (var b = 0u; b < nb; b++) {
        let k = b * QK_K + tid;
        let xv = select(0.0, X[row * K + k], k < K);
        var amax = abs(xv);
        amax = max(amax, subgroupShuffleXor(amax, 16u));
        amax = max(amax, subgroupShuffleXor(amax, 8u));
        amax = max(amax, subgroupShuffleXor(amax, 4u));
        amax = max(amax, subgroupShuffleXor(amax, 2u));
        amax = max(amax, subgroupShuffleXor(amax, 1u));
        let scale = amax / 127.0;
        if (elem32 == 0u) { xs[block32] = scale; }
        let safe_scale = select(1.0, scale, scale != 0.0);
        let qi = clamp(i32(round(xv / safe_scale)), -127, 127);
        var packed = u32(qi & 255) << (pack_lane * 8u);
        packed |= subgroupShuffleXor(packed, 1u);
        packed |= subgroupShuffleXor(packed, 2u);
        if (pack_lane == 0u) {
            xq[block32 * 8u + pack_group] = packed;
        }
        workgroupBarrier();

        if (col < N) {
            let base = col * rs + b * BLOCK_WORDS;
            let dm = unpack2x16float(W[base]);
            let sb = lane / 4u;
            let j = sb & 3u;
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
            let qgroup = sb / 2u;
            let high = (sb & 1u) != 0u;
            let elem0 = (lane & 3u) * 8u;
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
            let aq0 = xq[lane * 2u];
            let aq1 = xq[lane * 2u + 1u];
            let dot = dot4I8Packed(aq0, w0) + dot4I8Packed(aq1, w1);
            let sum = dot4I8Packed(aq0, 0x01010101u) +
                      dot4I8Packed(aq1, 0x01010101u);
            acc += xs[sb] * (dm.x * f32(sc) * f32(dot) -
                             dm.y * f32(mn) * f32(sum));
        }
        workgroupBarrier();
    }

    let total = subgroupAdd(acc);
    if (lane == 0u && col < N) {
        Y[row * N + col] = total + Bias[col];
    }
}
