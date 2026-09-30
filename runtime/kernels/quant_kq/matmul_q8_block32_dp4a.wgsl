// @meta noregistry=true
requires packed_4x8_integer_dot_product;
enable subgroups;

// Decode-specialized ORT block-Q8 projection. Quantize each 32-value
// activation block once per workgroup, then share it across eight output rows
// and use DP4A for the dot products. ORT weights are unsigned bytes with an
// implicit zero point of 128; XOR 0x80 converts them to signed packed bytes.
struct Params { M: u32, N: u32, K: u32, _pad: u32 };
@group(0) @binding(0) var<storage, read> X: array<f32>;
@group(0) @binding(1) var<storage, read> W: array<u32>;
@group(0) @binding(2) var<storage, read> scales: array<u32>;
@group(0) @binding(3) var<storage, read_write> Y: array<f32>;
@group(0) @binding(4) var<uniform> p: Params;

var<workgroup> xq: array<u32, 64>;
var<workgroup> xs: array<f32, 8>;

fn scale_at(index: u32) -> f32 {
    let pair = unpack2x16float(scales[index >> 1u]);
    return select(pair.x, pair.y, (index & 1u) != 0u);
}

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3<u32>,
        @builtin(local_invocation_id) lid: vec3<u32>) {
    let logical_warp = lid.x / 32u;
    let lane = lid.x & 31u;
    let n = wid.x * 8u + logical_warp;
    let valid = n < p.N;

    let words_per_row = p.K / 4u;
    let groups_per_row = p.K / 32u;
    var acc = 0.0;
    let block = lid.x / 32u;
    let elem = lid.x & 31u;
    let pack_lane = elem & 3u;
    let pack = elem / 4u;

    for (var k0 = 0u; k0 < p.K; k0 += 256u) {
        let xv = X[k0 + lid.x];
        var xmax = abs(xv);
        xmax = max(xmax, subgroupShuffleXor(xmax, 16u));
        xmax = max(xmax, subgroupShuffleXor(xmax, 8u));
        xmax = max(xmax, subgroupShuffleXor(xmax, 4u));
        xmax = max(xmax, subgroupShuffleXor(xmax, 2u));
        xmax = max(xmax, subgroupShuffleXor(xmax, 1u));
        let scale_x = xmax / 127.0;
        if (elem == 0u) { xs[block] = scale_x; }
        let safe_scale = select(1.0, scale_x, scale_x != 0.0);
        let qx = u32(clamp(i32(round(xv / safe_scale)), -127, 127)) & 255u;
        var packed_x = qx << (pack_lane * 8u);
        packed_x |= subgroupShuffleXor(packed_x, 1u);
        packed_x |= subgroupShuffleXor(packed_x, 2u);
        if (pack_lane == 0u) { xq[block * 8u + pack] = packed_x; }
        workgroupBarrier();

        if (valid) {
            let word = k0 / 4u + lane * 2u;
            let w0 = W[n * words_per_row + word] ^ 0x80808080u;
            let w1 = W[n * words_per_row + word + 1u] ^ 0x80808080u;
            let idot = dot4I8Packed(xq[lane * 2u], w0) +
                       dot4I8Packed(xq[lane * 2u + 1u], w1);
            let group = k0 / 32u + lane / 4u;
            acc += f32(idot) * xs[lane / 4u] *
                   scale_at(n * groups_per_row + group);
        }
        workgroupBarrier();
    }

    // Explicit logical-warp reduction is valid for both wave32 and wave64.
    acc += subgroupShuffleXor(acc, 16u);
    acc += subgroupShuffleXor(acc, 8u);
    acc += subgroupShuffleXor(acc, 4u);
    acc += subgroupShuffleXor(acc, 2u);
    acc += subgroupShuffleXor(acc, 1u);
    if (lane == 0u && valid) { Y[n] = acc; }
}
