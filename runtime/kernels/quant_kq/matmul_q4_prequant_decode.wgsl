// @meta noregistry=true
requires packed_4x8_integer_dot_product;
enable subgroups;

@group(0) @binding(0) var<storage, read> XQ: array<u32>;
@group(0) @binding(1) var<storage, read> XS: array<u32>;
@group(0) @binding(2) var<storage, read> W: array<u32>;
@group(0) @binding(3) var<storage, read> WS: array<u32>;
@group(0) @binding(4) var<storage, read_write> Y: array<f32>;
struct Params { N: u32, K: u32, _pad0: u32, _pad1: u32 };
@group(0) @binding(5) var<uniform> p: Params;

fn fp16_at(buf: ptr<storage, array<u32>, read>, index: u32) -> f32 {
    let pair = unpack2x16float((*buf)[index >> 1u]);
    return select(pair.x, pair.y, (index & 1u) != 0u);
}

fn signed_first4(value: u32) -> u32 {
    let b0 = value & 0xFFu;
    let b1 = (value >> 8u) & 0xFFu;
    return pack4xI8(vec4<i32>(
        i32(b0 & 0xFu) - 8, i32(b0 >> 4u) - 8,
        i32(b1 & 0xFu) - 8, i32(b1 >> 4u) - 8));
}

fn signed_second4(value: u32) -> u32 {
    let b2 = (value >> 16u) & 0xFFu;
    let b3 = (value >> 24u) & 0xFFu;
    return pack4xI8(vec4<i32>(
        i32(b2 & 0xFu) - 8, i32(b2 >> 4u) - 8,
        i32(b3 & 0xFu) - 8, i32(b3 >> 4u) - 8));
}

@compute @workgroup_size(128)
fn main(@builtin(workgroup_id) wid: vec3<u32>,
        @builtin(local_invocation_id) lid: vec3<u32>) {
    let output_in_tile = lid.x / 32u;
    let lane = lid.x & 31u;
    let n = wid.x * 4u + output_in_tile;
    let blocks = p.K / 32u;
    let words_per_row = p.K / 8u;
    var acc = 0.0;

    if (n < p.N) {
        for (var block = lane; block < blocks; block += 32u) {
            let x_scale = fp16_at(&XS, block / 4u);
            let w_scale = fp16_at(&WS, n * blocks + block);
            let x_base = block * 8u;
            let w_base = n * words_per_row + block * 4u;
            var idot = 0;
            for (var word = 0u; word < 4u; word++) {
                let packed = W[w_base + word];
                idot += dot4I8Packed(XQ[x_base + word * 2u], signed_first4(packed));
                idot += dot4I8Packed(XQ[x_base + word * 2u + 1u], signed_second4(packed));
            }
            acc += f32(idot) * x_scale * w_scale;
        }
    }

    acc += subgroupShuffleXor(acc, 16u);
    acc += subgroupShuffleXor(acc, 8u);
    acc += subgroupShuffleXor(acc, 4u);
    acc += subgroupShuffleXor(acc, 2u);
    acc += subgroupShuffleXor(acc, 1u);
    if (lane == 0u && n < p.N) { Y[n] = acc; }
}
