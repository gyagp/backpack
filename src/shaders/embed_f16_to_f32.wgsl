struct Params {
    offset: u32,
    count: u32,
};

@group(0) @binding(0) var<storage, read> input : array<u32>;
@group(0) @binding(1) var<storage, read_write> output : array<f32>;
@group(0) @binding(2) var<uniform> params : Params;

fn f16_to_f32(bits: u32) -> f32 {
    let sign = (bits >> 15u) & 1u;
    let exp = (bits >> 10u) & 0x1Fu;
    let frac = bits & 0x3FFu;
    var f32_bits: u32;
    if (exp == 0u) {
        if (frac == 0u) {
            f32_bits = sign << 31u;
        } else {
            var e = 1u;
            var f = frac;
            loop {
                if ((f & 0x400u) != 0u) { break; }
                f = f << 1u;
                e = e - 1u;
            }
            f = f & 0x3FFu;
            f32_bits = (sign << 31u) | ((e + 112u) << 23u) | (f << 13u);
        }
    } else if (exp == 31u) {
        f32_bits = (sign << 31u) | 0x7F800000u | (frac << 13u);
    } else {
        f32_bits = (sign << 31u) | ((exp + 112u) << 23u) | (frac << 13u);
    }
    return bitcast<f32>(f32_bits);
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid : vec3u) {
    let i = gid.x;
    if (i >= params.count) { return; }
    let word_idx = (params.offset + i) >> 1u;
    let word = input[word_idx];
    let is_high = (params.offset + i) & 1u;
    let bits = select(word & 0xFFFFu, (word >> 16u) & 0xFFFFu, is_high == 1u);
    output[i] = f16_to_f32(bits);
}
