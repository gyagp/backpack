enable subgroups;

// Decode-specialized ORT block-Q8 projection. One logical 32-lane warp
// computes one output row; a 256-thread workgroup therefore produces 8 rows.
// Weights are unsigned bytes with implicit zero point 128 and one fp16 scale
// per 32 K values. Activations remain f32, avoiding an additional quantization.
struct Params { M: u32, N: u32, K: u32, _pad: u32 };
@group(0) @binding(0) var<storage, read> X: array<f32>;
@group(0) @binding(1) var<storage, read> W: array<u32>;
@group(0) @binding(2) var<storage, read> scales: array<u32>;
@group(0) @binding(3) var<storage, read_write> Y: array<f32>;
@group(0) @binding(4) var<uniform> p: Params;

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
    if (valid) {
        for (var word = lane; word < words_per_row; word += 32u) {
            let packed = W[n * words_per_row + word];
            let q = vec4<f32>(
                f32(i32(packed & 255u) - 128),
                f32(i32((packed >> 8u) & 255u) - 128),
                f32(i32((packed >> 16u) & 255u) - 128),
                f32(i32((packed >> 24u) & 255u) - 128));
            let k = word * 4u;
            let x = vec4<f32>(X[k], X[k + 1u], X[k + 2u], X[k + 3u]);
            acc += dot(x, q) * scale_at(n * groups_per_row + word / 8u);
        }
    }

    // Explicit logical-warp reduction is valid for both wave32 and wave64.
    acc += subgroupShuffleXor(acc, 16u);
    acc += subgroupShuffleXor(acc, 8u);
    acc += subgroupShuffleXor(acc, 4u);
    acc += subgroupShuffleXor(acc, 2u);
    acc += subgroupShuffleXor(acc, 1u);
    if (lane == 0u && valid) { Y[n] = acc; }
}
