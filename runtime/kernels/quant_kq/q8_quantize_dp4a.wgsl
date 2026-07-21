requires packed_4x8_integer_dot_product;
enable subgroups;

// Quantize one f32 activation vector into the Q8 layout consumed by the
// prequantized Q4_K DP4A matvec. One workgroup handles 256 values.
@group(0) @binding(0) var<storage, read> X: array<f32>;
@group(0) @binding(1) var<storage, read_write> XQ: array<u32>;
@group(0) @binding(2) var<storage, read_write> XS: array<f32>;
@group(0) @binding(3) var<storage, read> P: array<u32>;

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) lid: vec3<u32>,
        @builtin(workgroup_id) wid: vec3<u32>) {
    let tid = lid.x;
    let lane = tid & 31u;
    let block32 = tid / 32u;
    let pack_lane = lane & 3u;
    let pack_group = lane / 4u;
    let K = P[0];
    let k = wid.x * 256u + tid;
    let xv = select(0.0, X[k], k < K);
    var amax = abs(xv);
    amax = max(amax, subgroupShuffleXor(amax, 16u));
    amax = max(amax, subgroupShuffleXor(amax, 8u));
    amax = max(amax, subgroupShuffleXor(amax, 4u));
    amax = max(amax, subgroupShuffleXor(amax, 2u));
    amax = max(amax, subgroupShuffleXor(amax, 1u));
    let scale = amax / 127.0;
    let global_block = wid.x * 8u + block32;
    if (lane == 0u) { XS[global_block] = scale; }
    let safe_scale = select(1.0, scale, scale != 0.0);
    let qi = clamp(i32(round(xv / safe_scale)), -127, 127);
    var packed = u32(qi & 255) << (pack_lane * 8u);
    packed |= subgroupShuffleXor(packed, 1u);
    packed |= subgroupShuffleXor(packed, 2u);
    if (pack_lane == 0u) {
        XQ[wid.x * 64u + block32 * 8u + pack_group] = packed;
    }
}
