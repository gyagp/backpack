@group(0) @binding(0) var<storage, read> W: array<u32>;
@group(0) @binding(1) var<storage, read> Token: array<i32>;
@group(0) @binding(2) var<storage, read_write> X: array<f32>;
@group(0) @binding(3) var<storage, read> P: array<u32>;

fn u8_at(addr: u32) -> u32 {
    return (W[addr / 4u] >> ((addr & 3u) * 8u)) & 0xffu;
}

fn i8_at(addr: u32) -> i32 {
    let v = u8_at(addr);
    return select(i32(v), i32(v) - 256, v >= 128u);
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let k = gid.x;
    let K = P[0];
    if (k >= K) { return; }

    let row_stride_bytes = P[1] * 4u;
    let token = u32(max(Token[0], 0));
    let local = k & 255u;
    let group = local / 128u;
    let r = local & 127u;
    let quarter = r / 32u;
    let lane = r & 31u;
    let block = k / 256u;
    let base = token * row_stride_bytes + block * 210u;

    let ql_base = base + group * 64u;
    let qh = u8_at(base + 128u + group * 32u + lane);
    let ql = u8_at(ql_base + select(lane, 32u + lane,
                                    quarter == 1u || quarter == 3u));
    let low = select(ql & 15u, ql >> 4u, quarter >= 2u);
    let high_shift = quarter * 2u;
    let q = i32(low | (((qh >> high_shift) & 3u) << 4u)) - 32;

    let scale_index = group * 8u + lane / 16u + quarter * 2u;
    let scale = i8_at(base + 192u + scale_index);
    let dh = u8_at(base + 208u) | (u8_at(base + 209u) << 8u);
    let d = unpack2x16float(dh).x;
    X[k] = d * f32(scale) * f32(q);
}
