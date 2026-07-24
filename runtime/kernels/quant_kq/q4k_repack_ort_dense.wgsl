// Convert raw row-major GGUF Q4_K blocks into the dense nibble and affine
// sidecar layout consumed by ORT's 64x64 DP4A matmul tile.
@group(0) @binding(0) var<storage, read> Raw: array<u32>;
@group(0) @binding(1) var<storage, read_write> Dense: array<u32>;
@group(0) @binding(2) var<storage, read_write> ScaleMin: array<f32>;
@group(0) @binding(3) var<storage, read> P: array<u32>;

fn pack8(p0: u32, p1: u32, high: bool) -> u32 {
    var result = 0u;
    for (var i = 0u; i < 4u; i++) {
        let byte = (p0 >> (i * 8u)) & 255u;
        let q = select(byte & 15u, byte >> 4u, high);
        result |= q << (i * 4u);
    }
    for (var i = 0u; i < 4u; i++) {
        let byte = (p1 >> (i * 8u)) & 255u;
        let q = select(byte & 15u, byte >> 4u, high);
        result |= q << ((i + 4u) * 4u);
    }
    return result;
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let K = P[0]; let N = P[1]; let blocksPerRow = P[2]; let rowStrideWords = P[3];
    let groupsPerRow = K / 32u; let group = gid.x;
    if (group >= N * groupsPerRow) { return; }
    let row = group / groupsPerRow; let groupInRow = group % groupsPerRow;
    let block = groupInRow / 8u; let subBlock = groupInRow & 7u;
    if (block >= blocksPerRow) { return; }
    let base = row * rowStrideWords + block * 36u;
    let qbase = base + 4u + (subBlock / 2u) * 8u;
    let high = (subBlock & 1u) != 0u;
    for (var part = 0u; part < 4u; part++) {
        Dense[group * 4u + part] = pack8(Raw[qbase + part * 2u], Raw[qbase + part * 2u + 1u], high);
    }
    let dm = unpack2x16float(Raw[base]); let shift = (subBlock & 3u) * 8u;
    let scaleByte = (Raw[base + 1u] >> shift) & 255u;
    let minByte = (Raw[base + 2u] >> shift) & 255u;
    var scale: u32; var minValue: u32;
    if (subBlock < 4u) { scale = scaleByte & 63u; minValue = minByte & 63u; }
    else { let highBits = (Raw[base + 3u] >> shift) & 255u;
        scale = (highBits & 15u) | ((scaleByte >> 2u) & 48u);
        minValue = (highBits >> 4u) | ((minByte >> 2u) & 48u); }
    ScaleMin[group * 2u] = dm.x * f32(scale);
    ScaleMin[group * 2u + 1u] = dm.y * f32(minValue);
}
