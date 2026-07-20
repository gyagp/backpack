// @meta bindings=5

// Gather M rows from a repacked native-Q4 table and dequantize to fp32.
// Dispatch: (ceil(M*D/256), 1, 1).

@group(0) @binding(0) var<storage, read> W: array<u32>;
@group(0) @binding(1) var<storage, read> Scales: array<u32>;
@group(0) @binding(2) var<storage, read> Tokens: array<i32>;
@group(0) @binding(3) var<storage, read_write> Out: array<f32>;
@group(0) @binding(4) var<storage, read> P: array<u32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let M = P[0]; let D = P[1]; let vocab = P[2];
    let scale_out = bitcast<f32>(P[3]);
    let flat = gid.x;
    if (flat >= M * D) { return; }
    let row = flat / D; let i = flat % D;
    let token_i = Tokens[row];
    let token = select(0u, u32(token_i), token_i >= 0 && u32(token_i) < vocab);
    let wi = token * (D / 8u) + i / 8u;
    let q = (W[wi] >> (4u * (i & 7u))) & 15u;
    let block = token * (D / 32u) + i / 32u;
    let sp = unpack2x16float(Scales[block / 2u]);
    let s = select(sp.x, sp.y, (block & 1u) != 0u);
    Out[flat] = (f32(q) - 8.0) * s * scale_out;
}
