// Scale — in-place element-wise multiply by scalar
// data[i] *= scale
// Dispatch: (ceil(N/512), 1, 1) — each thread handles 2 elements
// Params: [N, bitcast<u32>(scale)]

${T_READ_RW}
${T_WRITE2}

@group(0) @binding(0) var<storage, read_write> data: array<${T}>;
@group(0) @binding(1) var<storage, read> params: array<u32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let N = params[0];
    let scale = bitcast<f32>(params[1]);

    let base = gid.x * 2u;
    if (base >= N) { return; }

    let v0 = t_read_rw(&data, base) * scale;
    var v1: f32 = 0.0;
    if (base + 1u < N) {
        v1 = t_read_rw(&data, base + 1u) * scale;
    }
    t_write2(&data, base, v0, v1);
}
