@group(0) @binding(0) var<storage, read> a : array<f32>;
@group(0) @binding(1) var<storage, read> b : array<f32>;
@group(0) @binding(2) var<storage, read_write> output : array<f32>;
@group(0) @binding(3) var<uniform> params : vec4u;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid : vec3u) {
    let idx = gid.x;
    let total = params.x;
    let a_count = params.y;
    if (idx >= total) {
        return;
    }
    if (idx < a_count) {
        output[idx] = a[idx];
    } else {
        output[idx] = b[idx - a_count];
    }
}
