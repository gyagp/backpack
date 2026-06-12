@group(0) @binding(0) var<storage, read> input : array<f32>;
@group(0) @binding(1) var<storage, read_write> output : array<f32>;
@group(0) @binding(2) var<uniform> scale : f32;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid : vec3u) {
    let i = gid.x;
    if (i < arrayLength(&input)) {
        output[i] = input[i] * scale;
    }
}
