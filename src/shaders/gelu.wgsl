@group(0) @binding(0) var<storage, read> input : array<f32>;
@group(0) @binding(1) var<storage, read_write> output : array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid : vec3u) {
    let i = gid.x;
    if (i < arrayLength(&input)) {
        let x = input[i];
        let c = 0.7978845608028654;
        output[i] = 0.5 * x * (1.0 + tanh(c * (x + 0.044715 * x * x * x)));
    }
}
