@group(0) @binding(0) var<storage, read> input : array<f32>;
@group(0) @binding(1) var<storage, read_write> output : array<u32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid : vec3u) {
    let pair = gid.x;
    let i = pair * 2u;
    let n = arrayLength(&input);
    if (i >= n) { return; }
    let a = input[i];
    var b = 0.0;
    if (i + 1u < n) { b = input[i + 1u]; }
    output[pair] = pack2x16float(vec2f(a, b));
}
