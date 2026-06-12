struct Params {
    row_length : u32,
    epsilon : f32,
};

@group(0) @binding(0) var<storage, read> input : array<f32>;
@group(0) @binding(1) var<storage, read_write> output : array<f32>;
@group(0) @binding(2) var<uniform> params : Params;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid : vec3u) {
    let row = gid.x;
    let offset = row * params.row_length;
    let end = offset + params.row_length;
    if (end > arrayLength(&input)) {
        return;
    }

    var sum_sq : f32 = 0.0;
    for (var i = offset; i < end; i = i + 1u) {
        let v = input[i];
        sum_sq = sum_sq + v * v;
    }

    let rms = sqrt(sum_sq / f32(params.row_length) + params.epsilon);
    let inv_rms = 1.0 / rms;

    for (var i = offset; i < end; i = i + 1u) {
        output[i] = input[i] * inv_rms;
    }
}
