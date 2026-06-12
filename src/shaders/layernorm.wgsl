struct Params {
    row_length : u32,
    epsilon : f32,
}

@group(0) @binding(0) var<storage, read> input : array<f32>;
@group(0) @binding(1) var<storage, read> scale : array<f32>;
@group(0) @binding(2) var<storage, read> bias : array<f32>;
@group(0) @binding(3) var<storage, read_write> output : array<f32>;
@group(0) @binding(4) var<uniform> params : Params;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid : vec3u) {
    let row = gid.x;
    let offset = row * params.row_length;
    let end = offset + params.row_length;
    if (end > arrayLength(&input)) {
        return;
    }

    var mean : f32 = 0.0;
    for (var i = offset; i < end; i = i + 1u) {
        mean = mean + input[i];
    }
    mean = mean / f32(params.row_length);

    var variance : f32 = 0.0;
    for (var i = offset; i < end; i = i + 1u) {
        let d = input[i] - mean;
        variance = variance + d * d;
    }
    variance = variance / f32(params.row_length);

    let inv_std = 1.0 / sqrt(variance + params.epsilon);
    for (var i = offset; i < end; i = i + 1u) {
        let j = i - offset;
        output[i] = (input[i] - mean) * inv_std * scale[j] + bias[j];
    }
}
