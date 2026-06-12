struct Params {
    row_length : u32,
    epsilon : f32,
};

@group(0) @binding(0) var<storage, read> input : array<f32>;
@group(0) @binding(1) var<storage, read> weights : array<f32>;
@group(0) @binding(2) var<storage, read_write> output : array<f32>;
@group(0) @binding(3) var<uniform> params : Params;

var<workgroup> shared_sums : array<f32, 64>;

@compute @workgroup_size(64)
fn main(
    @builtin(local_invocation_id) lid : vec3u,
    @builtin(workgroup_id) wid : vec3u,
) {
    let row = wid.x;
    let local_id = lid.x;
    let offset = row * params.row_length;
    let end = offset + params.row_length;
    if (end > arrayLength(&input)) {
        return;
    }

    // Parallel reduction for sum of squares
    var partial_sum : f32 = 0.0;
    for (var i = local_id; i < params.row_length; i = i + 64u) {
        let v = input[offset + i];
        partial_sum = partial_sum + v * v;
    }
    shared_sums[local_id] = partial_sum;
    workgroupBarrier();

    // Tree reduction
    for (var stride = 32u; stride > 0u; stride = stride >> 1u) {
        if (local_id < stride) {
            shared_sums[local_id] = shared_sums[local_id] + shared_sums[local_id + stride];
        }
        workgroupBarrier();
    }

    let inv_rms = 1.0 / sqrt(shared_sums[0] / f32(params.row_length) + params.epsilon);

    // Fused normalize + scale by weights
    for (var i = local_id; i < params.row_length; i = i + 64u) {
        output[offset + i] = input[offset + i] * inv_rms * weights[i];
    }
}
