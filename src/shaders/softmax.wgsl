@group(0) @binding(0) var<storage, read> input : array<f32>;
@group(0) @binding(1) var<storage, read_write> output : array<f32>;
@group(0) @binding(2) var<uniform> params : vec2u; // (row_length, num_rows)

var<workgroup> shared_max : array<f32, 256>;
var<workgroup> shared_sum : array<f32, 256>;

@compute @workgroup_size(256)
fn main(
    @builtin(local_invocation_id) lid : vec3u,
    @builtin(workgroup_id) wid : vec3u,
) {
    let row_len = params.x;
    let row = wid.x;
    let tid = lid.x;
    let row_offset = row * row_len;

    // Pass 1: find max across the row
    var local_max = -3.402823e+38f; // -FLT_MAX
    for (var i = tid; i < row_len; i += 256u) {
        local_max = max(local_max, input[row_offset + i]);
    }
    shared_max[tid] = local_max;
    workgroupBarrier();

    // Reduce max
    for (var stride = 128u; stride > 0u; stride >>= 1u) {
        if (tid < stride) {
            shared_max[tid] = max(shared_max[tid], shared_max[tid + stride]);
        }
        workgroupBarrier();
    }
    let row_max = shared_max[0];
    workgroupBarrier();

    // Pass 2: compute sum of exp(x - max)
    var local_sum = 0.0f;
    for (var i = tid; i < row_len; i += 256u) {
        local_sum += exp(input[row_offset + i] - row_max);
    }
    shared_sum[tid] = local_sum;
    workgroupBarrier();

    // Reduce sum
    for (var stride = 128u; stride > 0u; stride >>= 1u) {
        if (tid < stride) {
            shared_sum[tid] += shared_sum[tid + stride];
        }
        workgroupBarrier();
    }
    let row_sum = shared_sum[0];
    workgroupBarrier();

    // Pass 3: normalize
    let inv_sum = 1.0 / row_sum;
    for (var i = tid; i < row_len; i += 256u) {
        output[row_offset + i] = exp(input[row_offset + i] - row_max) * inv_sum;
    }
}
