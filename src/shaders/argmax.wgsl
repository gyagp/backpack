@group(0) @binding(0) var<storage, read> input : array<f32>;
@group(0) @binding(1) var<storage, read_write> output : array<u32>;
@group(0) @binding(2) var<uniform> params : u32; // element count

var<workgroup> shared_val : array<f32, 256>;
var<workgroup> shared_idx : array<u32, 256>;

@compute @workgroup_size(256)
fn main(
    @builtin(local_invocation_id) lid : vec3u,
) {
    let n = params;
    let tid = lid.x;

    var local_max = -3.402823e+38f;
    var local_idx = 0u;
    for (var i = tid; i < n; i += 256u) {
        let v = input[i];
        if (v > local_max) {
            local_max = v;
            local_idx = i;
        }
    }
    shared_val[tid] = local_max;
    shared_idx[tid] = local_idx;
    workgroupBarrier();

    for (var stride = 128u; stride > 0u; stride >>= 1u) {
        if (tid < stride) {
            if (shared_val[tid + stride] > shared_val[tid]) {
                shared_val[tid] = shared_val[tid + stride];
                shared_idx[tid] = shared_idx[tid + stride];
            }
        }
        workgroupBarrier();
    }

    if (tid == 0u) {
        output[0] = shared_idx[0];
    }
}
