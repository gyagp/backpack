@group(0) @binding(0) var<storage, read> input : array<f32>;
@group(0) @binding(1) var<storage, read_write> output : array<f32>;
@group(0) @binding(2) var<uniform> params : array<vec4u, 3>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid : vec3u) {
    let total = params[0].x;
    let ndim = params[0].y;
    let idx = gid.x;
    if (idx >= total) {
        return;
    }

    let out_shape = params[1];
    let in_strides = params[2];

    var remaining = idx;
    var in_idx = 0u;

    if (ndim >= 1u) {
        let d0 = remaining / out_shape.y / out_shape.z / out_shape.w;
        remaining = remaining - d0 * out_shape.y * out_shape.z * out_shape.w;
        in_idx = in_idx + d0 * in_strides.x;
    }
    if (ndim >= 2u) {
        let d1 = remaining / out_shape.z / out_shape.w;
        remaining = remaining - d1 * out_shape.z * out_shape.w;
        in_idx = in_idx + d1 * in_strides.y;
    }
    if (ndim >= 3u) {
        let d2 = remaining / out_shape.w;
        remaining = remaining - d2 * out_shape.w;
        in_idx = in_idx + d2 * in_strides.z;
    }
    if (ndim >= 4u) {
        in_idx = in_idx + remaining * in_strides.w;
    }

    output[idx] = input[in_idx];
}
