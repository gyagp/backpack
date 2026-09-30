// @meta noregistry=true
enable f16;

@group(0) @binding(0) var<storage, read> X: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> Skip: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> W: array<vec4<f16>>;
@group(0) @binding(3) var<storage, read_write> Y: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read_write> SkipOut: array<vec4<f32>>;
@group(0) @binding(5) var<storage, read> _params_: array<u32>;
var<workgroup> sum_sq_shared: array<f32, 128>;

@compute @workgroup_size(128)
fn main(@builtin(workgroup_id) wid: vec3<u32>,
        @builtin(local_invocation_id) lid: vec3<u32>) {
    let N = _params_[0];
    let nRows = _params_[1];
    let eps = bitcast<f32>(_params_[2]);
    let row = wid.x;
    if (row >= nRows) { return; }
    let N4 = N / 4u;
    let base = row * N4;
    var sum_sq: f32 = 0.0;
    for (var i = lid.x; i < N4; i += 128u) {
        let v = X[base + i] + Skip[base + i];
        SkipOut[base + i] = v;
        sum_sq += dot(v, v);
    }
    sum_sq_shared[lid.x] = sum_sq;
    workgroupBarrier();
    for (var stride = 64u; stride > 0u; stride >>= 1u) {
        if (lid.x < stride) {
            sum_sq_shared[lid.x] += sum_sq_shared[lid.x + stride];
        }
        workgroupBarrier();
    }
    let inv_rms = inverseSqrt(sum_sq_shared[0] / f32(N) + eps);
    for (var i = lid.x; i < N4; i += 128u) {
        Y[base + i] = SkipOut[base + i] * inv_rms * vec4<f32>(W[i]);
    }
}
