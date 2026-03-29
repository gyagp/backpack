enable f16;

@group(0) @binding(0) var<storage, read> X: array<f32>;
@group(0) @binding(1) var<storage, read_write> Y: array<f32>;
@group(0) @binding(2) var<storage, read> W: array<f16>;
@group(0) @binding(3) var<storage, read_write> Rstd: array<f32>;

struct Params { stride: i32, N: i32, eps: f32, };
@group(0) @binding(4) var<storage, read> params: Params;

var<workgroup> shared_sum: array<f32, 256>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(local_invocation_id) lid: vec3<u32>,
        @builtin(workgroup_id) wg: vec3<u32>) {
    let row = wg.x;
    let N = u32(params.N);
    let nRows = u32(params.stride) / N;
    let base = row * N;
    let tid = lid.x;

    // Phase 1: parallel sum_sq reduction
    var local_sum: f32 = 0.0;
    if (row < nRows) {
        for (var i = tid; i < N; i += 256u) {
            let v = X[base + i];
            local_sum += v * v;
        }
    }
    shared_sum[tid] = local_sum;
    workgroupBarrier();

    // Tree reduction
    for (var s = 128u; s > 0u; s >>= 1u) {
        if (tid < s) {
            shared_sum[tid] += shared_sum[tid + s];
        }
        workgroupBarrier();
    }

    if (row >= nRows) { return; }

    let inv_rms = 1.0 / sqrt(shared_sum[0] / f32(N) + params.eps);

    // Phase 2: parallel normalize + scale
    for (var i = tid; i < N; i += 256u) {
        Y[base + i] = X[base + i] * inv_rms * f32(W[i]);
    }
    if (tid == 0u) { Rstd[row] = inv_rms; }
}
