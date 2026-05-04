enable subgroups;

// Fused PLE norm + add + optional scale:
//   X = (X + RMSNorm(A, W)) * scale
// Replaces 2-3 dispatches: ple_norm + ple_add + layer_scalar
// Grid: (1, 1, 1) — single workgroup for single-row decode
// WG: 128 threads (4 warps)

var<workgroup> _smem: array<i32, 4>;

@group(0) @binding(0) var<storage, read> A: array<f32>;       // projOutBuf (PLE projection output)
@group(0) @binding(1) var<storage, read_write> X: array<f32>; // xBuf (residual, updated in-place)
@group(0) @binding(2) var<storage, read> W: array<f32>;       // pleNormW
@group(0) @binding(3) var<storage, read> params: array<u32>;  // [N, scale_as_u32, eps_as_u32, 0]

@compute @workgroup_size(128)
fn main(@builtin(local_invocation_id) lid: vec3<u32>) {
    let tid = lid.x;
    let N = params[0];
    let scale = bitcast<f32>(params[1]);
    let eps = bitcast<f32>(params[2]);
    let warp_id = tid / 32u;
    let lane_id = tid % 32u;

    // Pass 1: Compute RMSNorm of A
    var sum_sq: f32 = 0.0;
    for (var idx = tid; idx < N; idx += 128u) {
        let v = A[idx];
        sum_sq += v * v;
    }

    let warp_sum = subgroupAdd(sum_sq);
    if (lane_id == 0u) {
        _smem[warp_id] = bitcast<i32>(warp_sum);
    }
    workgroupBarrier();

    let final_sum = bitcast<f32>(_smem[0]) + bitcast<f32>(_smem[1])
                  + bitcast<f32>(_smem[2]) + bitcast<f32>(_smem[3]);
    let rstd = 1.0 / sqrt(final_sum / f32(N) + eps);

    // Pass 2: norm + add + scale
    for (var idx = tid; idx < N; idx += 128u) {
        let normed = A[idx] * rstd * W[idx];
        X[idx] = (X[idx] + normed) * scale;
    }
}
