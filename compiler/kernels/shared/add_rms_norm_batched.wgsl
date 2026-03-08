enable subgroups;

// Batched add + RMSNorm for T rows:
//   X[t] = X[t] + A[t]  (residual add, in-place)
//   Y[t] = X[t] * W / rms(X[t])  (RMSNorm for next op)
// Grid: (T, 1, 1)
// WG: 128 threads

var<workgroup> _smem: array<i32, 4>;

@group(0) @binding(0) var<storage, read_write> X: array<f32>;  // residual [T × N], updated in-place
@group(0) @binding(1) var<storage, read> A: array<f32>;  // addend [T × N]
@group(0) @binding(2) var<storage, read_write> Y: array<f32>;  // normed output [T × N]
@group(0) @binding(3) var<storage, read> W: array<f32>;  // norm weight [N]
@group(0) @binding(4) var<storage, read_write> Rstd: array<f32>;  // [T]

struct Params {
    stride: i32,
    N: i32,
    eps: f32,
};
@group(0) @binding(5) var<storage, read> params: Params;

const MAX_N: u32 = 8192u;

@compute @workgroup_size(128)
fn main(
    @builtin(workgroup_id) _wg_id: vec3<u32>,
    @builtin(local_invocation_id) _lid: vec3<u32>,
) {
    let row = i32(_wg_id.x);
    let tid = i32(_lid.x);
    let N = params.N;
    let stride = params.stride;
    let base = row * stride;

    // Pass 1: Add residual and compute sum of squares
    var sum_sq: f32 = 0.0;
    var idx = tid;
    for (; idx < i32(MAX_N); idx += 128) {
        if (idx < N) {
            let pos = u32(base + idx);
            let v = X[pos] + A[pos];
            X[pos] = v;
            sum_sq += v * v;
        }
    }

    let warp_sum = subgroupAdd(sum_sq);
    let warp_id = tid / 32;
    let lane_id = tid % 32;
    if (lane_id == 0) {
        _smem[warp_id] = bitcast<i32>(warp_sum);
    }
    workgroupBarrier();

    var total: f32 = 0.0;
    if (tid < 4) {
        total = bitcast<f32>(_smem[tid]);
    }
    let final_sum = subgroupAdd(total);
    let rstd = 1.0 / sqrt(final_sum / f32(N) + params.eps);
    if (tid == 0) {
        Rstd[u32(row)] = rstd;
    }

    // Pass 2: Apply norm + weight
    idx = tid;
    for (; idx < i32(MAX_N); idx += 128) {
        if (idx < N) {
            let pos = u32(base + idx);
            Y[pos] = X[pos] * rstd * W[u32(idx)];
        }
    }
}
