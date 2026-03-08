enable subgroups;

// Batched RMSNorm for T rows: Y[t] = X[t] * W / rms(X[t]) for each row t.
// Grid: (T, 1, 1) — one workgroup per row.
// WG: 128 threads.

var<workgroup> _smem: array<i32, 4>;

@group(0) @binding(0) var<storage, read> buf0: array<f32>;  // X [T × N]
@group(0) @binding(1) var<storage, read_write> buf1: array<f32>;  // Y [T × N]
@group(0) @binding(2) var<storage, read> buf2: array<f32>;  // W [N]
@group(0) @binding(3) var<storage, read_write> buf3: array<f32>;  // Rstd [T]

struct Params {
    stride: i32,
    N: i32,
    eps: f32,
};
@group(0) @binding(4) var<storage, read> params: Params;

const MAX_N: u32 = 8192u;  // supports up to E=8192

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

    var sum_sq: f32 = 0.0;
    var idx = tid;
    for (; idx < i32(MAX_N); idx += 128) {
        if (idx < N) {
            let v = buf0[u32(base + idx)];
            sum_sq += v * v;
        }
    }

    // Warp reduce
    let warp_sum = subgroupAdd(sum_sq);

    // Cross-warp reduce via shared memory
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
        buf3[u32(row)] = rstd;
    }

    // Apply norm + weight
    idx = tid;
    for (; idx < i32(MAX_N); idx += 128) {
        if (idx < N) {
            let v = buf0[u32(base + idx)];
            buf1[u32(base + idx)] = v * rstd * buf2[u32(idx)];
        }
    }
}
