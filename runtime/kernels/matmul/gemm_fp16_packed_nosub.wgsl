// Gemm fp16-packed weights, no subgroup — shared memory reduction
// Dispatch: (M, ceil(N/TILE_N), 1)

@group(0) @binding(0) var<storage, read_write> A: array<f32>;
@group(0) @binding(1) var<storage, read_write> B: array<u32>;
@group(0) @binding(2) var<storage, read_write> Bias: array<f32>;
@group(0) @binding(3) var<storage, read_write> Y: array<f32>;
@group(0) @binding(4) var<storage, read_write> _params_: array<u32>;

const TILE_N: u32 = 8u;
const WG_SIZE: u32 = 256u;
const WARP_SIZE: u32 = 32u;

var<workgroup> shared_reduce: array<f32, 256>;

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) lid: vec3<u32>,
        @builtin(workgroup_id) wid: vec3<u32>) {
    let row = wid.x;
    let tile_col = wid.y;
    let tid = lid.x;

    let M = _params_[0];
    let N = _params_[1];
    let K = _params_[2];
    let transB = _params_[3];

    if (row >= M) { return; }

    let a_base = row * K;
    let half_K = K / 2u;
    let half_N = N / 2u;

    let warp_id = tid / WARP_SIZE;
    let lane = tid % WARP_SIZE;
    let col = tile_col * TILE_N + warp_id;

    var acc: f32 = 0.0;

    if (col < N) {
        let stride = 128u; // 32 lanes × 4 elements

        if (transB != 0u) {
            // B layout: (N, K) packed as (N, K/2) u32. K is even.
            let b_base = col * half_K;
            var k = lane * 4u;
            for (; k + 3u < K; k = k + stride) {
                let a0 = A[a_base + k];
                let a1 = A[a_base + k + 1u];
                let a2 = A[a_base + k + 2u];
                let a3 = A[a_base + k + 3u];

                let w01 = unpack2x16float(B[b_base + k / 2u]);
                let w23 = unpack2x16float(B[b_base + k / 2u + 1u]);

                acc += dot(vec4<f32>(a0, a1, a2, a3),
                           vec4<f32>(w01.x, w01.y, w23.x, w23.y));
            }
            // Handle remaining elements (K not divisible by 4)
            for (; k < K; k = k + 1u) {
                let pair_idx = k / 2u;
                let w = unpack2x16float(B[b_base + pair_idx]);
                let v = select(w.x, w.y, (k & 1u) == 1u);
                acc += A[a_base + k] * v;
            }
        } else {
            // B layout: (K, N) packed as (K, N/2) u32. N is even (enforced by host).
            let col_pair = col / 2u;
            let col_odd = col & 1u;
            var k = lane * 4u;
            for (; k + 3u < K; k = k + stride) {
                let a0 = A[a_base + k];
                let a1 = A[a_base + k + 1u];
                let a2 = A[a_base + k + 2u];
                let a3 = A[a_base + k + 3u];

                let w0 = unpack2x16float(B[(k) * half_N + col_pair]);
                let w1 = unpack2x16float(B[(k + 1u) * half_N + col_pair]);
                let w2 = unpack2x16float(B[(k + 2u) * half_N + col_pair]);
                let w3 = unpack2x16float(B[(k + 3u) * half_N + col_pair]);

                let v0 = select(w0.x, w0.y, col_odd == 1u);
                let v1 = select(w1.x, w1.y, col_odd == 1u);
                let v2 = select(w2.x, w2.y, col_odd == 1u);
                let v3 = select(w3.x, w3.y, col_odd == 1u);

                acc += dot(vec4<f32>(a0, a1, a2, a3),
                           vec4<f32>(v0, v1, v2, v3));
            }
            // Handle remaining elements (K not divisible by 4)
            for (; k < K; k = k + 1u) {
                let w = unpack2x16float(B[k * half_N + col_pair]);
                let v = select(w.x, w.y, col_odd == 1u);
                acc += A[a_base + k] * v;
            }
        }
    }

    // Shared memory reduction within each warp (replace subgroupAdd)
    shared_reduce[tid] = acc;
    workgroupBarrier();

    let warp_base = warp_id * WARP_SIZE;
    // Tree reduction within warp
    if (lane < 16u) { shared_reduce[warp_base + lane] += shared_reduce[warp_base + lane + 16u]; }
    workgroupBarrier();
    if (lane < 8u) { shared_reduce[warp_base + lane] += shared_reduce[warp_base + lane + 8u]; }
    workgroupBarrier();
    if (lane < 4u) { shared_reduce[warp_base + lane] += shared_reduce[warp_base + lane + 4u]; }
    workgroupBarrier();
    if (lane < 2u) { shared_reduce[warp_base + lane] += shared_reduce[warp_base + lane + 2u]; }
    workgroupBarrier();
    if (lane < 1u) { shared_reduce[warp_base + lane] += shared_reduce[warp_base + lane + 1u]; }
    workgroupBarrier();

    if (lane == 0u && col < N) {
        Y[row * N + col] = shared_reduce[warp_base] + Bias[col];
    }
}
