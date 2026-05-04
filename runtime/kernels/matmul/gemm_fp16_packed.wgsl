enable subgroups;

// Gemm with FP16-packed weights: Y = A * B^T + Bias
// A: (M, K) f32 activations
// B: (N, K/2) u32 — each u32 holds 2 fp16 weights via unpack2x16float
// Bias: (N) f32, added to output (pass zeros for no bias)
// Y: (M, N) f32 output
// Dispatch: (M, ceil(N/TILE_N), 1)

@group(0) @binding(0) var<storage, read_write> A: array<f32>;
@group(0) @binding(1) var<storage, read_write> B: array<u32>;
@group(0) @binding(2) var<storage, read_write> Bias: array<f32>;
@group(0) @binding(3) var<storage, read_write> Y: array<f32>;
@group(0) @binding(4) var<storage, read_write> _params_: array<u32>;

const TILE_N: u32 = 8u;

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

    let warp_id = tid / 32u;
    let lane = tid % 32u;
    let col = tile_col * TILE_N + warp_id;

    var acc: f32 = 0.0;

    if (col < N) {
        // transB=1: B is (N, K), row-major → b_base = col * half_K
        // transB=0: B is (K, N), row-major → b[k,col] = B[(k/2)*N + col] (interleaved)
        // The packed layout only makes sense for transB=1 (contiguous K dim per row).
        // For transB=0, we still support it but stride differently.

        let stride = 128u; // 32 lanes × 4 elements
        var k = lane * 4u;

        if (transB != 0u) {
            let b_base = col * half_K;
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
            for (; k < K; k = k + stride) {
                let pair = unpack2x16float(B[b_base + k / 2u]);
                let w_val = select(pair.x, pair.y, (k & 1u) == 1u);
                acc += A[a_base + k] * w_val;
            }
        } else {
            // transB=0: B is (K, N) row-major, packed along N dimension
            // B[k][col] → u32 at index (k * N + col) / 2, but col pairs share a u32
            // For simplicity, iterate per-element
            for (; k < K; k = k + stride) {
                let a_val = A[a_base + k];
                let b_idx = k * N + col;
                let pair = unpack2x16float(B[b_idx / 2u]);
                let w_val = select(pair.x, pair.y, (b_idx & 1u) == 1u);
                acc += a_val * w_val;
            }
        }
    }

    let warp_sum = subgroupAdd(acc);

    if (lane == 0u && col < N) {
        Y[row * N + col] = warp_sum + Bias[col];
    }
}
