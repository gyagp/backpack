enable subgroups;

@group(0) @binding(0) var<storage, read_write> X: array<f32>;
@group(0) @binding(1) var<storage, read_write> W: array<u32>;
@group(0) @binding(2) var<storage, read_write> Bias: array<f32>;
@group(0) @binding(3) var<storage, read_write> Y: array<f32>;
@group(0) @binding(4) var<storage, read_write> _params_: array<u32>;

// Tiled FP16-packed matmul: Y = X * W^T + Bias
// X: (M, K) f32 activations
// W: (N, K/2) u32 — each u32 holds 2 fp16 weights, unpacked via unpack2x16float
// Y: (M, N) f32 output
// No 'enable f16' required.

const TILE_M: u32 = 4u;
const TILE_N: u32 = 4u;

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) lid: vec3<u32>,
        @builtin(workgroup_id) wid: vec3<u32>) {
    let tile_row = wid.x;
    let tile_col = wid.y;
    let tid = lid.x;

    let K = _params_[0];
    let N = _params_[1];
    let M = _params_[2];

    let half_K = K / 2u;

    // 256 threads = 8 warps of 32
    // Each warp handles one (row, col) pair from the TILE_M × TILE_N tile
    // 8 warps → 4 rows × 2 col-pairs (each warp does 2 columns)
    let warp_id = tid / 32u;
    let lane = tid % 32u;

    let local_row = warp_id / 2u;   // 0..3
    let local_col2 = warp_id % 2u;  // 0..1, each handles 2 cols

    let row = tile_row * TILE_M + local_row;
    let col0 = tile_col * TILE_N + local_col2 * 2u;

    var acc0: f32 = 0.0;
    var acc1: f32 = 0.0;

    if (row < M) {
        let x_base = row * K;
        let stride = 128u;  // 32 lanes * 4 elements
        var k = lane * 4u;

        for (; k + 3u < K; k = k + stride) {
            let x0 = X[x_base + k];
            let x1 = X[x_base + k + 1u];
            let x2 = X[x_base + k + 2u];
            let x3 = X[x_base + k + 3u];
            let xv = vec4<f32>(x0, x1, x2, x3);

            if (col0 < N) {
                let w_base0 = col0 * half_K;
                let w01 = unpack2x16float(W[w_base0 + k / 2u]);
                let w23 = unpack2x16float(W[w_base0 + k / 2u + 1u]);
                acc0 += dot(xv, vec4<f32>(w01.x, w01.y, w23.x, w23.y));
            }

            if (col0 + 1u < N) {
                let w_base1 = (col0 + 1u) * half_K;
                let w01 = unpack2x16float(W[w_base1 + k / 2u]);
                let w23 = unpack2x16float(W[w_base1 + k / 2u + 1u]);
                acc1 += dot(xv, vec4<f32>(w01.x, w01.y, w23.x, w23.y));
            }
        }

        // Handle remaining elements when K is not a multiple of 4
        for (; k < K; k = k + stride) {
            let xv = X[x_base + k];
            if (col0 < N) {
                let w_base0 = col0 * half_K;
                let pair = unpack2x16float(W[w_base0 + k / 2u]);
                let w_val = select(pair.x, pair.y, (k & 1u) == 1u);
                acc0 += xv * w_val;
            }
            if (col0 + 1u < N) {
                let w_base1 = (col0 + 1u) * half_K;
                let pair = unpack2x16float(W[w_base1 + k / 2u]);
                let w_val = select(pair.x, pair.y, (k & 1u) == 1u);
                acc1 += xv * w_val;
            }
        }
    }

    let sum0 = subgroupAdd(acc0);
    let sum1 = subgroupAdd(acc1);

    if (lane == 0u) {
        if (row < M && col0 < N) {
            Y[row * N + col0] = sum0 + Bias[col0];
        }
        if (row < M && col0 + 1u < N) {
            Y[row * N + col0 + 1u] = sum1 + Bias[col0 + 1u];
        }
    }
}
