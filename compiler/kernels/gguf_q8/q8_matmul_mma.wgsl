enable f16;
enable subgroups;
enable chromium_experimental_subgroup_matrix;

// Subgroup-matrix tiled Q8_0 GEMM for batched prefill:
//   Y[M×N] = X[M×K] × W[N×K]^T + Bias[N]
//
// Uses cooperative matrix multiply-accumulate (MMA) with fp16 tiles:
//   - Activations (fp32) staged to fp16 in shared memory
//   - Q8_0 weights dequantized to fp16 in shared memory
//   - MMA: f16×f16 → f32 accumulation, 16×16×16 tiles
//
// 4 subgroups per WG, each computing a 16×16 sub-tile of the 32×32 output.
// Grid: (ceil(N/32), ceil(M/32))
//
// params: [M, N, K, 0]

@group(0) @binding(0) var<storage, read_write> X: array<f32>;
@group(0) @binding(1) var<storage, read_write> W_Q8: array<u32>;
@group(0) @binding(2) var<storage, read_write> Scales: array<u32>;
@group(0) @binding(3) var<storage, read_write> Bias: array<f32>;
@group(0) @binding(4) var<storage, read_write> Y: array<f32>;
@group(0) @binding(5) var<storage, read_write> _params_: array<u32>;

const TILE_ROWS: u32 = 32u;
const TILE_COLS: u32 = 32u;
const TILE_K: u32 = 16u;
const SCALE_BLOCK: u32 = 32u;
const MAX_K_TILES: u32 = 384u;  // supports K up to 6144
const WG_SIZE: u32 = 128u;

var<workgroup> tile_A: array<f16, 512>;   // 32 × 16
var<workgroup> tile_B: array<f16, 512>;   // 32 × 16
var<workgroup> tile_C: array<f32, 1024>;  // 32 × 32

@compute @workgroup_size(128)
fn main(@builtin(local_invocation_id) lid: vec3<u32>,
        @builtin(workgroup_id) wid: vec3<u32>,
        @builtin(subgroup_id) subgroup_id: u32) {
    let M = _params_[0];
    let N = _params_[1];
    let K = _params_[2];
    let local_idx = lid.x;

    let row_base = wid.y * TILE_ROWS;
    let col_base = wid.x * TILE_COLS;

    let sg_row = subgroup_id & 1u;
    let sg_col = subgroup_id >> 1u;

    let weight_stride = K / 4u;
    let scale_stride = K / SCALE_BLOCK;

    var matC: subgroup_matrix_result<f32, 16, 16>;
    for (var tile_idx = 0u; tile_idx < MAX_K_TILES; tile_idx += 1u) {
        let k_base = tile_idx * TILE_K;

        // Load tile_A: X[32×16] → f16
        for (var i = local_idx; i < TILE_ROWS * TILE_K; i += WG_SIZE) {
            let local_row = i / TILE_K;
            let local_k = i % TILE_K;
            let global_row = row_base + local_row;
            let global_k = k_base + local_k;
            if (global_row < M && global_k < K) {
                tile_A[i] = f16(X[global_row * K + global_k]);
            } else {
                tile_A[i] = 0.0h;
            }
        }

        // Load tile_B: W_Q8[32×16] dequantized to f16
        for (var i = local_idx; i < TILE_COLS * TILE_K; i += WG_SIZE) {
            let local_col = i / TILE_K;
            let local_k = i % TILE_K;
            let global_col = col_base + local_col;
            let global_k = k_base + local_k;
            if (global_col < N && global_k < K) {
                let packed = W_Q8[global_col * weight_stride + global_k / 4u];
                let shift = (global_k & 3u) * 8u;
                let q = f32(extractBits(i32(packed), shift, 8u));
                let scale_idx = global_col * scale_stride + global_k / SCALE_BLOCK;
                let sp = unpack2x16float(Scales[scale_idx / 2u]);
                let scale = select(sp.x, sp.y, (scale_idx & 1u) != 0u);
                tile_B[i] = f16(q * scale);
            } else {
                tile_B[i] = 0.0h;
            }
        }

        workgroupBarrier();

        let a_offset = sg_row * 16u * TILE_K;
        let b_offset = sg_col * 16u * TILE_K;

        let matA = subgroupMatrixLoad<subgroup_matrix_left<f16, 16, 16>>(
            &tile_A, a_offset, false, TILE_K);
        let matB = subgroupMatrixLoad<subgroup_matrix_right<f16, 16, 16>>(
            &tile_B, b_offset, true, TILE_K);
        matC = subgroupMatrixMultiplyAccumulate(matA, matB, matC);

        workgroupBarrier();
    }

    let c_offset = sg_row * 16u * TILE_COLS + sg_col * 16u;
    subgroupMatrixStore(&tile_C, c_offset, matC, false, TILE_COLS);
    workgroupBarrier();

    for (var i = local_idx; i < TILE_ROWS * TILE_COLS; i += WG_SIZE) {
        let local_row = i / TILE_COLS;
        let local_col = i % TILE_COLS;
        let global_row = row_base + local_row;
        let global_col = col_base + local_col;
        if (global_row < M && global_col < N) {
            Y[global_row * N + global_col] = tile_C[i] + Bias[global_col];
        }
    }
}
