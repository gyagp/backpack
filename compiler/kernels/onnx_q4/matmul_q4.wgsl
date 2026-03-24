// MatMulNBits Q4 — ONNX GenAI quantized matmul
// Y[m,n] = sum_k X[m,k] * dequant(W_Q4[n,k])
//
// Weight layout: W_Q4[N, blocks_per_col, 16] uint8 (Q4 packed, 8 nibbles per u32)
// Scale layout: Scales[N * blocks_per_col] fp16 (packed 2 per u32)
// block_size = 32, blocks_per_col = K / 32
//
// Dispatch: (ceil(N/8), M, 1)
// Workgroup: 128 threads = 8 output columns × 16 K-reduction threads

@group(0) @binding(0) var<storage, read> A: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> B: array<u32>;
@group(0) @binding(2) var<storage, read> Scales: array<u32>;
@group(0) @binding(3) var<storage, read_write> Y: array<f32>;
@group(0) @binding(4) var<storage, read> _params_: array<u32>;

const TILE_N: u32 = 8u;
const TILE_K_VEC: u32 = 16u;
const WG_SIZE: u32 = 128u;

var<workgroup> tile_A: array<vec4<f32>, TILE_K_VEC>;
var<workgroup> inter_results: array<array<f32, TILE_K_VEC>, TILE_N>;

@compute @workgroup_size(128)
fn main(@builtin(workgroup_id) wid: vec3<u32>,
        @builtin(local_invocation_id) lid: vec3<u32>) {
    let M = _params_[0];
    let N = _params_[1];
    let K = _params_[2];
    let blocks_per_col = K / 32u;

    let row = wid.y;
    let n_tile = wid.x;
    let local_id = lid.x;
    let k_idx = local_id % TILE_K_VEC;
    let n_idx = local_id / TILE_K_VEC;

    inter_results[n_idx][k_idx] = 0.0;
    workgroupBarrier();

    let a_base = row * (K / 4u);
    let b_col = n_tile * TILE_N + n_idx;

    for (var k_start = 0u; k_start < K; k_start += TILE_K_VEC * 8u) {
        if (local_id < TILE_K_VEC) {
            let a_offset = (k_start / 4u) + local_id * 2u;
            if (row < M && a_offset < K / 4u) {
                tile_A[local_id] = A[a_base + a_offset];
            } else {
                tile_A[local_id] = vec4<f32>(0.0);
            }
        }
        workgroupBarrier();

        let k_elem = k_start + k_idx * 8u;
        if (b_col < N && k_elem < K && row < M) {
            let b_offset = b_col * blocks_per_col * 4u + k_elem / 8u;
            let b_packed = B[b_offset];

            let block_idx = k_elem / 32u;
            let scale_flat = b_col * blocks_per_col + block_idx;
            let scale_u32 = Scales[scale_flat / 2u];
            let scale_half = select(scale_u32 & 0xFFFFu, (scale_u32 >> 16u) & 0xFFFFu, (scale_flat & 1u) != 0u);
            let scale = unpack2x16float(scale_half | (scale_half << 16u)).x;

            let lo = unpack4xU8(b_packed & 0x0F0F0F0Fu);
            let hi = unpack4xU8((b_packed >> 4u) & 0x0F0F0F0Fu);
            let b0 = vec4<f32>(f32(lo[0]) - 8.0, f32(hi[0]) - 8.0,
                               f32(lo[1]) - 8.0, f32(hi[1]) - 8.0) * scale;
            let b1 = vec4<f32>(f32(lo[2]) - 8.0, f32(hi[2]) - 8.0,
                               f32(lo[3]) - 8.0, f32(hi[3]) - 8.0) * scale;

            let a_local_offset = k_idx * 2u;
            var sum: f32 = 0.0;
            if (a_local_offset < TILE_K_VEC) {
                sum += dot(tile_A[a_local_offset], b0);
            }
            if (a_local_offset + 1u < TILE_K_VEC) {
                sum += dot(tile_A[a_local_offset + 1u], b1);
            }
            inter_results[n_idx][k_idx] += sum;
        }
        workgroupBarrier();
    }

    if (k_idx == 0u && b_col < N && row < M) {
        var total: f32 = 0.0;
        for (var k = 0u; k < TILE_K_VEC; k++) {
            total += inter_results[n_idx][k];
        }
        Y[row * N + b_col] = total;
    }
}
