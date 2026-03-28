// QMoE gate_up Q4 matmul — Q4 matmul for one MoE expert's gate_up projection.
// Y[n] = sum_k X[k] * dequant(W[expert, n, k])
//
// Weight layout: W_q4[num_experts, N, K/2] uint8
// Scale layout:  S[num_experts, N, K/block_size] fp16
// Params: [0]=N (output dim = 2*intermediate), [1]=K (input dim = hidden),
//         [2]=expertIdx, [3]=blocks_per_col (K/32)
//
// Dispatch: (ceil(N/256), 1, 1) — one thread per output element

@group(0) @binding(0) var<storage, read> X: array<f32>;
@group(0) @binding(1) var<storage, read> W: array<u32>;
@group(0) @binding(2) var<storage, read> Scales: array<u32>;
@group(0) @binding(3) var<storage, read_write> Y: array<f32>;
@group(0) @binding(4) var<storage, read> _params_: array<u32>;
@group(0) @binding(5) var<storage, read> _params2_: array<u32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let N = _params_[0];
    let K = _params_[1];
    let expert = _params_[2];
    let blocks_per_col = _params_[3];

    let n = gid.x;
    if (n >= N) { return; }

    let expert_w_offset = expert * N * (K / 2u);
    let expert_s_offset = expert * N * blocks_per_col;

    var acc: f32 = 0.0;
    let w_base = expert_w_offset + n * (K / 2u);

    for (var blk = 0u; blk < blocks_per_col; blk++) {
        let scale_flat = expert_s_offset + n * blocks_per_col + blk;
        let scale_u32 = Scales[scale_flat / 2u];
        let scale_half = select(scale_u32 & 0xFFFFu, (scale_u32 >> 16u) & 0xFFFFu, (scale_flat & 1u) != 0u);
        let scale = unpack2x16float(scale_half | (scale_half << 16u)).x;

        let k_base = blk * 32u;
        let w_blk_base = w_base + k_base / 2u;

        for (var j = 0u; j < 16u; j++) {
            let byte_idx = w_blk_base + j;
            let byte_u32 = W[byte_idx / 4u];
            let byte_val = (byte_u32 >> ((byte_idx % 4u) * 8u)) & 0xFFu;
            let lo = f32(byte_val & 0xFu) - 8.0;
            let hi = f32((byte_val >> 4u) & 0xFu) - 8.0;
            acc += X[k_base + j * 2u] * lo * scale;
            acc += X[k_base + j * 2u + 1u] * hi * scale;
        }
    }

    Y[n] = acc;
}
