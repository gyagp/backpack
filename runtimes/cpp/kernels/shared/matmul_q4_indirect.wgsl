// Q4 matmul with expert index from GPU buffer.
// Like MATMUL_Q4 but reads expert index from binding 5 to compute weight offset.
// Y[n] = sum_k X[k] * dequant(W[expert, n, k])
// Weights: W[num_experts, N, K/2] packed uint8
// Scales:  S[num_experts, N, blocks_per_col] packed fp16
// Params: [0]=N, [1]=K, [2]=blocks_per_col, [3]=slot
// Dispatch: (ceil(N/256), 1, 1)

${T_READ}
${T_WRITE}

@group(0) @binding(0) var<storage, read> X: array<${T}>;
@group(0) @binding(1) var<storage, read> W: array<u32>;
@group(0) @binding(2) var<storage, read> Scales: array<u32>;
@group(0) @binding(3) var<storage, read_write> Y: array<${T}>;
@group(0) @binding(4) var<storage, read> _params_: array<u32>;
@group(0) @binding(5) var<storage, read> expert_indices: array<u32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let N = _params_[0];
    let K = _params_[1];
    let blocks_per_col = _params_[2];
    let slot = _params_[3];

    let n = gid.x;
    if (n >= N) { return; }

    let expert = expert_indices[slot];

    // Byte offsets into the [num_experts, N, K/2] weight buffer
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
            acc += t_read(&X, k_base + j * 2u) * lo * scale;
            acc += t_read(&X, k_base + j * 2u + 1u) * hi * scale;
        }
    }

    t_write(&Y, n, acc);
}
