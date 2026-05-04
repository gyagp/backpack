// @meta bindings=6
requires packed_4x8_integer_dot_product;
enable subgroups;

// Fused: GELU·mul + Q4 matmul + residual add for decode (M=1).
// DP4A-accelerated. Reads gateUpBuf (2*IM elements), applies gelu(gate)*up,
// quantizes to int8, then DP4A with Q4 W_down and adds to residual Y.
// GELU variant of q4_down_silu_add_decode for Gemma-4 and similar models.
//
// Dispatch: (ceil(N/32), 1, 1) where N = E (n_embd)
//
// Bindings:
//   0: GateUp (read) — 2*IM floats (gate || up concatenated)
//   1: B (read) — packed Q4 weights [N × K/8] as u32
//   2: Scales (read) — fp16 scales [N × nGroups] packed as u32
//   3: Bias (read) — per-output bias
//   4: Y (read_write) — output += matmul result (residual add)
//   5: _params_ — [K=IM, N=E, IM, 0]

@group(0) @binding(0) var<storage, read> GateUp: array<f32>;
@group(0) @binding(1) var<storage, read> B: array<u32>;
@group(0) @binding(2) var<storage, read> Scales: array<u32>;
@group(0) @binding(3) var<storage, read> Bias: array<f32>;
@group(0) @binding(4) var<storage, read_write> Y: array<f32>;
@group(0) @binding(5) var<storage, read> _params_: array<u32>;

const BK: u32 = 256u;
const COLS_PER_WARP: u32 = 4u;

var<workgroup> smem_xq: array<u32, 64>;
var<workgroup> smem_xs: array<f32, 8>;

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) lid: vec3<u32>,
        @builtin(workgroup_id) wid: vec3<u32>) {
    let K = _params_[0];   // IM (intermediate_size)
    let N = _params_[1];   // E (n_embd)
    let IM = _params_[2];  // same as K
    let tid = lid.x;

    let warp_id = tid / 32u;
    let lane = tid % 32u;

    let n_blocks = K / 32u;
    let q4_words_per_row = K / 8u;

    var cols: array<u32, 4>;
    var col_valid: array<bool, 4>;
    for (var c = 0u; c < COLS_PER_WARP; c++) {
        cols[c] = wid.x * 32u + warp_id * COLS_PER_WARP + c;
        col_valid[c] = cols[c] < N;
    }

    let block_id = tid / 32u;
    let elem_in_block = tid % 32u;
    let pack_lane = elem_in_block % 4u;
    let pack_group = elem_in_block / 4u;

    var acc: array<f32, 4>;
    for (var c = 0u; c < COLS_PER_WARP; c++) { acc[c] = 0.0; }

    let nk = K / BK;
    for (var g = 0u; g < nk; g++) {
        let k_base = g * BK;
        let k = k_base + tid;

        // ── Compute gelu(gate) * up and quantize to int8 ──
        let gate = GateUp[k];
        let up = GateUp[IM + k];
        let inner = 0.7978845608 * (gate + 0.044715 * gate * gate * gate);
        let gelu = 0.5 * gate * (1.0 + tanh(inner));
        let x_val = gelu * up;

        var max_val = abs(x_val);
        max_val = max(max_val, subgroupShuffleXor(max_val, 16u));
        max_val = max(max_val, subgroupShuffleXor(max_val, 8u));
        max_val = max(max_val, subgroupShuffleXor(max_val, 4u));
        max_val = max(max_val, subgroupShuffleXor(max_val, 2u));
        max_val = max(max_val, subgroupShuffleXor(max_val, 1u));

        let x_scale = max_val / 127.0;
        if (elem_in_block == 0u) {
            smem_xs[block_id] = x_scale;
        }

        let safe_scale = select(1.0, x_scale, x_scale != 0.0);
        let q_val = clamp(i32(round(x_val / safe_scale)), -127, 127);

        let byte_val = u32(q_val & 0xFF);
        let shifted = byte_val << (pack_lane * 8u);
        var packed = shifted;
        packed = packed | subgroupShuffleXor(packed, 1u);
        packed = packed | subgroupShuffleXor(packed, 2u);

        if (pack_lane == 0u) {
            smem_xq[block_id * 8u + pack_group] = packed;
        }

        workgroupBarrier();

        // ── DP4A matmul ──
        let x_block = lane / 4u;

        for (var c = 0u; c < COLS_PER_WARP; c++) {
            if (col_valid[c]) {
                let n = cols[c];
                let q4_off = n * q4_words_per_row + (g * BK + lane * 8u) / 8u;
                let q4_packed = B[q4_off];

                let b0 = q4_packed & 0xFFu;
                let b1 = (q4_packed >> 8u) & 0xFFu;
                let b2 = (q4_packed >> 16u) & 0xFFu;
                let b3 = (q4_packed >> 24u) & 0xFFu;

                let w0 = (u32(b0 & 0xFu) - 8u) & 0xFFu;
                let w1 = (u32(b0 >> 4u) - 8u) & 0xFFu;
                let w2 = (u32(b1 & 0xFu) - 8u) & 0xFFu;
                let w3 = (u32(b1 >> 4u) - 8u) & 0xFFu;
                let w4 = (u32(b2 & 0xFu) - 8u) & 0xFFu;
                let w5 = (u32(b2 >> 4u) - 8u) & 0xFFu;
                let w6 = (u32(b3 & 0xFu) - 8u) & 0xFFu;
                let w7 = (u32(b3 >> 4u) - 8u) & 0xFFu;

                let wq0 = w0 | (w1 << 8u) | (w2 << 16u) | (w3 << 24u);
                let wq1 = w4 | (w5 << 8u) | (w6 << 16u) | (w7 << 24u);

                let w_block = g * 8u + x_block;
                let sp = unpack2x16float(Scales[(n * n_blocks + w_block) / 2u]);
                let w_scale = select(sp.x, sp.y, ((n * n_blocks + w_block) & 1u) != 0u);

                let xq0 = smem_xq[lane * 2u];
                let xq1 = smem_xq[lane * 2u + 1u];
                let x_scale = smem_xs[x_block];

                let idot = dot4I8Packed(xq0, wq0) + dot4I8Packed(xq1, wq1);
                acc[c] += f32(idot) * w_scale * x_scale;
            }
        }

        workgroupBarrier();
    }

    for (var c = 0u; c < COLS_PER_WARP; c++) {
        let warp_sum = subgroupAdd(acc[c]);
        if (lane == 0u && col_valid[c]) {
            Y[cols[c]] += warp_sum + Bias[cols[c]];
        }
    }
}
