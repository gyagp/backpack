// @meta bindings=6
enable subgroups;

// Fused: SiLU·mul + Q8_0 matmul + residual add.
// Reads gateUpBuf (2*IM elements), applies silu(gate)*up on-the-fly,
// then multiplies by W_down and adds to residual Y.
// Eliminates the separate silu_mul dispatch entirely.
//
// Input X layout: [gate_0..gate_{IM-1}, up_0..up_{IM-1}]
// Effective input after silu: silu(gate_i) * up_i for i in [0, IM)
//
// Bindings:
//   0: GateUp (read) — 2*IM floats (gate || up concatenated)
//   1: W_Q8 (read) — quantized weight matrix (N × K/4 u32)
//   2: Scales (read) — fp16 scales packed as u32
//   3: Bias (read) — per-output bias (zeros for no bias)
//   4: Y (read_write) — output += matmul result (residual add)
//   5: _params_ — [K=IM, N=E, IM, 0]

@group(0) @binding(0) var<storage, read_write> GateUp: array<f32>;
@group(0) @binding(1) var<storage, read_write> W_Q8: array<u32>;
@group(0) @binding(2) var<storage, read_write> Scales: array<u32>;
@group(0) @binding(3) var<storage, read_write> Bias: array<f32>;
@group(0) @binding(4) var<storage, read_write> Y: array<f32>;
@group(0) @binding(5) var<storage, read_write> _params_: array<u32>;

const TILE_N: u32 = 8u;
const STRIDE: u32 = 256u;
const MAX_STRIDES: u32 = 24u;  // ceil(6144 / 256)

var<workgroup> smem_x: array<f32, 256>;

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) lid: vec3<u32>,
        @builtin(workgroup_id) wid: vec3<u32>) {
    let row = wid.x;
    let tile_col = wid.y;
    let tid = lid.x;

    let K = _params_[0];   // IM (intermediate_size)
    let N = _params_[1];   // E (n_embd)
    let IM = _params_[2];  // same as K, used for up offset

    let warp_id = tid / 32u;
    let lane = tid % 32u;
    let col = tile_col * TILE_N + warp_id;
    let valid = col < N;
    let stride_w = K / 4u;
    let n_blocks = K / 32u;
    let w_base = select(0u, col * stride_w, valid);
    let s_base = select(0u, col * n_blocks, valid);
    var acc: f32 = 0.0;

    for (var g = 0u; g < MAX_STRIDES; g = g + 1u) {
        let k_off = g * STRIDE;
        let in_range = k_off < K;

        // Load gate and up values, apply silu·mul, store to shared memory
        if (in_range) {
            let idx = k_off + tid;
            let gate = GateUp[row * 2u * IM + idx];         // gate[idx]
            let up   = GateUp[row * 2u * IM + IM + idx];    // up[idx]
            // silu(gate) * up = gate / (1 + exp(-gate)) * up
            let silu_gate = gate / (1.0 + exp(-gate));
            smem_x[tid] = silu_gate * up;
        }
        workgroupBarrier();

        if (valid && in_range) {
            let k_base = lane * 8u;
            let xv0 = vec4<f32>(smem_x[k_base], smem_x[k_base+1u],
                                smem_x[k_base+2u], smem_x[k_base+3u]);
            let xv1 = vec4<f32>(smem_x[k_base+4u], smem_x[k_base+5u],
                                smem_x[k_base+6u], smem_x[k_base+7u]);

            let w_off = w_base + g * 64u + lane * 2u;
            let pw0 = W_Q8[w_off];
            let pw1 = W_Q8[w_off + 1u];

            let wv0 = vec4<f32>(f32(extractBits(i32(pw0),0u,8u)),f32(extractBits(i32(pw0),8u,8u)),
                                f32(extractBits(i32(pw0),16u,8u)),f32(extractBits(i32(pw0),24u,8u)));
            let wv1 = vec4<f32>(f32(extractBits(i32(pw1),0u,8u)),f32(extractBits(i32(pw1),8u,8u)),
                                f32(extractBits(i32(pw1),16u,8u)),f32(extractBits(i32(pw1),24u,8u)));

            let block0 = g * 8u + (lane * 8u) / 32u;
            let block1 = g * 8u + (lane * 8u + 4u) / 32u;
            let sp0 = unpack2x16float(Scales[(s_base+block0)/2u]);
            let scale0 = select(sp0.x, sp0.y, ((s_base+block0)&1u)!=0u);
            let sp1 = unpack2x16float(Scales[(s_base+block1)/2u]);
            let scale1 = select(sp1.x, sp1.y, ((s_base+block1)&1u)!=0u);

            acc += dot(xv0, wv0) * scale0 + dot(xv1, wv1) * scale1;
        }
        workgroupBarrier();
    }

    let warp_sum = subgroupAdd(acc);
    if (lane == 0u && valid) {
        Y[row * N + col] += warp_sum + Bias[col];
    }
}
