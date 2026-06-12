// Flash-attention style tiled attention kernel
// Processes K/V in tiles with online softmax to avoid materializing the full attention matrix.
// K/V are stored in f16 packed as array<u32> (2 x f16 per u32).
// Layout: Q[batch * num_heads * seq_len * head_dim] (f32)
//         K/V[kv_head * max_seq_len * (head_dim/2)] (u32, packed f16)
// Output: [batch * num_heads * seq_len * head_dim] (f32)

const TILE_SIZE: u32 = 64;

@group(0) @binding(0) var<storage, read> Q : array<f32>;
@group(0) @binding(1) var<storage, read> K : array<u32>;
@group(0) @binding(2) var<storage, read> V : array<u32>;
@group(0) @binding(3) var<storage, read_write> output : array<f32>;
@group(0) @binding(4) var<uniform> params : array<vec4u, 2>;

var<workgroup> tile_scores : array<f32, 256>;
var<workgroup> shared_buf : array<f32, 256>;

fn load_k(k_head_base: u32, col: u32, head_dim: u32, d: u32) -> f32 {
    let half_hd = head_dim / 2u;
    let pair_idx = k_head_base + col * half_hd + d / 2u;
    let packed = K[pair_idx];
    let unpacked = unpack2x16float(packed);
    if ((d & 1u) == 0u) { return unpacked.x; } else { return unpacked.y; }
}

fn load_v(v_head_base: u32, col: u32, head_dim: u32, d: u32) -> f32 {
    let half_hd = head_dim / 2u;
    let pair_idx = v_head_base + col * half_hd + d / 2u;
    let packed = V[pair_idx];
    let unpacked = unpack2x16float(packed);
    if ((d & 1u) == 0u) { return unpacked.x; } else { return unpacked.y; }
}

@compute @workgroup_size(256)
fn main(
    @builtin(local_invocation_id) lid : vec3u,
    @builtin(workgroup_id) wid : vec3u,
) {
    let seq_len = params[0].x;
    let head_dim = params[0].y;
    let num_heads = params[0].z;
    let num_kv_heads = params[0].w;
    let scale_bits = params[1].x;
    let scale = bitcast<f32>(scale_bits);
    let q_seq_offset = params[1].y;

    let tid = lid.x;
    let head_idx = wid.x;
    let q_row = wid.y;
    let causal_len = q_seq_offset + q_row + 1u;

    let kv_head = head_idx % num_kv_heads;

    let q_base = (q_row * num_heads + head_idx) * head_dim;
    let half_hd = head_dim / 2u;
    let k_head_base = kv_head * seq_len * half_hd;
    let v_head_base = kv_head * seq_len * half_hd;
    let out_base = (q_row * num_heads + head_idx) * head_dim;

    for (var d = tid; d < head_dim; d += 256u) {
        output[out_base + d] = 0.0;
    }

    var running_max = -3.402823e+38f;
    var running_sum = 0.0f;

    let num_tiles = (seq_len + TILE_SIZE - 1u) / TILE_SIZE;

    for (var tile = 0u; tile < num_tiles; tile++) {
        let tile_start = tile * TILE_SIZE;
        let tile_end = min(tile_start + TILE_SIZE, seq_len);
        let tile_len = tile_end - tile_start;

        if (tid < tile_len) {
            let k_col = tile_start + tid;
            if (k_col < causal_len) {
                var dot = 0.0f;
                for (var d = 0u; d < head_dim; d += 1u) {
                    dot += Q[q_base + d] * load_k(k_head_base, k_col, head_dim, d);
                }
                tile_scores[tid] = dot * scale;
            } else {
                tile_scores[tid] = -3.402823e+38f;
            }
        }
        if (tid < TILE_SIZE && tid >= tile_len) {
            tile_scores[tid] = -3.402823e+38f;
        }
        workgroupBarrier();

        if (tid < TILE_SIZE) {
            shared_buf[tid] = tile_scores[tid];
        } else {
            shared_buf[tid] = -3.402823e+38f;
        }
        workgroupBarrier();

        for (var stride = 128u; stride > 0u; stride >>= 1u) {
            if (tid < stride) {
                shared_buf[tid] = max(shared_buf[tid], shared_buf[tid + stride]);
            }
            workgroupBarrier();
        }
        let tile_max = shared_buf[0];
        workgroupBarrier();

        var local_exp_sum = 0.0f;
        if (tid < tile_len) {
            let e = exp(tile_scores[tid] - tile_max);
            tile_scores[tid] = e;
            local_exp_sum = e;
        }
        shared_buf[tid] = local_exp_sum;
        workgroupBarrier();

        for (var stride = 128u; stride > 0u; stride >>= 1u) {
            if (tid < stride) {
                shared_buf[tid] += shared_buf[tid + stride];
            }
            workgroupBarrier();
        }
        let tile_exp_sum = shared_buf[0];
        workgroupBarrier();

        let new_max = max(running_max, tile_max);
        let correction_old = exp(running_max - new_max);
        let correction_new = exp(tile_max - new_max);
        let new_sum = running_sum * correction_old + tile_exp_sum * correction_new;

        for (var d = tid; d < head_dim; d += 256u) {
            var acc = output[out_base + d] * correction_old;
            for (var k = 0u; k < tile_len; k++) {
                let weight = tile_scores[k] * correction_new;
                acc += weight * load_v(v_head_base, tile_start + k, head_dim, d);
            }
            output[out_base + d] = acc;
        }
        workgroupBarrier();

        running_max = new_max;
        running_sum = new_sum;
    }

    let inv_sum = 1.0 / running_sum;
    for (var d = tid; d < head_dim; d += 256u) {
        output[out_base + d] *= inv_sum;
    }
}
