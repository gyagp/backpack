// Fused multi-head attention: Q*K^T / sqrt(head_dim), softmax, * V
// Supports grouped-query attention (num_kv_heads <= num_heads)
// K/V stored as packed f16 in array<u32> (2 x f16 per u32)
// Layout: Q[batch * num_heads * seq_len * head_dim] (f32)
//         K/V[kv_head * seq_len * (head_dim/2)] (u32, packed f16)
// Output: [batch * num_heads * seq_len * head_dim] (f32)

@group(0) @binding(0) var<storage, read> Q : array<f32>;
@group(0) @binding(1) var<storage, read> K : array<u32>;
@group(0) @binding(2) var<storage, read> V : array<u32>;
@group(0) @binding(3) var<storage, read_write> output : array<f32>;
@group(0) @binding(4) var<uniform> params : array<vec4u, 2>;

var<workgroup> shared_max : array<f32, 256>;
var<workgroup> shared_sum : array<f32, 256>;
var<workgroup> attn_row : array<f32, 1024>;

fn load_k(k_head_base: u32, col: u32, head_dim: u32, d: u32) -> f32 {
    let half_hd = head_dim / 2u;
    let packed = K[k_head_base + col * half_hd + d / 2u];
    let unpacked = unpack2x16float(packed);
    if ((d & 1u) == 0u) { return unpacked.x; } else { return unpacked.y; }
}

fn load_v(v_head_base: u32, col: u32, head_dim: u32, d: u32) -> f32 {
    let half_hd = head_dim / 2u;
    let packed = V[v_head_base + col * half_hd + d / 2u];
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

    let tid = lid.x;

    let head_idx = wid.x;
    let q_row = wid.y;

    let kv_head = head_idx % num_kv_heads;

    let q_base = (head_idx * seq_len + q_row) * head_dim;
    let half_hd = head_dim / 2u;
    let k_head_base = kv_head * seq_len * half_hd;
    let v_head_base = kv_head * seq_len * half_hd;

    for (var k_col = tid; k_col < seq_len; k_col += 256u) {
        var dot = 0.0f;
        for (var d = 0u; d < head_dim; d += 1u) {
            dot += Q[q_base + d] * load_k(k_head_base, k_col, head_dim, d);
        }
        attn_row[k_col] = dot * scale;
    }
    workgroupBarrier();

    var local_max = -3.402823e+38f;
    for (var i = tid; i < seq_len; i += 256u) {
        local_max = max(local_max, attn_row[i]);
    }
    shared_max[tid] = local_max;
    workgroupBarrier();

    for (var stride = 128u; stride > 0u; stride >>= 1u) {
        if (tid < stride) {
            shared_max[tid] = max(shared_max[tid], shared_max[tid + stride]);
        }
        workgroupBarrier();
    }
    let row_max = shared_max[0];
    workgroupBarrier();

    var local_sum = 0.0f;
    for (var i = tid; i < seq_len; i += 256u) {
        let e = exp(attn_row[i] - row_max);
        attn_row[i] = e;
        local_sum += e;
    }
    shared_sum[tid] = local_sum;
    workgroupBarrier();

    for (var stride = 128u; stride > 0u; stride >>= 1u) {
        if (tid < stride) {
            shared_sum[tid] += shared_sum[tid + stride];
        }
        workgroupBarrier();
    }
    let row_sum = shared_sum[0];
    workgroupBarrier();

    let inv_sum = 1.0 / row_sum;
    for (var i = tid; i < seq_len; i += 256u) {
        attn_row[i] = attn_row[i] * inv_sum;
    }
    workgroupBarrier();

    let out_base = (head_idx * seq_len + q_row) * head_dim;
    for (var d = tid; d < head_dim; d += 256u) {
        var acc = 0.0f;
        for (var s = 0u; s < seq_len; s += 1u) {
            acc += attn_row[s] * load_v(v_head_base, s, head_dim, d);
        }
        output[out_base + d] = acc;
    }
}
