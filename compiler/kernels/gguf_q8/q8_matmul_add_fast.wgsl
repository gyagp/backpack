enable subgroups;
@group(0) @binding(0) var<storage, read_write> X: array<f32>;
@group(0) @binding(1) var<storage, read_write> W_Q8: array<u32>;
@group(0) @binding(2) var<storage, read_write> Scales: array<u32>;
@group(0) @binding(3) var<storage, read_write> Bias: array<f32>;
@group(0) @binding(4) var<storage, read_write> Y: array<f32>;
@group(0) @binding(5) var<storage, read_write> _params_: array<u32>;
const TILE_N: u32 = 8u;
@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) lid: vec3<u32>,
        @builtin(workgroup_id) wid: vec3<u32>) {
    let row = wid.x; let tile_col = wid.y; let tid = lid.x;
    let K = _params_[0]; let N = _params_[1]; let x_base = row * K;
    let warp_id = tid/32u; let lane = tid%32u;
    let col = tile_col * TILE_N + warp_id;
    let n_strides = K / 512u; let stride_w = K / 4u;
    var acc: f32 = 0.0;
    if (col < N) {
        let w_base = col * stride_w; let n_blocks = K / 32u; let s_base = col * n_blocks;
        for (var g = 0u; g < n_strides; g = g + 1u) {
            let k_base = g*512u + lane*16u;
            let xv0 = vec4<f32>(X[x_base+k_base],X[x_base+k_base+1u],X[x_base+k_base+2u],X[x_base+k_base+3u]);
            let xv1 = vec4<f32>(X[x_base+k_base+4u],X[x_base+k_base+5u],X[x_base+k_base+6u],X[x_base+k_base+7u]);
            let xv2 = vec4<f32>(X[x_base+k_base+8u],X[x_base+k_base+9u],X[x_base+k_base+10u],X[x_base+k_base+11u]);
            let xv3 = vec4<f32>(X[x_base+k_base+12u],X[x_base+k_base+13u],X[x_base+k_base+14u],X[x_base+k_base+15u]);
            let w_off = w_base + g*128u + lane*4u;
            let pw0=W_Q8[w_off]; let pw1=W_Q8[w_off+1u]; let pw2=W_Q8[w_off+2u]; let pw3=W_Q8[w_off+3u];
            let wv0 = vec4<f32>(f32(extractBits(i32(pw0),0u,8u)),f32(extractBits(i32(pw0),8u,8u)),f32(extractBits(i32(pw0),16u,8u)),f32(extractBits(i32(pw0),24u,8u)));
            let wv1 = vec4<f32>(f32(extractBits(i32(pw1),0u,8u)),f32(extractBits(i32(pw1),8u,8u)),f32(extractBits(i32(pw1),16u,8u)),f32(extractBits(i32(pw1),24u,8u)));
            let wv2 = vec4<f32>(f32(extractBits(i32(pw2),0u,8u)),f32(extractBits(i32(pw2),8u,8u)),f32(extractBits(i32(pw2),16u,8u)),f32(extractBits(i32(pw2),24u,8u)));
            let wv3 = vec4<f32>(f32(extractBits(i32(pw3),0u,8u)),f32(extractBits(i32(pw3),8u,8u)),f32(extractBits(i32(pw3),16u,8u)),f32(extractBits(i32(pw3),24u,8u)));
            let b0=g*16u+(lane*16u)/32u; let b1=g*16u+(lane*16u+4u)/32u;
            let b2=g*16u+(lane*16u+8u)/32u; let b3=g*16u+(lane*16u+12u)/32u;
            let sp0=unpack2x16float(Scales[(s_base+b0)/2u]); let sc0=select(sp0.x,sp0.y,((s_base+b0)&1u)!=0u);
            let sp1=unpack2x16float(Scales[(s_base+b1)/2u]); let sc1=select(sp1.x,sp1.y,((s_base+b1)&1u)!=0u);
            let sp2=unpack2x16float(Scales[(s_base+b2)/2u]); let sc2=select(sp2.x,sp2.y,((s_base+b2)&1u)!=0u);
            let sp3=unpack2x16float(Scales[(s_base+b3)/2u]); let sc3=select(sp3.x,sp3.y,((s_base+b3)&1u)!=0u);
            acc += dot(xv0,wv0)*sc0+dot(xv1,wv1)*sc1+dot(xv2,wv2)*sc2+dot(xv3,wv3)*sc3;
        }
    }
    let warp_sum = subgroupAdd(acc);
    if (lane==0u && col<N) { Y[row*N+col] += warp_sum + Bias[col]; }
}
