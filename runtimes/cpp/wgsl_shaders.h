#pragma once
/**
 * wgsl_shaders.h -- Auto-generated from compiler/kernels/*.wgsl
 * Do not edit manually. Regenerate with: python compiler/gen_wgsl_shaders.py
 */

#include <string>
#include <unordered_map>

struct ShaderInfo {
    const char* source;
    uint32_t numBindings;
};

// [gguf_q8] q8_matmul (6 bindings)
static const char* WGSL_Q8_MATMUL = R"WGSL(
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
    let row = wid.x;
    let tile_col = wid.y;
    let tid = lid.x;

    let K = _params_[0];
    let N = _params_[1];
    let x_base = row * K;

    // 8 warps × 32 threads: each warp computes one output
    let warp_id = tid / 32u;
    let lane = tid % 32u;
    let col = tile_col * TILE_N + warp_id;

    // K_PER_ITER=8: each lane processes 8 elements per 256-element stride
    // (32 lanes × 8 elem/lane = 256 elements = 8 Q8_0 blocks per stride)
    let n_strides = K / 256u;
    let stride_w = K / 4u;  // u32 per weight row

    var acc: f32 = 0.0;

    if (col < N) {
        let w_base = col * stride_w;
        let n_blocks = K / 32u;
        let s_base = col * n_blocks;

        for (var g = 0u; g < n_strides; g = g + 1u) {
            // Read 8 fp32 activations (two vec4 loads)
            let k_base = g * 256u + lane * 8u;
            let xv0 = vec4<f32>(X[x_base + k_base],
                                X[x_base + k_base + 1u],
                                X[x_base + k_base + 2u],
                                X[x_base + k_base + 3u]);
            let xv1 = vec4<f32>(X[x_base + k_base + 4u],
                                X[x_base + k_base + 5u],
                                X[x_base + k_base + 6u],
                                X[x_base + k_base + 7u]);

            // Read 2 packed u32 weights (8 int8 values)
            let w_off = w_base + g * 64u + lane * 2u;
            let pw0 = W_Q8[w_off];
            let pw1 = W_Q8[w_off + 1u];

            // Extract int8 → f32 and form vec4 for dot product
            let wv0 = vec4<f32>(f32(extractBits(i32(pw0), 0u, 8u)),
                                f32(extractBits(i32(pw0), 8u, 8u)),
                                f32(extractBits(i32(pw0), 16u, 8u)),
                                f32(extractBits(i32(pw0), 24u, 8u)));
            let wv1 = vec4<f32>(f32(extractBits(i32(pw1), 0u, 8u)),
                                f32(extractBits(i32(pw1), 8u, 8u)),
                                f32(extractBits(i32(pw1), 16u, 8u)),
                                f32(extractBits(i32(pw1), 24u, 8u)));

            // Per-block scales: 2 blocks per 8 elements
            // block_idx = (g * 256 + lane * 8) / 32
            let bi = g * 8u + lane / 4u;
            let si0 = s_base + bi;
            let si1 = si0 + 1u;
            // Correction: lane * 8 spans TWO 32-element blocks when lane >= 4
            // Block for first 4 elements: (g*256 + lane*8) / 32 = g*8 + lane/4
            // Block for second 4 elements: (g*256 + lane*8 + 4) / 32
            let block0 = g * 8u + (lane * 8u) / 32u;
            let block1 = g * 8u + (lane * 8u + 4u) / 32u;
            let sp0 = unpack2x16float(Scales[(s_base + block0) / 2u]);
            let scale0 = select(sp0.x, sp0.y, ((s_base + block0) & 1u) != 0u);
            let sp1 = unpack2x16float(Scales[(s_base + block1) / 2u]);
            let scale1 = select(sp1.x, sp1.y, ((s_base + block1) & 1u) != 0u);

            // vec4 dot products with per-block scaling
            acc += dot(xv0, wv0) * scale0 + dot(xv1, wv1) * scale1;
        }
    }

    let warp_sum = subgroupAdd(acc);

    if (lane == 0u && col < N) {
        Y[row * N + col] = warp_sum + Bias[col];
    }
}
)WGSL";

// [gguf_q8] q8_matmul_add (6 bindings)
static const char* WGSL_Q8_MATMUL_ADD = R"WGSL(
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
    let row = wid.x;
    let tile_col = wid.y;
    let tid = lid.x;

    let K = _params_[0];
    let N = _params_[1];
    let x_base = row * K;

    // 8 warps × 32 threads: each warp computes one output
    let warp_id = tid / 32u;
    let lane = tid % 32u;
    let col = tile_col * TILE_N + warp_id;

    // K_PER_ITER=8: each lane processes 8 elements per 256-element stride
    // (32 lanes × 8 elem/lane = 256 elements = 8 Q8_0 blocks per stride)
    let n_strides = K / 256u;
    let stride_w = K / 4u;  // u32 per weight row

    var acc: f32 = 0.0;

    if (col < N) {
        let w_base = col * stride_w;
        let n_blocks = K / 32u;
        let s_base = col * n_blocks;

        for (var g = 0u; g < n_strides; g = g + 1u) {
            // Read 8 fp32 activations (two vec4 loads)
            let k_base = g * 256u + lane * 8u;
            let xv0 = vec4<f32>(X[x_base + k_base],
                                X[x_base + k_base + 1u],
                                X[x_base + k_base + 2u],
                                X[x_base + k_base + 3u]);
            let xv1 = vec4<f32>(X[x_base + k_base + 4u],
                                X[x_base + k_base + 5u],
                                X[x_base + k_base + 6u],
                                X[x_base + k_base + 7u]);

            // Read 2 packed u32 weights (8 int8 values)
            let w_off = w_base + g * 64u + lane * 2u;
            let pw0 = W_Q8[w_off];
            let pw1 = W_Q8[w_off + 1u];

            // Extract int8 → f32 and form vec4 for dot product
            let wv0 = vec4<f32>(f32(extractBits(i32(pw0), 0u, 8u)),
                                f32(extractBits(i32(pw0), 8u, 8u)),
                                f32(extractBits(i32(pw0), 16u, 8u)),
                                f32(extractBits(i32(pw0), 24u, 8u)));
            let wv1 = vec4<f32>(f32(extractBits(i32(pw1), 0u, 8u)),
                                f32(extractBits(i32(pw1), 8u, 8u)),
                                f32(extractBits(i32(pw1), 16u, 8u)),
                                f32(extractBits(i32(pw1), 24u, 8u)));

            // Per-block scales: 2 blocks per 8 elements
            // block_idx = (g * 256 + lane * 8) / 32
            let bi = g * 8u + lane / 4u;
            let si0 = s_base + bi;
            let si1 = si0 + 1u;
            // Correction: lane * 8 spans TWO 32-element blocks when lane >= 4
            // Block for first 4 elements: (g*256 + lane*8) / 32 = g*8 + lane/4
            // Block for second 4 elements: (g*256 + lane*8 + 4) / 32
            let block0 = g * 8u + (lane * 8u) / 32u;
            let block1 = g * 8u + (lane * 8u + 4u) / 32u;
            let sp0 = unpack2x16float(Scales[(s_base + block0) / 2u]);
            let scale0 = select(sp0.x, sp0.y, ((s_base + block0) & 1u) != 0u);
            let sp1 = unpack2x16float(Scales[(s_base + block1) / 2u]);
            let scale1 = select(sp1.x, sp1.y, ((s_base + block1) & 1u) != 0u);

            // vec4 dot products with per-block scaling
            acc += dot(xv0, wv0) * scale0 + dot(xv1, wv1) * scale1;
        }
    }

    let warp_sum = subgroupAdd(acc);

    if (lane == 0u && col < N) {
        Y[row * N + col] += warp_sum + Bias[col];
    }
}
)WGSL";

// [gguf_q8] q8_matmul_add_fast (6 bindings)
static const char* WGSL_Q8_MATMUL_ADD_FAST = R"WGSL(
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
)WGSL";

// [gguf_q8] q8_matmul_fast (6 bindings)
static const char* WGSL_Q8_MATMUL_FAST = R"WGSL(
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
    let row = wid.x;
    let tile_col = wid.y;
    let tid = lid.x;
    let K = _params_[0];
    let N = _params_[1];
    let x_base = row * K;
    let warp_id = tid / 32u;
    let lane = tid % 32u;
    let col = tile_col * TILE_N + warp_id;
    let n_strides = K / 512u;
    let stride_w = K / 4u;
    var acc: f32 = 0.0;
    if (col < N) {
        let w_base = col * stride_w;
        let n_blocks = K / 32u;
        let s_base = col * n_blocks;
        for (var g = 0u; g < n_strides; g = g + 1u) {
            let k_base = g * 512u + lane * 16u;
            let xv0 = vec4<f32>(X[x_base+k_base], X[x_base+k_base+1u],
                                X[x_base+k_base+2u], X[x_base+k_base+3u]);
            let xv1 = vec4<f32>(X[x_base+k_base+4u], X[x_base+k_base+5u],
                                X[x_base+k_base+6u], X[x_base+k_base+7u]);
            let xv2 = vec4<f32>(X[x_base+k_base+8u], X[x_base+k_base+9u],
                                X[x_base+k_base+10u], X[x_base+k_base+11u]);
            let xv3 = vec4<f32>(X[x_base+k_base+12u], X[x_base+k_base+13u],
                                X[x_base+k_base+14u], X[x_base+k_base+15u]);
            let w_off = w_base + g * 128u + lane * 4u;
            let pw0 = W_Q8[w_off]; let pw1 = W_Q8[w_off+1u];
            let pw2 = W_Q8[w_off+2u]; let pw3 = W_Q8[w_off+3u];
            let wv0 = vec4<f32>(f32(extractBits(i32(pw0),0u,8u)),f32(extractBits(i32(pw0),8u,8u)),
                                f32(extractBits(i32(pw0),16u,8u)),f32(extractBits(i32(pw0),24u,8u)));
            let wv1 = vec4<f32>(f32(extractBits(i32(pw1),0u,8u)),f32(extractBits(i32(pw1),8u,8u)),
                                f32(extractBits(i32(pw1),16u,8u)),f32(extractBits(i32(pw1),24u,8u)));
            let wv2 = vec4<f32>(f32(extractBits(i32(pw2),0u,8u)),f32(extractBits(i32(pw2),8u,8u)),
                                f32(extractBits(i32(pw2),16u,8u)),f32(extractBits(i32(pw2),24u,8u)));
            let wv3 = vec4<f32>(f32(extractBits(i32(pw3),0u,8u)),f32(extractBits(i32(pw3),8u,8u)),
                                f32(extractBits(i32(pw3),16u,8u)),f32(extractBits(i32(pw3),24u,8u)));
            let block0 = g*16u + (lane*16u)/32u;
            let block1 = g*16u + (lane*16u+4u)/32u;
            let block2 = g*16u + (lane*16u+8u)/32u;
            let block3 = g*16u + (lane*16u+12u)/32u;
            let sp0 = unpack2x16float(Scales[(s_base+block0)/2u]);
            let sc0 = select(sp0.x, sp0.y, ((s_base+block0)&1u)!=0u);
            let sp1 = unpack2x16float(Scales[(s_base+block1)/2u]);
            let sc1 = select(sp1.x, sp1.y, ((s_base+block1)&1u)!=0u);
            let sp2 = unpack2x16float(Scales[(s_base+block2)/2u]);
            let sc2 = select(sp2.x, sp2.y, ((s_base+block2)&1u)!=0u);
            let sp3 = unpack2x16float(Scales[(s_base+block3)/2u]);
            let sc3 = select(sp3.x, sp3.y, ((s_base+block3)&1u)!=0u);
            acc += dot(xv0,wv0)*sc0 + dot(xv1,wv1)*sc1
                 + dot(xv2,wv2)*sc2 + dot(xv3,wv3)*sc3;
        }
    }
    let warp_sum = subgroupAdd(acc);
    if (lane==0u && col<N) { Y[row*N+col] = warp_sum + Bias[col]; }
}
)WGSL";

// [shared] add_rms_norm (6 bindings)
static const char* WGSL_ADD_RMS_NORM = R"WGSL(
enable subgroups;

// Auto-generated by Triton WebGPU Backend
// Kernel: add_rms_norm_loop_kernel
// Workgroup size: 128 (4 warps x 32 threads)

var<workgroup> _smem: array<i32, 4>;

@group(0) @binding(0) var<storage, read_write> buf0: array<f32>;  // X
@group(0) @binding(1) var<storage, read> buf1: array<f32>;  // Residual
@group(0) @binding(2) var<storage, read_write> buf2: array<f32>;  // Y
@group(0) @binding(3) var<storage, read> buf3: array<f32>;  // W
@group(0) @binding(4) var<storage, read_write> buf4: array<f32>;  // Rstd

struct Params {
    stride: i32,
    N: i32,
    eps: f32,
};
@group(0) @binding(5) var<storage, read> params: Params;

@compute @workgroup_size(128)
fn main(
    @builtin(workgroup_id) _wg_id: vec3<u32>,
    @builtin(local_invocation_id) _lid: vec3<u32>,
    @builtin(num_workgroups) _num_wg: vec3<u32>,
) {
    let v11: i32 = i32(_wg_id.x);
    let v12: i32 = params.N + 127;
    let v13: i32 = v12 / 128;
    var v15: i32 = 0;
    var v16_0: f32 = f32(0);
    loop {
        let v17: bool = v15 < v13;
        if !v17 { break; }
        let v19: i32 = v15 * 128;
        let v20: i32 = i32(_lid.x);
        let v21: i32 = v20 & 127;
        let v22: i32 = i32(u32(v21) % u32(32));
        let v23: i32 = i32(_lid.x);
        let v24: i32 = i32(u32(v23) / u32(32));
        let v25: i32 = v22 << u32(0);
        let v26: i32 = 0 | v25;
        let v27: i32 = v24 << u32(5);
        let v28: i32 = v26 | v27;
        let v29: i32 = v28 & 127;
        let v30: i32 = i32(u32(v29) >> u32(0));
        let v31: i32 = v30 | 0;
        let v32: i32 = 0 ^ v31;
        let v33: i32 = v32 ^ 0;
        let v34: i32 = v33 + 0;
        let v35: i32 = v19 + v34;
        let v36: bool = v35 < params.N;
        let v37: i32 = v11 * params.stride;
        let v40: f32 = buf0[u32((v37 + v35))];
        let v41: f32 = select(f32(0.0), v40, v36);
        let v44: f32 = buf1[u32((v37 + v35))];
        let v45: f32 = select(f32(0.0), v44, v36);
        let v46: f32 = v41 + v45;
        let v47: f32 = select(f32(0), v46, v36);
        if v36 { buf0[u32((v37 + v35))] = v47; }
        let v48: f32 = v46 * v46;
        let v50: f32 = v16_0 + v48;
        let v52: i32 = v15 + 1;
        v15 = v52;
        v16_0 = v50;
    }
    let v55: i32 = bitcast<i32>(v16_0);
    let v56: i32 = subgroupShuffleXor(v55, u32(16));
    let v57: f32 = bitcast<f32>(v56);
    let v58: f32 = v16_0 + v57;
    let v59: i32 = bitcast<i32>(v58);
    let v60: i32 = subgroupShuffleXor(v59, u32(8));
    let v61: f32 = bitcast<f32>(v60);
    let v62: f32 = v58 + v61;
    let v63: i32 = bitcast<i32>(v62);
    let v64: i32 = subgroupShuffleXor(v63, u32(4));
    let v65: f32 = bitcast<f32>(v64);
    let v66: f32 = v62 + v65;
    let v67: i32 = bitcast<i32>(v66);
    let v68: i32 = subgroupShuffleXor(v67, u32(2));
    let v69: f32 = bitcast<f32>(v68);
    let v70: f32 = v66 + v69;
    let v71: i32 = bitcast<i32>(v70);
    let v72: i32 = subgroupShuffleXor(v71, u32(1));
    let v73: f32 = bitcast<f32>(v72);
    let v74: f32 = v70 + v73;
    let v75: i32 = i32(_lid.x);
    let v76: i32 = v75 & 127;
    let v77: i32 = i32(u32(v76) % u32(32));
    let v78: i32 = i32(_lid.x);
    let v79: i32 = i32(u32(v78) / u32(32));
    let v80: i32 = v77 << u32(0);
    let v81: i32 = 0 | v80;
    let v82: i32 = v79 << u32(5);
    let v83: i32 = v81 | v82;
    let v84: i32 = v83 & 96;
    let v85: i32 = i32(u32(v84) >> u32(3));
    let v86: i32 = 0 | v85;
    let v87: i32 = 0 ^ v86;
    let v88: i32 = v87 ^ 0;
    let v89: i32 = v88 ^ 0;
    let v90: i32 = v89 + 0;
    _smem[u32(v90) >> 2u] = bitcast<i32>(v74);
    workgroupBarrier();
    let v94: i32 = i32(_lid.x);
    let v95: i32 = v94 & 127;
    let v96: i32 = i32(u32(v95) % u32(32));
    let v97: i32 = i32(_lid.x);
    let v98: i32 = i32(u32(v97) / u32(32));
    let v99: i32 = v96 << u32(0);
    let v100: i32 = 0 | v99;
    let v101: i32 = v98 << u32(5);
    let v102: i32 = v100 | v101;
    let v103: i32 = v102 & 3;
    let v104: i32 = v103 << u32(2);
    let v105: i32 = v104 | 0;
    let v106: i32 = 0 ^ v105;
    let v107: i32 = v106 ^ 0;
    let v108: i32 = v107 ^ 0;
    let v109: i32 = v108 + 0;
    let v111: f32 = bitcast<f32>(_smem[u32(v109) >> 2u]);
    let v114: i32 = bitcast<i32>(v111);
    let v115: i32 = subgroupShuffleXor(v114, u32(2));
    let v116: f32 = bitcast<f32>(v115);
    let v117: f32 = v111 + v116;
    let v118: i32 = bitcast<i32>(v117);
    let v119: i32 = subgroupShuffleXor(v118, u32(1));
    let v120: f32 = bitcast<f32>(v119);
    let v121: f32 = v117 + v120;
    let v122: f32 = f32(params.N);
    let v123: f32 = v121 / v122;
    let v124: f32 = v123 + params.eps;
    let v125: f32 = sqrt(v124);
    let v126: f32 = f32(1.0) / v125;
    var v128: i32 = 0;
    loop {
        let v129: bool = v128 < v13;
        if !v129 { break; }
        let v131: i32 = v128 * 128;
        let v132: i32 = i32(_lid.x);
        let v133: i32 = v132 & 127;
        let v134: i32 = i32(u32(v133) % u32(32));
        let v135: i32 = i32(_lid.x);
        let v136: i32 = i32(u32(v135) / u32(32));
        let v137: i32 = v134 << u32(0);
        let v138: i32 = 0 | v137;
        let v139: i32 = v136 << u32(5);
        let v140: i32 = v138 | v139;
        let v141: i32 = v140 & 127;
        let v142: i32 = i32(u32(v141) >> u32(0));
        let v143: i32 = v142 | 0;
        let v144: i32 = 0 ^ v143;
        let v145: i32 = v144 ^ 0;
        let v146: i32 = v145 + 0;
        let v147: i32 = v131 + v146;
        let v148: bool = v147 < params.N;
        let v149: i32 = v11 * params.stride;
        let v152: f32 = buf0[u32((v149 + v147))];
        let v153: f32 = select(f32(0.0), v152, v148);
        let v155: f32 = buf3[u32(v147)];
        let v156: f32 = select(f32(1.0), v155, v148);
        let v159: f32 = v153 * v126;
        let v160: f32 = v159 * v156;
        let v161: f32 = select(f32(0), v160, v148);
        if v148 { buf2[u32((v149 + v147))] = v161; }
        let v162: i32 = v128 + 1;
        v128 = v162;
    }
    if _lid.x == 0u { buf4[u32(v11)] = v126; }
}
)WGSL";

// [shared] argmax (3 bindings)
static const char* WGSL_ARGMAX = R"WGSL(
enable subgroups;

@group(0) @binding(0) var<storage, read> Logits: array<f32>;
@group(0) @binding(1) var<storage, read_write> Result: array<i32>;
@group(0) @binding(2) var<storage, read> _params_: array<u32>;

var<workgroup> wg_max_val: array<f32, 8>;
var<workgroup> wg_max_idx: array<i32, 8>;

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) lid: vec3<u32>,
        @builtin(workgroup_id) wid: vec3<u32>) {
    let N = _params_[0];
    let tid = lid.x;
    let warp_id = tid / 32u;
    let lane = tid % 32u;

    // Each thread scans N/256 elements
    var local_max: f32 = -1e30;
    var local_idx: i32 = 0;
    var i = tid;
    for (; i < N; i = i + 256u) {
        let v = Logits[i];
        if (v > local_max) {
            local_max = v;
            local_idx = i32(i);
        }
    }

    // Warp reduce: find max within 32 lanes
    for (var offset = 16u; offset > 0u; offset = offset >> 1u) {
        let other_val = subgroupShuffleXor(bitcast<i32>(local_max), offset);
        let other_idx = subgroupShuffleXor(local_idx, offset);
        let ov = bitcast<f32>(other_val);
        if (ov > local_max) {
            local_max = ov;
            local_idx = other_idx;
        }
    }

    // Write warp results to shared memory
    if (lane == 0u) {
        wg_max_val[warp_id] = local_max;
        wg_max_idx[warp_id] = local_idx;
    }
    workgroupBarrier();

    // Final reduce across 8 warps (done by thread 0)
    if (tid == 0u) {
        var best_val = wg_max_val[0];
        var best_idx = wg_max_idx[0];
        for (var w = 1u; w < 8u; w = w + 1u) {
            if (wg_max_val[w] > best_val) {
                best_val = wg_max_val[w];
                best_idx = wg_max_idx[w];
            }
        }
        Result[0] = best_idx;
    }
}
)WGSL";

// [shared] embed_gather (4 bindings)
static const char* WGSL_EMBED_GATHER = R"WGSL(
@group(0) @binding(0) var<storage, read> EmbeddingTable: array<f32>;
@group(0) @binding(1) var<storage, read> TokenId: array<i32>;
@group(0) @binding(2) var<storage, read_write> X: array<f32>;
@group(0) @binding(3) var<storage, read> _params_: array<u32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let E = _params_[0];
    let idx = gid.x;
    if (idx >= E) { return; }
    let token = u32(TokenId[0]);
    X[idx] = EmbeddingTable[token * E + idx];
}
)WGSL";

// [shared] fp16_gemm (5 bindings)
static const char* WGSL_FP16_GEMM = R"WGSL(
enable subgroups;

@group(0) @binding(0) var<storage, read_write> X: array<f32>;
@group(0) @binding(1) var<storage, read_write> W: array<u32>;
@group(0) @binding(2) var<storage, read_write> Bias: array<f32>;
@group(0) @binding(3) var<storage, read_write> Y: array<f32>;
@group(0) @binding(4) var<storage, read_write> _params_: array<u32>;

const TILE_N: u32 = 8u;

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) lid: vec3<u32>,
        @builtin(workgroup_id) wid: vec3<u32>) {
    let row = wid.x;
    let tile_col = wid.y;
    let tid = lid.x;

    let K = _params_[0];
    let N = _params_[1];
    let x_base = row * K;

    // 8 warps × 32 threads: each warp computes one output
    let warp_id = tid / 32u;
    let lane = tid % 32u;
    let col = tile_col * TILE_N + warp_id;

    // Each lane handles K/32 elements, strided by 4 (vec4 processing)
    // lane processes elements: lane*4, lane*4 + 32*4, lane*4 + 64*4, ...
    var acc: f32 = 0.0;

    if (col < N) {
        // W layout: (N, K) row-major, stored as u32 (2 fp16 per u32)
        // w_base = col * (K/2) — index into u32 array
        let w_base = col * (K / 2u);

        // Each lane processes 4 consecutive elements starting at lane*4
        // Stride between lane iterations: 32 lanes * 4 elements = 128
        let stride = 128u;
        var k = lane * 4u;

        for (; k + 3u < K; k = k + stride) {
            // Load 4 fp32 activations
            let x0 = X[x_base + k];
            let x1 = X[x_base + k + 1u];
            let x2 = X[x_base + k + 2u];
            let x3 = X[x_base + k + 3u];

            // Load 2 u32 = 4 fp16 weights, unpack to fp32
            let w01 = unpack2x16float(W[w_base + k / 2u]);
            let w23 = unpack2x16float(W[w_base + k / 2u + 1u]);

            // dot(vec4<f32>, vec4<f32>) — 4 FMAs
            acc += dot(vec4<f32>(x0, x1, x2, x3),
                       vec4<f32>(w01.x, w01.y, w23.x, w23.y));
        }
    }

    // Reduce across 32 lanes in warp
    let warp_sum = subgroupAdd(acc);

    if (lane == 0u && col < N) {
        Y[row * N + col] = warp_sum + Bias[col];
    }
}
)WGSL";

// [shared] fp16_gemm_wide (5 bindings)
static const char* WGSL_FP16_GEMM_WIDE = R"WGSL(
enable subgroups;

@group(0) @binding(0) var<storage, read_write> X: array<f32>;
@group(0) @binding(1) var<storage, read_write> W: array<u32>;
@group(0) @binding(2) var<storage, read_write> Bias: array<f32>;
@group(0) @binding(3) var<storage, read_write> Y: array<f32>;
@group(0) @binding(4) var<storage, read_write> _params_: array<u32>;

const TILE_N: u32 = 32u;
const COLS_PER_WARP: u32 = 4u;

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) lid: vec3<u32>,
        @builtin(workgroup_id) wid: vec3<u32>) {
    let row = wid.x;
    let tile_col = wid.y;
    let tid = lid.x;

    let K = _params_[0];
    let N = _params_[1];
    let x_base = row * K;

    let warp_id = tid / 32u;
    let lane = tid % 32u;

    // Each warp handles COLS_PER_WARP=4 columns
    let base_col = tile_col * TILE_N + warp_id * COLS_PER_WARP;

    var acc0: f32 = 0.0;
    var acc1: f32 = 0.0;
    var acc2: f32 = 0.0;
    var acc3: f32 = 0.0;

    let half_K = K / 2u;
    let stride = 128u;  // 32 lanes * 4 elements = 128
    var k = lane * 4u;

    for (; k + 3u < K; k = k + stride) {
        // Load X once, reuse for all 4 columns
        let x0 = X[x_base + k];
        let x1 = X[x_base + k + 1u];
        let x2 = X[x_base + k + 2u];
        let x3 = X[x_base + k + 3u];
        let xv = vec4<f32>(x0, x1, x2, x3);

        // Column 0
        if (base_col < N) {
            let w_base0 = base_col * half_K;
            let w01 = unpack2x16float(W[w_base0 + k / 2u]);
            let w23 = unpack2x16float(W[w_base0 + k / 2u + 1u]);
            acc0 += dot(xv, vec4<f32>(w01.x, w01.y, w23.x, w23.y));
        }
        // Column 1
        if (base_col + 1u < N) {
            let w_base1 = (base_col + 1u) * half_K;
            let w01 = unpack2x16float(W[w_base1 + k / 2u]);
            let w23 = unpack2x16float(W[w_base1 + k / 2u + 1u]);
            acc1 += dot(xv, vec4<f32>(w01.x, w01.y, w23.x, w23.y));
        }
        // Column 2
        if (base_col + 2u < N) {
            let w_base2 = (base_col + 2u) * half_K;
            let w01 = unpack2x16float(W[w_base2 + k / 2u]);
            let w23 = unpack2x16float(W[w_base2 + k / 2u + 1u]);
            acc2 += dot(xv, vec4<f32>(w01.x, w01.y, w23.x, w23.y));
        }
        // Column 3
        if (base_col + 3u < N) {
            let w_base3 = (base_col + 3u) * half_K;
            let w01 = unpack2x16float(W[w_base3 + k / 2u]);
            let w23 = unpack2x16float(W[w_base3 + k / 2u + 1u]);
            acc3 += dot(xv, vec4<f32>(w01.x, w01.y, w23.x, w23.y));
        }
    }

    // Warp reduce
    let sum0 = subgroupAdd(acc0);
    let sum1 = subgroupAdd(acc1);
    let sum2 = subgroupAdd(acc2);
    let sum3 = subgroupAdd(acc3);

    if (lane == 0u) {
        if (base_col      < N) { Y[row * N + base_col]      = sum0 + Bias[base_col]; }
        if (base_col + 1u < N) { Y[row * N + base_col + 1u] = sum1 + Bias[base_col + 1u]; }
        if (base_col + 2u < N) { Y[row * N + base_col + 2u] = sum2 + Bias[base_col + 2u]; }
        if (base_col + 3u < N) { Y[row * N + base_col + 3u] = sum3 + Bias[base_col + 3u]; }
    }
}
)WGSL";

// [shared] fused_qknorm_rope (9 bindings)
static const char* WGSL_FUSED_QKNORM_ROPE = R"WGSL(
enable subgroups;

// Auto-generated by Triton WebGPU Backend
// Kernel: fused_qknorm_rope_qkv_kernel
// Workgroup size: 128 (4 warps x 32 threads)

var<workgroup> _smem: array<i32, 4>;

@group(0) @binding(0) var<storage, read> buf0: array<f32>;  // QKV
@group(0) @binding(1) var<storage, read_write> buf1: array<f32>;  // Q_out
@group(0) @binding(2) var<storage, read_write> buf2: array<f32>;  // K_cache
@group(0) @binding(3) var<storage, read_write> buf3: array<f32>;  // V_cache
@group(0) @binding(4) var<storage, read> buf4: array<f32>;  // CosTable
@group(0) @binding(5) var<storage, read> buf5: array<f32>;  // SinTable
@group(0) @binding(6) var<storage, read> buf6: array<f32>;  // NormQ
@group(0) @binding(7) var<storage, read> buf7: array<f32>;  // NormK

struct Params {
    n_head: i32,
    q_size: i32,
    kv_size: i32,
    pos: i32,
    half_rot: i32,
    cache_offset: i32,
    eps: f32,
};
@group(0) @binding(8) var<storage, read> params: Params;

@compute @workgroup_size(128)
fn main(
    @builtin(workgroup_id) _wg_id: vec3<u32>,
    @builtin(local_invocation_id) _lid: vec3<u32>,
    @builtin(num_workgroups) _num_wg: vec3<u32>,
) {
    let v18: i32 = i32(_wg_id.x);
    let v19: i32 = i32(_lid.x);
    let v20: i32 = v19 & 127;
    let v21: i32 = i32(u32(v20) % u32(32));
    let v22: i32 = i32(_lid.x);
    let v23: i32 = i32(u32(v22) / u32(32));
    let v24: i32 = v21 << u32(0);
    let v25: i32 = 0 | v24;
    let v26: i32 = v23 << u32(5);
    let v27: i32 = v25 | v26;
    let v28: i32 = v27 & 127;
    let v29: i32 = i32(u32(v28) >> u32(0));
    let v30: i32 = v29 | 0;
    let v31: i32 = 0 ^ v30;
    let v32: i32 = v31 ^ 0;
    let v33: i32 = v32 + 0;
    let v34: bool = v18 < params.n_head;
    let v35: i32 = v18 - params.n_head;
    let v36: i32 = v18 * 128;
    let v37: i32 = v35 * 128;
    let v38: i32 = params.q_size + v37;
    let v39: i32 = select(v38, v36, v34);
    let v42: f32 = buf0[u32((v39 + v33))];
    let v43: f32 = v42 * v42;
    let v44: i32 = bitcast<i32>(v43);
    let v45: i32 = subgroupShuffleXor(v44, u32(16));
    let v46: f32 = bitcast<f32>(v45);
    let v47: f32 = v43 + v46;
    let v48: i32 = bitcast<i32>(v47);
    let v49: i32 = subgroupShuffleXor(v48, u32(8));
    let v50: f32 = bitcast<f32>(v49);
    let v51: f32 = v47 + v50;
    let v52: i32 = bitcast<i32>(v51);
    let v53: i32 = subgroupShuffleXor(v52, u32(4));
    let v54: f32 = bitcast<f32>(v53);
    let v55: f32 = v51 + v54;
    let v56: i32 = bitcast<i32>(v55);
    let v57: i32 = subgroupShuffleXor(v56, u32(2));
    let v58: f32 = bitcast<f32>(v57);
    let v59: f32 = v55 + v58;
    let v60: i32 = bitcast<i32>(v59);
    let v61: i32 = subgroupShuffleXor(v60, u32(1));
    let v62: f32 = bitcast<f32>(v61);
    let v63: f32 = v59 + v62;
    let v64: i32 = i32(_lid.x);
    let v65: i32 = v64 & 127;
    let v66: i32 = i32(u32(v65) % u32(32));
    let v67: i32 = i32(_lid.x);
    let v68: i32 = i32(u32(v67) / u32(32));
    let v69: i32 = v66 << u32(0);
    let v70: i32 = 0 | v69;
    let v71: i32 = v68 << u32(5);
    let v72: i32 = v70 | v71;
    let v73: i32 = v72 & 96;
    let v74: i32 = i32(u32(v73) >> u32(3));
    let v75: i32 = 0 | v74;
    let v76: i32 = 0 ^ v75;
    let v77: i32 = v76 ^ 0;
    let v78: i32 = v77 ^ 0;
    let v79: i32 = v78 + 0;
    _smem[u32(v79) >> 2u] = bitcast<i32>(v63);
    workgroupBarrier();
    let v83: i32 = i32(_lid.x);
    let v84: i32 = v83 & 127;
    let v85: i32 = i32(u32(v84) % u32(32));
    let v86: i32 = i32(_lid.x);
    let v87: i32 = i32(u32(v86) / u32(32));
    let v88: i32 = v85 << u32(0);
    let v89: i32 = 0 | v88;
    let v90: i32 = v87 << u32(5);
    let v91: i32 = v89 | v90;
    let v92: i32 = v91 & 3;
    let v93: i32 = v92 << u32(2);
    let v94: i32 = v93 | 0;
    let v95: i32 = 0 ^ v94;
    let v96: i32 = v95 ^ 0;
    let v97: i32 = v96 ^ 0;
    let v98: i32 = v97 + 0;
    let v100: f32 = bitcast<f32>(_smem[u32(v98) >> 2u]);
    let v103: i32 = bitcast<i32>(v100);
    let v104: i32 = subgroupShuffleXor(v103, u32(2));
    let v105: f32 = bitcast<f32>(v104);
    let v106: f32 = v100 + v105;
    let v107: i32 = bitcast<i32>(v106);
    let v108: i32 = subgroupShuffleXor(v107, u32(1));
    let v109: f32 = bitcast<f32>(v108);
    let v110: f32 = v106 + v109;
    let v111: f32 = v110 / f32(128.0);
    let v112: f32 = v111 + params.eps;
    let v113: f32 = sqrt(v112);
    let v114: f32 = f32(1.0) / v113;
    let v116: f32 = buf6[u32(v33)];
    let v118: f32 = buf7[u32(v33)];
    let v119: f32 = select(v118, v116, v34);
    let v120: f32 = v42 * v114;
    let v121: f32 = v120 * v119;
    let v122: i32 = v33 % params.half_rot;
    let v123: i32 = params.half_rot * 2;
    let v124: bool = v33 < v123;
    let v125: i32 = params.pos * params.half_rot;
    let v128: f32 = buf4[u32((v125 + v122))];
    let v129: f32 = select(f32(1.0), v128, v124);
    let v132: f32 = buf5[u32((v125 + v122))];
    let v133: f32 = select(f32(0.0), v132, v124);
    let v134: bool = v33 < params.half_rot;
    let v135: f32 = select(f32(1.0), f32(-1.0), v134);
    let v136: i32 = v33 + params.half_rot;
    let v137: i32 = v33 - params.half_rot;
    let v138: i32 = select(v137, v136, v134);
    let v139: i32 = select(v33, v138, v124);
    let v141: f32 = buf0[u32((v39 + v139))];
    let v143: f32 = buf6[u32(v139)];
    let v145: f32 = buf7[u32(v139)];
    let v146: f32 = select(v145, v143, v34);
    let v147: f32 = v141 * v114;
    let v148: f32 = v147 * v146;
    let v149: f32 = v121 * v129;
    let v150: f32 = v135 * v148;
    let v151: f32 = v150 * v133;
    let v152: f32 = v149 + v151;
    let v153: f32 = select(v121, v152, v124);
    let v156: f32 = select(f32(0), v153, v34);
    if v34 { buf1[u32((v36 + v33))] = v156; }
    let v157: i32 = select(0, 1, v34);
    let v158: bool = v157 == 0;
    let v162: f32 = select(f32(0), v153, v158);
    if v158 { buf2[u32(((params.cache_offset + v37) + v33))] = v162; }
    let v163: i32 = params.q_size + params.kv_size;
    let v164: i32 = v163 + v37;
    let v167: f32 = buf0[u32((v164 + v33))];
    let v168: f32 = select(f32(0.0), v167, v158);
    let v172: f32 = select(f32(0), v168, v158);
    if v158 { buf3[u32(((params.cache_offset + v37) + v33))] = v172; }
}
)WGSL";

// [shared] gqa_chunked_pass1 (5 bindings)
static const char* WGSL_GQA_CHUNKED_PASS1 = R"WGSL(
enable subgroups;

@group(0) @binding(0) var<storage, read_write> Q: array<f32>;
@group(0) @binding(1) var<storage, read_write> K_cache: array<f32>;
@group(0) @binding(2) var<storage, read_write> V_cache: array<f32>;
@group(0) @binding(3) var<storage, read_write> Partials: array<f32>;
@group(0) @binding(4) var<storage, read_write> _params_: array<u32>;

const HD: u32 = 128u;
const CHUNK: u32 = 64u;
// Each thread processes HD_PER_THREAD elements, total 32 * 4 = 128
const HD_PER_THREAD: u32 = 4u;

@compute @workgroup_size(32)
fn main(@builtin(local_invocation_id) lid: vec3<u32>,
        @builtin(workgroup_id) wid: vec3<u32>) {
    let head = wid.x;
    let chunk_id = wid.y;
    let lane = lid.x;

    let kv_stride = _params_[0];
    let n_rep = _params_[1];
    let T_total = _params_[2];
    let n_chunks = _params_[4];
    let scale = bitcast<f32>(_params_[5]);
    let neg_inf = bitcast<f32>(_params_[6]);
    let max_chunks = _params_[7];

    let kv_head = head / n_rep;
    let kv_off = kv_head * HD;
    let q_base = head * HD;

    // Load Q into registers (4 elements per thread)
    let q0 = Q[q_base + lane * HD_PER_THREAD];
    let q1 = Q[q_base + lane * HD_PER_THREAD + 1u];
    let q2 = Q[q_base + lane * HD_PER_THREAD + 2u];
    let q3 = Q[q_base + lane * HD_PER_THREAD + 3u];

    let t_start = chunk_id * CHUNK;

    var acc0: f32 = 0.0; var acc1: f32 = 0.0;
    var acc2: f32 = 0.0; var acc3: f32 = 0.0;
    var m_prev: f32 = neg_inf;
    var l_prev: f32 = 0.0;

    for (var i = 0u; i < CHUNK; i = i + 1u) {
        let t = t_start + i;
        let valid = t < T_total;

        let k_base = select(0u, t * kv_stride + kv_off, valid);
        let k0 = select(0.0, K_cache[k_base + lane * HD_PER_THREAD], valid);
        let k1 = select(0.0, K_cache[k_base + lane * HD_PER_THREAD + 1u], valid);
        let k2 = select(0.0, K_cache[k_base + lane * HD_PER_THREAD + 2u], valid);
        let k3 = select(0.0, K_cache[k_base + lane * HD_PER_THREAD + 3u], valid);

        // Full 128-element dot product via 4-element partial + subgroupAdd
        let partial = q0 * k0 + q1 * k1 + q2 * k2 + q3 * k3;
        let dot = subgroupAdd(partial);
        let raw_score = dot * scale;
        let score = select(neg_inf, raw_score, valid);

        let m_new = max(m_prev, score);
        let exp_prev = exp(m_prev - m_new);
        let exp_score = exp(score - m_new);
        let l_new = l_prev * exp_prev + exp_score;
        let rescale = l_prev * exp_prev / max(l_new, 1e-10);
        let w = exp_score / max(l_new, 1e-10);

        let v_base = select(0u, t * kv_stride + kv_off, valid);
        let v0 = select(0.0, V_cache[v_base + lane * HD_PER_THREAD], valid);
        let v1 = select(0.0, V_cache[v_base + lane * HD_PER_THREAD + 1u], valid);
        let v2 = select(0.0, V_cache[v_base + lane * HD_PER_THREAD + 2u], valid);
        let v3 = select(0.0, V_cache[v_base + lane * HD_PER_THREAD + 3u], valid);

        acc0 = acc0 * rescale + v0 * w;
        acc1 = acc1 * rescale + v1 * w;
        acc2 = acc2 * rescale + v2 * w;
        acc3 = acc3 * rescale + v3 * w;

        m_prev = m_new;
        l_prev = l_new;
    }

    // Use max_chunks for the stride to prevent overlapping writes
    // across heads. Only write if this chunk is within n_chunks.
    let partial_stride = HD + 2u;
    let base = head * max_chunks * partial_stride + chunk_id * partial_stride;
    if (chunk_id < n_chunks) {
        if (lane == 0u) {
            Partials[base] = m_prev;
            Partials[base + 1u] = l_prev;
        }
        Partials[base + 2u + lane * HD_PER_THREAD] = acc0;
        Partials[base + 2u + lane * HD_PER_THREAD + 1u] = acc1;
        Partials[base + 2u + lane * HD_PER_THREAD + 2u] = acc2;
        Partials[base + 2u + lane * HD_PER_THREAD + 3u] = acc3;
    }
}
)WGSL";

// [shared] gqa_chunked_pass2 (3 bindings)
static const char* WGSL_GQA_CHUNKED_PASS2 = R"WGSL(
enable subgroups;

@group(0) @binding(0) var<storage, read_write> Partials: array<f32>;
@group(0) @binding(1) var<storage, read_write> Out: array<f32>;
@group(0) @binding(2) var<storage, read_write> _params_: array<u32>;

const HD: u32 = 128u;
const HD_PER_THREAD: u32 = 4u;

@compute @workgroup_size(32)
fn main(@builtin(local_invocation_id) lid: vec3<u32>,
        @builtin(workgroup_id) wid: vec3<u32>) {
    let head = wid.x;
    let lane = lid.x;

    let n_chunks = _params_[4];
    let neg_inf = bitcast<f32>(_params_[6]);
    let max_chunks = _params_[7];

    let partial_stride = HD + 2u;
    let head_base = head * max_chunks * partial_stride;

    var acc0: f32 = 0.0; var acc1: f32 = 0.0;
    var acc2: f32 = 0.0; var acc3: f32 = 0.0;
    var m_prev: f32 = neg_inf;
    var l_prev: f32 = 0.0;

    for (var c = 0u; c < n_chunks; c = c + 1u) {
        let base = head_base + c * partial_stride;
        let m_chunk = Partials[base];
        let l_chunk = Partials[base + 1u];
        let v0 = Partials[base + 2u + lane * HD_PER_THREAD];
        let v1 = Partials[base + 2u + lane * HD_PER_THREAD + 1u];
        let v2 = Partials[base + 2u + lane * HD_PER_THREAD + 2u];
        let v3 = Partials[base + 2u + lane * HD_PER_THREAD + 3u];

        if (l_chunk > 0.0) {
            let m_new = max(m_prev, m_chunk);
            let exp_prev = exp(m_prev - m_new);
            let exp_chunk = exp(m_chunk - m_new);
            let l_new = l_prev * exp_prev + l_chunk * exp_chunk;
            let rescale = l_prev * exp_prev / max(l_new, 1e-10);
            let w = l_chunk * exp_chunk / max(l_new, 1e-10);

            acc0 = acc0 * rescale + v0 * w;
            acc1 = acc1 * rescale + v1 * w;
            acc2 = acc2 * rescale + v2 * w;
            acc3 = acc3 * rescale + v3 * w;

            m_prev = m_new;
            l_prev = l_new;
        }
    }

    Out[head * HD + lane * HD_PER_THREAD] = acc0;
    Out[head * HD + lane * HD_PER_THREAD + 1u] = acc1;
    Out[head * HD + lane * HD_PER_THREAD + 2u] = acc2;
    Out[head * HD + lane * HD_PER_THREAD + 3u] = acc3;
}
)WGSL";

// [shared] rms_norm (5 bindings)
static const char* WGSL_RMS_NORM = R"WGSL(
enable subgroups;

// Auto-generated by Triton WebGPU Backend
// Kernel: rms_norm_loop_kernel
// Workgroup size: 128 (4 warps x 32 threads)

var<workgroup> _smem: array<i32, 4>;

@group(0) @binding(0) var<storage, read> buf0: array<f32>;  // X
@group(0) @binding(1) var<storage, read_write> buf1: array<f32>;  // Y
@group(0) @binding(2) var<storage, read> buf2: array<f32>;  // W
@group(0) @binding(3) var<storage, read_write> buf3: array<f32>;  // Rstd

struct Params {
    stride: i32,
    N: i32,
    eps: f32,
};
@group(0) @binding(4) var<storage, read> params: Params;

@compute @workgroup_size(128)
fn main(
    @builtin(workgroup_id) _wg_id: vec3<u32>,
    @builtin(local_invocation_id) _lid: vec3<u32>,
    @builtin(num_workgroups) _num_wg: vec3<u32>,
) {
    let v10: i32 = i32(_wg_id.x);
    let v11: i32 = params.N + 127;
    let v12: i32 = v11 / 128;
    var v14: i32 = 0;
    var v15_0: f32 = f32(0);
    loop {
        let v16: bool = v14 < v12;
        if !v16 { break; }
        let v18: i32 = v14 * 128;
        let v19: i32 = i32(_lid.x);
        let v20: i32 = v19 & 127;
        let v21: i32 = i32(u32(v20) % u32(32));
        let v22: i32 = i32(_lid.x);
        let v23: i32 = i32(u32(v22) / u32(32));
        let v24: i32 = v21 << u32(0);
        let v25: i32 = 0 | v24;
        let v26: i32 = v23 << u32(5);
        let v27: i32 = v25 | v26;
        let v28: i32 = v27 & 127;
        let v29: i32 = i32(u32(v28) >> u32(0));
        let v30: i32 = v29 | 0;
        let v31: i32 = 0 ^ v30;
        let v32: i32 = v31 ^ 0;
        let v33: i32 = v32 + 0;
        let v34: i32 = v18 + v33;
        let v35: bool = v34 < params.N;
        let v36: i32 = v10 * params.stride;
        let v39: f32 = buf0[u32((v36 + v34))];
        let v40: f32 = select(f32(0.0), v39, v35);
        let v41: f32 = v40 * v40;
        let v43: f32 = v15_0 + v41;
        let v45: i32 = v14 + 1;
        v14 = v45;
        v15_0 = v43;
    }
    let v48: i32 = bitcast<i32>(v15_0);
    let v49: i32 = subgroupShuffleXor(v48, u32(16));
    let v50: f32 = bitcast<f32>(v49);
    let v51: f32 = v15_0 + v50;
    let v52: i32 = bitcast<i32>(v51);
    let v53: i32 = subgroupShuffleXor(v52, u32(8));
    let v54: f32 = bitcast<f32>(v53);
    let v55: f32 = v51 + v54;
    let v56: i32 = bitcast<i32>(v55);
    let v57: i32 = subgroupShuffleXor(v56, u32(4));
    let v58: f32 = bitcast<f32>(v57);
    let v59: f32 = v55 + v58;
    let v60: i32 = bitcast<i32>(v59);
    let v61: i32 = subgroupShuffleXor(v60, u32(2));
    let v62: f32 = bitcast<f32>(v61);
    let v63: f32 = v59 + v62;
    let v64: i32 = bitcast<i32>(v63);
    let v65: i32 = subgroupShuffleXor(v64, u32(1));
    let v66: f32 = bitcast<f32>(v65);
    let v67: f32 = v63 + v66;
    let v68: i32 = i32(_lid.x);
    let v69: i32 = v68 & 127;
    let v70: i32 = i32(u32(v69) % u32(32));
    let v71: i32 = i32(_lid.x);
    let v72: i32 = i32(u32(v71) / u32(32));
    let v73: i32 = v70 << u32(0);
    let v74: i32 = 0 | v73;
    let v75: i32 = v72 << u32(5);
    let v76: i32 = v74 | v75;
    let v77: i32 = v76 & 96;
    let v78: i32 = i32(u32(v77) >> u32(3));
    let v79: i32 = 0 | v78;
    let v80: i32 = 0 ^ v79;
    let v81: i32 = v80 ^ 0;
    let v82: i32 = v81 ^ 0;
    let v83: i32 = v82 + 0;
    _smem[u32(v83) >> 2u] = bitcast<i32>(v67);
    workgroupBarrier();
    let v87: i32 = i32(_lid.x);
    let v88: i32 = v87 & 127;
    let v89: i32 = i32(u32(v88) % u32(32));
    let v90: i32 = i32(_lid.x);
    let v91: i32 = i32(u32(v90) / u32(32));
    let v92: i32 = v89 << u32(0);
    let v93: i32 = 0 | v92;
    let v94: i32 = v91 << u32(5);
    let v95: i32 = v93 | v94;
    let v96: i32 = v95 & 3;
    let v97: i32 = v96 << u32(2);
    let v98: i32 = v97 | 0;
    let v99: i32 = 0 ^ v98;
    let v100: i32 = v99 ^ 0;
    let v101: i32 = v100 ^ 0;
    let v102: i32 = v101 + 0;
    let v104: f32 = bitcast<f32>(_smem[u32(v102) >> 2u]);
    let v107: i32 = bitcast<i32>(v104);
    let v108: i32 = subgroupShuffleXor(v107, u32(2));
    let v109: f32 = bitcast<f32>(v108);
    let v110: f32 = v104 + v109;
    let v111: i32 = bitcast<i32>(v110);
    let v112: i32 = subgroupShuffleXor(v111, u32(1));
    let v113: f32 = bitcast<f32>(v112);
    let v114: f32 = v110 + v113;
    let v115: f32 = f32(params.N);
    let v116: f32 = v114 / v115;
    let v117: f32 = v116 + params.eps;
    let v118: f32 = sqrt(v117);
    let v119: f32 = f32(1.0) / v118;
    var v121: i32 = 0;
    loop {
        let v122: bool = v121 < v12;
        if !v122 { break; }
        let v124: i32 = v121 * 128;
        let v125: i32 = i32(_lid.x);
        let v126: i32 = v125 & 127;
        let v127: i32 = i32(u32(v126) % u32(32));
        let v128: i32 = i32(_lid.x);
        let v129: i32 = i32(u32(v128) / u32(32));
        let v130: i32 = v127 << u32(0);
        let v131: i32 = 0 | v130;
        let v132: i32 = v129 << u32(5);
        let v133: i32 = v131 | v132;
        let v134: i32 = v133 & 127;
        let v135: i32 = i32(u32(v134) >> u32(0));
        let v136: i32 = v135 | 0;
        let v137: i32 = 0 ^ v136;
        let v138: i32 = v137 ^ 0;
        let v139: i32 = v138 + 0;
        let v140: i32 = v124 + v139;
        let v141: bool = v140 < params.N;
        let v142: i32 = v10 * params.stride;
        let v145: f32 = buf0[u32((v142 + v140))];
        let v146: f32 = select(f32(0.0), v145, v141);
        let v148: f32 = buf2[u32(v140)];
        let v149: f32 = select(f32(1.0), v148, v141);
        let v152: f32 = v146 * v119;
        let v153: f32 = v152 * v149;
        let v154: f32 = select(f32(0), v153, v141);
        if v141 { buf1[u32((v142 + v140))] = v154; }
        let v155: i32 = v121 + 1;
        v121 = v155;
    }
    if _lid.x == 0u { buf3[u32(v10)] = v119; }
}
)WGSL";

// [shared] silu_mul_fused (3 bindings)
static const char* WGSL_SILU_MUL_FUSED = R"WGSL(
// Auto-generated by Triton WebGPU Backend
// Kernel: silu_mul_fused_kernel
// Workgroup size: 128 (4 warps x 32 threads)

@group(0) @binding(0) var<storage, read> buf0: array<f32>;  // GateUp
@group(0) @binding(1) var<storage, read_write> buf1: array<f32>;  // Out

struct Params {
    N: i32,
};
@group(0) @binding(2) var<storage, read> params: Params;

@compute @workgroup_size(128)
fn main(
    @builtin(workgroup_id) _wg_id: vec3<u32>,
    @builtin(local_invocation_id) _lid: vec3<u32>,
    @builtin(num_workgroups) _num_wg: vec3<u32>,
) {
    let v6: i32 = i32(_wg_id.x);
    let v7: i32 = v6 * 128;
    let v8: i32 = i32(_lid.x);
    let v9: i32 = v8 & 127;
    let v10: i32 = i32(u32(v9) % u32(32));
    let v11: i32 = i32(_lid.x);
    let v12: i32 = i32(u32(v11) / u32(32));
    let v13: i32 = v10 << u32(0);
    let v14: i32 = 0 | v13;
    let v15: i32 = v12 << u32(5);
    let v16: i32 = v14 | v15;
    let v17: i32 = v16 & 127;
    let v18: i32 = i32(u32(v17) >> u32(0));
    let v19: i32 = v18 | 0;
    let v20: i32 = 0 ^ v19;
    let v21: i32 = v20 ^ 0;
    let v22: i32 = v21 + 0;
    let v23: i32 = v7 + v22;
    let v24: bool = v23 < params.N;
    let v26: f32 = buf0[u32(v23)];
    let v27: f32 = select(f32(0.0), v26, v24);
    let v30: f32 = buf0[u32((params.N + v23))];
    let v31: f32 = select(f32(0.0), v30, v24);
    let v32: f32 = f32(0.0) - v27;
    let v33: f32 = exp(v32);
    let v34: f32 = v33 + f32(1.0);
    let v35: f32 = v27 / v34;
    let v37: f32 = v35 * v31;
    let v38: f32 = select(f32(0), v37, v24);
    if v24 { buf1[u32(v23)] = v38; }
}
)WGSL";


inline const std::unordered_map<std::string, ShaderInfo>& getEmbeddedKernels() {
    static const std::unordered_map<std::string, ShaderInfo> kernels = {
        {"q8_matmul", {WGSL_Q8_MATMUL, 6}},
        {"q8_matmul_add", {WGSL_Q8_MATMUL_ADD, 6}},
        {"q8_matmul_add_fast", {WGSL_Q8_MATMUL_ADD_FAST, 6}},
        {"q8_matmul_fast", {WGSL_Q8_MATMUL_FAST, 6}},
        {"add_rms_norm", {WGSL_ADD_RMS_NORM, 6}},
        {"argmax", {WGSL_ARGMAX, 3}},
        {"embed_gather", {WGSL_EMBED_GATHER, 4}},
        {"fp16_gemm", {WGSL_FP16_GEMM, 5}},
        {"fp16_gemm_wide", {WGSL_FP16_GEMM_WIDE, 5}},
        {"fused_qknorm_rope", {WGSL_FUSED_QKNORM_ROPE, 9}},
        {"gqa_chunked_pass1", {WGSL_GQA_CHUNKED_PASS1, 5}},
        {"gqa_chunked_pass2", {WGSL_GQA_CHUNKED_PASS2, 3}},
        {"rms_norm", {WGSL_RMS_NORM, 5}},
        {"silu_mul_fused", {WGSL_SILU_MUL_FUSED, 3}},
    };
    return kernels;
}
