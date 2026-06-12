"""Benchmark optimized matmul variants to find >50% peak FLOPS."""
import sys
import time
import struct
import numpy as np

try:
    import wgpu
except ImportError:
    print("ERROR: wgpu not installed. pip install wgpu", file=sys.stderr)
    sys.exit(1)

WARMUP_RUNS = 3
BENCH_RUNS = 5

# Variant 1: f32 input to isolate f16 unpack overhead
SHADER_F32 = """
const TILE_SIZE: u32 = 16u;
const TM: u32 = 4u;
const TN: u32 = 4u;
const BM: u32 = 64u;
const BN: u32 = 64u;
const BK: u32 = 16u;

struct Params { M: u32, N: u32, K: u32, }

@group(0) @binding(0) var<storage, read> A: array<f32>;
@group(0) @binding(1) var<storage, read> B: array<f32>;
@group(0) @binding(2) var<storage, read_write> C: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;

var<workgroup> tileA: array<f32, 1024>;
var<workgroup> tileB: array<f32, 1024>;

@compute @workgroup_size(16, 16)
fn main(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wgid: vec3<u32>,
) {
    let lr = lid.x;
    let lc = lid.y;
    let thread_id = lr * TILE_SIZE + lc;

    let block_row = wgid.x * BM;
    let block_col = wgid.y * BN;

    var results: array<f32, 16>;
    for (var i = 0u; i < 16u; i++) { results[i] = 0.0; }

    let numTiles = (params.K + BK - 1u) / BK;

    for (var t: u32 = 0u; t < numTiles; t = t + 1u) {
        let tile_k = t * BK;

        for (var load = 0u; load < 4u; load++) {
            let flat = load * 256u + thread_id;
            let a_r = flat / BK;
            let a_c = flat % BK;
            let gr = block_row + a_r;
            let gc = tile_k + a_c;
            if gr < params.M && gc < params.K {
                tileA[a_r * BK + a_c] = A[gr * params.K + gc];
            } else {
                tileA[a_r * BK + a_c] = 0.0;
            }
        }

        for (var load = 0u; load < 4u; load++) {
            let flat = load * 256u + thread_id;
            let b_r = flat / BN;
            let b_c = flat % BN;
            let gr = tile_k + b_r;
            let gc = block_col + b_c;
            if gr < params.K && gc < params.N {
                tileB[b_r * BN + b_c] = B[gr * params.N + gc];
            } else {
                tileB[b_r * BN + b_c] = 0.0;
            }
        }

        workgroupBarrier();

        for (var k: u32 = 0u; k < BK; k = k + 1u) {
            for (var rm = 0u; rm < TM; rm++) {
                let a_val = tileA[(lr * TM + rm) * BK + k];
                for (var rn = 0u; rn < TN; rn++) {
                    results[rm * TN + rn] += a_val * tileB[k * BN + lc * TN + rn];
                }
            }
        }

        workgroupBarrier();
    }

    for (var rm = 0u; rm < TM; rm++) {
        for (var rn = 0u; rn < TN; rn++) {
            let row = block_row + lr * TM + rm;
            let col = block_col + lc * TN + rn;
            if row < params.M && col < params.N {
                C[row * params.N + col] = results[rm * TN + rn];
            }
        }
    }
}
"""

# Variant 2: vec4 loads from shared memory
SHADER_VEC4 = """
const TILE_SIZE: u32 = 16u;
const TM: u32 = 4u;
const TN: u32 = 4u;
const BM: u32 = 64u;
const BN: u32 = 64u;
const BK: u32 = 16u;

struct Params { M: u32, N: u32, K: u32, }

@group(0) @binding(0) var<storage, read> A: array<f32>;
@group(0) @binding(1) var<storage, read> B: array<f32>;
@group(0) @binding(2) var<storage, read_write> C: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;

var<workgroup> tileA: array<f32, 1024>;
var<workgroup> tileB: array<vec4<f32>, 256>;  // BK * (BN/4)

@compute @workgroup_size(16, 16)
fn main(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wgid: vec3<u32>,
) {
    let lr = lid.x;
    let lc = lid.y;
    let thread_id = lr * TILE_SIZE + lc;

    let block_row = wgid.x * BM;
    let block_col = wgid.y * BN;

    var results: array<f32, 16>;
    for (var i = 0u; i < 16u; i++) { results[i] = 0.0; }

    let numTiles = (params.K + BK - 1u) / BK;

    for (var t: u32 = 0u; t < numTiles; t = t + 1u) {
        let tile_k = t * BK;

        for (var load = 0u; load < 4u; load++) {
            let flat = load * 256u + thread_id;
            let a_r = flat / BK;
            let a_c = flat % BK;
            let gr = block_row + a_r;
            let gc = tile_k + a_c;
            if gr < params.M && gc < params.K {
                tileA[a_r * BK + a_c] = A[gr * params.K + gc];
            } else {
                tileA[a_r * BK + a_c] = 0.0;
            }
        }

        // Load B as vec4 — 16 rows × 16 vec4s = 256
        for (var load = 0u; load < 1u; load++) {
            let flat = thread_id;
            let b_r = flat / 16u;  // BN/4 = 16 vec4s per row
            let b_c4 = flat % 16u;
            let gr = tile_k + b_r;
            let base_gc = block_col + b_c4 * 4u;
            if gr < params.K && base_gc + 3u < params.N {
                let idx = gr * params.N;
                tileB[b_r * 16u + b_c4] = vec4<f32>(
                    B[idx + base_gc],
                    B[idx + base_gc + 1u],
                    B[idx + base_gc + 2u],
                    B[idx + base_gc + 3u]
                );
            } else {
                tileB[b_r * 16u + b_c4] = vec4<f32>(0.0);
            }
        }

        workgroupBarrier();

        for (var k: u32 = 0u; k < BK; k = k + 1u) {
            for (var rm = 0u; rm < TM; rm++) {
                let a_val = tileA[(lr * TM + rm) * BK + k];
                let b_vec = tileB[k * 16u + lc];
                results[rm * TN + 0u] += a_val * b_vec.x;
                results[rm * TN + 1u] += a_val * b_vec.y;
                results[rm * TN + 2u] += a_val * b_vec.z;
                results[rm * TN + 3u] += a_val * b_vec.w;
            }
        }

        workgroupBarrier();
    }

    for (var rm = 0u; rm < TM; rm++) {
        let row = block_row + lr * TM + rm;
        if row < params.M {
            let base_col = block_col + lc * TN;
            for (var rn = 0u; rn < TN; rn++) {
                let col = base_col + rn;
                if col < params.N {
                    C[row * params.N + col] = results[rm * TN + rn];
                }
            }
        }
    }
}
"""

# Variant 3: larger BK (32) to do more compute per barrier
SHADER_BK32 = """
const TILE_SIZE: u32 = 16u;
const TM: u32 = 4u;
const TN: u32 = 4u;
const BM: u32 = 64u;
const BN: u32 = 64u;
const BK: u32 = 32u;

struct Params { M: u32, N: u32, K: u32, }

@group(0) @binding(0) var<storage, read> A: array<f32>;
@group(0) @binding(1) var<storage, read> B: array<f32>;
@group(0) @binding(2) var<storage, read_write> C: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;

var<workgroup> tileA: array<f32, 2048>;  // BM * BK = 64 * 32
var<workgroup> tileB: array<f32, 2048>;  // BK * BN = 32 * 64

@compute @workgroup_size(16, 16)
fn main(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wgid: vec3<u32>,
) {
    let lr = lid.x;
    let lc = lid.y;
    let thread_id = lr * TILE_SIZE + lc;

    let block_row = wgid.x * BM;
    let block_col = wgid.y * BN;

    var results: array<f32, 16>;
    for (var i = 0u; i < 16u; i++) { results[i] = 0.0; }

    let numTiles = (params.K + BK - 1u) / BK;

    for (var t: u32 = 0u; t < numTiles; t = t + 1u) {
        let tile_k = t * BK;

        for (var load = 0u; load < 8u; load++) {
            let flat = load * 256u + thread_id;
            let a_r = flat / BK;
            let a_c = flat % BK;
            let gr = block_row + a_r;
            let gc = tile_k + a_c;
            if gr < params.M && gc < params.K {
                tileA[a_r * BK + a_c] = A[gr * params.K + gc];
            } else {
                tileA[a_r * BK + a_c] = 0.0;
            }
        }

        for (var load = 0u; load < 8u; load++) {
            let flat = load * 256u + thread_id;
            let b_r = flat / BN;
            let b_c = flat % BN;
            let gr = tile_k + b_r;
            let gc = block_col + b_c;
            if gr < params.K && gc < params.N {
                tileB[b_r * BN + b_c] = B[gr * params.N + gc];
            } else {
                tileB[b_r * BN + b_c] = 0.0;
            }
        }

        workgroupBarrier();

        for (var k: u32 = 0u; k < BK; k = k + 1u) {
            for (var rm = 0u; rm < TM; rm++) {
                let a_val = tileA[(lr * TM + rm) * BK + k];
                for (var rn = 0u; rn < TN; rn++) {
                    results[rm * TN + rn] += a_val * tileB[k * BN + lc * TN + rn];
                }
            }
        }

        workgroupBarrier();
    }

    for (var rm = 0u; rm < TM; rm++) {
        for (var rn = 0u; rn < TN; rn++) {
            let row = block_row + lr * TM + rm;
            let col = block_col + lc * TN + rn;
            if row < params.M && col < params.N {
                C[row * params.N + col] = results[rm * TN + rn];
            }
        }
    }
}
"""


def make_params(M, N, K):
    return struct.pack('III', M, N, K)


BLOCK_M = 64
BLOCK_N = 64


def setup_pipeline_f32(device, shader_code, M, N, K, A_f32, B_f32):
    params_bytes = make_params(M, N, K)
    out_size = M * N * 4

    buf_a = device.create_buffer_with_data(data=A_f32.tobytes(), usage=wgpu.BufferUsage.STORAGE)
    buf_b = device.create_buffer_with_data(data=B_f32.tobytes(), usage=wgpu.BufferUsage.STORAGE)
    buf_c = device.create_buffer(size=out_size, usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC)
    buf_params = device.create_buffer_with_data(data=params_bytes, usage=wgpu.BufferUsage.UNIFORM)

    shader_module = device.create_shader_module(code=shader_code)

    bind_group_layout = device.create_bind_group_layout(entries=[
        {"binding": 0, "visibility": wgpu.ShaderStage.COMPUTE, "buffer": {"type": wgpu.BufferBindingType.read_only_storage}},
        {"binding": 1, "visibility": wgpu.ShaderStage.COMPUTE, "buffer": {"type": wgpu.BufferBindingType.read_only_storage}},
        {"binding": 2, "visibility": wgpu.ShaderStage.COMPUTE, "buffer": {"type": wgpu.BufferBindingType.storage}},
        {"binding": 3, "visibility": wgpu.ShaderStage.COMPUTE, "buffer": {"type": wgpu.BufferBindingType.uniform}},
    ])

    pipeline_layout = device.create_pipeline_layout(bind_group_layouts=[bind_group_layout])
    pipeline = device.create_compute_pipeline(layout=pipeline_layout, compute={"module": shader_module, "entry_point": "main"})

    bind_group = device.create_bind_group(layout=bind_group_layout, entries=[
        {"binding": 0, "resource": {"buffer": buf_a, "offset": 0, "size": buf_a.size}},
        {"binding": 1, "resource": {"buffer": buf_b, "offset": 0, "size": buf_b.size}},
        {"binding": 2, "resource": {"buffer": buf_c, "offset": 0, "size": buf_c.size}},
        {"binding": 3, "resource": {"buffer": buf_params, "offset": 0, "size": buf_params.size}},
    ])

    wg_x = (M + BLOCK_M - 1) // BLOCK_M
    wg_y = (N + BLOCK_N - 1) // BLOCK_N

    return {
        "pipeline": pipeline, "bind_group": bind_group,
        "buf_c": buf_c, "out_size": out_size,
        "wg_x": wg_x, "wg_y": wg_y,
        "bufs": [buf_a, buf_b, buf_c, buf_params],
    }


def dispatch_and_sync(device, ctx):
    encoder = device.create_command_encoder()
    pass_enc = encoder.begin_compute_pass()
    pass_enc.set_pipeline(ctx["pipeline"])
    pass_enc.set_bind_group(0, ctx["bind_group"])
    pass_enc.dispatch_workgroups(ctx["wg_x"], ctx["wg_y"])
    pass_enc.end()
    device.queue.submit([encoder.finish()])
    _data = device.queue.read_buffer(ctx["buf_c"])


def dispatch_batched(device, ctx, n):
    encoder = device.create_command_encoder()
    for _ in range(n):
        pass_enc = encoder.begin_compute_pass()
        pass_enc.set_pipeline(ctx["pipeline"])
        pass_enc.set_bind_group(0, ctx["bind_group"])
        pass_enc.dispatch_workgroups(ctx["wg_x"], ctx["wg_y"])
        pass_enc.end()
    t0 = time.perf_counter()
    device.queue.submit([encoder.finish()])
    _data = device.queue.read_buffer(ctx["buf_c"])
    t1 = time.perf_counter()
    return (t1 - t0) / n


def bench_variant(device, name, shader_code, M, N, K, peak_gflops):
    rng = np.random.default_rng(42)
    A = rng.uniform(-1, 1, (M, K)).astype(np.float32)
    B = rng.uniform(-1, 1, (K, N)).astype(np.float32)

    ctx = setup_pipeline_f32(device, shader_code, M, N, K, A, B)

    for _ in range(WARMUP_RUNS):
        dispatch_and_sync(device, ctx)

    times = [dispatch_batched(device, ctx, 50) for _ in range(BENCH_RUNS)]

    for buf in ctx["bufs"]:
        buf.destroy()

    flops = 2 * M * N * K
    median_s = np.median(times)
    gflops = flops / median_s / 1e9
    pct = gflops / peak_gflops * 100 if peak_gflops > 0 else 0

    print(f"  {name:25s}  {np.median(times)*1000:8.2f} ms  {gflops:8.1f} GFLOPS  {pct:5.1f}%")
    return gflops


def main():
    adapter = wgpu.gpu.request_adapter_sync(power_preference="high-performance")
    info = adapter.info
    device = adapter.request_device_sync()

    gpu_name = info.get('device', '') or 'Unknown'
    desc = (info.get('device', '') + ' ' + info.get('description', '')).lower()
    peaks = {
        'rtx 5090': 104800, 'rtx 5080': 69200, 'rtx 5070 ti': 44100, 'rtx 5070': 31300,
        'rtx 4090': 82600, 'rtx 4080': 48700, 'rtx 4070': 29100,
        'rtx 3090': 35600, 'rtx 3080': 29800, 'rtx 3070': 20300,
        'rtx 3060': 12700,
    }
    peak_gflops = 0.0
    for name, gf in peaks.items():
        if name in desc:
            peak_gflops = float(gf)
            break

    print(f"GPU: {gpu_name}, Peak: {peak_gflops:.0f} GFLOPS")
    print()

    for M in [2048, 4096]:
        print(f"--- {M}x{M}x{M} ---")
        bench_variant(device, "baseline (f32, BK=16)", SHADER_F32, M, M, M, peak_gflops)
        bench_variant(device, "BK=32", SHADER_BK32, M, M, M, peak_gflops)
        bench_variant(device, "vec4 shared B", SHADER_VEC4, M, M, M, peak_gflops)
        print()


if __name__ == "__main__":
    main()
