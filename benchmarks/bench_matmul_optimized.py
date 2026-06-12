"""Benchmark for optimized matmul — 128x128 tiles, 8x8 register tiling, vectorized f16 loads."""
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

SHADER_CODE = """
const TILE_SIZE: u32 = 16u;
const TM: u32 = 8u;
const TN: u32 = 8u;
const BM: u32 = 128u;
const BN: u32 = 128u;
const BK: u32 = 16u;

struct Params {
    M: u32,
    N: u32,
    K: u32,
}

@group(0) @binding(0) var<storage, read> A: array<u32>;
@group(0) @binding(1) var<storage, read> B: array<u32>;
@group(0) @binding(2) var<storage, read_write> C: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;

var<workgroup> tileA: array<f32, 2048>;  // BM * BK = 128 * 16
var<workgroup> tileB: array<f32, 2048>;  // BK * BN = 16 * 128

fn load_f16_A(index: u32) -> f32 {
    let word = A[index / 2u];
    let bits = (word >> ((index % 2u) * 16u)) & 0xFFFFu;
    return unpack2x16float(bits).x;
}

fn load_f16_B(index: u32) -> f32 {
    let word = B[index / 2u];
    let bits = (word >> ((index % 2u) * 16u)) & 0xFFFFu;
    return unpack2x16float(bits).x;
}

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

    var results: array<f32, 64>;
    for (var i = 0u; i < 64u; i++) {
        results[i] = 0.0;
    }

    let numTiles = (params.K + BK - 1u) / BK;

    for (var t: u32 = 0u; t < numTiles; t = t + 1u) {
        let tile_k = t * BK;

        // Load tileA: 128 rows x 16 cols = 2048 elements, 256 threads -> 8 per thread
        for (var load = 0u; load < 8u; load++) {
            let flat = load * 256u + thread_id;
            let a_r = flat / BK;
            let a_c = flat % BK;
            let global_a_row = block_row + a_r;
            let global_a_col = tile_k + a_c;
            if global_a_row < params.M && global_a_col < params.K {
                tileA[a_r * BK + a_c] = load_f16_A(global_a_row * params.K + global_a_col);
            } else {
                tileA[a_r * BK + a_c] = 0.0;
            }
        }

        // Load tileB: 16 rows x 128 cols = 2048 elements, 256 threads -> 8 per thread
        for (var load = 0u; load < 8u; load++) {
            let flat = load * 256u + thread_id;
            let b_r = flat / BN;
            let b_c = flat % BN;
            let global_b_row = tile_k + b_r;
            let global_b_col = block_col + b_c;
            if global_b_row < params.K && global_b_col < params.N {
                tileB[b_r * BN + b_c] = load_f16_B(global_b_row * params.N + global_b_col);
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


def pack_f16_to_u32(arr: np.ndarray) -> bytes:
    flat = arr.flatten().astype(np.float16)
    raw = flat.view(np.uint16).astype(np.uint32)
    if len(raw) % 2:
        raw = np.append(raw, np.uint32(0))
    packed = ((raw[1::2] << 16) | raw[0::2]).astype(np.uint32)
    return packed.tobytes()


def make_params(M: int, N: int, K: int) -> bytes:
    return struct.pack('III', M, N, K)


BLOCK_M = 128
BLOCK_N = 128


def setup_pipeline(device, shader_code, M, N, K, A_bytes, B_bytes):
    params_bytes = make_params(M, N, K)
    out_size = M * N * 4

    buf_a = device.create_buffer_with_data(data=A_bytes, usage=wgpu.BufferUsage.STORAGE)
    buf_b = device.create_buffer_with_data(data=B_bytes, usage=wgpu.BufferUsage.STORAGE)
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
    t0 = time.perf_counter()
    encoder = device.create_command_encoder()
    pass_enc = encoder.begin_compute_pass()
    pass_enc.set_pipeline(ctx["pipeline"])
    pass_enc.set_bind_group(0, ctx["bind_group"])
    pass_enc.dispatch_workgroups(ctx["wg_x"], ctx["wg_y"])
    pass_enc.end()
    device.queue.submit([encoder.finish()])
    _data = device.queue.read_buffer(ctx["buf_c"])
    t1 = time.perf_counter()
    return t1 - t0


def dispatch_batched(device, ctx, n_dispatches):
    encoder = device.create_command_encoder()
    for _ in range(n_dispatches):
        pass_enc = encoder.begin_compute_pass()
        pass_enc.set_pipeline(ctx["pipeline"])
        pass_enc.set_bind_group(0, ctx["bind_group"])
        pass_enc.dispatch_workgroups(ctx["wg_x"], ctx["wg_y"])
        pass_enc.end()

    t0 = time.perf_counter()
    device.queue.submit([encoder.finish()])
    _data = device.queue.read_buffer(ctx["buf_c"])
    t1 = time.perf_counter()
    return (t1 - t0) / n_dispatches


def verify_correctness(device, shader_code, M, N, K):
    rng = np.random.default_rng(42)
    A = rng.uniform(-1, 1, (M, K)).astype(np.float16)
    B = rng.uniform(-1, 1, (K, N)).astype(np.float16)
    A_bytes = pack_f16_to_u32(A)
    B_bytes = pack_f16_to_u32(B)

    ctx = setup_pipeline(device, shader_code, M, N, K, A_bytes, B_bytes)
    dispatch_and_sync(device, ctx)

    raw = device.queue.read_buffer(ctx["buf_c"])
    gpu_result = np.frombuffer(raw, dtype=np.float32).reshape(M, N)
    expected = A.astype(np.float32) @ B.astype(np.float32)

    for buf in ctx["bufs"]:
        buf.destroy()

    max_err = np.max(np.abs(gpu_result - expected))
    rel_err = max_err / (np.max(np.abs(expected)) + 1e-8)
    return max_err, rel_err


def benchmark_size(device, shader_code, M, N, K):
    rng = np.random.default_rng(42)
    A = rng.uniform(-1, 1, (M, K)).astype(np.float16)
    B = rng.uniform(-1, 1, (K, N)).astype(np.float16)
    A_bytes = pack_f16_to_u32(A)
    B_bytes = pack_f16_to_u32(B)

    ctx = setup_pipeline(device, shader_code, M, N, K, A_bytes, B_bytes)

    for _ in range(WARMUP_RUNS):
        dispatch_and_sync(device, ctx)

    n_batch = 50
    times = []
    for _ in range(BENCH_RUNS):
        times.append(dispatch_batched(device, ctx, n_batch))

    for buf in ctx["bufs"]:
        buf.destroy()

    return times


def get_theoretical_peak_gflops(info):
    desc = (info.get('device', '') + ' ' + info.get('description', '')).lower()
    peaks = {
        'rtx 5090': 104800, 'rtx 5080': 69200, 'rtx 5070 ti': 44100, 'rtx 5070': 31300,
        'rtx 4090': 82600, 'rtx 4080': 48700, 'rtx 4070': 29100,
        'rtx 3090': 35600, 'rtx 3080': 29800, 'rtx 3070': 20300,
        'rtx 3060': 12700, 'rtx 2080': 14200, 'rtx 2070': 9100,
        'rx 7900 xtx': 61400, 'rx 7900 xt': 51600,
        'rx 6900 xt': 23040, 'rx 6800 xt': 20740,
        'apple m1': 2600, 'apple m2': 3600, 'apple m3': 4100,
    }
    for name, gf in peaks.items():
        if name in desc:
            return float(gf)
    return 0.0


def main():
    adapter = wgpu.gpu.request_adapter_sync(power_preference="high-performance")
    if not adapter:
        print("ERROR: No WebGPU adapter found", file=sys.stderr)
        sys.exit(1)

    info = adapter.info
    device = adapter.request_device_sync()

    gpu_name = info.get('device', '') or info.get('description', '') or 'Unknown GPU'
    peak_gflops = get_theoretical_peak_gflops(info)

    print("=" * 80)
    print(f"Optimized Matmul Benchmark (128x128 tiles, 8x8 register tiling)")
    print(f"GPU: {gpu_name}")
    if peak_gflops > 0:
        print(f"Theoretical peak (f32): {peak_gflops:.0f} GFLOPS")
    print("=" * 80)

    print("\nCorrectness check...")
    for sz in [128, 256, 512]:
        max_err, rel_err = verify_correctness(device, SHADER_CODE, sz, sz, sz)
        status = "PASS" if rel_err < 0.01 else "FAIL"
        print(f"  {sz}x{sz}: max_err={max_err:.6f} rel_err={rel_err:.6f} [{status}]")

    print()
    print(f"{'Size':>15s}  {'FLOPS':>12s}  {'Min (ms)':>10s}  {'Med (ms)':>10s}  {'Max (ms)':>10s}  {'GFLOPS':>10s}  {'% Peak':>8s}")
    print("-" * 80)

    sizes = [
        (1024, 1024, 1024),
        (2048, 2048, 2048),
        (4096, 4096, 4096),
    ]

    for M, N, K in sizes:
        flops = 2 * M * N * K
        times = benchmark_size(device, SHADER_CODE, M, N, K)
        times_ms = [t * 1000 for t in times]
        median_s = np.median(times)
        gflops = flops / median_s / 1e9

        pct = f"{gflops / peak_gflops * 100:.1f}%" if peak_gflops > 0 else "N/A"
        print(f"{M}x{N}x{K:>5d}  {flops:>12.2e}  {min(times_ms):>10.2f}  {np.median(times_ms):>10.2f}  {max(times_ms):>10.2f}  {gflops:>10.1f}  {pct:>8s}")

    print()


if __name__ == "__main__":
    main()
