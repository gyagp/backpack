"""Benchmark for matmul_tiled_f16.wgsl — measures achieved GFLOPS on real GPU hardware."""
import os
import sys
import struct
import re
import numpy as np

try:
    import wgpu
except ImportError:
    print("ERROR: wgpu not installed. pip install wgpu", file=sys.stderr)
    sys.exit(1)

BLOCK_M = 128
BLOCK_N = 128
WARMUP_RUNS = 5
BENCH_RUNS = 10

SHADER_PATH = os.path.join(os.path.dirname(__file__), '..', 'src', 'shaders', 'matmul_tiled_f16.wgsl')


def load_shader() -> str:
    """Load the actual shader and strip batch-related code for non-batched benchmark."""
    with open(SHADER_PATH) as f:
        code = f.read()

    code = re.sub(r'^\s*batch_size:\s*u32,?\s*\n', '', code, flags=re.MULTILINE)
    code = re.sub(r'^\s*stride_A:\s*u32,?\s*\n', '', code, flags=re.MULTILINE)
    code = re.sub(r'^\s*stride_C:\s*u32,?\s*\n', '', code, flags=re.MULTILINE)

    code = re.sub(r'^\s*let batch\b.*\n', '', code, flags=re.MULTILINE)
    code = re.sub(r'^\s*if batch\b.*\n\s*return;\s*\n\s*\}\s*\n', '', code, flags=re.MULTILINE)

    code = re.sub(r'^\s*let a_offset\b.*\n', '', code, flags=re.MULTILINE)
    code = re.sub(r'^\s*let c_offset\b.*\n', '', code, flags=re.MULTILINE)

    code = code.replace('a_offset + ', '')
    code = code.replace('c_offset + ', '')

    return code


def pack_f16_to_u32(arr: np.ndarray) -> bytes:
    flat = arr.flatten().astype(np.float16)
    raw = flat.view(np.uint16).astype(np.uint32)
    if len(raw) % 2:
        raw = np.append(raw, np.uint32(0))
    packed = ((raw[1::2] << 16) | raw[0::2]).astype(np.uint32)
    return packed.tobytes()


def make_params(M: int, N: int, K: int) -> bytes:
    return struct.pack('III', M, N, K)


def setup_pipeline(device, shader_code: str, M: int, N: int, K: int,
                   A_bytes: bytes, B_bytes: bytes):
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


def dispatch_with_timestamps(device, ctx: dict, query_set, resolve_buf) -> float:
    encoder = device.create_command_encoder()
    pass_enc = encoder.begin_compute_pass(timestamp_writes={
        'query_set': query_set,
        'beginning_of_pass_write_index': 0,
        'end_of_pass_write_index': 1,
    })
    pass_enc.set_pipeline(ctx["pipeline"])
    pass_enc.set_bind_group(0, ctx["bind_group"])
    pass_enc.dispatch_workgroups(ctx["wg_x"], ctx["wg_y"])
    pass_enc.end()
    encoder.resolve_query_set(query_set, 0, 2, resolve_buf, 0)
    device.queue.submit([encoder.finish()])

    data = device.queue.read_buffer(resolve_buf)
    t0, t1 = struct.unpack('QQ', data)
    return (t1 - t0) / 1e9  # ns -> seconds


def benchmark_size(device, shader_code: str, M: int, N: int, K: int, query_set, resolve_buf):
    rng = np.random.default_rng(42)
    A = rng.uniform(-1, 1, (M, K)).astype(np.float16)
    B = rng.uniform(-1, 1, (K, N)).astype(np.float16)
    A_bytes = pack_f16_to_u32(A)
    B_bytes = pack_f16_to_u32(B)

    ctx = setup_pipeline(device, shader_code, M, N, K, A_bytes, B_bytes)

    for _ in range(WARMUP_RUNS):
        dispatch_with_timestamps(device, ctx, query_set, resolve_buf)

    times = []
    for _ in range(BENCH_RUNS):
        times.append(dispatch_with_timestamps(device, ctx, query_set, resolve_buf))

    for buf in ctx["bufs"]:
        buf.destroy()

    return times


def compute_roofline_gflops(M, N, K, mem_bw_gbs):
    BM, BN = 128, 128
    mem_bytes = M * K * 2 * (N / BN) + K * N * 2 * (M / BM)
    flops = 2 * M * N * K
    ai = flops / mem_bytes
    return mem_bw_gbs * ai


def get_gpu_specs(info) -> dict:
    desc = (info.get('device', '') + ' ' + info.get('description', '')).lower()

    specs = {
        'rtx 5090': {'peak': 104800, 'bw': 1792},
        'rtx 5080': {'peak': 69200, 'bw': 960},
        'rtx 5070 ti': {'peak': 44100, 'bw': 896},
        'rtx 5070': {'peak': 31300, 'bw': 672},
        'rtx 4090': {'peak': 82600, 'bw': 1008},
        'rtx 4080': {'peak': 48700, 'bw': 717},
        'rtx 4070': {'peak': 29100, 'bw': 504},
        'rtx 3090': {'peak': 35600, 'bw': 936},
        'rtx 3080': {'peak': 29800, 'bw': 760},
        'rtx 3070': {'peak': 20300, 'bw': 448},
        'rtx 3060': {'peak': 12700, 'bw': 360},
        'apple m1': {'peak': 2600, 'bw': 68},
        'apple m2': {'peak': 3600, 'bw': 100},
        'apple m3': {'peak': 4100, 'bw': 150},
    }
    for name, s in specs.items():
        if name in desc:
            return s
    return {'peak': 0, 'bw': 0}


def main():
    adapter = wgpu.gpu.request_adapter_sync(power_preference="high-performance")
    if not adapter:
        print("ERROR: No WebGPU adapter found", file=sys.stderr)
        sys.exit(1)

    info = adapter.info
    device = adapter.request_device_sync(required_features=['timestamp-query'])

    query_set = device.create_query_set(type='timestamp', count=2)
    resolve_buf = device.create_buffer(size=16, usage=wgpu.BufferUsage.QUERY_RESOLVE | wgpu.BufferUsage.COPY_SRC)

    shader_code = load_shader()

    gpu_name = info.get('device', '') or info.get('description', '') or 'Unknown GPU'
    specs = get_gpu_specs(info)
    peak_gflops = specs['peak']
    mem_bw = specs['bw']

    print("=" * 90)
    print(f"Matmul Tiled F16 Benchmark (GPU timestamp queries)")
    print(f"GPU: {gpu_name}")
    print(f"Block size: {BLOCK_M}x{BLOCK_N}, register tiling 8x8 per thread")
    print(f"Warmup runs: {WARMUP_RUNS}, Bench runs: {BENCH_RUNS}")
    if peak_gflops > 0:
        print(f"Theoretical FP32 peak: {peak_gflops:.0f} GFLOPS")
    if mem_bw > 0:
        print(f"Memory bandwidth: {mem_bw} GB/s")
    print("=" * 90)
    print()
    print(f"{'Size':>15s}  {'Med (ms)':>10s}  {'GFLOPS':>10s}  {'% Peak':>8s}  {'Roofline':>10s}  {'% Roof':>8s}")
    print("-" * 75)

    sizes = [
        (1024, 1024, 1024),
        (2048, 2048, 2048),
        (4096, 4096, 4096),
    ]

    for M, N, K in sizes:
        flops = 2 * M * N * K
        times = benchmark_size(device, shader_code, M, N, K, query_set, resolve_buf)
        times_ms = [t * 1000 for t in times]
        median_s = np.median(times)
        gflops = flops / median_s / 1e9

        pct_peak = f"{gflops / peak_gflops * 100:.1f}%" if peak_gflops > 0 else "N/A"
        roof = compute_roofline_gflops(M, N, K, mem_bw) if mem_bw > 0 else 0
        pct_roof = f"{gflops / roof * 100:.1f}%" if roof > 0 else "N/A"
        roof_str = f"{roof:.0f}" if roof > 0 else "N/A"

        print(f"{M}x{N}x{K:>5d}  {np.median(times_ms):>10.2f}  {gflops:>10.1f}  {pct_peak:>8s}  {roof_str:>10s}  {pct_roof:>8s}")

    print()
    print("Optimizations applied:")
    print("  - Transposed tileA storage [BK][BM_PAD] to eliminate shared memory bank conflicts")
    print("  - 8x8 register tiling (64 accumulators per thread)")
    print("  - Fully unrolled inner loop with scalar register accumulators")
    print("  - Vectorized f16 pair loading via unpack2x16float")
    print("  - BM_PAD=132 (BM+4) padding to avoid bank conflicts")


if __name__ == "__main__":
    main()
