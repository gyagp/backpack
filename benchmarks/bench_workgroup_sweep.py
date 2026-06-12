"""Auto-benchmarking harness that sweeps workgroup sizes for key kernels
(matmul, attention, rmsnorm) and selects optimal config per GPU."""
import json
import os
import platform
import re
import struct
import sys
import time

import numpy as np

try:
    import wgpu
except ImportError:
    print("ERROR: wgpu not installed. pip install wgpu", file=sys.stderr)
    sys.exit(1)

SHADER_DIR = os.path.join(os.path.dirname(__file__), '..', 'src', 'shaders')
WARMUP_RUNS = 3
BENCH_RUNS = 10


def pack_f16_to_u32(arr: np.ndarray) -> bytes:
    flat = arr.flatten().astype(np.float16)
    raw = flat.view(np.uint16).astype(np.uint32)
    if len(raw) % 2:
        raw = np.append(raw, np.uint32(0))
    packed = ((raw[1::2] << 16) | raw[0::2]).astype(np.uint32)
    return packed.tobytes()


def get_device_and_info():
    adapter = wgpu.gpu.request_adapter_sync(power_preference="high-performance")
    if not adapter:
        print("ERROR: No WebGPU adapter found", file=sys.stderr)
        sys.exit(1)
    info = adapter.info
    device = adapter.request_device_sync(required_features=['timestamp-query'])
    query_set = device.create_query_set(type='timestamp', count=2)
    resolve_buf = device.create_buffer(
        size=16,
        usage=wgpu.BufferUsage.QUERY_RESOLVE | wgpu.BufferUsage.COPY_SRC,
    )
    gpu_name = info.get('device', '') or info.get('description', '') or 'Unknown GPU'
    return device, gpu_name, query_set, resolve_buf


def dispatch_timed(device, pipeline, bind_group, wg_x, wg_y, wg_z, query_set, resolve_buf):
    encoder = device.create_command_encoder()
    pass_enc = encoder.begin_compute_pass(timestamp_writes={
        'query_set': query_set,
        'beginning_of_pass_write_index': 0,
        'end_of_pass_write_index': 1,
    })
    pass_enc.set_pipeline(pipeline)
    pass_enc.set_bind_group(0, bind_group)
    pass_enc.dispatch_workgroups(wg_x, wg_y, wg_z)
    pass_enc.end()
    encoder.resolve_query_set(query_set, 0, 2, resolve_buf, 0)
    device.queue.submit([encoder.finish()])
    data = device.queue.read_buffer(resolve_buf)
    t0, t1 = struct.unpack('QQ', data)
    return (t1 - t0) / 1e9


def bench_dispatch(device, pipeline, bind_group, wg_x, wg_y, wg_z, query_set, resolve_buf):
    for _ in range(WARMUP_RUNS):
        dispatch_timed(device, pipeline, bind_group, wg_x, wg_y, wg_z, query_set, resolve_buf)
    times = []
    for _ in range(BENCH_RUNS):
        times.append(dispatch_timed(device, pipeline, bind_group, wg_x, wg_y, wg_z, query_set, resolve_buf))
    return float(np.median(times))


# ---------------------------------------------------------------------------
# Matmul kernel sweep — parameterized BM, BN, BK, workgroup_size
# ---------------------------------------------------------------------------

MATMUL_TEMPLATE = """\
const WG_X: u32 = {wg_x}u;
const WG_Y: u32 = {wg_y}u;
const TM: u32 = {tm}u;
const TN: u32 = {tn}u;
const BM: u32 = {bm}u;
const BN: u32 = {bn}u;
const BK: u32 = {bk}u;
const BM_PAD: u32 = {bm_pad}u;

struct Params {{
    M: u32,
    N: u32,
    K: u32,
}}

@group(0) @binding(0) var<storage, read> A: array<u32>;
@group(0) @binding(1) var<storage, read> B: array<u32>;
@group(0) @binding(2) var<storage, read_write> C: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;

var<workgroup> tileA: array<f32, {smem_a}>;
var<workgroup> tileB: array<f32, {smem_b}>;

@compute @workgroup_size({wg_x}, {wg_y})
fn main(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wgid: vec3<u32>,
) {{
    let lr = lid.x;
    let lc = lid.y;
    let thread_id = lr * WG_Y + lc;

    let block_row = wgid.x * BM;
    let block_col = wgid.y * BN;

    var acc: array<f32, {acc_count}>;
    for (var i = 0u; i < {acc_count}u; i++) {{
        acc[i] = 0.0;
    }}

    let threads_per_block = WG_X * WG_Y;
    let a_loads = (BK * BM + threads_per_block - 1u) / threads_per_block;
    let b_loads = (BK * BN + threads_per_block - 1u) / threads_per_block;

    for (var bk = 0u; bk < params.K; bk += BK) {{
        // Load tileA: BK x BM (transposed storage for bank conflict avoidance)
        for (var load = 0u; load < a_loads; load++) {{
            let idx = thread_id + load * threads_per_block;
            let bk_idx = idx / BM;
            let bm_idx = idx % BM;
            if (bk_idx < BK && (block_row + bm_idx) < params.M && (bk + bk_idx) < params.K) {{
                let a_row = block_row + bm_idx;
                let a_col = bk + bk_idx;
                let packed_idx = (a_row * params.K + a_col) / 2u;
                let word = A[packed_idx];
                let pair = unpack2x16float(word);
                if ((a_row * params.K + a_col) % 2u == 0u) {{
                    tileA[bk_idx * BM_PAD + bm_idx] = pair.x;
                }} else {{
                    tileA[bk_idx * BM_PAD + bm_idx] = pair.y;
                }}
            }}
        }}

        // Load tileB: BK x BN
        for (var load = 0u; load < b_loads; load++) {{
            let idx = thread_id + load * threads_per_block;
            let bk_idx = idx / BN;
            let bn_idx = idx % BN;
            if (bk_idx < BK && (block_col + bn_idx) < params.N && (bk + bk_idx) < params.K) {{
                let b_row = bk + bk_idx;
                let b_col = block_col + bn_idx;
                let packed_idx = (b_row * params.N + b_col) / 2u;
                let word = B[packed_idx];
                let pair = unpack2x16float(word);
                if ((b_row * params.N + b_col) % 2u == 0u) {{
                    tileB[bk_idx * BN + bn_idx] = pair.x;
                }} else {{
                    tileB[bk_idx * BN + bn_idx] = pair.y;
                }}
            }}
        }}

        workgroupBarrier();

        let row_base = lr * TM;
        let col_base = lc * TN;
        for (var k = 0u; k < BK; k++) {{
            for (var tm = 0u; tm < TM; tm++) {{
                let a_val = tileA[k * BM_PAD + row_base + tm];
                for (var tn = 0u; tn < TN; tn++) {{
                    acc[tm * TN + tn] += a_val * tileB[k * BN + col_base + tn];
                }}
            }}
        }}

        workgroupBarrier();
    }}

    let row_base = lr * TM;
    let col_base = lc * TN;
    for (var tm = 0u; tm < TM; tm++) {{
        for (var tn = 0u; tn < TN; tn++) {{
            let out_row = block_row + row_base + tm;
            let out_col = block_col + col_base + tn;
            if (out_row < params.M && out_col < params.N) {{
                C[out_row * params.N + out_col] = acc[tm * TN + tn];
            }}
        }}
    }}
}}
"""


def generate_matmul_shader(bm, bn, bk, wg_x, wg_y):
    tm = bm // wg_x
    tn = bn // wg_y
    bm_pad = bm + 4
    return MATMUL_TEMPLATE.format(
        wg_x=wg_x, wg_y=wg_y,
        tm=tm, tn=tn,
        bm=bm, bn=bn, bk=bk,
        bm_pad=bm_pad,
        smem_a=bk * bm_pad,
        smem_b=bk * bn,
        acc_count=tm * tn,
    )


MATMUL_CONFIGS = [
    # workgroup_size 64 (8x8)
    {"bm": 128, "bn": 128, "bk": 32, "wg_x": 8,  "wg_y": 8,  "label": "128x128_bk32_wg64"},
    {"bm": 64,  "bn": 64,  "bk": 32, "wg_x": 8,  "wg_y": 8,  "label": "64x64_bk32_wg64"},
    # workgroup_size 128 (8x16)
    {"bm": 128, "bn": 128, "bk": 32, "wg_x": 8,  "wg_y": 16, "label": "128x128_bk32_wg128"},
    {"bm": 64,  "bn": 128, "bk": 32, "wg_x": 8,  "wg_y": 16, "label": "64x128_bk32_wg128"},
    # workgroup_size 256 (16x16)
    {"bm": 64,  "bn": 64,  "bk": 16, "wg_x": 16, "wg_y": 16, "label": "64x64_bk16_wg256"},
    {"bm": 64,  "bn": 64,  "bk": 32, "wg_x": 16, "wg_y": 16, "label": "64x64_bk32_wg256"},
    {"bm": 128, "bn": 128, "bk": 16, "wg_x": 16, "wg_y": 16, "label": "128x128_bk16_wg256"},
    {"bm": 128, "bn": 128, "bk": 32, "wg_x": 16, "wg_y": 16, "label": "128x128_bk32_wg256"},
    {"bm": 256, "bn": 128, "bk": 16, "wg_x": 16, "wg_y": 16, "label": "256x128_bk16_wg256"},
    # workgroup_size 512 (16x32)
    {"bm": 128, "bn": 128, "bk": 32, "wg_x": 16, "wg_y": 32, "label": "128x128_bk32_wg512"},
    {"bm": 256, "bn": 128, "bk": 16, "wg_x": 16, "wg_y": 32, "label": "256x128_bk16_wg512"},
]


def sweep_matmul(device, query_set, resolve_buf):
    M, N, K = 2048, 2048, 2048
    flops = 2 * M * N * K
    rng = np.random.default_rng(42)
    A = rng.uniform(-1, 1, (M, K)).astype(np.float16)
    B = rng.uniform(-1, 1, (K, N)).astype(np.float16)
    A_bytes = pack_f16_to_u32(A)
    B_bytes = pack_f16_to_u32(B)

    buf_a = device.create_buffer_with_data(data=A_bytes, usage=wgpu.BufferUsage.STORAGE)
    buf_b = device.create_buffer_with_data(data=B_bytes, usage=wgpu.BufferUsage.STORAGE)
    params_bytes = struct.pack('III', M, N, K)
    buf_params = device.create_buffer_with_data(data=params_bytes, usage=wgpu.BufferUsage.UNIFORM)

    results = []
    for cfg in MATMUL_CONFIGS:
        bm, bn, bk = cfg["bm"], cfg["bn"], cfg["bk"]
        wg_x, wg_y = cfg["wg_x"], cfg["wg_y"]
        label = cfg["label"]
        try:
            shader_code = generate_matmul_shader(bm, bn, bk, wg_x, wg_y)
            out_size = M * N * 4
            buf_c = device.create_buffer(
                size=out_size,
                usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC,
            )
            shader_module = device.create_shader_module(code=shader_code)
            bgl = device.create_bind_group_layout(entries=[
                {"binding": 0, "visibility": wgpu.ShaderStage.COMPUTE, "buffer": {"type": wgpu.BufferBindingType.read_only_storage}},
                {"binding": 1, "visibility": wgpu.ShaderStage.COMPUTE, "buffer": {"type": wgpu.BufferBindingType.read_only_storage}},
                {"binding": 2, "visibility": wgpu.ShaderStage.COMPUTE, "buffer": {"type": wgpu.BufferBindingType.storage}},
                {"binding": 3, "visibility": wgpu.ShaderStage.COMPUTE, "buffer": {"type": wgpu.BufferBindingType.uniform}},
            ])
            pl = device.create_pipeline_layout(bind_group_layouts=[bgl])
            pipeline = device.create_compute_pipeline(layout=pl, compute={"module": shader_module, "entry_point": "main"})
            bg = device.create_bind_group(layout=bgl, entries=[
                {"binding": 0, "resource": {"buffer": buf_a, "offset": 0, "size": buf_a.size}},
                {"binding": 1, "resource": {"buffer": buf_b, "offset": 0, "size": buf_b.size}},
                {"binding": 2, "resource": {"buffer": buf_c, "offset": 0, "size": buf_c.size}},
                {"binding": 3, "resource": {"buffer": buf_params, "offset": 0, "size": buf_params.size}},
            ])
            wg_dispatch_x = (M + bm - 1) // bm
            wg_dispatch_y = (N + bn - 1) // bn
            median_s = bench_dispatch(device, pipeline, bg, wg_dispatch_x, wg_dispatch_y, 1, query_set, resolve_buf)
            gflops = flops / median_s / 1e9
            buf_c.destroy()
            results.append({
                "config": label,
                "bm": bm, "bn": bn, "bk": bk,
                "workgroup_size": wg_x * wg_y,
                "wg_dims": [wg_x, wg_y],
                "median_ms": round(median_s * 1000, 3),
                "gflops": round(gflops, 1),
            })
            print(f"  matmul {label:>25s}: {median_s*1000:8.2f} ms  {gflops:8.1f} GFLOPS")
        except Exception as e:
            print(f"  matmul {label:>25s}: FAILED ({e})")
            results.append({"config": label, "error": str(e)})

    buf_a.destroy()
    buf_b.destroy()
    buf_params.destroy()
    return results


# ---------------------------------------------------------------------------
# 1D kernel sweep (attention, rmsnorm) — vary workgroup_size
# ---------------------------------------------------------------------------

def load_shader_with_wg_size(path, wg_size):
    with open(path) as f:
        code = f.read()
    code = re.sub(r'@workgroup_size\(\d+\)', f'@workgroup_size({wg_size})', code)
    return code


WORKGROUP_SIZES_1D = [64, 128, 256, 512]


def sweep_rmsnorm(device, query_set, resolve_buf):
    shader_path = os.path.join(SHADER_DIR, 'rmsnorm_scaled.wgsl')
    if not os.path.exists(shader_path):
        return []

    dim = 4096
    rows = 512
    total = rows * dim
    epsilon = 1e-5
    rng = np.random.default_rng(42)
    data_np = rng.uniform(-1, 1, total).astype(np.float32)
    weights_np = rng.uniform(0.5, 1.5, dim).astype(np.float32)

    buf_in = device.create_buffer_with_data(data=data_np.tobytes(), usage=wgpu.BufferUsage.STORAGE)
    buf_w = device.create_buffer_with_data(data=weights_np.tobytes(), usage=wgpu.BufferUsage.STORAGE)
    buf_out = device.create_buffer(size=total * 4, usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC)
    params_bytes = struct.pack('If', dim, epsilon)
    buf_params = device.create_buffer_with_data(data=params_bytes, usage=wgpu.BufferUsage.UNIFORM)

    results = []
    for wg in WORKGROUP_SIZES_1D:
        try:
            code = load_shader_with_wg_size(shader_path, wg)
            sm = device.create_shader_module(code=code)
            bgl = device.create_bind_group_layout(entries=[
                {"binding": 0, "visibility": wgpu.ShaderStage.COMPUTE, "buffer": {"type": wgpu.BufferBindingType.read_only_storage}},
                {"binding": 1, "visibility": wgpu.ShaderStage.COMPUTE, "buffer": {"type": wgpu.BufferBindingType.read_only_storage}},
                {"binding": 2, "visibility": wgpu.ShaderStage.COMPUTE, "buffer": {"type": wgpu.BufferBindingType.storage}},
                {"binding": 3, "visibility": wgpu.ShaderStage.COMPUTE, "buffer": {"type": wgpu.BufferBindingType.uniform}},
            ])
            pl = device.create_pipeline_layout(bind_group_layouts=[bgl])
            pipeline = device.create_compute_pipeline(layout=pl, compute={"module": sm, "entry_point": "main"})
            bg = device.create_bind_group(layout=bgl, entries=[
                {"binding": 0, "resource": {"buffer": buf_in, "offset": 0, "size": buf_in.size}},
                {"binding": 1, "resource": {"buffer": buf_w, "offset": 0, "size": buf_w.size}},
                {"binding": 2, "resource": {"buffer": buf_out, "offset": 0, "size": buf_out.size}},
                {"binding": 3, "resource": {"buffer": buf_params, "offset": 0, "size": buf_params.size}},
            ])
            median_s = bench_dispatch(device, pipeline, bg, rows, 1, 1, query_set, resolve_buf)
            results.append({
                "workgroup_size": wg,
                "median_ms": round(median_s * 1000, 3),
            })
            print(f"  rmsnorm wg={wg:>4d}: {median_s*1000:8.3f} ms")
        except Exception as e:
            print(f"  rmsnorm wg={wg:>4d}: FAILED ({e})")
            results.append({"workgroup_size": wg, "error": str(e)})

    buf_in.destroy()
    buf_w.destroy()
    buf_out.destroy()
    buf_params.destroy()
    return results


def sweep_attention(device, query_set, resolve_buf):
    shader_path = os.path.join(SHADER_DIR, 'attention.wgsl')
    if not os.path.exists(shader_path):
        return []

    seq_len = 512
    head_dim = 128
    n_heads = 32
    n_kv_heads = 32
    scale = 1.0 / (head_dim ** 0.5)
    rng = np.random.default_rng(42)

    Q = rng.uniform(-1, 1, (n_heads * head_dim,)).astype(np.float32)
    K = rng.uniform(-1, 1, (n_kv_heads * seq_len * head_dim,)).astype(np.float32)
    V = rng.uniform(-1, 1, (n_kv_heads * seq_len * head_dim,)).astype(np.float32)
    out_size = n_heads * head_dim

    buf_q = device.create_buffer_with_data(data=Q.tobytes(), usage=wgpu.BufferUsage.STORAGE)
    buf_k = device.create_buffer_with_data(data=K.tobytes(), usage=wgpu.BufferUsage.STORAGE)
    buf_v = device.create_buffer_with_data(data=V.tobytes(), usage=wgpu.BufferUsage.STORAGE)
    buf_out = device.create_buffer(size=out_size * 4, usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC)
    scale_bits = struct.unpack('I', struct.pack('f', scale))[0]
    params_data = struct.pack('IIIIIIII',
        seq_len, head_dim, n_heads, n_kv_heads,
        scale_bits, 0, 0, 0)
    buf_params = device.create_buffer_with_data(data=params_data, usage=wgpu.BufferUsage.UNIFORM)

    results = []
    for wg in WORKGROUP_SIZES_1D:
        try:
            code = load_shader_with_wg_size(shader_path, wg)
            sm = device.create_shader_module(code=code)
            bgl = device.create_bind_group_layout(entries=[
                {"binding": 0, "visibility": wgpu.ShaderStage.COMPUTE, "buffer": {"type": wgpu.BufferBindingType.read_only_storage}},
                {"binding": 1, "visibility": wgpu.ShaderStage.COMPUTE, "buffer": {"type": wgpu.BufferBindingType.read_only_storage}},
                {"binding": 2, "visibility": wgpu.ShaderStage.COMPUTE, "buffer": {"type": wgpu.BufferBindingType.read_only_storage}},
                {"binding": 3, "visibility": wgpu.ShaderStage.COMPUTE, "buffer": {"type": wgpu.BufferBindingType.storage}},
                {"binding": 4, "visibility": wgpu.ShaderStage.COMPUTE, "buffer": {"type": wgpu.BufferBindingType.uniform}},
            ])
            pl = device.create_pipeline_layout(bind_group_layouts=[bgl])
            pipeline = device.create_compute_pipeline(layout=pl, compute={"module": sm, "entry_point": "main"})
            bg = device.create_bind_group(layout=bgl, entries=[
                {"binding": 0, "resource": {"buffer": buf_q, "offset": 0, "size": buf_q.size}},
                {"binding": 1, "resource": {"buffer": buf_k, "offset": 0, "size": buf_k.size}},
                {"binding": 2, "resource": {"buffer": buf_v, "offset": 0, "size": buf_v.size}},
                {"binding": 3, "resource": {"buffer": buf_out, "offset": 0, "size": buf_out.size}},
                {"binding": 4, "resource": {"buffer": buf_params, "offset": 0, "size": buf_params.size}},
            ])
            median_s = bench_dispatch(device, pipeline, bg, n_heads, 1, 1, query_set, resolve_buf)
            results.append({
                "workgroup_size": wg,
                "median_ms": round(median_s * 1000, 3),
            })
            print(f"  attention wg={wg:>4d}: {median_s*1000:8.3f} ms")
        except Exception as e:
            print(f"  attention wg={wg:>4d}: FAILED ({e})")
            results.append({"workgroup_size": wg, "error": str(e)})

    buf_q.destroy()
    buf_k.destroy()
    buf_v.destroy()
    buf_out.destroy()
    buf_params.destroy()
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def select_best(results, metric_key="median_ms"):
    valid = [r for r in results if "error" not in r]
    if not valid:
        return None
    return min(valid, key=lambda r: r[metric_key])


def main():
    device, gpu_name, query_set, resolve_buf = get_device_and_info()

    print("=" * 70)
    print(f"Workgroup Size Sweep Benchmark")
    print(f"GPU: {gpu_name}")
    print(f"Warmup: {WARMUP_RUNS}, Bench runs: {BENCH_RUNS}")
    print("=" * 70)

    output = {
        "gpu": gpu_name,
        "platform": platform.platform(),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "kernels": {},
    }

    print("\n--- Matmul (2048x2048x2048, f16 packed) ---")
    matmul_results = sweep_matmul(device, query_set, resolve_buf)
    best_matmul = select_best(matmul_results)
    output["kernels"]["matmul"] = {
        "sweep": matmul_results,
        "best": best_matmul,
    }

    print("\n--- RMSNorm (4096 dim, 512 rows) ---")
    rmsnorm_results = sweep_rmsnorm(device, query_set, resolve_buf)
    best_rmsnorm = select_best(rmsnorm_results)
    output["kernels"]["rmsnorm"] = {
        "sweep": rmsnorm_results,
        "best": best_rmsnorm,
    }

    print("\n--- Attention (32 heads, seq=512, dim=128) ---")
    attention_results = sweep_attention(device, query_set, resolve_buf)
    best_attention = select_best(attention_results)
    output["kernels"]["attention"] = {
        "sweep": attention_results,
        "best": best_attention,
    }

    print("\n" + "=" * 70)
    print("BEST CONFIGURATIONS:")
    if best_matmul:
        print(f"  matmul:    {best_matmul['config']} — {best_matmul['gflops']} GFLOPS ({best_matmul['median_ms']} ms)")
    if best_rmsnorm:
        print(f"  rmsnorm:   wg={best_rmsnorm['workgroup_size']} — {best_rmsnorm['median_ms']} ms")
    if best_attention:
        print(f"  attention: wg={best_attention['workgroup_size']} — {best_attention['median_ms']} ms")
    print("=" * 70)

    out_path = os.path.join(os.path.dirname(__file__), 'workgroup_sweep_results.json')
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nResults written to {out_path}")


if __name__ == "__main__":
    main()
