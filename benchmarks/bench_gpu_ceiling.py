"""Test raw GPU compute throughput to establish a ceiling for matmul optimization."""
import sys
import time
import struct
import numpy as np

try:
    import wgpu
except ImportError:
    print("ERROR: wgpu not installed", file=sys.stderr)
    sys.exit(1)

# Pure compute: just do a bunch of FMAs per thread with no memory access
SHADER_RAW_COMPUTE = """
struct Params { N: u32, }

@group(0) @binding(0) var<storage, read_write> out: array<f32>;
@group(0) @binding(1) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if gid.x >= params.N { return; }
    var a: f32 = f32(gid.x) * 0.001;
    var b: f32 = 1.0001;
    // 1024 FMAs
    for (var i = 0u; i < 256u; i++) {
        a = a * b + b;
        a = a * b + b;
        a = a * b + b;
        a = a * b + b;
    }
    out[gid.x] = a;
}
"""

# Shared memory bandwidth test
SHADER_SMEM_BW = """
struct Params { N: u32, }

@group(0) @binding(0) var<storage, read_write> out: array<f32>;
@group(0) @binding(1) var<uniform> params: Params;

var<workgroup> smem: array<f32, 4096>;

@compute @workgroup_size(256)
fn main(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(global_invocation_id) gid: vec3<u32>,
) {
    // Write 16 values per thread
    for (var i = 0u; i < 16u; i++) {
        smem[lid.x * 16u + i] = f32(lid.x + i);
    }
    workgroupBarrier();

    // Read 16 values per thread (different pattern to avoid trivial optimization)
    var sum: f32 = 0.0;
    for (var i = 0u; i < 16u; i++) {
        sum += smem[i * 256u + lid.x];
    }

    if gid.x < params.N {
        out[gid.x] = sum;
    }
}
"""


def dispatch_batched(device, pipeline, bind_group, wg_count, n_dispatches, buf_out):
    encoder = device.create_command_encoder()
    for _ in range(n_dispatches):
        p = encoder.begin_compute_pass()
        p.set_pipeline(pipeline)
        p.set_bind_group(0, bind_group)
        p.dispatch_workgroups(wg_count)
        p.end()
    t0 = time.perf_counter()
    device.queue.submit([encoder.finish()])
    _ = device.queue.read_buffer(buf_out, size=4)
    t1 = time.perf_counter()
    return (t1 - t0) / n_dispatches


def test_raw_compute(device):
    N = 1024 * 1024  # 1M threads
    flops_per_thread = 1024 * 2  # 1024 FMAs = 2048 FLOP
    total_flops = N * flops_per_thread

    params = struct.pack('I', N)
    buf_out = device.create_buffer(size=N * 4, usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC)
    buf_params = device.create_buffer_with_data(data=params, usage=wgpu.BufferUsage.UNIFORM)

    shader = device.create_shader_module(code=SHADER_RAW_COMPUTE)
    bgl = device.create_bind_group_layout(entries=[
        {"binding": 0, "visibility": wgpu.ShaderStage.COMPUTE, "buffer": {"type": wgpu.BufferBindingType.storage}},
        {"binding": 1, "visibility": wgpu.ShaderStage.COMPUTE, "buffer": {"type": wgpu.BufferBindingType.uniform}},
    ])
    pl = device.create_pipeline_layout(bind_group_layouts=[bgl])
    pipeline = device.create_compute_pipeline(layout=pl, compute={"module": shader, "entry_point": "main"})
    bg = device.create_bind_group(layout=bgl, entries=[
        {"binding": 0, "resource": {"buffer": buf_out, "offset": 0, "size": buf_out.size}},
        {"binding": 1, "resource": {"buffer": buf_params, "offset": 0, "size": buf_params.size}},
    ])

    wg_count = (N + 255) // 256

    # warmup
    for _ in range(3):
        dispatch_batched(device, pipeline, bg, wg_count, 1, buf_out)

    t = dispatch_batched(device, pipeline, bg, wg_count, 100, buf_out)
    gflops = total_flops / t / 1e9

    buf_out.destroy()
    buf_params.destroy()
    return gflops


def main():
    adapter = wgpu.gpu.request_adapter_sync(power_preference="high-performance")
    info = adapter.info
    device = adapter.request_device_sync()
    gpu = info.get('device', 'Unknown')

    print(f"GPU: {gpu}")
    print()

    gflops = test_raw_compute(device)
    print(f"Raw compute throughput (1M threads × 1024 FMAs): {gflops:.1f} GFLOPS")
    print(f"  (RTX 5080 theoretical FP32: ~56,000 GFLOPS)")
    print(f"  Utilization: {gflops/56000*100:.1f}%")


if __name__ == "__main__":
    main()
