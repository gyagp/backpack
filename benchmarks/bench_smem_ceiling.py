"""Measure actual FP32 compute peak and shared-memory-limited FMA throughput."""
import sys
import time
import struct
import numpy as np

try:
    import wgpu
except ImportError:
    sys.exit(1)

# Test 1: Pure FMA throughput (no shared memory) - establishes true peak
SHADER_PURE_FMA = """
@group(0) @binding(0) var<storage, read_write> out: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    var a0: f32 = f32(gid.x) * 0.001 + 1.0;
    var a1: f32 = a0 + 0.1;
    var a2: f32 = a0 + 0.2;
    var a3: f32 = a0 + 0.3;
    let b: f32 = 1.00001;
    // 4 independent chains, 256 FMAs each = 1024 FMAs total
    for (var i = 0u; i < 256u; i++) {
        a0 = a0 * b + b;
        a1 = a1 * b + b;
        a2 = a2 * b + b;
        a3 = a3 * b + b;
    }
    out[gid.x] = a0 + a1 + a2 + a3;
}
"""

# Test 2: FMA with shared memory reads (simulates matmul inner loop)
SHADER_SMEM_FMA = """
var<workgroup> smem: array<f32, 4096>;

@group(0) @binding(0) var<storage, read_write> out: array<f32>;

@compute @workgroup_size(256)
fn main(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(global_invocation_id) gid: vec3<u32>,
) {
    // Fill shared memory
    for (var i = 0u; i < 16u; i++) {
        smem[lid.x * 16u + i] = f32(lid.x + i) * 0.001;
    }
    workgroupBarrier();

    // Simulate matmul: read from smem, do FMAs
    // 16 iterations, each reads 2 smem values, does 4 FMAs
    var r0: f32 = 0.0; var r1: f32 = 0.0; var r2: f32 = 0.0; var r3: f32 = 0.0;
    for (var k = 0u; k < 16u; k++) {
        let a = smem[lid.x + k * 16u];  // stride access
        let b0 = smem[k * 4u]; let b1 = smem[k * 4u + 1u]; let b2 = smem[k * 4u + 2u]; let b3 = smem[k * 4u + 3u];
        r0 += a * b0;
        r1 += a * b1;
        r2 += a * b2;
        r3 += a * b3;
    }
    // Repeat to increase FLOP count
    for (var rep = 0u; rep < 63u; rep++) {
        for (var k = 0u; k < 16u; k++) {
            let a = smem[lid.x + k * 16u];
            let b0 = smem[k * 4u]; let b1 = smem[k * 4u + 1u]; let b2 = smem[k * 4u + 2u]; let b3 = smem[k * 4u + 3u];
            r0 += a * b0;
            r1 += a * b1;
            r2 += a * b2;
            r3 += a * b3;
        }
    }

    out[gid.x] = r0 + r1 + r2 + r3;
}
"""

# Test 3: Matmul inner loop pattern with higher arithmetic intensity
SHADER_SMEM_FMA_HI = """
var<workgroup> smemA: array<f32, 4096>;
var<workgroup> smemB: array<f32, 4096>;

@group(0) @binding(0) var<storage, read_write> out: array<f32>;

@compute @workgroup_size(256)
fn main(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(global_invocation_id) gid: vec3<u32>,
) {
    for (var i = 0u; i < 16u; i++) {
        smemA[lid.x * 16u + i] = f32(lid.x + i) * 0.001;
        smemB[lid.x * 16u + i] = f32(lid.x - i) * 0.001;
    }
    workgroupBarrier();

    // 4x4 register tile, 16 iterations of k
    // Per k: 4 A reads + 4 B reads + 16 FMAs = 16/8 = 2 FMAs per read
    var r: array<f32, 16>;
    for (var i = 0u; i < 16u; i++) { r[i] = 0.0; }

    let row = lid.x / 16u;  // 0..15
    let col = lid.x % 16u;  // 0..15

    for (var rep = 0u; rep < 64u; rep++) {
        for (var k = 0u; k < 16u; k++) {
            let a0 = smemA[k * 64u + row * 4u];
            let a1 = smemA[k * 64u + row * 4u + 1u];
            let a2 = smemA[k * 64u + row * 4u + 2u];
            let a3 = smemA[k * 64u + row * 4u + 3u];
            let b0 = smemB[k * 64u + col * 4u];
            let b1 = smemB[k * 64u + col * 4u + 1u];
            let b2 = smemB[k * 64u + col * 4u + 2u];
            let b3 = smemB[k * 64u + col * 4u + 3u];
            r[0]+=a0*b0; r[1]+=a0*b1; r[2]+=a0*b2; r[3]+=a0*b3;
            r[4]+=a1*b0; r[5]+=a1*b1; r[6]+=a1*b2; r[7]+=a1*b3;
            r[8]+=a2*b0; r[9]+=a2*b1; r[10]+=a2*b2; r[11]+=a2*b3;
            r[12]+=a3*b0; r[13]+=a3*b1; r[14]+=a3*b2; r[15]+=a3*b3;
        }
    }

    var sum: f32 = 0.0;
    for (var i = 0u; i < 16u; i++) { sum += r[i]; }
    out[gid.x] = sum;
}
"""


def make_pipeline(device, shader_code, N):
    buf = device.create_buffer(size=N*4, usage=wgpu.BufferUsage.STORAGE|wgpu.BufferUsage.COPY_SRC)
    sm = device.create_shader_module(code=shader_code)
    bgl = device.create_bind_group_layout(entries=[
        {"binding":0,"visibility":wgpu.ShaderStage.COMPUTE,"buffer":{"type":wgpu.BufferBindingType.storage}},
    ])
    pl = device.create_pipeline_layout(bind_group_layouts=[bgl])
    pipe = device.create_compute_pipeline(layout=pl, compute={"module":sm,"entry_point":"main"})
    bg = device.create_bind_group(layout=bgl, entries=[
        {"binding":0,"resource":{"buffer":buf,"offset":0,"size":buf.size}},
    ])
    return pipe, bg, buf, (N+255)//256


def bench(device, pipe, bg, buf, wg, n):
    enc = device.create_command_encoder()
    for _ in range(n):
        p = enc.begin_compute_pass()
        p.set_pipeline(pipe)
        p.set_bind_group(0, bg)
        p.dispatch_workgroups(wg)
        p.end()
    t0 = time.perf_counter()
    device.queue.submit([enc.finish()])
    _ = device.queue.read_buffer(buf, size=4)
    t1 = time.perf_counter()
    return (t1-t0)/n


def main():
    adapter = wgpu.gpu.request_adapter_sync(power_preference="high-performance")
    device = adapter.request_device_sync()
    gpu = adapter.info.get('device','?')
    print(f"GPU: {gpu}")
    N = 1024*1024

    # Test 1: Pure FMA
    pipe, bg, buf, wg = make_pipeline(device, SHADER_PURE_FMA, N)
    for _ in range(3): bench(device, pipe, bg, buf, wg, 1)
    t = bench(device, pipe, bg, buf, wg, 100)
    flops_per_thread = 1024 * 2
    gflops = N * flops_per_thread / t / 1e9
    print(f"\n1. Pure FMA (no smem): {gflops:.0f} GFLOPS ({t*1e6:.0f} us)")
    buf.destroy()

    # Test 2: smem + FMA (low arithmetic intensity)
    pipe, bg, buf, wg = make_pipeline(device, SHADER_SMEM_FMA, N)
    for _ in range(3): bench(device, pipe, bg, buf, wg, 1)
    t = bench(device, pipe, bg, buf, wg, 100)
    flops_per_thread = 64 * 16 * 4 * 2  # 64 reps * 16 k * 4 FMAs * 2 flop
    gflops = N * flops_per_thread / t / 1e9
    print(f"2. Smem + FMA (4 FMAs/5 reads per k): {gflops:.0f} GFLOPS ({t*1e6:.0f} us)")
    buf.destroy()

    # Test 3: smem + FMA (matmul-like, high intensity)
    pipe, bg, buf, wg = make_pipeline(device, SHADER_SMEM_FMA_HI, N)
    for _ in range(3): bench(device, pipe, bg, buf, wg, 1)
    t = bench(device, pipe, bg, buf, wg, 100)
    flops_per_thread = 64 * 16 * 16 * 2  # 64 reps * 16 k * 16 FMAs * 2 flop
    gflops = N * flops_per_thread / t / 1e9
    print(f"3. Smem + FMA (16 FMAs/8 reads per k): {gflops:.0f} GFLOPS ({t*1e6:.0f} us)")
    buf.destroy()


if __name__ == "__main__":
    main()
