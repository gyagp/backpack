"""Test matmul with transposed tileA to avoid bank conflicts."""
import sys
import time
import struct
import numpy as np

try:
    import wgpu
except ImportError:
    print("ERROR: wgpu not installed", file=sys.stderr)
    sys.exit(1)

WARMUP_RUNS = 3
BENCH_RUNS = 5

# Key optimization: transpose tileA storage so reads are along columns (no bank conflict)
# Also preload a_val array per k to reduce repeated indexing
SHADER_TRANSPOSED = """
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

// tileA stored transposed: [BK][BM] so inner loop reads contiguous BM
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

    var r0: f32 = 0.0; var r1: f32 = 0.0; var r2: f32 = 0.0; var r3: f32 = 0.0;
    var r4: f32 = 0.0; var r5: f32 = 0.0; var r6: f32 = 0.0; var r7: f32 = 0.0;
    var r8: f32 = 0.0; var r9: f32 = 0.0; var r10: f32 = 0.0; var r11: f32 = 0.0;
    var r12: f32 = 0.0; var r13: f32 = 0.0; var r14: f32 = 0.0; var r15: f32 = 0.0;

    let numTiles = (params.K + BK - 1u) / BK;

    for (var t: u32 = 0u; t < numTiles; t = t + 1u) {
        let tile_k = t * BK;

        // Load tileA transposed: A[row][col] -> tileA[col][row] = tileA[a_c * BM + a_r]
        for (var load = 0u; load < 4u; load++) {
            let flat = load * 256u + thread_id;
            let a_r = flat / BK;   // row in tile (0..63)
            let a_c = flat % BK;   // col in tile (0..15)
            let gr = block_row + a_r;
            let gc = tile_k + a_c;
            if gr < params.M && gc < params.K {
                tileA[a_c * BM + a_r] = A[gr * params.K + gc];
            } else {
                tileA[a_c * BM + a_r] = 0.0;
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

        let a_base = lr * TM;  // start row in tileA for this thread
        let b_base = lc * TN;  // start col in tileB for this thread

        for (var k: u32 = 0u; k < BK; k = k + 1u) {
            let a0 = tileA[k * BM + a_base];
            let a1 = tileA[k * BM + a_base + 1u];
            let a2 = tileA[k * BM + a_base + 2u];
            let a3 = tileA[k * BM + a_base + 3u];

            let b0 = tileB[k * BN + b_base];
            let b1 = tileB[k * BN + b_base + 1u];
            let b2 = tileB[k * BN + b_base + 2u];
            let b3 = tileB[k * BN + b_base + 3u];

            r0 += a0 * b0; r1 += a0 * b1; r2 += a0 * b2; r3 += a0 * b3;
            r4 += a1 * b0; r5 += a1 * b1; r6 += a1 * b2; r7 += a1 * b3;
            r8 += a2 * b0; r9 += a2 * b1; r10 += a2 * b2; r11 += a2 * b3;
            r12 += a3 * b0; r13 += a3 * b1; r14 += a3 * b2; r15 += a3 * b3;
        }

        workgroupBarrier();
    }

    // Write results
    var results: array<f32, 16>;
    results[0]=r0; results[1]=r1; results[2]=r2; results[3]=r3;
    results[4]=r4; results[5]=r5; results[6]=r6; results[7]=r7;
    results[8]=r8; results[9]=r9; results[10]=r10; results[11]=r11;
    results[12]=r12; results[13]=r13; results[14]=r14; results[15]=r15;

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

BLOCK_M = 64
BLOCK_N = 64


def make_params(M, N, K):
    return struct.pack('III', M, N, K)


def setup_pipeline(device, shader_code, M, N, K, A, B):
    params_bytes = make_params(M, N, K)
    out_size = M * N * 4
    buf_a = device.create_buffer_with_data(data=A.tobytes(), usage=wgpu.BufferUsage.STORAGE)
    buf_b = device.create_buffer_with_data(data=B.tobytes(), usage=wgpu.BufferUsage.STORAGE)
    buf_c = device.create_buffer(size=out_size, usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC)
    buf_params = device.create_buffer_with_data(data=params_bytes, usage=wgpu.BufferUsage.UNIFORM)
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
    wg_x = (M + BLOCK_M - 1) // BLOCK_M
    wg_y = (N + BLOCK_N - 1) // BLOCK_N
    return {"pipeline": pipeline, "bind_group": bg, "buf_c": buf_c,
            "wg_x": wg_x, "wg_y": wg_y, "bufs": [buf_a, buf_b, buf_c, buf_params]}


def dispatch_and_sync(device, ctx):
    enc = device.create_command_encoder()
    p = enc.begin_compute_pass()
    p.set_pipeline(ctx["pipeline"])
    p.set_bind_group(0, ctx["bind_group"])
    p.dispatch_workgroups(ctx["wg_x"], ctx["wg_y"])
    p.end()
    device.queue.submit([enc.finish()])
    return device.queue.read_buffer(ctx["buf_c"])


def dispatch_batched(device, ctx, n):
    enc = device.create_command_encoder()
    for _ in range(n):
        p = enc.begin_compute_pass()
        p.set_pipeline(ctx["pipeline"])
        p.set_bind_group(0, ctx["bind_group"])
        p.dispatch_workgroups(ctx["wg_x"], ctx["wg_y"])
        p.end()
    t0 = time.perf_counter()
    device.queue.submit([enc.finish()])
    _ = device.queue.read_buffer(ctx["buf_c"])
    t1 = time.perf_counter()
    return (t1 - t0) / n


def main():
    adapter = wgpu.gpu.request_adapter_sync(power_preference="high-performance")
    info = adapter.info
    device = adapter.request_device_sync()
    gpu = info.get('device', 'Unknown')
    print(f"GPU: {gpu}")

    # Correctness check
    for sz in [64, 128, 256, 512]:
        rng = np.random.default_rng(42)
        A = rng.uniform(-1, 1, (sz, sz)).astype(np.float32)
        B = rng.uniform(-1, 1, (sz, sz)).astype(np.float32)
        ctx = setup_pipeline(device, SHADER_TRANSPOSED, sz, sz, sz, A, B)
        raw = dispatch_and_sync(device, ctx)
        gpu_result = np.frombuffer(raw, dtype=np.float32).reshape(sz, sz)
        expected = A @ B
        max_err = np.max(np.abs(gpu_result - expected))
        rel = max_err / (np.max(np.abs(expected)) + 1e-8)
        print(f"  {sz}x{sz}: max_err={max_err:.6f} rel={rel:.8f} {'PASS' if rel < 0.001 else 'FAIL'}")
        for buf in ctx["bufs"]:
            buf.destroy()

    print()
    for M in [2048, 4096]:
        rng = np.random.default_rng(42)
        A = rng.uniform(-1, 1, (M, M)).astype(np.float32)
        B = rng.uniform(-1, 1, (M, M)).astype(np.float32)
        ctx = setup_pipeline(device, SHADER_TRANSPOSED, M, M, M, A, B)
        for _ in range(WARMUP_RUNS):
            dispatch_and_sync(device, ctx)
        times = [dispatch_batched(device, ctx, 50) for _ in range(BENCH_RUNS)]
        for buf in ctx["bufs"]:
            buf.destroy()
        flops = 2 * M * M * M
        med = np.median(times)
        gflops = flops / med / 1e9
        print(f"  {M}x{M}: {med*1000:.2f} ms, {gflops:.1f} GFLOPS ({gflops/69200*100:.1f}%)")


if __name__ == "__main__":
    main()
