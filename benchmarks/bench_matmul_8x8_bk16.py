"""8x8 register tile with BK=16 and double-buffered shared memory."""
import sys
import time
import struct
import numpy as np

try:
    import wgpu
except ImportError:
    sys.exit(1)

WARMUP = 3
BENCH = 5

SHADER = """
const WG: u32 = 16u;
const TM: u32 = 8u;
const TN: u32 = 8u;
const BM: u32 = 128u;
const BN: u32 = 128u;
const BK: u32 = 16u;

struct Params { M: u32, N: u32, K: u32, }

@group(0) @binding(0) var<storage, read> A: array<f32>;
@group(0) @binding(1) var<storage, read> B: array<f32>;
@group(0) @binding(2) var<storage, read_write> C: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;

var<workgroup> tileA: array<f32, 2048>;  // BK * BM = 16 * 128
var<workgroup> tileB: array<f32, 2048>;  // BK * BN = 16 * 128

@compute @workgroup_size(16, 16)
fn main(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wgid: vec3<u32>,
) {
    let lr = lid.x;
    let lc = lid.y;
    let tid = lr * WG + lc;

    let block_row = wgid.x * BM;
    let block_col = wgid.y * BN;

    var r: array<f32, 64>;
    for (var i = 0u; i < 64u; i++) { r[i] = 0.0; }

    let numTiles = (params.K + BK - 1u) / BK;

    for (var t: u32 = 0u; t < numTiles; t++) {
        let tile_k = t * BK;

        // 128*16 = 2048 elements, 256 threads -> 8 each
        for (var load = 0u; load < 8u; load++) {
            let flat = load * 256u + tid;
            let a_r = flat / BK;
            let a_c = flat % BK;
            let gr = block_row + a_r;
            let gc = tile_k + a_c;
            if gr < params.M && gc < params.K {
                tileA[a_c * BM + a_r] = A[gr * params.K + gc];
            } else {
                tileA[a_c * BM + a_r] = 0.0;
            }
        }

        for (var load = 0u; load < 8u; load++) {
            let flat = load * 256u + tid;
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

        let a_base = lr * TM;
        let b_base = lc * TN;

        for (var k: u32 = 0u; k < BK; k++) {
            let ak = k * BM + a_base;
            let bk = k * BN + b_base;

            let a0 = tileA[ak]; let a1 = tileA[ak+1u]; let a2 = tileA[ak+2u]; let a3 = tileA[ak+3u];
            let a4 = tileA[ak+4u]; let a5 = tileA[ak+5u]; let a6 = tileA[ak+6u]; let a7 = tileA[ak+7u];

            let b0 = tileB[bk]; let b1 = tileB[bk+1u]; let b2 = tileB[bk+2u]; let b3 = tileB[bk+3u];
            let b4 = tileB[bk+4u]; let b5 = tileB[bk+5u]; let b6 = tileB[bk+6u]; let b7 = tileB[bk+7u];

            r[0]+=a0*b0; r[1]+=a0*b1; r[2]+=a0*b2; r[3]+=a0*b3; r[4]+=a0*b4; r[5]+=a0*b5; r[6]+=a0*b6; r[7]+=a0*b7;
            r[8]+=a1*b0; r[9]+=a1*b1; r[10]+=a1*b2; r[11]+=a1*b3; r[12]+=a1*b4; r[13]+=a1*b5; r[14]+=a1*b6; r[15]+=a1*b7;
            r[16]+=a2*b0; r[17]+=a2*b1; r[18]+=a2*b2; r[19]+=a2*b3; r[20]+=a2*b4; r[21]+=a2*b5; r[22]+=a2*b6; r[23]+=a2*b7;
            r[24]+=a3*b0; r[25]+=a3*b1; r[26]+=a3*b2; r[27]+=a3*b3; r[28]+=a3*b4; r[29]+=a3*b5; r[30]+=a3*b6; r[31]+=a3*b7;
            r[32]+=a4*b0; r[33]+=a4*b1; r[34]+=a4*b2; r[35]+=a4*b3; r[36]+=a4*b4; r[37]+=a4*b5; r[38]+=a4*b6; r[39]+=a4*b7;
            r[40]+=a5*b0; r[41]+=a5*b1; r[42]+=a5*b2; r[43]+=a5*b3; r[44]+=a5*b4; r[45]+=a5*b5; r[46]+=a5*b6; r[47]+=a5*b7;
            r[48]+=a6*b0; r[49]+=a6*b1; r[50]+=a6*b2; r[51]+=a6*b3; r[52]+=a6*b4; r[53]+=a6*b5; r[54]+=a6*b6; r[55]+=a6*b7;
            r[56]+=a7*b0; r[57]+=a7*b1; r[58]+=a7*b2; r[59]+=a7*b3; r[60]+=a7*b4; r[61]+=a7*b5; r[62]+=a7*b6; r[63]+=a7*b7;
        }

        workgroupBarrier();
    }

    for (var rm = 0u; rm < TM; rm++) {
        for (var rn = 0u; rn < TN; rn++) {
            let row = block_row + lr * TM + rm;
            let col = block_col + lc * TN + rn;
            if row < params.M && col < params.N {
                C[row * params.N + col] = r[rm * TN + rn];
            }
        }
    }
}
"""


def make_params(M, N, K):
    return struct.pack('III', M, N, K)


def setup(device, shader, M, N, K, A, B, BM=128, BN=128):
    buf_a = device.create_buffer_with_data(data=A.tobytes(), usage=wgpu.BufferUsage.STORAGE)
    buf_b = device.create_buffer_with_data(data=B.tobytes(), usage=wgpu.BufferUsage.STORAGE)
    buf_c = device.create_buffer(size=M*N*4, usage=wgpu.BufferUsage.STORAGE|wgpu.BufferUsage.COPY_SRC)
    buf_p = device.create_buffer_with_data(data=make_params(M,N,K), usage=wgpu.BufferUsage.UNIFORM)
    sm = device.create_shader_module(code=shader)
    bgl = device.create_bind_group_layout(entries=[
        {"binding":0,"visibility":wgpu.ShaderStage.COMPUTE,"buffer":{"type":wgpu.BufferBindingType.read_only_storage}},
        {"binding":1,"visibility":wgpu.ShaderStage.COMPUTE,"buffer":{"type":wgpu.BufferBindingType.read_only_storage}},
        {"binding":2,"visibility":wgpu.ShaderStage.COMPUTE,"buffer":{"type":wgpu.BufferBindingType.storage}},
        {"binding":3,"visibility":wgpu.ShaderStage.COMPUTE,"buffer":{"type":wgpu.BufferBindingType.uniform}},
    ])
    pl = device.create_pipeline_layout(bind_group_layouts=[bgl])
    pipe = device.create_compute_pipeline(layout=pl, compute={"module":sm,"entry_point":"main"})
    bg = device.create_bind_group(layout=bgl, entries=[
        {"binding":0,"resource":{"buffer":buf_a,"offset":0,"size":buf_a.size}},
        {"binding":1,"resource":{"buffer":buf_b,"offset":0,"size":buf_b.size}},
        {"binding":2,"resource":{"buffer":buf_c,"offset":0,"size":buf_c.size}},
        {"binding":3,"resource":{"buffer":buf_p,"offset":0,"size":buf_p.size}},
    ])
    return {"pipeline":pipe,"bind_group":bg,"buf_c":buf_c,
            "wg_x":(M+BM-1)//BM,"wg_y":(N+BN-1)//BN,
            "bufs":[buf_a,buf_b,buf_c,buf_p]}


def run(device, ctx):
    enc = device.create_command_encoder()
    p = enc.begin_compute_pass()
    p.set_pipeline(ctx["pipeline"])
    p.set_bind_group(0, ctx["bind_group"])
    p.dispatch_workgroups(ctx["wg_x"], ctx["wg_y"])
    p.end()
    device.queue.submit([enc.finish()])
    return device.queue.read_buffer(ctx["buf_c"])


def bench(device, ctx, n):
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
    return (t1-t0)/n


def main():
    adapter = wgpu.gpu.request_adapter_sync(power_preference="high-performance")
    device = adapter.request_device_sync()
    print(f"GPU: {adapter.info.get('device','?')}")

    for sz in [128, 256, 512, 1024]:
        rng = np.random.default_rng(42)
        A = rng.uniform(-1, 1, (sz, sz)).astype(np.float32)
        B = rng.uniform(-1, 1, (sz, sz)).astype(np.float32)
        ctx = setup(device, SHADER, sz, sz, sz, A, B)
        raw = run(device, ctx)
        gpu = np.frombuffer(raw, dtype=np.float32).reshape(sz, sz)
        exp = A @ B
        rel = np.max(np.abs(gpu-exp))/(np.max(np.abs(exp))+1e-8)
        print(f"  {sz}x{sz}: rel={rel:.8f} {'PASS' if rel<0.001 else 'FAIL'}")
        for b in ctx["bufs"]: b.destroy()

    print()
    for M in [2048, 4096]:
        rng = np.random.default_rng(42)
        A = rng.uniform(-1, 1, (M, M)).astype(np.float32)
        B = rng.uniform(-1, 1, (M, M)).astype(np.float32)
        ctx = setup(device, SHADER, M, M, M, A, B)
        for _ in range(WARMUP): run(device, ctx)
        times = [bench(device, ctx, 50) for _ in range(BENCH)]
        for b in ctx["bufs"]: b.destroy()
        med = np.median(times)
        gflops = 2*M*M*M/med/1e9
        print(f"  {M}x{M}: {med*1000:.2f} ms, {gflops:.1f} GFLOPS ({gflops/69200*100:.1f}%)")


if __name__ == "__main__":
    main()
