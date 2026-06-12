"""Optimized matmul: vectorized global loads, transposed tileA, 4x4 register tile."""
import sys
import time
import struct
import numpy as np

try:
    import wgpu
except ImportError:
    sys.exit(1)

# Vectorized global loads using vec4<f32> where possible
# Transposed tileA for bank-conflict-free reads
SHADER = """
const WG: u32 = 16u;
const TM: u32 = 4u;
const TN: u32 = 4u;
const BM: u32 = 64u;
const BN: u32 = 64u;
const BK: u32 = 16u;

struct Params { M: u32, N: u32, K: u32, }

@group(0) @binding(0) var<storage, read> A: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> B: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> C: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;

var<workgroup> tileA: array<f32, 1024>;  // stored transposed: [BK][BM]
var<workgroup> tileB: array<f32, 1024>;  // [BK][BN]

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

    var r0: f32=0.0; var r1: f32=0.0; var r2: f32=0.0; var r3: f32=0.0;
    var r4: f32=0.0; var r5: f32=0.0; var r6: f32=0.0; var r7: f32=0.0;
    var r8: f32=0.0; var r9: f32=0.0; var r10: f32=0.0; var r11: f32=0.0;
    var r12: f32=0.0; var r13: f32=0.0; var r14: f32=0.0; var r15: f32=0.0;

    let K4 = params.K / 4u;
    let N4 = params.N / 4u;
    let numTiles = (params.K + BK - 1u) / BK;

    for (var t: u32 = 0u; t < numTiles; t++) {
        let tile_k = t * BK;

        // Load A: 64 rows × 16 cols = 1024 elements = 256 vec4s
        // Each thread loads 1 vec4 (4 consecutive cols) per pass
        // tid maps to: row = tid / 4, col_group = tid % 4 (each group = 4 cols)
        // Need 64*16/256/4 = 1 vec4 per thread... no, 1024/4 = 256 vec4s = 1 per thread
        // But cols are along K which may not be aligned to vec4
        // Simpler: load scalar, 4 per thread
        for (var load = 0u; load < 4u; load++) {
            let flat = load * 256u + tid;
            let a_r = flat / BK;
            let a_c = flat % BK;
            let gr = block_row + a_r;
            let gc = tile_k + a_c;
            if gr < params.M && gc < params.K {
                // A stored as vec4 along K dimension
                let idx = gr * K4 + gc / 4u;
                let comp = gc % 4u;
                let v = A[idx];
                if comp == 0u { tileA[a_c * BM + a_r] = v.x; }
                else if comp == 1u { tileA[a_c * BM + a_r] = v.y; }
                else if comp == 2u { tileA[a_c * BM + a_r] = v.z; }
                else { tileA[a_c * BM + a_r] = v.w; }
            } else {
                tileA[a_c * BM + a_r] = 0.0;
            }
        }

        // Load B: 16 rows × 64 cols = 1024 elements
        // B cols are along N, load as vec4 along N
        // 1024 / 4 = 256 vec4s = 1 per thread
        {
            let flat = tid;
            let b_r = flat / 16u;  // 256/16 = 16 rows covered
            let b_c4 = flat % 16u; // 16 vec4s per row
            let gr = tile_k + b_r;
            let gc4 = block_col / 4u + b_c4;
            if gr < params.K && (block_col + b_c4 * 4u + 3u) < params.N {
                let v = B[gr * N4 + gc4];
                let base = b_r * BN + b_c4 * 4u;
                tileB[base] = v.x;
                tileB[base + 1u] = v.y;
                tileB[base + 2u] = v.z;
                tileB[base + 3u] = v.w;
            } else {
                let base_gc = block_col + b_c4 * 4u;
                for (var i = 0u; i < 4u; i++) {
                    let gc = base_gc + i;
                    let base = b_r * BN + b_c4 * 4u + i;
                    if gr < params.K && gc < params.N {
                        // scalar fallback
                        let arr_idx = gr * params.N + gc;
                        let v = A[arr_idx / 4u]; // wrong buffer, use scalar path
                        tileB[base] = 0.0;
                    } else {
                        tileB[base] = 0.0;
                    }
                }
            }
        }

        workgroupBarrier();

        let a_base = lr * TM;
        let b_base = lc * TN;

        for (var k: u32 = 0u; k < BK; k++) {
            let ak = k * BM + a_base;
            let bk = k * BN + b_base;
            let a0 = tileA[ak]; let a1 = tileA[ak+1u]; let a2 = tileA[ak+2u]; let a3 = tileA[ak+3u];
            let b0 = tileB[bk]; let b1 = tileB[bk+1u]; let b2 = tileB[bk+2u]; let b3 = tileB[bk+3u];
            r0+=a0*b0; r1+=a0*b1; r2+=a0*b2; r3+=a0*b3;
            r4+=a1*b0; r5+=a1*b1; r6+=a1*b2; r7+=a1*b3;
            r8+=a2*b0; r9+=a2*b1; r10+=a2*b2; r11+=a2*b3;
            r12+=a3*b0; r13+=a3*b1; r14+=a3*b2; r15+=a3*b3;
        }

        workgroupBarrier();
    }

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

# Simpler version: f32 arrays with scalar loads, transposed tileA, unrolled inner loop
# This is our known-good 14% baseline
SHADER_SIMPLE = """
const WG: u32 = 16u;
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
    let tid = lr * WG + lc;

    let block_row = wgid.x * BM;
    let block_col = wgid.y * BN;

    var r0: f32=0.0; var r1: f32=0.0; var r2: f32=0.0; var r3: f32=0.0;
    var r4: f32=0.0; var r5: f32=0.0; var r6: f32=0.0; var r7: f32=0.0;
    var r8: f32=0.0; var r9: f32=0.0; var r10: f32=0.0; var r11: f32=0.0;
    var r12: f32=0.0; var r13: f32=0.0; var r14: f32=0.0; var r15: f32=0.0;

    let numTiles = (params.K + BK - 1u) / BK;

    for (var t: u32 = 0u; t < numTiles; t++) {
        let tile_k = t * BK;

        for (var load = 0u; load < 4u; load++) {
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

        for (var load = 0u; load < 4u; load++) {
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
            let b0 = tileB[bk]; let b1 = tileB[bk+1u]; let b2 = tileB[bk+2u]; let b3 = tileB[bk+3u];
            r0+=a0*b0; r1+=a0*b1; r2+=a0*b2; r3+=a0*b3;
            r4+=a1*b0; r5+=a1*b1; r6+=a1*b2; r7+=a1*b3;
            r8+=a2*b0; r9+=a2*b1; r10+=a2*b2; r11+=a2*b3;
            r12+=a3*b0; r13+=a3*b1; r14+=a3*b2; r15+=a3*b3;
        }

        workgroupBarrier();
    }

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


# Double-buffered version: overlap loading next tile with computing current tile
SHADER_DOUBLE_BUF = """
const WG: u32 = 16u;
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

// Double buffer: 2 sets of tiles
var<workgroup> tileA: array<f32, 2048>;  // [2][BK*BM]
var<workgroup> tileB: array<f32, 2048>;  // [2][BK*BN]

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

    var r0: f32=0.0; var r1: f32=0.0; var r2: f32=0.0; var r3: f32=0.0;
    var r4: f32=0.0; var r5: f32=0.0; var r6: f32=0.0; var r7: f32=0.0;
    var r8: f32=0.0; var r9: f32=0.0; var r10: f32=0.0; var r11: f32=0.0;
    var r12: f32=0.0; var r13: f32=0.0; var r14: f32=0.0; var r15: f32=0.0;

    let numTiles = (params.K + BK - 1u) / BK;

    // Load first tile into buffer 0
    {
        let tile_k = 0u;
        for (var load = 0u; load < 4u; load++) {
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
        for (var load = 0u; load < 4u; load++) {
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
    }

    workgroupBarrier();

    for (var t: u32 = 0u; t < numTiles; t++) {
        let cur = (t % 2u) * 1024u;
        let nxt = ((t + 1u) % 2u) * 1024u;

        // Load next tile into alternate buffer (if there is one)
        if t + 1u < numTiles {
            let tile_k = (t + 1u) * BK;
            for (var load = 0u; load < 4u; load++) {
                let flat = load * 256u + tid;
                let a_r = flat / BK;
                let a_c = flat % BK;
                let gr = block_row + a_r;
                let gc = tile_k + a_c;
                if gr < params.M && gc < params.K {
                    tileA[nxt + a_c * BM + a_r] = A[gr * params.K + gc];
                } else {
                    tileA[nxt + a_c * BM + a_r] = 0.0;
                }
            }
            for (var load = 0u; load < 4u; load++) {
                let flat = load * 256u + tid;
                let b_r = flat / BN;
                let b_c = flat % BN;
                let gr = tile_k + b_r;
                let gc = block_col + b_c;
                if gr < params.K && gc < params.N {
                    tileB[nxt + b_r * BN + b_c] = B[gr * params.N + gc];
                } else {
                    tileB[nxt + b_r * BN + b_c] = 0.0;
                }
            }
        }

        // Compute on current tile
        let a_base = cur + lr * TM;
        let b_base = cur;

        for (var k: u32 = 0u; k < BK; k++) {
            let ak = cur + k * BM + lr * TM;
            let bk = cur + k * BN + lc * TN;
            let a0 = tileA[ak]; let a1 = tileA[ak+1u]; let a2 = tileA[ak+2u]; let a3 = tileA[ak+3u];
            let b0 = tileB[bk]; let b1 = tileB[bk+1u]; let b2 = tileB[bk+2u]; let b3 = tileB[bk+3u];
            r0+=a0*b0; r1+=a0*b1; r2+=a0*b2; r3+=a0*b3;
            r4+=a1*b0; r5+=a1*b1; r6+=a1*b2; r7+=a1*b3;
            r8+=a2*b0; r9+=a2*b1; r10+=a2*b2; r11+=a2*b3;
            r12+=a3*b0; r13+=a3*b1; r14+=a3*b2; r15+=a3*b3;
        }

        workgroupBarrier();
    }

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


def make_params(M, N, K):
    return struct.pack('III', M, N, K)

BM, BN = 64, 64

def setup(device, shader, M, N, K, A, B):
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


def test_variant(device, name, shader, M=4096):
    rng = np.random.default_rng(42)
    A = rng.uniform(-1, 1, (M, M)).astype(np.float32)
    B = rng.uniform(-1, 1, (M, M)).astype(np.float32)

    # Correctness at small size
    sz = 256
    A_s = rng.uniform(-1, 1, (sz, sz)).astype(np.float32)
    B_s = rng.uniform(-1, 1, (sz, sz)).astype(np.float32)
    ctx = setup(device, shader, sz, sz, sz, A_s, B_s)
    raw = run(device, ctx)
    gpu = np.frombuffer(raw, dtype=np.float32).reshape(sz, sz)
    exp = A_s @ B_s
    rel = np.max(np.abs(gpu-exp))/(np.max(np.abs(exp))+1e-8)
    ok = "PASS" if rel < 0.001 else "FAIL"
    for b in ctx["bufs"]: b.destroy()

    # Performance
    ctx = setup(device, shader, M, M, M, A, B)
    for _ in range(3): run(device, ctx)
    times = [bench(device, ctx, 50) for _ in range(5)]
    for b in ctx["bufs"]: b.destroy()
    med = np.median(times)
    gflops = 2*M*M*M/med/1e9
    print(f"  {name:30s}  {ok}  {med*1000:8.2f} ms  {gflops:8.1f} GFLOPS  {gflops/69200*100:5.1f}%")


def main():
    adapter = wgpu.gpu.request_adapter_sync(power_preference="high-performance")
    device = adapter.request_device_sync()
    print(f"GPU: {adapter.info.get('device','?')}\n")

    test_variant(device, "baseline (transposed)", SHADER_SIMPLE)
    test_variant(device, "double-buffered", SHADER_DOUBLE_BUF)


if __name__ == "__main__":
    main()
