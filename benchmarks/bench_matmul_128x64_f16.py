"""128x64 tile with f16 inputs, transposed tileA, 8x4 register tile."""
import sys, time, struct
import numpy as np
try:
    import wgpu
except ImportError:
    sys.exit(1)

SHADER = """
const WG: u32 = 16u;
const TM: u32 = 8u;
const TN: u32 = 4u;
const BM: u32 = 128u;
const BN: u32 = 64u;
const BK: u32 = 16u;

struct Params { M: u32, N: u32, K: u32, }

@group(0) @binding(0) var<storage, read> A: array<u32>;
@group(0) @binding(1) var<storage, read> B: array<u32>;
@group(0) @binding(2) var<storage, read_write> C: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;

var<workgroup> tileA: array<f32, 2048>;  // transposed [BK][BM] = 16*128
var<workgroup> tileB: array<f32, 1024>;  // [BK][BN] = 16*64

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
    let tid = lr * WG + lc;
    let block_row = wgid.x * BM;
    let block_col = wgid.y * BN;

    var r0:f32=0.0;var r1:f32=0.0;var r2:f32=0.0;var r3:f32=0.0;
    var r4:f32=0.0;var r5:f32=0.0;var r6:f32=0.0;var r7:f32=0.0;
    var r8:f32=0.0;var r9:f32=0.0;var r10:f32=0.0;var r11:f32=0.0;
    var r12:f32=0.0;var r13:f32=0.0;var r14:f32=0.0;var r15:f32=0.0;
    var r16:f32=0.0;var r17:f32=0.0;var r18:f32=0.0;var r19:f32=0.0;
    var r20:f32=0.0;var r21:f32=0.0;var r22:f32=0.0;var r23:f32=0.0;
    var r24:f32=0.0;var r25:f32=0.0;var r26:f32=0.0;var r27:f32=0.0;
    var r28:f32=0.0;var r29:f32=0.0;var r30:f32=0.0;var r31:f32=0.0;

    let numTiles = (params.K + BK - 1u) / BK;

    for (var t: u32 = 0u; t < numTiles; t++) {
        let tile_k = t * BK;

        // Load tileA transposed: 128*16 = 2048 / 256 = 8 per thread
        for (var load = 0u; load < 8u; load++) {
            let flat = load * 256u + tid;
            let a_r = flat / BK;
            let a_c = flat % BK;
            let gr = block_row + a_r;
            let gc = tile_k + a_c;
            if gr < params.M && gc < params.K {
                tileA[a_c * BM + a_r] = load_f16_A(gr * params.K + gc);
            } else {
                tileA[a_c * BM + a_r] = 0.0;
            }
        }

        // Load tileB: 16*64 = 1024 / 256 = 4 per thread
        for (var load = 0u; load < 4u; load++) {
            let flat = load * 256u + tid;
            let b_r = flat / BN;
            let b_c = flat % BN;
            let gr = tile_k + b_r;
            let gc = block_col + b_c;
            if gr < params.K && gc < params.N {
                tileB[b_r * BN + b_c] = load_f16_B(gr * params.N + gc);
            } else {
                tileB[b_r * BN + b_c] = 0.0;
            }
        }

        workgroupBarrier();

        for (var k: u32 = 0u; k < BK; k++) {
            let ak = k * BM + lr * TM;
            let bk = k * BN + lc * TN;
            let a0=tileA[ak];let a1=tileA[ak+1u];let a2=tileA[ak+2u];let a3=tileA[ak+3u];
            let a4=tileA[ak+4u];let a5=tileA[ak+5u];let a6=tileA[ak+6u];let a7=tileA[ak+7u];
            let b0=tileB[bk];let b1=tileB[bk+1u];let b2=tileB[bk+2u];let b3=tileB[bk+3u];
            r0+=a0*b0;r1+=a0*b1;r2+=a0*b2;r3+=a0*b3;
            r4+=a1*b0;r5+=a1*b1;r6+=a1*b2;r7+=a1*b3;
            r8+=a2*b0;r9+=a2*b1;r10+=a2*b2;r11+=a2*b3;
            r12+=a3*b0;r13+=a3*b1;r14+=a3*b2;r15+=a3*b3;
            r16+=a4*b0;r17+=a4*b1;r18+=a4*b2;r19+=a4*b3;
            r20+=a5*b0;r21+=a5*b1;r22+=a5*b2;r23+=a5*b3;
            r24+=a6*b0;r25+=a6*b1;r26+=a6*b2;r27+=a6*b3;
            r28+=a7*b0;r29+=a7*b1;r30+=a7*b2;r31+=a7*b3;
        }

        workgroupBarrier();
    }

    var results: array<f32, 32>;
    results[0]=r0;results[1]=r1;results[2]=r2;results[3]=r3;
    results[4]=r4;results[5]=r5;results[6]=r6;results[7]=r7;
    results[8]=r8;results[9]=r9;results[10]=r10;results[11]=r11;
    results[12]=r12;results[13]=r13;results[14]=r14;results[15]=r15;
    results[16]=r16;results[17]=r17;results[18]=r18;results[19]=r19;
    results[20]=r20;results[21]=r21;results[22]=r22;results[23]=r23;
    results[24]=r24;results[25]=r25;results[26]=r26;results[27]=r27;
    results[28]=r28;results[29]=r29;results[30]=r30;results[31]=r31;

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

BM, BN = 128, 64

def pack_f16_to_u32(arr):
    flat = arr.flatten().astype(np.float16)
    raw = flat.view(np.uint16).astype(np.uint32)
    if len(raw) % 2: raw = np.append(raw, np.uint32(0))
    return ((raw[1::2] << 16) | raw[0::2]).astype(np.uint32).tobytes()

def make_params(M,N,K): return struct.pack('III',M,N,K)

def setup(device, M, N, K, A_bytes, B_bytes):
    buf_a=device.create_buffer_with_data(data=A_bytes,usage=wgpu.BufferUsage.STORAGE)
    buf_b=device.create_buffer_with_data(data=B_bytes,usage=wgpu.BufferUsage.STORAGE)
    buf_c=device.create_buffer(size=M*N*4,usage=wgpu.BufferUsage.STORAGE|wgpu.BufferUsage.COPY_SRC)
    buf_p=device.create_buffer_with_data(data=make_params(M,N,K),usage=wgpu.BufferUsage.UNIFORM)
    sm=device.create_shader_module(code=SHADER)
    bgl=device.create_bind_group_layout(entries=[
        {"binding":0,"visibility":wgpu.ShaderStage.COMPUTE,"buffer":{"type":wgpu.BufferBindingType.read_only_storage}},
        {"binding":1,"visibility":wgpu.ShaderStage.COMPUTE,"buffer":{"type":wgpu.BufferBindingType.read_only_storage}},
        {"binding":2,"visibility":wgpu.ShaderStage.COMPUTE,"buffer":{"type":wgpu.BufferBindingType.storage}},
        {"binding":3,"visibility":wgpu.ShaderStage.COMPUTE,"buffer":{"type":wgpu.BufferBindingType.uniform}},
    ])
    pl=device.create_pipeline_layout(bind_group_layouts=[bgl])
    pipe=device.create_compute_pipeline(layout=pl,compute={"module":sm,"entry_point":"main"})
    bg=device.create_bind_group(layout=bgl,entries=[
        {"binding":0,"resource":{"buffer":buf_a,"offset":0,"size":buf_a.size}},
        {"binding":1,"resource":{"buffer":buf_b,"offset":0,"size":buf_b.size}},
        {"binding":2,"resource":{"buffer":buf_c,"offset":0,"size":buf_c.size}},
        {"binding":3,"resource":{"buffer":buf_p,"offset":0,"size":buf_p.size}},
    ])
    return {"pipeline":pipe,"bind_group":bg,"buf_c":buf_c,
            "wg_x":(M+BM-1)//BM,"wg_y":(N+BN-1)//BN,"bufs":[buf_a,buf_b,buf_c,buf_p]}

def run(device,ctx):
    enc=device.create_command_encoder()
    p=enc.begin_compute_pass();p.set_pipeline(ctx["pipeline"]);p.set_bind_group(0,ctx["bind_group"])
    p.dispatch_workgroups(ctx["wg_x"],ctx["wg_y"]);p.end()
    device.queue.submit([enc.finish()])
    return device.queue.read_buffer(ctx["buf_c"])

def bench(device,ctx,n):
    enc=device.create_command_encoder()
    for _ in range(n):
        p=enc.begin_compute_pass();p.set_pipeline(ctx["pipeline"]);p.set_bind_group(0,ctx["bind_group"])
        p.dispatch_workgroups(ctx["wg_x"],ctx["wg_y"]);p.end()
    t0=time.perf_counter()
    device.queue.submit([enc.finish()])
    _=device.queue.read_buffer(ctx["buf_c"])
    t1=time.perf_counter()
    return (t1-t0)/n

def main():
    adapter=wgpu.gpu.request_adapter_sync(power_preference="high-performance")
    device=adapter.request_device_sync()
    print(f"GPU: {adapter.info.get('device','?')}")

    for sz in [128,256,512]:
        rng=np.random.default_rng(42)
        A=rng.uniform(-1,1,(sz,sz)).astype(np.float16)
        B=rng.uniform(-1,1,(sz,sz)).astype(np.float16)
        ctx=setup(device,sz,sz,sz,pack_f16_to_u32(A),pack_f16_to_u32(B))
        raw=run(device,ctx)
        gpu=np.frombuffer(raw,dtype=np.float32).reshape(sz,sz)
        exp=A.astype(np.float32)@B.astype(np.float32)
        rel=np.max(np.abs(gpu-exp))/(np.max(np.abs(exp))+1e-8)
        print(f"  {sz}x{sz}: rel={rel:.8f} {'PASS' if rel<0.01 else 'FAIL'}")
        for b in ctx["bufs"]: b.destroy()

    print()
    for M in [2048,4096]:
        rng=np.random.default_rng(42)
        A=rng.uniform(-1,1,(M,M)).astype(np.float16)
        B=rng.uniform(-1,1,(M,M)).astype(np.float16)
        ctx=setup(device,M,M,M,pack_f16_to_u32(A),pack_f16_to_u32(B))
        for _ in range(3): run(device,ctx)
        times=[bench(device,ctx,50) for _ in range(5)]
        for b in ctx["bufs"]: b.destroy()
        med=np.median(times)
        gflops=2*M*M*M/med/1e9
        # Use roofline model for peak
        ai_f16 = (BM*BN) / (2*(BM+BN))  # f16 halves bytes
        roofline = min(56000, 726 * ai_f16)
        print(f"  {M}x{M}: {med*1000:.2f}ms {gflops:.1f} GFLOPS  roofline={roofline:.0f} GFLOPS ({gflops/roofline*100:.1f}% of roofline)")

if __name__=="__main__":
    main()
