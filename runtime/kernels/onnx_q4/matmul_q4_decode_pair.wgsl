// @meta bindings=8
requires packed_4x8_integer_dot_product;
enable subgroups;

// Paired Q4 decode projection for gate/up matrices sharing one activation.
// Quantizes X once and computes matching columns from both matrices.
@group(0) @binding(0) var<storage, read> X: array<f32>;
@group(0) @binding(1) var<storage, read> B0: array<u32>;
@group(0) @binding(2) var<storage, read> S0: array<u32>;
@group(0) @binding(3) var<storage, read_write> Y0: array<f32>;
@group(0) @binding(4) var<storage, read> P: array<u32>;
@group(0) @binding(5) var<storage, read> B1: array<u32>;
@group(0) @binding(6) var<storage, read> S1: array<u32>;
@group(0) @binding(7) var<storage, read_write> Y1: array<f32>;

const BK: u32 = 256u;
const COLS_PER_WARP: u32 = 2u;
var<workgroup> xq: array<u32, 64>;
var<workgroup> xs: array<f32, 8>;

fn scale_at(s: ptr<storage, array<u32>, read>, index: u32) -> f32 {
    let pair = unpack2x16float((*s)[index >> 1u]);
    return select(pair.x, pair.y, (index & 1u) != 0u);
}

fn pack4(a: u32, b: u32, c: u32, d: u32) -> u32 {
    return ((a - 8u) & 255u) | (((b - 8u) & 255u) << 8u) |
           (((c - 8u) & 255u) << 16u) | (((d - 8u) & 255u) << 24u);
}
fn q4_lo(p: u32) -> u32 {
    return pack4(p & 15u, (p >> 4u) & 15u,
                 (p >> 8u) & 15u, (p >> 12u) & 15u);
}
fn q4_hi(p: u32) -> u32 {
    return pack4((p >> 16u) & 15u, (p >> 20u) & 15u,
                 (p >> 24u) & 15u, (p >> 28u) & 15u);
}

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) lid: vec3<u32>,
        @builtin(workgroup_id) wid: vec3<u32>) {
    let N = P[1]; let K = P[2]; let tid = lid.x;
    let warp = tid / 32u; let lane = tid & 31u;
    let blocks = K / 32u; let words = K / 8u;
    var cols: array<u32, COLS_PER_WARP>;
    var valid: array<bool, COLS_PER_WARP>;
    var a0: array<f32, COLS_PER_WARP>;
    var a1: array<f32, COLS_PER_WARP>;
    for (var c=0u; c<COLS_PER_WARP; c++) {
        cols[c] = wid.x * (8u * COLS_PER_WARP) + warp * COLS_PER_WARP + c;
        valid[c] = cols[c] < N; a0[c] = 0.0; a1[c] = 0.0;
    }
    let block = tid / 32u; let e = tid & 31u;
    let byte = e & 3u; let pack = e / 4u;

    for (var g=0u; g<K/BK; g++) {
        let xv = X[g*BK + tid];
        var mx = abs(xv);
        mx=max(mx,subgroupShuffleXor(mx,16u)); mx=max(mx,subgroupShuffleXor(mx,8u));
        mx=max(mx,subgroupShuffleXor(mx,4u)); mx=max(mx,subgroupShuffleXor(mx,2u));
        mx=max(mx,subgroupShuffleXor(mx,1u));
        let sx=mx/127.0; if(e==0u){xs[block]=sx;}
        let safe=select(1.0,sx,sx!=0.0);
        let qi=clamp(i32(round(xv/safe)),-127,127);
        var pq=u32(qi&255)<<(byte*8u);
        pq|=subgroupShuffleXor(pq,1u); pq|=subgroupShuffleXor(pq,2u);
        if(byte==0u){xq[block*8u+pack]=pq;}
        workgroupBarrier();

        let xb=lane/4u; let x0=xq[lane*2u]; let x1=xq[lane*2u+1u];
        for(var c=0u;c<COLS_PER_WARP;c++){
            if(valid[c]){
                let n=cols[c]; let off=n*words+g*32u+lane;
                let w0=B0[off]; let w1=B1[off];
                let d0=dot4I8Packed(x0,q4_lo(w0))+dot4I8Packed(x1,q4_hi(w0));
                let d1=dot4I8Packed(x0,q4_lo(w1))+dot4I8Packed(x1,q4_hi(w1));
                let si=n*blocks+g*8u+xb;
                a0[c]+=f32(d0)*xs[xb]*scale_at(&S0,si);
                a1[c]+=f32(d1)*xs[xb]*scale_at(&S1,si);
            }
        }
        workgroupBarrier();
    }
    for(var c=0u;c<COLS_PER_WARP;c++){
        var v0=a0[c]; var v1=a1[c];
        v0+=subgroupShuffleXor(v0,16u);v1+=subgroupShuffleXor(v1,16u);
        v0+=subgroupShuffleXor(v0,8u);v1+=subgroupShuffleXor(v1,8u);
        v0+=subgroupShuffleXor(v0,4u);v1+=subgroupShuffleXor(v1,4u);
        v0+=subgroupShuffleXor(v0,2u);v1+=subgroupShuffleXor(v1,2u);
        v0+=subgroupShuffleXor(v0,1u);v1+=subgroupShuffleXor(v1,1u);
        if(lane==0u&&valid[c]){Y0[cols[c]]=v0;Y1[cols[c]]=v1;}
    }
}
