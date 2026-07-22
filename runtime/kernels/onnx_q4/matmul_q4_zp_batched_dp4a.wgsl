// @meta bindings=6
requires packed_4x8_integer_dot_product;
enable subgroups;

// Asymmetric ONNX Q4G32 prefill matmul. Four activation rows share each
// packed weight tile; Q4 values are centered by their per-block zero point
// and accumulated with DP4A. Grid: (ceil(N/32), ceil(M/4), 1).
@group(0) @binding(0) var<storage, read> X: array<f32>;
@group(0) @binding(1) var<storage, read> B: array<u32>;
@group(0) @binding(2) var<storage, read> Scales: array<u32>;
@group(0) @binding(3) var<storage, read_write> Y: array<f32>;
@group(0) @binding(4) var<storage, read> P: array<u32>;
@group(0) @binding(5) var<storage, read> ZeroPoints: array<u32>;

const ROWS: u32 = 4u;
const BK: u32 = 256u;
const COLS_PER_WARP: u32 = 4u;
var<workgroup> xq: array<u32, ROWS * 64u>;
var<workgroup> xs: array<f32, ROWS * 8u>;

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) lid: vec3<u32>,
        @builtin(workgroup_id) wid: vec3<u32>) {
    let M=P[0]; let N=P[1]; let K=P[2]; let tid=lid.x;
    let warp=tid/32u; let lane=tid&31u; let row0=wid.y*ROWS;
    let nblocks=K/32u; let words=K/8u;
    var cols: array<u32,COLS_PER_WARP>;
    var valid: array<bool,COLS_PER_WARP>;
    var acc: array<f32,ROWS*COLS_PER_WARP>;
    for(var c=0u;c<COLS_PER_WARP;c++){
        cols[c]=wid.x*32u+warp*COLS_PER_WARP+c; valid[c]=cols[c]<N;
    }
    for(var i=0u;i<ROWS*COLS_PER_WARP;i++){acc[i]=0.0;}
    for(var kb=0u;kb<K;kb+=BK){
        let block=tid/32u; let elem=tid&31u;
        let packlane=elem&3u; let packgroup=elem/4u;
        for(var r=0u;r<ROWS;r++){
            let row=row0+r; let xv=select(0.0,X[row*K+kb+tid],row<M);
            var amax=abs(xv);
            amax=max(amax,subgroupShuffleXor(amax,16u));
            amax=max(amax,subgroupShuffleXor(amax,8u));
            amax=max(amax,subgroupShuffleXor(amax,4u));
            amax=max(amax,subgroupShuffleXor(amax,2u));
            amax=max(amax,subgroupShuffleXor(amax,1u));
            let scale=amax/127.0;
            if(elem==0u){xs[r*8u+block]=scale;}
            let safe=select(1.0,scale,scale!=0.0);
            let q=u32(clamp(i32(round(xv/safe)),-127,127))&255u;
            var packed=q<<(packlane*8u);
            packed|=subgroupShuffleXor(packed,1u);
            packed|=subgroupShuffleXor(packed,2u);
            if(packlane==0u){xq[r*64u+block*8u+packgroup]=packed;}
        }
        workgroupBarrier();
        for(var c=0u;c<COLS_PER_WARP;c++){
            if(valid[c]){
                let col=cols[c]; let q4=B[col*words+kb/8u+lane];
                let xblock=lane/4u; let wb=kb/32u+xblock;
                let si=col*nblocks+wb;
                let zbyte=(ZeroPoints[(si/2u)/4u]>>(((si/2u)%4u)*8u))&255u;
                let zp=select(zbyte&15u,(zbyte>>4u)&15u,(si&1u)!=0u);
                let b0=q4&255u; let b1=(q4>>8u)&255u;
                let b2=(q4>>16u)&255u; let b3=(q4>>24u)&255u;
                let wq0=(((b0&15u)-zp)&255u)|((((b0>>4u)-zp)&255u)<<8u)|
                    ((((b1&15u)-zp)&255u)<<16u)|((((b1>>4u)-zp)&255u)<<24u);
                let wq1=(((b2&15u)-zp)&255u)|((((b2>>4u)-zp)&255u)<<8u)|
                    ((((b3&15u)-zp)&255u)<<16u)|((((b3>>4u)-zp)&255u)<<24u);
                let sp=unpack2x16float(Scales[si/2u]);
                let ws=select(sp.x,sp.y,(si&1u)!=0u);
                for(var r=0u;r<ROWS;r++){
                    let base=r*64u+lane*2u;
                    let dot=dot4I8Packed(xq[base],wq0)+dot4I8Packed(xq[base+1u],wq1);
                    acc[c*ROWS+r]+=f32(dot)*ws*xs[r*8u+xblock];
                }
            }
        }
        workgroupBarrier();
    }
    for(var c=0u;c<COLS_PER_WARP;c++){for(var r=0u;r<ROWS;r++){
        let sum=subgroupAdd(acc[c*ROWS+r]);
        if(lane==0u&&valid[c]&&row0+r<M){Y[(row0+r)*N+cols[c]]=sum;}
    }}
}
