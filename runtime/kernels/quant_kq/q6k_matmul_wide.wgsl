enable subgroups;

@group(0) @binding(0) var<storage, read> X: array<f32>;
@group(0) @binding(1) var<storage, read> W: array<u32>;
@group(0) @binding(2) var<storage, read> Bias: array<f32>;
@group(0) @binding(3) var<storage, read_write> Y: array<f32>;
@group(0) @binding(4) var<storage, read> P: array<u32>;

var<workgroup> sx: array<f32, 256>;

fn u8_at(base_byte: u32, off: u32) -> u32 {
    let a = base_byte + off;
    return (W[a / 4u] >> ((a & 3u) * 8u)) & 255u;
}
fn i8_at(base_byte: u32, off: u32) -> i32 {
    let v=u8_at(base_byte,off);return select(i32(v),i32(v)-256,v>=128u);
}

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) lid: vec3<u32>,
        @builtin(workgroup_id) wid: vec3<u32>) {
    let tid=lid.x;let warp=tid/32u;let lane=tid&31u;
    let K=P[0];let N=P[1];let nb=P[2];let rs=P[3];let yo=P[4];
    let col0=wid.y*16u+warp*2u;
    var acc: array<f32,2>;
    acc[0]=0.0;acc[1]=0.0;
    for(var b=0u;b<nb;b++){
        let ki=b*256u+tid;sx[tid]=select(0.0,X[wid.x*K+ki],ki<K);
        workgroupBarrier();
        for(var c=0u;c<2u;c++){
            let col=col0+c;
            if(col<N){
                let bb=col*rs*4u+b*210u;
                let dh=u8_at(bb,208u)|(u8_at(bb,209u)<<8u);
                let d=unpack2x16float(dh).x;
                for(var g=0u;g<2u;g++){
                    let qlo=g*64u;let qho=128u+g*32u;let sco=192u+g*8u;
                    let ql0=u8_at(bb,qlo+lane);let ql1=u8_at(bb,qlo+32u+lane);
                    let qh=u8_at(bb,qho+lane);let si=lane/16u;
                    let q0=i32((ql0&15u)|(((qh>>0u)&3u)<<4u))-32;
                    let q1=i32((ql1&15u)|(((qh>>2u)&3u)<<4u))-32;
                    let q2=i32((ql0>>4u)|(((qh>>4u)&3u)<<4u))-32;
                    let q3=i32((ql1>>4u)|(((qh>>6u)&3u)<<4u))-32;
                    let lb=g*128u;
                    acc[c]+=sx[lb+lane]*d*f32(i8_at(bb,sco+si))*f32(q0);
                    acc[c]+=sx[lb+32u+lane]*d*f32(i8_at(bb,sco+si+2u))*f32(q1);
                    acc[c]+=sx[lb+64u+lane]*d*f32(i8_at(bb,sco+si+4u))*f32(q2);
                    acc[c]+=sx[lb+96u+lane]*d*f32(i8_at(bb,sco+si+6u))*f32(q3);
                }
            }
        }
        workgroupBarrier();
    }
    for(var c=0u;c<2u;c++){
        let col=col0+c;let sum=subgroupAdd(acc[c]);
        if(lane==0u&&col<N){Y[wid.x*N+col+yo]=sum+Bias[col];}
    }
}
