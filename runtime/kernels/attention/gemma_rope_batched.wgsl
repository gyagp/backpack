// @meta bindings=9
enable f16;
enable subgroups;

// Gemma batched Q/K normalization, RoPE and cache write. Params:
// [n_head, q_dim, kv_dim, pos_offset, half_dim, cache_len, n_kv, flags]
// flags bit 0: Q-only/shared-KV layer, bit 1: normalize V.

@group(0) @binding(0) var<storage, read> QKV: array<f32>;
@group(0) @binding(1) var<storage, read_write> QRot: array<f32>;
@group(0) @binding(2) var<storage, read_write> KCache: array<f16>;
@group(0) @binding(3) var<storage, read_write> VCache: array<f16>;
@group(0) @binding(4) var<storage, read> Cos: array<f32>;
@group(0) @binding(5) var<storage, read> Sin: array<f32>;
@group(0) @binding(6) var<storage, read> QW: array<f32>;
@group(0) @binding(7) var<storage, read> KW: array<f32>;
@group(0) @binding(8) var<storage, read> P: array<u32>;

const HD: u32 = 128u;
var<workgroup> sums: array<f32, 4>;

fn row_rms(base: u32, tid: u32) -> f32 {
    var ss=0.0;
    for(var i=tid;i<HD;i+=128u){let v=QKV[base+i];ss+=v*v;}
    let lane=tid&31u;let warp=tid/32u;let ws=subgroupAdd(ss);
    if(lane==0u){sums[warp]=ws;}
    workgroupBarrier();
    let total=sums[0]+sums[1]+sums[2]+sums[3];
    workgroupBarrier();
    return inverseSqrt(total/f32(HD)+1e-6);
}

@compute @workgroup_size(128)
fn main(@builtin(workgroup_id) wid: vec3<u32>,
        @builtin(local_invocation_id) lid: vec3<u32>) {
    let h=wid.x;let t=wid.y;let tid=lid.x;
    let nh=P[0];let qdim=P[1];let kvdim=P[2];let pos=P[3]+t;
    let half=P[4];let cache_pos=P[5]+t;let nkv=P[6];let flags=P[7];
    let qonly=(flags&1u)!=0u;let vnorm=(flags&2u)!=0u;
    let stride=select(qdim+2u*kvdim,qdim,qonly);
    if(h<nh){
        let src=t*stride+h*HD;let dst=t*qdim+h*HD;
        let r=row_rms(src,tid);let cb=pos*half;
        for(var i=tid;i<half;i+=128u){let j=i+half;
            let a=QKV[src+i]*r*QW[i];let b=QKV[src+j]*r*QW[j];
            let c=Cos[cb+i];let s=Sin[cb+i];QRot[dst+i]=a*c-b*s;QRot[dst+j]=b*c+a*s;
        }
    }else if(!qonly){
        let kh=h-nh;let ks=t*stride+qdim+kh*HD;let vs=ks+kvdim;
        let dst=cache_pos*nkv*HD+kh*HD;let kr=row_rms(ks,tid);let cb=pos*half;
        for(var i=tid;i<half;i+=128u){let j=i+half;
            let a=QKV[ks+i]*kr*KW[i];let b=QKV[ks+j]*kr*KW[j];let c=Cos[cb+i];let s=Sin[cb+i];
            KCache[dst+i]=f16(a*c-b*s);KCache[dst+j]=f16(b*c+a*s);
        }
        var vr=1.0;
        if(vnorm){vr=row_rms(vs,tid);}
        for(var i=tid;i<HD;i+=128u){VCache[dst+i]=f16(QKV[vs+i]*vr);}
    }
}
