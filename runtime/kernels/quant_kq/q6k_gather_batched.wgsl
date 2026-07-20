@group(0) @binding(0) var<storage, read> W: array<u32>;
@group(0) @binding(1) var<storage, read> Tokens: array<i32>;
@group(0) @binding(2) var<storage, read_write> X: array<f32>;
@group(0) @binding(3) var<storage, read> P: array<u32>;

fn u8_at(a:u32)->u32{return(W[a/4u]>>((a&3u)*8u))&255u;}
fn i8_at(a:u32)->i32{let v=u8_at(a);return select(i32(v),i32(v)-256,v>=128u);}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid:vec3<u32>){
 let idx=gid.x;let M=P[0];let K=P[1];if(idx>=M*K){return;}
 let m=idx/K;let k=idx-m*K;let rs=P[2]*4u;let token=u32(max(Tokens[m],0));
 let local=k&255u;let g=local/128u;let r=local&127u;let qtr=r/32u;let lane=r&31u;
 let base=token*rs+(k/256u)*210u;let qlb=base+g*64u;
 let qh=u8_at(base+128u+g*32u+lane);
 let ql=u8_at(qlb+select(lane,32u+lane,qtr==1u||qtr==3u));
 let low=select(ql&15u,ql>>4u,qtr>=2u);let q=i32(low|(((qh>>(qtr*2u))&3u)<<4u))-32;
 let sc=i8_at(base+192u+g*8u+lane/16u+qtr*2u);
 let dh=u8_at(base+208u)|(u8_at(base+209u)<<8u);
 X[idx]=unpack2x16float(dh).x*f32(sc)*f32(q);
}
