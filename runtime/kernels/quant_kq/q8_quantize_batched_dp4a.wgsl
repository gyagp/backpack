// @meta bindings=4 generated=false registry=q8_quantize_batched_dp4a
requires packed_4x8_integer_dot_product;
enable subgroups;
@group(0) @binding(0) var<storage,read>X:array<f32>;
@group(0) @binding(1) var<storage,read_write>XQ:array<u32>;
@group(0) @binding(2) var<storage,read_write>XS:array<f32>;
@group(0) @binding(3) var<storage,read>P:array<u32>;
@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id)lid:vec3<u32>,@builtin(workgroup_id)wid:vec3<u32>){
 let tid=lid.x;let lane=tid&31u;let block32=tid/32u;let packLane=lane&3u;let packGroup=lane/4u;
 let K=P[0];let M=P[2];let row=wid.y;if(row>=M){return;}
 let k=wid.x*256u+tid;let xv=select(0.0,X[row*K+k],k<K);var amax=abs(xv);
 amax=max(amax,subgroupShuffleXor(amax,16u));amax=max(amax,subgroupShuffleXor(amax,8u));
 amax=max(amax,subgroupShuffleXor(amax,4u));amax=max(amax,subgroupShuffleXor(amax,2u));
 amax=max(amax,subgroupShuffleXor(amax,1u));let scale=amax/127.0;
 let blocks32=(K+31u)/32u;if(lane==0u){XS[row*blocks32+wid.x*8u+block32]=scale;}
 let safe=select(1.0,scale,scale!=0.0);let qi=clamp(i32(round(xv/safe)),-127,127);
 var packed=u32(qi&255)<<(packLane*8u);packed|=subgroupShuffleXor(packed,1u);packed|=subgroupShuffleXor(packed,2u);
 if(packLane==0u){XQ[row*((K+3u)/4u)+wid.x*64u+block32*8u+packGroup]=packed;}
}
