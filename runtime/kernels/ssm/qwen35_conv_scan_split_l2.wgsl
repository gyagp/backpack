// @meta bindings=8
enable subgroups;

// Fused Qwen 3.5 prefill convolution scan, SiLU, Q/K split+L2 norm.
// Grid: (3, max(nk,nv), 1). Each workgroup scans T in causal order.
@group(0) @binding(0) var<storage,read_write> State:array<f32>;
@group(0) @binding(1) var<storage,read> X:array<f32>;
@group(0) @binding(2) var<storage,read> CW:array<f32>;
@group(0) @binding(3) var<storage,read> Bias:array<f32>;
@group(0) @binding(4) var<storage,read_write> Q:array<f32>;
@group(0) @binding(5) var<storage,read_write> K:array<f32>;
@group(0) @binding(6) var<storage,read_write> V:array<f32>;
@group(0) @binding(7) var<storage,read> P:array<u32>;
var<workgroup>sums:array<f32,4>;
@compute @workgroup_size(128)
fn main(@builtin(workgroup_id)wid:vec3<u32>,@builtin(local_invocation_id)lid:vec3<u32>){
 let nk=P[0];let nv=P[1];let dk=P[2];let dv=P[3];let convK=P[4];let eps=bitcast<f32>(P[5]);let T=P[6];
 let kind=wid.x;let h=wid.y;let d=lid.x;let dim=select(dk,dv,kind==2u);let heads=select(nk,nv,kind==2u);
 if(h>=heads){return;}let qsize=nk*dk;let coff=select(select(0u,qsize,kind==1u),2u*qsize,kind==2u);let channels=2u*qsize+nv*dv;let c=coff+h*dim+d;let sb=c*convK;
 for(var t=0u;t<T;t++){
  var a=Bias[c];for(var j=0u;j+1u<convK;j++){let old=State[sb+j+1u];State[sb+j]=old;a+=CW[sb+j]*old;}
  let newest=X[t*channels+c];State[sb+convK-1u]=newest;a+=CW[sb+convK-1u]*newest;let value=a/(1.0+exp(-a));
  if(kind==2u){V[(t*nv+h)*dv+d]=value;}else{let ss=subgroupAdd(value*value);if((d&31u)==0u){sums[d/32u]=ss;}workgroupBarrier();let inv=1.0/max(sqrt(sums[0]+sums[1]+sums[2]+sums[3]),eps);let out=(t*nk+h)*dk+d;if(kind==0u){Q[out]=value*inv;}else{K[out]=value*inv;}workgroupBarrier();}
 }
}
