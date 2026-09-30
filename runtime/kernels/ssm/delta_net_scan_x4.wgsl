// @meta bindings=8 generated=false registry=delta_net_scan_x4
enable subgroups;
@group(0) @binding(0)var<storage,read>Q:array<f32>;
@group(0) @binding(1)var<storage,read>K:array<f32>;
@group(0) @binding(2)var<storage,read>V:array<f32>;
@group(0) @binding(3)var<storage,read>Beta:array<f32>;
@group(0) @binding(4)var<storage,read>Gate:array<f32>;
@group(0) @binding(5)var<storage,read_write>State:array<f32>;
@group(0) @binding(6)var<storage,read_write>Y:array<f32>;
@group(0) @binding(7)var<storage,read>P:array<u32>;
var<workgroup>sums:array<f32,8>;
fn reduce128(x:f32,lane128:u32,pair:u32)->f32{let lane=lane128&31u;let warp=(lane128>>5u)+pair*4u;let s=subgroupAdd(x);if(lane==0u){sums[warp]=s;}workgroupBarrier();let b=pair*4u;let total=sums[b]+sums[b+1u]+sums[b+2u]+sums[b+3u];workgroupBarrier();return total;}
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id)wid:vec3<u32>,@builtin(local_invocation_id)lid:vec3<u32>){
 let nv=P[0];let nk=P[1];let dk=P[2];let dv=P[3];let T=P[4];let head=wid.x;let pair=lid.x>>7u;let ki=lid.x&127u;
 let vi0=wid.y*4u+pair*2u;let vi1=vi0+1u;let valid0=head<nv&&vi0<dv;let valid1=head<nv&&vi1<dv;let svi0=min(vi0,dv-1u);let svi1=min(vi1,dv-1u);
 let kh=head/max(1u,nv/nk);let sb=head*dk*dv;let qs=inverseSqrt(f32(dk));let idx0=sb+ki*dv+svi0;let idx1=sb+ki*dv+svi1;
 var st0=select(0.0,State[idx0],valid0&&ki<dk);var st1=select(0.0,State[idx1],valid1&&ki<dk);
 for(var t=0u;t<T;t++){
  let qbase=(t*nk+kh)*dk;let vbase=(t*nv+head)*dv;let bh=select(0.0,Beta[t*nv+head],head<nv);let gh=select(0.0,exp(Gate[t*nv+head]),head<nv);
  let kv=select(0.0,K[qbase+ki],head<nv&&ki<dk);let qv=select(0.0,Q[qbase+ki]*qs,head<nv&&ki<dk);
  let pred0=reduce128(gh*st0*kv,ki,pair);let pred1=reduce128(gh*st1*kv,ki,pair);
  let d0=select(0.0,(V[vbase+svi0]-pred0)*bh,valid0);let d1=select(0.0,(V[vbase+svi1]-pred1)*bh,valid1);
  st0=gh*st0+kv*d0;st1=gh*st1+kv*d1;
  let out0=reduce128(st0*qv,ki,pair);let out1=reduce128(st1*qv,ki,pair);
  if(ki==0u&&valid0){Y[vbase+vi0]=out0;}if(ki==0u&&valid1){Y[vbase+vi1]=out1;}
 }
 if(valid0&&ki<dk){State[idx0]=st0;}if(valid1&&ki<dk){State[idx1]=st1;}
}
