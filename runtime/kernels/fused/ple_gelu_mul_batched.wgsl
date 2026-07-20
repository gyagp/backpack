// @meta bindings=3

@group(0) @binding(0) var<storage, read_write> Gate: array<f32>;
@group(0) @binding(1) var<storage, read> PleInput: array<f32>;
@group(0) @binding(2) var<storage, read> P: array<u32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let M=P[0]; let D=P[1]; let layer=P[2]; let L=P[3];
    let flat=gid.x;
    if (flat>=M*D) { return; }
    let row=flat/D; let i=flat%D; let g=Gate[flat];
    let gelu=0.5*g*(1.0+tanh(0.7978845608*(g+0.044715*g*g*g)));
    Gate[flat]=gelu*PleInput[(row*L+layer)*D+i];
}
