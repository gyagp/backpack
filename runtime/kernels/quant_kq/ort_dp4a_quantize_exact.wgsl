// @meta noregistry=true
requires packed_4x8_integer_dot_product;
enable f16;
enable subgroups;
const workgroup_size_x: u32 = 64;
const workgroup_size_y: u32 = 1;
const workgroup_size_z: u32 = 1;
@group(0) @binding(0) var<storage, read> input_a: array<vec4<f16>>;
@group(0) @binding(1) var<storage, read_write> output: array<u32>;
@group(0) @binding(2) var<storage, read_write> scales: array<f16>;
struct Uniforms {
  output_size: u32
};
@group(0) @binding(3) var<uniform> uniforms: Uniforms;

alias input_a_value_t = vec4<f16>;
alias input_a_indices_t = vec3<u32>;
alias input_a_element_t = f16;

var<workgroup> a_values : array<array<input_a_value_t, 32>, 2>;
var<workgroup> max_values : array<input_a_value_t, 4>;

fn readInput(offset: u32) -> input_a_value_t
{
  if (offset >= uniforms.output_size) {
    return input_a_value_t(0);
  }
  return input_a[offset];
}


@compute @workgroup_size(workgroup_size_x, workgroup_size_y, workgroup_size_z)
fn main(@builtin(global_invocation_id) global_id : vec3<u32>,
        @builtin(workgroup_id) workgroup_id : vec3<u32>,
        @builtin(local_invocation_index) local_idx : u32,
        @builtin(local_invocation_id) local_id : vec3<u32>,
        @builtin(subgroup_invocation_id) sg_id : u32,
        @builtin(subgroup_size) sg_size : u32) {
  let global_idx = global_id.x;
  let workgroup_idx = workgroup_id.x;

  if (sg_size == 32) {
    let local_a = readInput(global_idx);
    let max_val = subgroupMax(abs(local_a));
    if (global_idx >= uniforms.output_size) {
      return;
    }
    let max_temp = max(max_val.xy, max_val.zw);
    let scale = max(max_temp[0], max_temp[1]);
    let norm_a = local_a/scale;
    output[global_idx]=pack4x8snorm(vec4<f32>(norm_a));;
    if (local_idx % 32 == 0)
    {

      scales[workgroup_idx * 2 + local_idx / 32]=scale/127;;
    }
  } else if (sg_size == 16) {
    let local_a = readInput(global_idx);
    let sub_max_value = subgroupMax(abs(local_a));
    if (local_idx % 16 == 0) {
      max_values[local_idx / 16] = sub_max_value;
    }
    workgroupBarrier();

    if (global_idx >= uniforms.output_size) {
      return;
    }

    var max_val = input_a_value_t(0);
    if (local_idx < 32) {
      max_val = max(max_values[0], max_values[1]);
    } else {
      max_val = max(max_values[2], max_values[3]);
    }
    let max_temp = max(max_val.xy, max_val.zw);
    let scale = max(max_temp[0], max_temp[1]);
    let norm_a = local_a/scale;
    output[global_idx]=pack4x8snorm(vec4<f32>(norm_a));;
    if (local_idx % 32 == 0)
    {

      scales[workgroup_idx * 2 + local_idx / 32]=scale/127;;
    }
  } else {
    let local_row = local_idx / 32u;
    let local_col = local_idx % 32u;
    a_values[local_row][local_col] = readInput(global_idx);
    workgroupBarrier();

    if (global_idx >= uniforms.output_size) {
      return;
    }

    var max_val = input_a_value_t(0);

    for (var i = 0u; i < 32u; i++)
    {
      max_val = max(max_val, abs(a_values[local_row][i]));
    }
    let max_temp = max(max_val.xy, max_val.zw);
    let scale = max(max_temp[0], max_temp[1]);
    let norm_a = a_values[local_row][local_col]/scale;
    output[global_idx]=pack4x8snorm(vec4<f32>(norm_a));;
    if (local_col == 0u)
    {

      scales[workgroup_idx * 2 + local_row]=scale/127;;
    }
  }

}
