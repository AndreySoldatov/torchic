@group(0) @binding(0) var<storage, read> input1: array<f32>;
@group(0) @binding(1) var<storage, read> input2: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(64)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>
) {
    let idx = global_id.x;
    if (idx >= arrayLength(&input1)) { return; }
    output[idx] = input1[idx] + input2[idx];
}