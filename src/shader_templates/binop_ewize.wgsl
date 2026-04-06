@group(0) @binding(0) var<storage, read> input1: array<f32>;
@group(0) @binding(1) var<storage, read> input2: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(64)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(num_workgroups) num_workgroups: vec3<u32>
) {
    let total_threads = num_workgroups.x * 64u;
    var idx = global_id.x;

    loop {
        if (idx >= arrayLength(&input1)) { return; }
        
        ${operation}
        
        idx += total_threads;
    }
}