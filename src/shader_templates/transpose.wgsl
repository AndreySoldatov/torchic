struct Params {
  M: u32,
  N: u32,
};

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<storage, read> p: Params;

@compute @workgroup_size(64)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(num_workgroups) num_workgroups: vec3<u32>
) {
    let total = p.M * p.N;
    let total_threads = num_workgroups.x * 64u;
    var idx = global_id.x;

    loop {
        if (idx >= total) { return; }
        
        let r = idx / p.N;
        let c = idx % p.N;

        output[c * p.M + r] = input[idx];
        
        idx += total_threads;
    }
}