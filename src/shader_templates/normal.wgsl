struct Params {
    fact: f32,
    seed: u32,
}

@group(0) @binding(0) var<storage, read> p: Params;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;

fn mix(x_in: u32) -> u32 {
    var x = x_in;
    x ^= x >> 16u;
    x *= 0x7feb352du;
    x ^= x >> 15u;
    x *= 0x846ca68bu;
    x ^= x >> 16u;
    return x;
}

fn rand(x: u32) -> f32 {
    return (f32(mix(x) >> 8u) + 0.5) * (1.0 / 16777216.0);
}

// Box-Muller Method
fn randn(x: u32) -> f32 {
    let u1 = rand(2u * x);
    let u2 = rand(2u * x + 1u);

    let r = sqrt(-2.0 * log(u1));
    let theta = 6.283185307179586 * u2;
    
    return r * cos(theta);
}

@compute @workgroup_size(64)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(num_workgroups) num_workgroups: vec3<u32>
) {
    let total_threads = num_workgroups.x * 64u;
    var idx = global_id.x;

    loop {
        if (idx >= arrayLength(&output)) { return; }

        output[idx] = randn(idx ^ p.seed) * p.fact;
        idx += total_threads;
    }
}