const threads : u32 = 256;

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;

var<workgroup> wgm: array<f32, threads>;

@compute @workgroup_size(threads)
fn main(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) nw: vec3<u32>,
    @builtin(local_invocation_id) lid : vec3<u32>,
    @builtin(workgroup_id) wid : vec3<u32>
) {
    let lidx = lid.x;
    let step_size = nw.x * threads;
    var i = gid.x;
    var acc = 0.0;

    loop {
        if (i >= arrayLength(&input)) { break; }
        acc += input[i];
        i += step_size;
    }

    wgm[lidx] = acc;
    workgroupBarrier();

    var stride = threads / 2u;
    loop {
        if (stride == 0u) { break; }
        if (lidx < stride) {
            wgm[lidx] += wgm[lidx + stride];
        }
        workgroupBarrier();
        stride /= 2u;
    }

    if (lidx == 0u) {
        output[wid.x] = wgm[0];
    }
}