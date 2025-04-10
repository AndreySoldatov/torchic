struct MatrixMultiplicationUniforms {
    rows1: u32,
    cols1: u32,
    cols2: u32,
}

@group(0) @binding(0)
var<storage, read> data1: array<f32>;
@group(0) @binding(1)
var<storage, read> data2: array<f32>;
@group(0) @binding(2)
var<storage, read_write> output: array<f32>;

@group(0) @binding(3)
var<uniform> mat_uniforms: MatrixMultiplicationUniforms;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let row = global_id.x;
    let col = global_id.y;

    if (row >= mat_uniforms.rows1 || col >= mat_uniforms.cols2) {
        return;
    }

    var sum: f32 = 0.0;
    for (var k: u32 = 0; k < mat_uniforms.cols1; k = k + 1) {
        sum = sum + data1[row * mat_uniforms.cols1 + k] * data2[k * mat_uniforms.cols2 + col];
    }
    
    output[row * mat_uniforms.cols2 + col] = sum;
}