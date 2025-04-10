use anyhow::bail;
use wgpu::util::DeviceExt;

use crate::runtime::GPURuntime;

use super::Tensor;

#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct MatrixMultiplicationUniforms {
    pub a_rows: u32,
    pub a_cols: u32,
    pub b_cols: u32,
}

impl Tensor {
    /// Performs matrix multiplication between two tensors.
    /// The tensors must have compatible shapes for matrix multiplication.
    pub fn matmul(&self, other: &Tensor, runtime: &GPURuntime) -> anyhow::Result<Self> {
        if self.shape.len() != 2 || other.shape.len() != 2 {
            bail!("Both tensors must be 2D for matrix multiplication.");
        }

        if self.shape[1] != other.shape[0] {
            bail!(
                "Incompatible shapes for matrix multiplication: {}x{} and {}x{}",
                self.shape[0],
                self.shape[1],
                other.shape[0],
                other.shape[1]
            );
        }

        let new_shape = vec![self.shape[0], other.shape[1]];
        let new_tensor = Tensor::new_zeroed(runtime, &new_shape);

        let operation = &runtime.tensor_operations.matrix_multiplication_operation;

        let uniforms = MatrixMultiplicationUniforms {
            a_rows: self.shape[0] as u32,
            a_cols: self.shape[1] as u32,
            b_cols: other.shape[1] as u32,
        };
        let uniforms_buffer =
            runtime
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("Matrix Multiplication Uniforms Buffer"),
                    contents: bytemuck::cast_slice(&[uniforms]),
                    usage: wgpu::BufferUsages::UNIFORM,
                });

        let bind_group = runtime
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some(
                    format!(
                        "Matrix Multiplication Bind Group; First Tensor: {}; Second Tensor: {}; Operation: matmul",
                        self.uuid, other.uuid
                    )
                    .as_str(),
                ),
                layout: &operation.bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: self.data_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: other.data_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: new_tensor.data_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: uniforms_buffer.as_entire_binding(),
                    },
                ],
            });

        let mut encoder = runtime
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some(
                    format!(
                        "Matrix Multiplication Encoder; {}x{}",
                        self.uuid, other.uuid
                    )
                    .as_str(),
                ),
            });

        let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some(
                format!(
                    "Matrix Multiplication Compute Pass; {}x{}",
                    self.uuid, other.uuid
                )
                .as_str(),
            ),
            timestamp_writes: None,
        });

        compute_pass.set_pipeline(&operation.pipeline);
        compute_pass.set_bind_group(0, &bind_group, &[]);
        compute_pass.dispatch_workgroups(
            self.shape[0].div_ceil(16) as u32,
            other.shape[1].div_ceil(16) as u32,
            1,
        );

        drop(compute_pass);

        runtime.queue.submit(Some(encoder.finish()));

        Ok(new_tensor)
    }
}
