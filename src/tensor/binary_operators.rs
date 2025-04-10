use anyhow::{Context, bail};

use super::Tensor;
use crate::runtime::GPURuntime;

impl Tensor {
    fn perform_binary_elementwise_operation(
        &self,
        other: &Tensor,
        name: &str,
        runtime: &GPURuntime,
    ) -> anyhow::Result<Self> {
        if self.shape != other.shape {
            bail!(
                "Tensor shapes do not match: {} vs {}",
                self.shape.iter().product::<usize>(),
                other.shape.iter().product::<usize>()
            );
        }

        let new_tensor = Tensor::new_zeroed(runtime, &self.shape);

        let operation = runtime
            .tensor_operations
            .binary_elementwise_operations
            .get(name)
            .context(format!(
                "Binary elementwise operation '{}' not found in runtime.",
                name
            ))?;

        let bind_group = runtime
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some(
                    format!(
                        "Binary elementwise operation Bind Group; First Tensor: {}; Second Tensor: {}; Operation: {}",
                        self.uuid, other.uuid, name
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
                ],
            });

        let mut encoder = runtime
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some(
                    format!("Binary elementwise operation encoder; name: {}", name).as_str(),
                ),
            });

        let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some(format!("Binary elementwise Opration Compute Pass: {}", name).as_str()),
            timestamp_writes: None,
        });

        compute_pass.set_pipeline(&operation.pipeline);
        compute_pass.set_bind_group(0, &bind_group, &[]);
        compute_pass.dispatch_workgroups(
            self.shape.iter().product::<usize>().div_ceil(64) as u32,
            1,
            1,
        );

        drop(compute_pass);

        runtime.queue.submit(Some(encoder.finish()));

        Ok(new_tensor)
    }

    pub fn add(&self, other: &Tensor, runtime: &GPURuntime) -> anyhow::Result<Self> {
        self.perform_binary_elementwise_operation(other, "add", runtime)
    }

    pub fn sub(&self, other: &Tensor, runtime: &GPURuntime) -> anyhow::Result<Self> {
        self.perform_binary_elementwise_operation(other, "sub", runtime)
    }

    pub fn mul(&self, other: &Tensor, runtime: &GPURuntime) -> anyhow::Result<Self> {
        self.perform_binary_elementwise_operation(other, "mul", runtime)
    }

    pub fn div(&self, other: &Tensor, runtime: &GPURuntime) -> anyhow::Result<Self> {
        self.perform_binary_elementwise_operation(other, "div", runtime)
    }

    pub fn pow(&self, other: &Tensor, runtime: &GPURuntime) -> anyhow::Result<Self> {
        self.perform_binary_elementwise_operation(other, "pow", runtime)
    }

    pub fn min(&self, other: &Tensor, runtime: &GPURuntime) -> anyhow::Result<Self> {
        self.perform_binary_elementwise_operation(other, "min", runtime)
    }

    pub fn max(&self, other: &Tensor, runtime: &GPURuntime) -> anyhow::Result<Self> {
        self.perform_binary_elementwise_operation(other, "max", runtime)
    }

    pub fn eq(&self, other: &Tensor, runtime: &GPURuntime) -> anyhow::Result<Self> {
        self.perform_binary_elementwise_operation(other, "eq", runtime)
    }

    pub fn ne(&self, other: &Tensor, runtime: &GPURuntime) -> anyhow::Result<Self> {
        self.perform_binary_elementwise_operation(other, "ne", runtime)
    }

    pub fn lt(&self, other: &Tensor, runtime: &GPURuntime) -> anyhow::Result<Self> {
        self.perform_binary_elementwise_operation(other, "lt", runtime)
    }

    pub fn le(&self, other: &Tensor, runtime: &GPURuntime) -> anyhow::Result<Self> {
        self.perform_binary_elementwise_operation(other, "le", runtime)
    }

    pub fn gt(&self, other: &Tensor, runtime: &GPURuntime) -> anyhow::Result<Self> {
        self.perform_binary_elementwise_operation(other, "gt", runtime)
    }

    pub fn ge(&self, other: &Tensor, runtime: &GPURuntime) -> anyhow::Result<Self> {
        self.perform_binary_elementwise_operation(other, "ge", runtime)
    }
}
