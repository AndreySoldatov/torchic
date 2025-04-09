use super::Tensor;
use crate::runtime::GPURuntime;
use anyhow::Context;

impl Tensor {
    fn perform_unary_operation(&self, name: &str, runtime: &GPURuntime) -> anyhow::Result<Self> {
        let new_tensor = Tensor::new_zeroed(runtime, &self.shape);

        let operation = runtime
            .unary_operations
            .get(name)
            .context(format!("Unary operation '{}' not found in runtime.", name))?;

        let bind_group = runtime
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some(
                    format!(
                        "Unary operation Bind Group; Tensor: {}; Operation: {}",
                        self.uuid, name
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
                        resource: new_tensor.data_buffer.as_entire_binding(),
                    },
                ],
            });

        let mut encoder = runtime
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some(format!("Unary Opration Encoder: {}", name).as_str()),
            });

        let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some(format!("Unary Opration Compute Pass: {}", name).as_str()),
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

    pub fn abs(&self, runtime: &GPURuntime) -> Self {
        self.perform_unary_operation("abs", runtime).unwrap()
    }

    pub fn acos(&self, runtime: &GPURuntime) -> Self {
        self.perform_unary_operation("acos", runtime).unwrap()
    }

    pub fn asin(&self, runtime: &GPURuntime) -> Self {
        self.perform_unary_operation("asin", runtime).unwrap()
    }

    pub fn atan(&self, runtime: &GPURuntime) -> Self {
        self.perform_unary_operation("atan", runtime).unwrap()
    }

    pub fn sin(&self, runtime: &GPURuntime) -> Self {
        self.perform_unary_operation("sin", runtime).unwrap()
    }

    pub fn sinh(&self, runtime: &GPURuntime) -> Self {
        self.perform_unary_operation("sinh", runtime).unwrap()
    }

    pub fn cos(&self, runtime: &GPURuntime) -> Self {
        self.perform_unary_operation("cos", runtime).unwrap()
    }

    pub fn cosh(&self, runtime: &GPURuntime) -> Self {
        self.perform_unary_operation("cosh", runtime).unwrap()
    }

    pub fn tan(&self, runtime: &GPURuntime) -> Self {
        self.perform_unary_operation("tan", runtime).unwrap()
    }

    pub fn tanh(&self, runtime: &GPURuntime) -> Self {
        self.perform_unary_operation("tanh", runtime).unwrap()
    }

    pub fn ceil(&self, runtime: &GPURuntime) -> Self {
        self.perform_unary_operation("ceil", runtime).unwrap()
    }

    pub fn floor(&self, runtime: &GPURuntime) -> Self {
        self.perform_unary_operation("floor", runtime).unwrap()
    }

    pub fn round(&self, runtime: &GPURuntime) -> Self {
        self.perform_unary_operation("round", runtime).unwrap()
    }

    pub fn trunc(&self, runtime: &GPURuntime) -> Self {
        self.perform_unary_operation("trunc", runtime).unwrap()
    }

    pub fn exp(&self, runtime: &GPURuntime) -> Self {
        self.perform_unary_operation("exp", runtime).unwrap()
    }

    pub fn log(&self, runtime: &GPURuntime) -> Self {
        self.perform_unary_operation("log", runtime).unwrap()
    }

    pub fn sqrt(&self, runtime: &GPURuntime) -> Self {
        self.perform_unary_operation("sqrt", runtime).unwrap()
    }

    pub fn inversesqrt(&self, runtime: &GPURuntime) -> Self {
        self.perform_unary_operation("inversesqrt", runtime)
            .unwrap()
    }

    pub fn relu(&self, runtime: &GPURuntime) -> Self {
        self.perform_unary_operation("relu", runtime).unwrap()
    }

    pub fn leaky_relu(&self, runtime: &GPURuntime) -> Self {
        self.perform_unary_operation("leaky_relu", runtime).unwrap()
    }

    pub fn sigmoid(&self, runtime: &GPURuntime) -> Self {
        self.perform_unary_operation("sigmoid", runtime).unwrap()
    }
}
