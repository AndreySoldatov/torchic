use std::num::NonZeroU64;

use anyhow::Context;

use crate::runtime::{self, GPURuntime};

use super::Tensor;

pub struct UnaryComputeOperation {
    pub(crate) bind_group_layout: wgpu::BindGroupLayout,
    pub(crate) pipeline: wgpu::ComputePipeline,
}

impl UnaryComputeOperation {
    pub fn new(device: &wgpu::Device, name: &str, code: &str) -> Self {
        let code_template = format!(
            "@group(0) @binding(0)
            var<storage, read> data: array<f32>;
            @group(0) @binding(1)
            var<storage, read_write> output: array<f32>;

            @compute @workgroup_size(64)
            fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {{
                let idx = global_id.x;

                if (idx >= arrayLength(&data)) {{
                    return;
                }}

                {}
            }}",
            code
        );

        let shader_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some(format!("Unary Compute Shader: {}", name).as_str()),
            source: wgpu::ShaderSource::Wgsl(code_template.into()),
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some(format!("Unary Opration Compute Bind Group Layout: {}", name).as_str()),
            entries: &[
                // Data buffer
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        min_binding_size: Some(
                            NonZeroU64::new(std::mem::size_of::<f32>() as u64).unwrap(),
                        ),
                        has_dynamic_offset: false,
                    },
                    count: None,
                },
                // Output buffer
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        min_binding_size: Some(
                            NonZeroU64::new(std::mem::size_of::<f32>() as u64).unwrap(),
                        ),
                        has_dynamic_offset: false,
                    },
                    count: None,
                },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some(format!("Unary Opration Compute Pipeline Layout: {}", name).as_str()),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some(format!("Unary Opration Compute Pipeline: {}", name).as_str()),
            layout: Some(&pipeline_layout),
            module: &shader_module,
            entry_point: Some("main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        Self {
            bind_group_layout,
            pipeline,
        }
    }
}

impl Tensor {
    fn perform_unary_operation(
        &mut self,
        name: &str,
        runtime: &GPURuntime,
    ) -> anyhow::Result<Self> {
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

    pub fn abs(&mut self, runtime: &GPURuntime) -> Self {
        self.perform_unary_operation("abs", runtime).unwrap()
    }

    pub fn acos(&mut self, runtime: &GPURuntime) -> Self {
        self.perform_unary_operation("acos", runtime).unwrap()
    }

    pub fn asin(&mut self, runtime: &GPURuntime) -> Self {
        self.perform_unary_operation("asin", runtime).unwrap()
    }

    pub fn atan(&mut self, runtime: &GPURuntime) -> Self {
        self.perform_unary_operation("atan", runtime).unwrap()
    }

    pub fn sin(&mut self, runtime: &GPURuntime) -> Self {
        self.perform_unary_operation("sin", runtime).unwrap()
    }

    pub fn sinh(&mut self, runtime: &GPURuntime) -> Self {
        self.perform_unary_operation("sinh", runtime).unwrap()
    }

    pub fn cos(&mut self, runtime: &GPURuntime) -> Self {
        self.perform_unary_operation("cos", runtime).unwrap()
    }

    pub fn cosh(&mut self, runtime: &GPURuntime) -> Self {
        self.perform_unary_operation("cosh", runtime).unwrap()
    }

    pub fn tan(&mut self, runtime: &GPURuntime) -> Self {
        self.perform_unary_operation("tan", runtime).unwrap()
    }

    pub fn tanh(&mut self, runtime: &GPURuntime) -> Self {
        self.perform_unary_operation("tanh", runtime).unwrap()
    }

    pub fn ceil(&mut self, runtime: &GPURuntime) -> Self {
        self.perform_unary_operation("ceil", runtime).unwrap()
    }

    pub fn floor(&mut self, runtime: &GPURuntime) -> Self {
        self.perform_unary_operation("floor", runtime).unwrap()
    }

    pub fn round(&mut self, runtime: &GPURuntime) -> Self {
        self.perform_unary_operation("round", runtime).unwrap()
    }

    pub fn trunc(&mut self, runtime: &GPURuntime) -> Self {
        self.perform_unary_operation("trunc", runtime).unwrap()
    }

    pub fn exp(&mut self, runtime: &GPURuntime) -> Self {
        self.perform_unary_operation("exp", runtime).unwrap()
    }

    pub fn log(&mut self, runtime: &GPURuntime) -> Self {
        self.perform_unary_operation("log", runtime).unwrap()
    }

    pub fn sqrt(&mut self, runtime: &GPURuntime) -> Self {
        self.perform_unary_operation("sqrt", runtime).unwrap()
    }

    pub fn inversesqrt(&mut self, runtime: &GPURuntime) -> Self {
        self.perform_unary_operation("inversesqrt", runtime)
            .unwrap()
    }

    pub fn relu(&mut self, runtime: &GPURuntime) -> Self {
        self.perform_unary_operation("relu", runtime).unwrap()
    }

    pub fn leaky_relu(&mut self, runtime: &GPURuntime) -> Self {
        self.perform_unary_operation("leaky_relu", runtime).unwrap()
    }

    pub fn sigmoid(&mut self, runtime: &GPURuntime) -> Self {
        self.perform_unary_operation("sigmoid", runtime).unwrap()
    }
}
