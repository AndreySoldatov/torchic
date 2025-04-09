use std::num::NonZeroU64;

pub struct ComputeOperation {
    pub(crate) bind_group_layout: wgpu::BindGroupLayout,
    pub(crate) pipeline: wgpu::ComputePipeline,
}

impl ComputeOperation {
    pub fn new_unary_operation(device: &wgpu::Device, name: &str, code: &str) -> Self {
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

    pub fn new_binary_elementwise(device: &wgpu::Device, name: &str, code: &str) -> Self {
        let code_template = format!(
            "@group(0) @binding(0)
            var<storage, read> data1: array<f32>;
            @group(0) @binding(1)
            var<storage, read> data2: array<f32>;
            @group(0) @binding(2)
            var<storage, read_write> output: array<f32>;

            @compute @workgroup_size(64)
            fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {{
                let idx = global_id.x;

                if (idx >= arrayLength(&data1)) {{
                    return;
                }}

                {}
            }}",
            code
        );

        let shader_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some(format!("Binary Compute Shader: {}", name).as_str()),
            source: wgpu::ShaderSource::Wgsl(code_template.into()),
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some(format!("Binary Opration Compute Bind Group Layout: {}", name).as_str()),
            entries: &[
                // Data buffer 1
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
                // Data buffer 2
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
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
                    binding: 2,
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
            label: Some(format!("Binary Opration Compute Pipeline Layout: {}", name).as_str()),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some(format!("Binary Opration Compute Pipeline: {}", name).as_str()),
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
