use std::collections::HashMap;

use anyhow::Context;

use crate::tensor::unary_operators::UnaryComputeOperation;

pub struct GPURuntime {
    pub(crate) device: wgpu::Device,
    pub(crate) queue: wgpu::Queue,
    pub(crate) unary_operations: HashMap<String, UnaryComputeOperation>,
}

impl GPURuntime {
    pub async fn new() -> anyhow::Result<Self> {
        // Initializing device and queue
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions::default())
            .await
            .context("Failed to find an appropriate GPU adapter")?;

        let name = adapter.get_info().name;

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: Some(format!("Runtime Device: {}", name).as_str()),
                    required_features: wgpu::Features::empty(),
                    required_limits: wgpu::Limits::downlevel_defaults(),
                    memory_hints: wgpu::MemoryHints::MemoryUsage,
                },
                None,
            )
            .await
            .context("Failed to create device")?;

        // Loading unary operations
        let mut unary_operations = HashMap::new();
        let unary_operations_list: HashMap<String, String> = ron::from_str(
            &std::fs::read_to_string("./shader_templates/unary_operations.ron").unwrap(),
        )
        .unwrap();

        for (name, code) in unary_operations_list.iter() {
            let operation = UnaryComputeOperation::new(&device, name, code);
            unary_operations.insert(name.clone(), operation);
        }

        Ok(GPURuntime {
            device,
            queue,
            unary_operations,
        })
    }
}
