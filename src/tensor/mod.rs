pub mod binary_operators;
pub mod matrix_multiplication;
pub mod unary_operators;

use anyhow::bail;
use uuid::Uuid;
use wgpu::util::DeviceExt;

use crate::runtime::{self, GPURuntime};

pub struct Tensor {
    pub uuid: Uuid,
    pub shape: Vec<usize>,
    pub(crate) data_buffer: wgpu::Buffer,
}

impl Tensor {
    /// Creates a new tensor with the given shape and initializes it with zeros.
    pub fn new_zeroed(runtime: &runtime::GPURuntime, shape: &[usize]) -> Self {
        let uuid = Uuid::new_v4();

        let size: u64 = (shape.iter().product::<usize>() * std::mem::size_of::<f32>()) as u64;

        let data_buffer = runtime.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(format!("Tensor data buffer: {}", uuid).as_str()),
            size,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Self {
            uuid,
            shape: shape.to_vec(),
            data_buffer,
        }
    }

    /// Creates a new tensor with the given shape and initializes it with random values.
    /// The values are in the range [0, 1).
    pub fn new_rand(runtime: &runtime::GPURuntime, shape: &[usize]) -> Self {
        let uuid = Uuid::new_v4();

        let mut data = vec![];
        data.reserve(shape.iter().product::<usize>());
        for _ in 0..shape.iter().product::<usize>() {
            data.push(rand::random::<f32>());
        }

        let data_buffer = runtime
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(format!("Tensor data buffer: {}", uuid).as_str()),
                contents: bytemuck::cast_slice(&data),
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST,
            });

        Self {
            uuid,
            shape: shape.to_vec(),
            data_buffer,
        }
    }

    /// Creates a new tensor with the given shape and initializes it with a constant value.
    pub fn new_constant(runtime: &runtime::GPURuntime, shape: &[usize], constant: f32) -> Self {
        let uuid = Uuid::new_v4();

        let data = vec![constant; shape.iter().product::<usize>()];

        let data_buffer = runtime
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(format!("Tensor data buffer: {}", uuid).as_str()),
                contents: bytemuck::cast_slice(&data),
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST,
            });

        Self {
            uuid,
            shape: shape.to_vec(),
            data_buffer,
        }
    }

    /// Creates a new tensor from an existing tensor.
    /// The new tensor will have the same shape and data as the original tensor.
    pub fn copy_from_tensor(runtime: &GPURuntime, tensor: &Tensor) -> Self {
        let uuid = Uuid::new_v4();

        let data_buffer = runtime.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(format!("Tensor data buffer: {}", uuid).as_str()),
            size: tensor.data_buffer.size(),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = runtime
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some(format!("Tensor copy encoder: {}", uuid).as_str()),
            });

        encoder.copy_buffer_to_buffer(
            &tensor.data_buffer,
            0,
            &data_buffer,
            0,
            tensor.data_buffer.size(),
        );

        runtime.queue.submit(Some(encoder.finish()));

        Self {
            uuid,
            shape: tensor.shape.clone(),
            data_buffer,
        }
    }

    /// Creates a new identity tensor with the given size.
    /// The identity tensor is a square matrix with ones on the diagonal and zeros elsewhere.
    pub fn new_identity(runtime: &GPURuntime, size: usize) -> Self {
        let uuid = Uuid::new_v4();

        let mut data = vec![0.0; size * size];
        for i in 0..size {
            data[i * size + i] = 1.0;
        }

        let data_buffer = runtime
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(format!("Tensor data buffer: {}", uuid).as_str()),
                contents: bytemuck::cast_slice(&data),
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST,
            });

        Self {
            uuid,
            shape: vec![size, size],
            data_buffer,
        }
    }

    pub fn from_cpu_with_shape(
        runtime: &GPURuntime,
        data: &[f32],
        shape: &[usize],
    ) -> anyhow::Result<Self> {
        if data.len() != shape.iter().product::<usize>() {
            bail!(
                "Data length does not match the product of the shape dimensions. Data length: {}, Shape product: {}",
                data.len(),
                shape.iter().product::<usize>()
            );
        }

        let uuid = Uuid::new_v4();

        let data_buffer = runtime
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(format!("Tensor data buffer: {}", uuid).as_str()),
                contents: bytemuck::cast_slice(data),
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST,
            });

        Ok(Self {
            uuid,
            shape: shape.to_vec(),
            data_buffer,
        })
    }

    // TODO: Implement a CPU tensor structure to preserve shape
    pub fn copy_to_cpu(&self, runtime: &runtime::GPURuntime) -> Vec<f32> {
        let download_buffer = runtime.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(format!("Tensor download buffer: {}", self.uuid).as_str()),
            size: self.data_buffer.size(),
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        let mut encoder = runtime
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some(format!("Tensor copy encoder: {}", self.uuid).as_str()),
            });

        encoder.copy_buffer_to_buffer(
            &self.data_buffer,
            0,
            &download_buffer,
            0,
            self.data_buffer.size(),
        );

        runtime.queue.submit(Some(encoder.finish()));

        let buffer_slice = download_buffer.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |_| tx.send(()).unwrap());
        runtime.device.poll(wgpu::Maintain::Wait);
        rx.recv().unwrap();

        let data = buffer_slice.get_mapped_range();
        bytemuck::cast_slice(&data).to_vec()
    }
}
