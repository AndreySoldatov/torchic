pub mod unary_operators;

use uuid::Uuid;
use wgpu::util::DeviceExt;

use crate::runtime;

pub struct Tensor {
    uuid: Uuid,
    pub shape: Vec<usize>,
    pub(crate) data_buffer: wgpu::Buffer,
}

impl Tensor {
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
