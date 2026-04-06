use std::num::NonZeroU64;

use wgpu::{BufferUsages, wgt::BufferDescriptor};

use crate::{AsBindingResource, runtime::WGPUContext};

#[derive(Debug)]
pub struct MetadataArena {
    ctx: WGPUContext,
    buf: wgpu::Buffer,
    cursor: u64,
    capacity: u64,
    alignment: u64,
}

pub struct MetdataHandle<'a> {
    pub(crate) offset: u64,
    pub(crate) size: u64,
    pub(crate) buf: &'a wgpu::Buffer,
}

impl<'a> AsBindingResource for MetdataHandle<'a> {
    fn as_binding_resource(&self) -> wgpu::BindingResource<'_> {
        wgpu::BindingResource::Buffer(wgpu::BufferBinding {
            buffer: self.buf,
            offset: self.offset,
            size: Some(NonZeroU64::new(self.size).unwrap()),
        })
    }
}

#[derive(Debug)]
pub enum AllocError {
    ArenaExhaustion,
}

fn align_up(x: u64, a: u64) -> u64 {
    x.div_ceil(a) * a
}

impl MetadataArena {
    pub fn new(ctx: WGPUContext, capacity: u64) -> Self {
        let alignment = ctx.device.limits().min_storage_buffer_offset_alignment as u64;
        let capacity = align_up(capacity, alignment);

        let buf = ctx.device.create_buffer(&BufferDescriptor {
            label: Some("metadata_arena"),
            mapped_at_creation: false,
            size: capacity,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
        });

        Self {
            ctx,
            buf,
            cursor: 0,
            capacity,
            alignment,
        }
    }

    pub fn allocate<'a>(&'a mut self, bytes: &[u8]) -> Result<MetdataHandle<'a>, AllocError> {
        let start = align_up(self.cursor, self.alignment);
        let end = align_up(start + bytes.len() as u64, self.alignment);

        if end > self.capacity {
            return Err(AllocError::ArenaExhaustion);
        }

        self.ctx.queue.write_buffer(&self.buf, start, bytes);
        self.cursor = end;

        Ok(MetdataHandle {
            offset: start,
            size: bytes.len() as u64,
            buf: &self.buf,
        })
    }

    pub fn reset(&mut self) {
        self.cursor = 0;
    }
}
