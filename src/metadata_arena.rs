use wgpu::{BufferUsages, wgt::BufferDescriptor};

use crate::runtime::WGPUContext;

const ALIGNMENT: u64 = 16;

#[derive(Debug)]
pub struct MetadataArena {
    ctx: WGPUContext,
    buf: wgpu::Buffer,
    cursor: u64,
    capacity: u64,
}

pub struct MetdataHandle {
    offset: u64,
    size: u64,
}

pub enum AllocError {
    ArenaExhaustion,
}

impl MetadataArena {
    pub fn new(ctx: WGPUContext, capacity: u64) -> Self {
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
        }
    }

    pub fn allocate(&mut self, bytes: &[u8]) -> Result<MetdataHandle, AllocError> {
        let start = self.cursor;

        let x = start + bytes.len() as u64;
        let end = (x + ALIGNMENT - 1) & !(ALIGNMENT - 1);

        if end > self.capacity {
            return Err(AllocError::ArenaExhaustion);
        }

        self.ctx.queue.write_buffer(&self.buf, start, bytes);

        self.cursor = end;

        Ok(MetdataHandle {
            offset: start,
            size: bytes.len() as u64,
        })
    }

    pub fn reset(&mut self) {
        self.cursor = 0;
    }
}
