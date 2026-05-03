use std::num::NonZeroU64;

use wgpu::{BufferUsages, wgt::BufferDescriptor};

use crate::{AsBindingResource, runtime::WGPUContext};

#[derive(Debug, Clone, Copy)]
struct ArenaCursor {
    page: usize,
    offset: u64,
}

#[derive(Debug)]
pub struct MetadataArena {
    ctx: WGPUContext,
    pages: Vec<wgpu::Buffer>,
    cursor: ArenaCursor,
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
    RequestedSizeBiggerThanPageSize,
}

fn align_up(x: u64, a: u64) -> u64 {
    x.div_ceil(a) * a
}

impl MetadataArena {
    pub fn new(ctx: WGPUContext, capacity: u64) -> Self {
        let alignment = ctx.device.limits().min_storage_buffer_offset_alignment as u64;
        let capacity = align_up(capacity, alignment);

        let buf = ctx.device.create_buffer(&BufferDescriptor {
            label: Some("metadata_arena 0"),
            mapped_at_creation: false,
            size: capacity,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
        });

        Self {
            ctx,
            pages: vec![buf],
            cursor: ArenaCursor { page: 0, offset: 0 },
            capacity,
            alignment,
        }
    }

    pub fn allocate<'a>(&'a mut self, bytes: &[u8]) -> Result<MetdataHandle<'a>, AllocError> {
        if align_up(bytes.len() as u64, self.alignment) > self.capacity {
            return Err(AllocError::RequestedSizeBiggerThanPageSize);
        }

        let mut start = align_up(self.cursor.offset, self.alignment);
        let mut end = align_up(start + bytes.len() as u64, self.alignment);

        if end > self.capacity {
            self.cursor.page += 1;
            self.cursor.offset = 0;

            assert!(self.cursor.page <= self.pages.len());

            if self.cursor.page == self.pages.len() {
                let label = format!("metadata_arena {}", self.cursor.page);
                self.pages
                    .push(self.ctx.device.create_buffer(&BufferDescriptor {
                        label: Some(label.as_str()),
                        mapped_at_creation: false,
                        size: self.capacity,
                        usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
                    }));
            }

            start = align_up(self.cursor.offset, self.alignment);
            end = align_up(start + bytes.len() as u64, self.alignment);
        }

        self.ctx
            .queue
            .write_buffer(&self.pages[self.cursor.page], start, bytes);
        self.cursor.offset = end;

        Ok(MetdataHandle {
            offset: start,
            size: bytes.len() as u64,
            buf: &self.pages[self.cursor.page],
        })
    }

    pub fn reset(&mut self) {
        self.cursor = ArenaCursor { page: 0, offset: 0 };
    }
}

#[cfg(test)]
mod tests {
    use std::sync::OnceLock;

    use super::*;

    fn test_ctx() -> WGPUContext {
        static CTX: OnceLock<WGPUContext> = OnceLock::new();

        CTX.get_or_init(|| {
            let adapter = WGPUContext::list_adapters()
                .into_iter()
                .next()
                .expect("No WGPU adapter available for metadata arena tests");
            WGPUContext::new(adapter)
        })
        .clone()
    }

    fn test_arena(capacity: u64) -> MetadataArena {
        MetadataArena::new(test_ctx(), capacity)
    }

    #[test]
    fn align_up_rounds_to_next_multiple() {
        assert_eq!(align_up(0, 256), 0);
        assert_eq!(align_up(1, 256), 256);
        assert_eq!(align_up(256, 256), 256);
        assert_eq!(align_up(257, 256), 512);
    }

    #[test]
    fn new_allocates_one_aligned_page() {
        let arena = test_arena(1);

        assert_eq!(arena.pages.len(), 1);
        assert_eq!(arena.cursor.page, 0);
        assert_eq!(arena.cursor.offset, 0);
        assert_eq!(arena.capacity % arena.alignment, 0);
        assert!(arena.capacity >= arena.alignment);
    }

    #[test]
    fn allocate_writes_metadata_and_advances_cursor() {
        let mut arena = test_arena(1024);
        let alignment = arena.alignment;

        {
            let handle = arena.allocate(&[1, 2, 3, 4]).unwrap();
            assert_eq!(handle.offset, 0);
            assert_eq!(handle.size, 4);
        }

        assert_eq!(arena.cursor.page, 0);
        assert_eq!(arena.cursor.offset, alignment);
    }

    #[test]
    fn allocate_uses_aligned_offsets_within_page() {
        let mut arena = test_arena(1024);
        let alignment = arena.alignment;

        {
            let first = arena.allocate(&[1, 0, 0, 0]).unwrap();
            assert_eq!(first.offset, 0);
        }

        {
            let second = arena.allocate(&[2, 0, 0, 0]).unwrap();
            assert_eq!(second.offset, alignment);
        }

        assert_eq!(arena.cursor.page, 0);
        assert_eq!(arena.cursor.offset, alignment * 2);
    }

    #[test]
    fn allocate_grows_to_new_page_when_current_page_is_full() {
        let mut arena = test_arena(512);
        let page_capacity = arena.capacity as usize;

        arena.allocate(&vec![1; page_capacity]).unwrap();
        assert_eq!(arena.pages.len(), 1);
        assert_eq!(arena.cursor.page, 0);
        assert_eq!(arena.cursor.offset, arena.capacity);

        {
            let handle = arena.allocate(&[2, 0, 0, 0]).unwrap();
            assert_eq!(handle.offset, 0);
        }

        assert_eq!(arena.pages.len(), 2);
        assert_eq!(arena.cursor.page, 1);
        assert_eq!(arena.cursor.offset, arena.alignment);
    }

    #[test]
    fn reset_rewinds_cursor_without_dropping_pages() {
        let mut arena = test_arena(512);
        let page_capacity = arena.capacity as usize;

        arena.allocate(&vec![1; page_capacity]).unwrap();
        arena.allocate(&[2, 0, 0, 0]).unwrap();
        assert_eq!(arena.pages.len(), 2);
        assert_eq!(arena.cursor.page, 1);

        arena.reset();

        assert_eq!(arena.pages.len(), 2);
        assert_eq!(arena.cursor.page, 0);
        assert_eq!(arena.cursor.offset, 0);
    }

    #[test]
    fn allocation_after_reset_reuses_existing_first_page() {
        let mut arena = test_arena(512);
        let page_capacity = arena.capacity as usize;

        arena.allocate(&vec![1; page_capacity]).unwrap();
        arena.allocate(&[2, 0, 0, 0]).unwrap();
        arena.reset();

        {
            let handle = arena.allocate(&[3, 0, 0, 0]).unwrap();
            assert_eq!(handle.offset, 0);
        }

        assert_eq!(arena.pages.len(), 2);
        assert_eq!(arena.cursor.page, 0);
        assert_eq!(arena.cursor.offset, arena.alignment);
    }

    #[test]
    fn allocation_reuses_existing_second_page_after_reset() {
        let mut arena = test_arena(512);
        let page_capacity = arena.capacity as usize;

        arena.allocate(&vec![1; page_capacity]).unwrap();
        arena.allocate(&[2, 0, 0, 0]).unwrap();
        arena.reset();

        arena.allocate(&vec![3; page_capacity]).unwrap();
        arena.allocate(&[4, 0, 0, 0]).unwrap();

        assert_eq!(arena.pages.len(), 2);
        assert_eq!(arena.cursor.page, 1);
        assert_eq!(arena.cursor.offset, arena.alignment);
    }

    #[test]
    fn allocation_larger_than_page_capacity_returns_error() {
        let mut arena = test_arena(512);
        let too_large = vec![0; arena.capacity as usize + 1];

        assert!(matches!(
            arena.allocate(&too_large),
            Err(AllocError::RequestedSizeBiggerThanPageSize)
        ));
        assert_eq!(arena.pages.len(), 1);
        assert_eq!(arena.cursor.page, 0);
        assert_eq!(arena.cursor.offset, 0);
    }
}
