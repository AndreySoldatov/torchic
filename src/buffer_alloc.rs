use std::{
    collections::{BTreeMap, HashSet},
    marker::PhantomData,
    sync::{Arc, LazyLock, Mutex, OnceLock, mpsc::channel},
};

use slotmap::{SlotMap, new_key_type};
use wgpu::{
    BufferDescriptor, BufferUsages,
    util::{BufferInitDescriptor, DeviceExt},
};

use crate::runtime::WGPUContext;

new_key_type! {
    struct BufferId;
}

#[derive(Debug)]
struct BufferEntry {
    raw: Arc<wgpu::Buffer>,
    capacity: u64,
}

#[derive(Debug)]
struct BufferDebugLabel {
    prefix: String,
    counter: u64,
}

#[derive(Debug)]
pub enum Usage {
    Storage,
    Readback,
}

#[derive(Debug)]
struct BufferAllocator {
    ctx: WGPUContext,

    debug_label: BufferDebugLabel,
    usage: Usage,

    store: SlotMap<BufferId, BufferEntry>,
    free_pool: BTreeMap<u64, Vec<BufferId>>,
    pending: HashSet<BufferId>,
}

const MAX_ABSOLUTE_LIMIT: u64 = 256 * 1024;

fn validate_size(cs: u64, rs: u64) -> bool {
    cs.saturating_sub(rs) < (rs / 2) && cs.saturating_sub(rs) <= MAX_ABSOLUTE_LIMIT
}

#[derive(Debug)]
enum BufferFreeError {
    DoubleFreeAttempt,
}

impl BufferAllocator {
    fn new(ctx: WGPUContext, usage: Usage) -> Self {
        Self {
            store: SlotMap::with_key(),
            free_pool: BTreeMap::new(),
            pending: HashSet::new(),
            ctx,
            debug_label: BufferDebugLabel {
                prefix: format!("{:?}", usage),
                counter: 0,
            },
            usage,
        }
    }

    fn request(&mut self, size: u64) -> BufferId {
        let mut delete_entry = None;

        let res_id =
            if let Some((candidate_size, entries)) = self.free_pool.range_mut(size..).next() {
                if validate_size(*candidate_size, size) {
                    let candidate = entries
                        .pop()
                        .expect("If the element in the map exists then it must be a non empty vec");

                    if entries.is_empty() {
                        delete_entry = Some(*candidate_size);
                    }

                    candidate
                } else {
                    self.allocate_new_buffer(size)
                }
            } else {
                self.allocate_new_buffer(size)
            };

        if let Some(k) = delete_entry {
            self.free_pool.remove(&k);
        }

        res_id
    }

    fn allocate_new_buffer(&mut self, size: u64) -> BufferId {
        let wgpu_usage = match self.usage {
            Usage::Storage => {
                BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST
            }
            Usage::Readback => BufferUsages::MAP_READ | BufferUsages::COPY_DST,
        };

        let label = format!("{}_{}", self.debug_label.prefix, self.debug_label.counter);

        let raw = self.ctx.device.create_buffer(&BufferDescriptor {
            label: Some(&label),
            mapped_at_creation: false,
            size,
            usage: wgpu_usage,
        });

        self.debug_label.counter += 1;

        self.store.insert(BufferEntry {
            raw: Arc::new(raw),
            capacity: size,
        })
    }

    fn free(&mut self, buf: BufferId) -> Result<(), BufferFreeError> {
        if self.pending.insert(buf) {
            Ok(())
        } else {
            Err(BufferFreeError::DoubleFreeAttempt)
        }
    }

    fn reclaim(&mut self) {
        // Wait for all pending work to be complete
        self.ctx
            .device
            .poll(wgpu::wgt::PollType::wait_indefinitely());

        for buf in self.pending.drain() {
            let capacity = self.store[buf].capacity;

            self.free_pool
                .entry(capacity)
                .and_modify(|v| v.push(buf))
                .or_insert(vec![buf]);
        }
    }
}

pub mod usage_marker {
    pub trait BufferUsageMarker {}

    #[derive(Clone, Copy, Debug)]
    pub struct Storage;
    impl BufferUsageMarker for Storage {}

    #[derive(Clone, Copy, Debug)]
    pub struct Readback;
    impl BufferUsageMarker for Readback {}
}

#[derive(Debug)]
pub struct BufferAllocatorRef<T: usage_marker::BufferUsageMarker> {
    alloc: Arc<Mutex<BufferAllocator>>,
    ctx: WGPUContext,
    _tag: PhantomData<T>,
}

impl BufferAllocatorRef<usage_marker::Storage> {
    pub fn new(ctx: WGPUContext) -> Self {
        Self {
            alloc: Arc::new(Mutex::new(BufferAllocator::new(
                ctx.clone(),
                Usage::Storage,
            ))),
            ctx,
            _tag: PhantomData,
        }
    }
}

impl BufferAllocatorRef<usage_marker::Readback> {
    pub fn new(ctx: WGPUContext) -> Self {
        Self {
            alloc: Arc::new(Mutex::new(BufferAllocator::new(
                ctx.clone(),
                Usage::Readback,
            ))),
            ctx,
            _tag: PhantomData,
        }
    }
}

impl<T: usage_marker::BufferUsageMarker> Clone for BufferAllocatorRef<T> {
    fn clone(&self) -> Self {
        Self {
            alloc: self.alloc.clone(),
            ctx: self.ctx.clone(),
            _tag: self._tag,
        }
    }
}

/// RAII Handle to a wgpu buffer. The buffer is placed in the "pending reuse" queue on the drop of this handle
pub struct BufferLease<T: usage_marker::BufferUsageMarker> {
    raw: Arc<wgpu::Buffer>,
    buf: BufferId,
    size: u64,
    alloc: BufferAllocatorRef<T>,
}

impl<T: usage_marker::BufferUsageMarker> BufferAllocatorRef<T> {
    pub fn request(&self, size: u64) -> BufferLease<T> {
        assert!(
            size % 4 == 0,
            "Only 4 bytes buffer size alignment supported for now"
        );

        let mut alloc = self.alloc.lock().unwrap();
        let buf = alloc.request(size);
        let raw = alloc.store[buf].raw.clone();
        drop(alloc);

        BufferLease {
            raw,
            buf,
            size,
            alloc: self.clone(),
        }
    }
}

impl<T: usage_marker::BufferUsageMarker> Drop for BufferLease<T> {
    fn drop(&mut self) {
        self.alloc
            .alloc
            .lock()
            .unwrap()
            .free(self.buf)
            .expect("buffer double-free detected");
    }
}

impl BufferLease<usage_marker::Readback> {
    pub fn download(&self) -> Vec<u8> {
        let (tx, rx) = channel();

        self.raw
            .map_async(wgpu::MapMode::Read, ..self.size, move |result| {
                tx.send(result).unwrap()
            });

        self.alloc
            .ctx
            .device
            .poll(wgpu::PollType::wait_indefinitely());

        rx.recv().unwrap().unwrap();

        let bytes = self.raw.get_mapped_range(..self.size).to_vec();

        self.raw.unmap();

        bytes
    }
}

impl BufferLease<usage_marker::Storage> {
    pub fn set(&self, data: &[u8]) {
        assert!(self.size == data.len() as u64);

        let queue = self.alloc.ctx.queue.clone();

        queue.write_buffer(&self.raw, 0, data);
    }
}
