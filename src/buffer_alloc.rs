use std::{
    collections::{BTreeMap, HashSet},
    marker::PhantomData,
    num::NonZeroU64,
    sync::{Arc, Mutex, mpsc::channel},
};

use slotmap::{SlotMap, new_key_type};
use wgpu::{BufferDescriptor, BufferUsages};

use crate::{
    AsBindingResource,
    buffer_alloc::usage_marker::Storage,
    runtime::{WGPUContext, rt},
};

new_key_type! {
    struct BufferId;
}

#[derive(Debug)]
struct BufferEntry {
    raw: Arc<wgpu::Buffer>,
    capacity: u64,
    timestamp: usize,
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

#[derive(Debug, Clone, Copy)]
pub(crate) struct MemoryCachePolicy {
    pub(crate) cache_to_used_proportion: f64,
    pub(crate) soft_cache_limit: u64,
    pub(crate) hard_cache_limit: u64,
    pub(crate) max_ttl: usize,
    pub(crate) eviction_debounce: usize,
}

#[derive(Debug)]
struct BufferAllocator {
    mcp: MemoryCachePolicy,

    ctx: WGPUContext,

    debug_label: BufferDebugLabel,
    usage: Usage,

    store: SlotMap<BufferId, BufferEntry>,
    free_pool: BTreeMap<u64, Vec<BufferId>>,
    pending: HashSet<BufferId>,

    clock: usize,
    last_eviction: usize,
}

const MAX_ABSOLUTE_LIMIT: u64 = 256 * 1024;

fn validate_size(cs: u64, rs: u64) -> bool {
    cs.saturating_sub(rs) < (rs / 2) && cs.saturating_sub(rs) <= MAX_ABSOLUTE_LIMIT
}

#[derive(Debug)]
enum AllocatorError {
    DoubleFreeAttempt,
    OutOfMemory,
}

fn try_create_buffer(
    device: &wgpu::Device,
    desc: &BufferDescriptor,
) -> Result<wgpu::Buffer, AllocatorError> {
    let scope = device.push_error_scope(wgpu::ErrorFilter::OutOfMemory);

    let buffer = device.create_buffer(desc);

    let err = pollster::block_on(scope.pop());

    match err {
        Some(wgpu::Error::OutOfMemory { source: _ }) => {
            return Err(AllocatorError::OutOfMemory);
        }
        Some(_) => {
            panic!("Something unexpected happened when creating a buffer")
        }
        None => return Ok(buffer),
    }
}

impl BufferAllocator {
    fn new(ctx: WGPUContext, usage: Usage, mcp: MemoryCachePolicy) -> Self {
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
            clock: 0,
            last_eviction: 0,
            mcp,
        }
    }

    fn request(&mut self, size: u64) -> Result<BufferId, AllocatorError> {
        self.clock += 1;

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

                    self.store[candidate].timestamp = self.clock;
                    Ok(candidate)
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

    fn allocate_new_buffer(&mut self, size: u64) -> Result<BufferId, AllocatorError> {
        let wgpu_usage = match self.usage {
            Usage::Storage => {
                BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST
            }
            Usage::Readback => BufferUsages::MAP_READ | BufferUsages::COPY_DST,
        };

        let label = format!("{}_{}", self.debug_label.prefix, self.debug_label.counter);

        let desc = BufferDescriptor {
            label: Some(&label),
            mapped_at_creation: false,
            size,
            usage: wgpu_usage,
        };
        let raw = try_create_buffer(&self.ctx.device, &desc);

        raw.map(|raw| {
            self.debug_label.counter += 1;
            self.store.insert(BufferEntry {
                // TODO: Change when cleanup is implemented
                raw: Arc::new(raw),
                capacity: size,
                timestamp: self.clock,
            })
        })
    }

    fn free(&mut self, buf: BufferId) -> Result<(), AllocatorError> {
        if self.pending.insert(buf) {
            Ok(())
        } else {
            Err(AllocatorError::DoubleFreeAttempt)
        }
    }

    fn reclaim(&mut self) {
        if self.pending.is_empty() {
            return;
        }

        // Wait for all pending work to be complete
        let _ = self
            .ctx
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

    fn evict(&mut self, hard: bool, target: Option<u64>) {
        if self.free_pool.is_empty() {
            return;
        };

        // TTL eviction pass
        if !hard {
            self.free_pool.iter_mut().for_each(|(_, v)| {
                v.retain(|id| {
                    let pred = (self.clock - self.store[*id].timestamp) < self.mcp.max_ttl;

                    if !pred {
                        self.store.remove(*id);
                    }

                    pred
                });
            });
            self.free_pool.retain(|_, v| !v.is_empty());
        }

        let cache_limit = (self.in_use_size() as f64 * self.mcp.cache_to_used_proportion) as u64;
        let cache_limit = cache_limit.clamp(self.mcp.soft_cache_limit, self.mcp.hard_cache_limit);

        // Cache limit pass
        let target = if let Some(target) = target {
            target
        } else {
            let fps = self
                .free_pool
                .iter()
                .fold(0, |sum, (k, v)| sum + k * v.len() as u64);

            fps.saturating_sub(cache_limit)
        };

        if target > 0 {
            let mut ids = self
                .free_pool
                .values()
                .flatten()
                .map(|id| (*id, self.store[*id].timestamp))
                .collect::<Vec<(BufferId, usize)>>();
            ids.sort_by_key(|(_, time)| *time);

            let mut for_deletion = HashSet::<BufferId>::new();
            let mut released = 0;
            for (id, _) in ids.iter() {
                for_deletion.insert(*id);
                released += self.store[*id].capacity;

                if released >= target {
                    break;
                }
            }

            self.free_pool.iter_mut().for_each(|(_, v)| {
                v.retain(|id| {
                    let pred = for_deletion.contains(id);

                    if pred {
                        self.store.remove(*id);
                    }

                    !pred
                });
            });
            self.free_pool.retain(|_, v| !v.is_empty());
        }

        self.last_eviction = self.clock;
    }

    fn try_evict(&mut self) {
        if self.clock - self.last_eviction > self.mcp.eviction_debounce {
            self.evict(false, None);
        }
    }

    fn total_size(&self) -> u64 {
        self.store.iter().fold(0, |acc, (_k, v)| acc + v.capacity)
    }

    fn pending_size(&self) -> u64 {
        self.pending
            .iter()
            .fold(0, |acc, id| acc + self.store[*id].capacity)
    }

    fn free_size(&self) -> u64 {
        self.free_pool
            .values()
            .flatten()
            .fold(0, |acc, id| acc + self.store[*id].capacity)
    }

    fn in_use_size(&self) -> u64 {
        self.total_size() - self.free_size() - self.pending_size()
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

pub enum BufferStatus {
    InUse,
    Pending,
    Free,
}

pub struct BufferStat {
    pub capacity: u64,
    pub status: BufferStatus,
}

pub struct BufferAllocStats {
    pub buffers: Vec<BufferStat>,
}

impl<T: usage_marker::BufferUsageMarker> BufferAllocatorRef<T> {
    pub fn stats(&self) -> BufferAllocStats {
        let alloc = self.alloc.lock().unwrap();
        BufferAllocStats {
            buffers: alloc
                .store
                .iter()
                .map(|(k, v)| {
                    let status = if alloc.pending.contains(&k) {
                        BufferStatus::Pending
                    } else if alloc.free_pool.contains_key(&v.capacity) {
                        if alloc.free_pool[&v.capacity].contains(&k) {
                            BufferStatus::Free
                        } else {
                            BufferStatus::InUse
                        }
                    } else {
                        BufferStatus::InUse
                    };
                    BufferStat {
                        capacity: v.capacity,
                        status,
                    }
                })
                .collect(),
        }
    }

    pub fn reclaim(&self) {
        self.alloc.lock().unwrap().reclaim();
    }

    pub fn try_evict(&self) {
        self.alloc.lock().unwrap().try_evict();
    }

    pub fn hard_evict(&self, target: Option<u64>) {
        self.alloc.lock().unwrap().evict(true, target);
    }
}

impl BufferAllocatorRef<usage_marker::Storage> {
    pub fn new(ctx: WGPUContext) -> Self {
        Self {
            alloc: Arc::new(Mutex::new(BufferAllocator::new(
                ctx.clone(),
                Usage::Storage,
                MemoryCachePolicy {
                    cache_to_used_proportion: 0.5,
                    soft_cache_limit: 1024 * 1024 * 1024,
                    hard_cache_limit: 1024 * 1024 * 1024 * 2,
                    max_ttl: 128,
                    eviction_debounce: 128,
                },
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
                MemoryCachePolicy {
                    cache_to_used_proportion: 0.5,
                    soft_cache_limit: 1024 * 1024 * 1024,
                    hard_cache_limit: 1024 * 1024 * 1024 * 2,
                    max_ttl: 128,
                    eviction_debounce: 128,
                },
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
#[derive(Debug)]
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

        let buf = self.alloc.lock().unwrap().request(size);
        let buf =
            match buf {
                Ok(buf) => buf,
                Err(_) => {
                    rt().hard_evict(Some(size));

                    self.alloc.lock().unwrap().request(size).expect(
                        "Buffer allocation failed after emergency eviction attempt. Aborting!",
                    )
                }
            };

        let alloc = self.alloc.lock().unwrap();
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

impl<T: usage_marker::BufferUsageMarker> BufferLease<T> {
    pub fn size(&self) -> u64 {
        self.size
    }
}

#[derive(Debug)]
pub enum DownloadError {
    IncompatibleSizes,
}

impl BufferLease<usage_marker::Readback> {
    pub fn download(&self, storage: &BufferLease<Storage>) -> Result<Vec<f32>, DownloadError> {
        if self.size() != storage.size() {
            return Err(DownloadError::IncompatibleSizes);
        }

        let mut encoder =
            self.alloc
                .ctx
                .device
                .create_command_encoder(&wgpu::wgt::CommandEncoderDescriptor {
                    label: Some("staging buffer copy encoder"),
                });
        encoder.copy_buffer_to_buffer(&storage.raw, 0, &self.raw, 0, storage.size);
        self.alloc.ctx.queue.submit(Some(encoder.finish()));

        let (tx, rx) = channel();

        self.raw
            .map_async(wgpu::MapMode::Read, ..self.size, move |result| {
                tx.send(result).unwrap()
            });

        let _ = self
            .alloc
            .ctx
            .device
            .poll(wgpu::PollType::wait_indefinitely());

        rx.recv().unwrap().unwrap();

        let result = {
            let bytes = self.raw.get_mapped_range(..self.size);
            bytemuck::cast_slice::<u8, f32>(&bytes).to_vec()
        };

        self.raw.unmap();

        Ok(result)
    }
}

impl BufferLease<usage_marker::Storage> {
    pub fn set(&self, data: &[u8]) {
        assert!(self.size == data.len() as u64);

        let queue = self.alloc.ctx.queue.clone();

        queue.write_buffer(&self.raw, 0, data);
    }

    pub(crate) fn raw(&self) -> &wgpu::Buffer {
        &self.raw
    }
}

impl AsBindingResource for BufferLease<usage_marker::Storage> {
    fn as_binding_resource(&self) -> wgpu::BindingResource<'_> {
        wgpu::BindingResource::Buffer(wgpu::BufferBinding {
            buffer: &self.raw,
            offset: 0,
            size: Some(NonZeroU64::new(self.size as u64).unwrap()),
        })
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
                .expect("No WGPU adapter available for buffer allocator tests");
            WGPUContext::new(adapter)
        })
        .clone()
    }

    fn test_policy() -> MemoryCachePolicy {
        MemoryCachePolicy {
            cache_to_used_proportion: 0.0,
            soft_cache_limit: 0,
            hard_cache_limit: 1024,
            max_ttl: 4,
            eviction_debounce: 2,
        }
    }

    fn storage_alloc() -> BufferAllocator {
        BufferAllocator::new(test_ctx(), Usage::Storage, test_policy())
    }

    #[test]
    fn validate_size_accepts_close_fit_and_rejects_wasteful_fit() {
        assert!(validate_size(16, 12));
        assert!(validate_size(1024 + 128, 1024));

        assert!(!validate_size(16, 8));
        assert!(!validate_size(1024 * 1024, 4));
    }

    #[test]
    fn request_allocates_new_buffer_and_tracks_active_bytes() {
        let mut alloc = storage_alloc();

        let id = alloc.request(16).unwrap();

        assert_eq!(alloc.clock, 1);
        assert_eq!(alloc.debug_label.counter, 1);
        assert_eq!(alloc.store[id].capacity, 16);
        assert_eq!(alloc.store[id].timestamp, 1);
        assert_eq!(alloc.total_size(), 16);
        assert_eq!(alloc.in_use_size(), 16);
        assert_eq!(alloc.pending_size(), 0);
        assert_eq!(alloc.free_size(), 0);
    }

    #[test]
    fn reclaim_moves_pending_buffers_to_free_pool() {
        let mut alloc = storage_alloc();
        let id = alloc.request(16).unwrap();

        alloc.free(id).unwrap();
        assert!(alloc.pending.contains(&id));
        assert_eq!(alloc.pending_size(), 16);
        assert_eq!(alloc.free_size(), 0);

        alloc.reclaim();

        assert!(alloc.pending.is_empty());
        assert_eq!(alloc.pending_size(), 0);
        assert_eq!(alloc.free_size(), 16);
        assert_eq!(alloc.free_pool.get(&16).unwrap(), &vec![id]);
    }

    #[test]
    fn request_reuses_compatible_free_buffer_and_refreshes_timestamp() {
        let mut alloc = storage_alloc();
        let id = alloc.request(16).unwrap();
        alloc.free(id).unwrap();
        alloc.reclaim();

        let reused = alloc.request(12).unwrap();

        assert_eq!(reused, id);
        assert_eq!(alloc.clock, 2);
        assert_eq!(alloc.debug_label.counter, 1);
        assert_eq!(alloc.store[id].timestamp, 2);
        assert!(alloc.free_pool.is_empty());
        assert_eq!(alloc.in_use_size(), 16);
    }

    #[test]
    fn request_keeps_wasteful_candidate_in_pool_and_allocates_new_buffer() {
        let mut alloc = storage_alloc();
        let large = alloc.request(1024).unwrap();
        alloc.free(large).unwrap();
        alloc.reclaim();

        let small = alloc.request(4).unwrap();

        assert_ne!(small, large);
        assert_eq!(alloc.debug_label.counter, 2);
        assert_eq!(alloc.store[small].capacity, 4);
        assert_eq!(alloc.free_pool.get(&1024).unwrap(), &vec![large]);
        assert_eq!(alloc.free_size(), 1024);
        assert_eq!(alloc.in_use_size(), 4);
    }

    #[test]
    fn soft_evict_removes_expired_free_buffers() {
        let mut alloc = storage_alloc();
        let id = alloc.request(16).unwrap();
        alloc.free(id).unwrap();
        alloc.reclaim();

        alloc.clock = alloc.store[id].timestamp + alloc.mcp.max_ttl;
        alloc.evict(false, None);

        assert!(!alloc.store.contains_key(id));
        assert!(alloc.free_pool.is_empty());
    }

    #[test]
    fn hard_evict_ignores_ttl_and_deletes_until_target_is_met() {
        let mut alloc = storage_alloc();
        let old = alloc.request(16).unwrap();
        alloc.free(old).unwrap();
        alloc.reclaim();

        let new = alloc.request(32).unwrap();
        alloc.free(new).unwrap();
        alloc.reclaim();

        alloc.evict(true, Some(17));

        assert!(!alloc.store.contains_key(old));
        assert!(!alloc.store.contains_key(new));
        assert!(alloc.free_pool.is_empty());
    }

    #[test]
    fn try_evict_respects_debounce() {
        let mut alloc = storage_alloc();
        let id = alloc.request(16).unwrap();
        alloc.free(id).unwrap();
        alloc.reclaim();

        alloc.clock = alloc.mcp.eviction_debounce;
        alloc.try_evict();
        assert!(alloc.store.contains_key(id));
        assert_eq!(alloc.last_eviction, 0);

        alloc.clock = alloc.mcp.eviction_debounce + 1;
        alloc.try_evict();
        assert!(!alloc.store.contains_key(id));
        assert_eq!(alloc.last_eviction, alloc.clock);
    }

    #[test]
    fn stats_reports_in_use_pending_and_free_buffers() {
        let alloc_ref = BufferAllocatorRef::<usage_marker::Storage> {
            alloc: Arc::new(Mutex::new(storage_alloc())),
            ctx: test_ctx(),
            _tag: PhantomData,
        };

        let in_use = alloc_ref.request(16);
        let pending = alloc_ref.request(32);
        drop(pending);
        let free = alloc_ref.request(64);
        drop(free);
        alloc_ref.reclaim();

        let stats = alloc_ref.stats();
        assert_eq!(stats.buffers.len(), 3);
        assert_eq!(
            stats
                .buffers
                .iter()
                .filter(|b| matches!(b.status, BufferStatus::InUse))
                .count(),
            1
        );
        assert_eq!(
            stats
                .buffers
                .iter()
                .filter(|b| matches!(b.status, BufferStatus::Free))
                .count(),
            2
        );

        drop(in_use);
        let stats = alloc_ref.stats();
        assert_eq!(
            stats
                .buffers
                .iter()
                .filter(|b| matches!(b.status, BufferStatus::Pending))
                .count(),
            1
        );
    }
}
