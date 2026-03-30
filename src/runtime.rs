use std::sync::{Arc, Mutex, OnceLock};

use crate::{
    autograd::GradStore,
    buffer_alloc::{
        BufferAllocatorRef,
        usage_marker::{Readback, Storage},
    },
    kernel_registry::KernelRegistry,
    metadata_arena::MetadataArena,
};

#[derive(Debug, Clone)]
pub struct WGPUContext {
    pub(crate) device: Arc<wgpu::Device>,
    pub(crate) queue: Arc<wgpu::Queue>,
}

impl WGPUContext {
    pub fn new(adapter: wgpu::Adapter) -> Self {
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("Torchic device"),
            required_limits: adapter.limits(),
            required_features: wgpu::Features::empty(),
            memory_hints: wgpu::MemoryHints::MemoryUsage,
            ..Default::default()
        }))
        .expect("This operation should be successfull, maybe there is a problem with your adapter");

        Self {
            device: Arc::new(device),
            queue: Arc::new(queue),
        }
    }

    pub fn list_adapters() -> Vec<wgpu::Adapter> {
        let instance = wgpu::Instance::default();
        pollster::block_on(instance.enumerate_adapters(wgpu::Backends::all()))
    }
}

#[derive(Debug)]
pub(crate) struct Runtime {
    pub(crate) ctx: WGPUContext,
    pub(crate) storage_buffer_alloc: BufferAllocatorRef<Storage>,
    pub(crate) readback_buffer_alloc: BufferAllocatorRef<Readback>,
    pub(crate) metadata_arena: MetadataArena,
    pub(crate) grad_store: GradStore,
    pub(crate) kernel_registry: Mutex<KernelRegistry>,
}

static RUNTIME: OnceLock<Arc<Runtime>> = OnceLock::new();

pub fn init_runtime(adapter: wgpu::Adapter) {
    let ctx = WGPUContext::new(adapter);

    let runtime = Runtime {
        ctx: ctx.clone(),
        storage_buffer_alloc: BufferAllocatorRef::<Storage>::new(ctx.clone()),
        readback_buffer_alloc: BufferAllocatorRef::<Readback>::new(ctx.clone()),
        metadata_arena: MetadataArena::new(ctx.clone(), 1024 * 1024 /* 1MB */),
        grad_store: GradStore::new(),
        kernel_registry: Mutex::new(KernelRegistry::new(ctx)),
    };

    RUNTIME
        .set(Arc::new(runtime))
        .expect("This should generally be ok");
}

pub(crate) fn rt() -> Arc<Runtime> {
    RUNTIME.get().expect("Runtime is not initialized! Initialize runtime with torchic::runtime::init_runtime(adapter)").clone()
}

pub fn dump_stats() {
    let rt = rt();
    println!("{:#?}", rt.storage_buffer_alloc.stats());
}
