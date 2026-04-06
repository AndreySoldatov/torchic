pub mod autograd;
pub mod buffer_alloc;
pub mod kernel_registry;
pub mod metadata_arena;
pub mod ops;
pub mod runtime;
pub mod tensor;

pub(crate) trait AsBindingResource {
    fn as_binding_resource(&self) -> wgpu::BindingResource<'_>;
}
