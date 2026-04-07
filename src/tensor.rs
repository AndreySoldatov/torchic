use std::sync::{Arc, atomic::AtomicU64};

use crate::{
    AsBindingResource,
    autograd::{self, GradNode},
    buffer_alloc::{BufferLease, usage_marker::Storage},
    ops::{self, TensorOpError},
    runtime::{cleanup, rt},
};

static TENSOR_ID_COUNTER: AtomicU64 = AtomicU64::new(0);

pub(crate) fn get_tensor_id() -> u64 {
    TENSOR_ID_COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
}

#[derive(Debug)]
pub(crate) struct TensorInner {
    pub(crate) id: u64,
    pub(crate) buf: BufferLease<Storage>,
    pub(crate) shape: Vec<usize>,

    pub(crate) requires_grad: bool,
    pub(crate) grad_node: Option<GradNode>,
}

impl Drop for TensorInner {
    fn drop(&mut self) {
        rt().grad_store.add_orphan(self.id);
    }
}

#[derive(Clone, Debug)]
pub struct Tensor {
    pub(crate) inner: Arc<TensorInner>,
}

pub const DTYPE_SIZE: usize = 4;

impl Tensor {
    pub(crate) fn id(&self) -> u64 {
        self.inner.id
    }

    pub(crate) fn bsize(&self) -> usize {
        self.numel() * DTYPE_SIZE
    }
}

impl AsBindingResource for Tensor {
    fn as_binding_resource(&self) -> wgpu::BindingResource<'_> {
        self.inner.buf.as_binding_resource()
    }
}

impl Tensor {
    pub fn shape(&self) -> &[usize] {
        &self.inner.shape
    }

    pub fn requires_grad(&self) -> bool {
        self.inner.requires_grad
    }

    pub fn numel(&self) -> usize {
        self.inner.shape.iter().product::<usize>()
    }
}

impl Tensor {
    pub fn new(shape: &[usize], data: &[f32], requires_grad: bool) -> Self {
        let bsize = shape.iter().product::<usize>() * DTYPE_SIZE;

        let rt = rt();
        let buf = rt.storage_buffer_alloc.request(bsize as u64);
        buf.set(bytemuck::cast_slice(data));

        Self {
            inner: Arc::new(TensorInner {
                id: get_tensor_id(),
                shape: shape.to_vec(),
                requires_grad,
                buf,
                grad_node: None,
            }),
        }
    }

    pub fn ones(shape: &[usize], requires_grad: bool) -> Self {
        let size = shape.iter().product();
        let data = vec![1.0; size];

        Self::new(shape, &data, requires_grad)
    }

    pub fn grad(&self) -> Option<Self> {
        rt().grad_store
            .map
            .lock()
            .unwrap()
            .get(&self.inner.id)
            .map(|t| t.clone())
    }

    pub fn backward(&self) {
        autograd::backward(self);
        cleanup();
    }

    pub fn to_vec(&self) -> Vec<f32> {
        let staging = rt().readback_buffer_alloc.request(self.bsize() as u64);
        let res = staging.download(&self.inner.buf).unwrap();

        cleanup();

        res
    }
}

impl Tensor {
    pub fn add(&self, other: &Tensor) -> Result<Tensor, ops::TensorOpError> {
        ops::add(self, other)
    }

    pub fn mul(&self, other: &Tensor) -> Result<Tensor, ops::TensorOpError> {
        ops::mul(self, other)
    }

    pub fn sum(&self) -> Result<Tensor, ops::TensorOpError> {
        ops::sum(self)
    }

    pub fn max(&self) -> Result<Tensor, ops::TensorOpError> {
        ops::max(self)
    }

    pub fn matmul(&self, other: &Tensor) -> Result<Tensor, ops::TensorOpError> {
        ops::matmul(self, other)
    }

    pub fn transposed(&self) -> Result<Tensor, ops::TensorOpError> {
        ops::transposed(self)
    }

    pub fn relu(&self) -> Result<Tensor, ops::TensorOpError> {
        ops::relu(self)
    }

    pub fn mul_s(&self, s: f32) -> Result<Tensor, TensorOpError> {
        ops::mul_scalar(self, s)
    }

    pub fn outer(&self, other: &Tensor) -> Result<Tensor, TensorOpError> {
        ops::outer(self, other)
    }
}
