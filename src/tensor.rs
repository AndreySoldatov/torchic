use std::sync::{Arc, atomic::AtomicU64};

use crate::{
    autograd::{self, GradNode},
    buffer_alloc::{BufferLease, usage_marker::Storage},
    ops,
    runtime::rt,
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

const DTYPE_SIZE: usize = 4;

impl Tensor {
    pub(crate) fn id(&self) -> u64 {
        self.inner.id
    }

    pub(crate) fn buf_binding(&self) -> wgpu::BindingResource<'_> {
        self.inner.buf.binding()
    }

    pub(crate) fn bsize(&self) -> usize {
        self.inner.shape.iter().product::<usize>() * DTYPE_SIZE
    }
}

impl Tensor {
    pub fn shape(&self) -> &[usize] {
        &self.inner.shape
    }

    pub fn requires_grad(&self) -> bool {
        self.inner.requires_grad
    }
}

impl Tensor {
    pub fn new(shape: Vec<usize>, data: &[f32], requires_grad: bool) -> Self {
        let bsize = shape.iter().product::<usize>() * DTYPE_SIZE;

        let rt = rt();
        let buf = rt.storage_buffer_alloc.request(bsize as u64);
        buf.set(bytemuck::cast_slice(data));

        Self {
            inner: Arc::new(TensorInner {
                id: get_tensor_id(),
                shape,
                requires_grad,
                buf,
                grad_node: None,
            }),
        }
    }

    pub fn ones(shape: Vec<usize>, requires_grad: bool) -> Self {
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
        rt().grad_store.cleanup();
        rt().storage_buffer_alloc.reclaim();
        rt().readback_buffer_alloc.reclaim();
    }

    pub fn to_vec(&self) -> Vec<f32> {
        let staging = rt().readback_buffer_alloc.request(self.bsize() as u64);
        staging.download(&self.inner.buf).unwrap()
    }
}

impl Tensor {
    pub fn add(&self, other: &Tensor) -> Result<Tensor, ops::TensorOpError> {
        ops::add(self, other, false)
    }

    pub fn mul(&self, other: &Tensor) -> Result<Tensor, ops::TensorOpError> {
        ops::mul(self, other, false)
    }
}
