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

    pub(crate) fn readback(&self) -> Vec<f32> {
        let staging = rt().readback_buffer_alloc.request(self.bsize() as u64);
        staging.download(&self.inner.buf).unwrap()
    }

    pub(crate) fn zero_grad(&self) {
        if !rt().grad_store.map.lock().unwrap().contains_key(&self.id()) {
            return;
        }

        rt().grad_store
            .map
            .lock()
            .unwrap()
            .remove(&self.id())
            .unwrap();
    }

    pub(crate) fn buf(&self) -> &BufferLease<Storage> {
        &self.inner.buf
    }

    // The only mutating operation. TO BE USED ONLY IN ADAM STEP FOR NOW
    pub(crate) fn assign(&mut self, other: &Tensor) {
        assert!(self.shape() == other.shape());
        let rt = rt();

        let mut encoder =
            rt.ctx
                .device
                .create_command_encoder(&wgpu::wgt::CommandEncoderDescriptor {
                    label: Some("Tensor copy encoder"),
                });
        encoder.copy_buffer_to_buffer(
            other.buf().raw(),
            0,
            self.inner.buf.raw(),
            0,
            Some(self.bsize() as u64),
        );

        rt.ctx.queue.submit(Some(encoder.finish()));
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
        let res = self.readback();
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

    pub(crate) fn div(&self, other: &Tensor) -> Result<Tensor, ops::TensorOpError> {
        ops::div(self, other)
    }

    pub(crate) fn sub(&self, other: &Tensor) -> Result<Tensor, ops::TensorOpError> {
        ops::sub(self, other)
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

    pub(crate) fn sqrt(&self) -> Result<Tensor, ops::TensorOpError> {
        ops::sqrt(self)
    }

    pub fn mul_s(&self, s: f32) -> Result<Tensor, TensorOpError> {
        ops::mul_scalar(self, s)
    }

    pub(crate) fn add_s(&self, s: f32) -> Result<Tensor, TensorOpError> {
        ops::add_scalar(self, s)
    }

    pub fn outer(&self, other: &Tensor) -> Result<Tensor, TensorOpError> {
        ops::outer(self, other)
    }

    pub fn cross_entropy_loss(&self, targets: &Tensor) -> Result<Tensor, TensorOpError> {
        ops::cross_entropy_loss(self, targets)
    }
}
