use std::sync::{Arc, atomic::AtomicU64};

use crate::{
    autograd::{self, GradNode},
    buffer_alloc::{BufferLease, usage_marker::Storage},
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
    // grad_fn: Option<Box<dyn Fn() + Send + Sync + 'static>>,
    pub(crate) grad_node: Option<GradNode>,
}

impl Drop for TensorInner {
    fn drop(&mut self) {
        rt().grad_store.lock().unwrap().map.remove(&self.id);
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

    pub(crate) fn buf_binding(&self) -> wgpu::BindingResource {
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

    fn grad(&self) -> Option<Self> {
        rt().grad_store
            .lock()
            .unwrap()
            .map
            .get(&self.inner.id)
            .map(|t| t.clone())
    }

    pub fn backward(&self) {
        autograd::backward(self.clone());
    }

    pub fn to_shape_and_vec(self) -> (Vec<usize>, Vec<f32>) {
        todo!()
    }
}
