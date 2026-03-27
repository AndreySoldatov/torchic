use std::sync::{Arc, atomic::AtomicU64};

use crate::{
    autograd::{self, GradNode},
    buffer_alloc::{BufferLease, usage_marker::Storage},
    runtime::rt,
};

static ID_COUNTER: AtomicU64 = AtomicU64::new(0);

fn get_id() -> u64 {
    ID_COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
}

#[derive(Debug)]
pub(crate) struct TensorInner {
    id: u64,
    buf: BufferLease<Storage>,
    shape: Vec<usize>,

    requires_grad: bool,
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
                id: get_id(),
                shape,
                requires_grad,
                buf,
                grad_node: None,
            }),
        }
    }

    pub fn ones(shape: Vec<usize>, requires_grad: bool) -> Self {
        let bsize = shape.iter().product();
        let data = vec![1.0; bsize];

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
