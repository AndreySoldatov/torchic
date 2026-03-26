use std::{
    collections::HashMap,
    sync::{Arc, LazyLock, Mutex, atomic::AtomicU64},
};

use wgpu::wgc::id::BufferId;

use crate::{
    buffer_alloc::{BufferLease, usage_marker::Storage},
    runtime::rt,
};

struct GradStore {
    map: HashMap<u64, Tensor>,
}

impl GradStore {
    fn new() -> Self {
        Self {
            map: HashMap::new(),
        }
    }

    fn acc(&mut self, id: u64, t: Tensor) {
        self.map
            .entry(id)
            .and_modify(|e| *e = e.clone().add(t.clone()))
            .or_insert(t);
    }
}

static GRAD_STORE: LazyLock<Mutex<GradStore>> = LazyLock::new(|| Mutex::new(GradStore::new()));

static ID_COUNTER: AtomicU64 = AtomicU64::new(0);

fn get_id() -> u64 {
    ID_COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
}

struct TensorInner {
    id: u64,
    buf: BufferLease<Storage>,
    shape: Vec<usize>,

    requires_grad: bool,
    grad_fn: Option<Box<dyn Fn() + Send + Sync + 'static>>,
}

impl Drop for TensorInner {
    fn drop(&mut self) {
        GRAD_STORE.lock().unwrap().map.remove(&self.id);
    }
}

#[derive(Clone)]
pub struct Tensor(Arc<TensorInner>);

const DTYPE_SIZE: usize = 4;

impl Tensor {
    fn _new(shape: Vec<usize>, data: &[f32], requires_grad: bool) -> Self {
        let bsize = shape.iter().product::<usize>() * DTYPE_SIZE;

        let rt = rt();
        let buf = rt.storage_buffer_alloc.request(bsize as u64);
        buf.set(bytemuck::cast_slice(data));

        Self(Arc::new(TensorInner {
            id: get_id(),
            shape,
            requires_grad,
            buf,
            grad_fn: None,
        }))
    }

    pub fn new(shape: Vec<usize>, data: &[f32]) -> Self {
        Self::_new(shape, data, false)
    }
    pub fn new_requires_grad(shape: Vec<usize>, data: &[f32]) -> Self {
        Self::_new(shape, data, true)
    }

    fn add(self, other: Tensor) -> Self {
        let requires_grad = self.0.requires_grad || other.0.requires_grad;
        let bsize = self.0.shape.iter().product::<usize>() * DTYPE_SIZE;

        let self_id = self.0.id;
        let other_id = other.0.id;
        let self_rg = self.0.requires_grad;
        let other_rg = other.0.requires_grad;
        let out_id = get_id();

        let grad_fn: Option<Box<dyn Fn() + Send + Sync + 'static>> = if requires_grad {
            Some(Box::new(move || {
                let out_grad = GRAD_STORE.lock().unwrap().map.get(&out_id).unwrap().clone();
                if self_rg {
                    GRAD_STORE.lock().unwrap().acc(self_id, out_grad.clone());
                }
                if other_rg {
                    GRAD_STORE.lock().unwrap().acc(other_id, out_grad);
                }
            }))
        } else {
            None
        };

        Self(Arc::new(TensorInner {
            id: out_id,
            buf: BUFFER_ALLOC.lock().unwrap().request(bsize),
            shape: self.0.shape.clone(),
            requires_grad,
            grad_fn,
        }))
    }

    fn grad(&self) -> Option<Self> {
        GRAD_STORE
            .lock()
            .unwrap()
            .map
            .get(&self.0.id)
            .map(|t| t.clone())
    }

    fn backward(&self) {}

    fn realize(self) -> (Vec<usize>, Vec<f32>) {
        todo!()
    }
}
