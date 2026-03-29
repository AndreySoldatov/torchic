use std::{
    collections::{HashMap, HashSet},
    sync::Mutex,
};

use crate::{
    ops::{self, OpType},
    runtime::rt,
    tensor::Tensor,
};

#[derive(Debug)]
pub(crate) struct GradNode {
    pub(crate) op: OpType,
    pub(crate) parents: Vec<Tensor>,
}

fn topo_recursive(tensor: Tensor, result: &mut Vec<Tensor>, visited: &mut HashSet<u64>) {
    if visited.insert(tensor.id()) {
        result.push(tensor.clone());
        if let Some(n) = &tensor.inner.grad_node {
            for parent in &n.parents {
                topo_recursive(parent.clone(), result, visited);
            }
        }
    }
}

fn topo(tensor: Tensor) -> Vec<Tensor> {
    let mut result = vec![];
    let mut visited: HashSet<u64> = HashSet::new();

    topo_recursive(tensor, &mut result, &mut visited);

    return result;
}

pub(crate) fn backward(tensor: Tensor) {
    let topo = topo(tensor.clone());

    rt().grad_store
        .map
        .lock()
        .unwrap()
        .insert(tensor.id(), Tensor::ones(tensor.shape().to_vec(), false));

    for t in topo {
        if let Some(n) = &t.inner.grad_node {
            match n.op {
                OpType::Add => {
                    add_backward(t.clone(), n.parents[0].clone(), n.parents[1].clone());
                }
            }
        }
    }
}

fn add_backward(out: Tensor, lhs: Tensor, rhs: Tensor) {
    let out_grad = rt()
        .grad_store
        .map
        .lock()
        .unwrap()
        .get(&out.id())
        .unwrap()
        .clone();

    if lhs.requires_grad() {
        rt().grad_store.acc(lhs.id(), out_grad.clone());
    }
    if rhs.requires_grad() {
        rt().grad_store.acc(rhs.id(), out_grad);
    }
}

#[derive(Debug)]
pub(crate) struct GradStore {
    pub(crate) map: Mutex<HashMap<u64, Tensor>>,
    pub(crate) orphans: Mutex<HashSet<u64>>,
}

impl GradStore {
    pub(crate) fn new() -> Self {
        Self {
            map: Mutex::new(HashMap::new()),
            orphans: Mutex::new(HashSet::new()),
        }
    }

    pub(crate) fn acc(&self, id: u64, t: Tensor) {
        self.map
            .lock()
            .unwrap()
            .entry(id)
            .and_modify(|e| *e = ops::add(e, &t).unwrap())
            .or_insert(t);
    }

    pub(crate) fn cleanup(&self) {
        let orphans = self.orphans.lock().unwrap().clone();
        for orphan in orphans {
            self.map.lock().unwrap().remove(&orphan);
        }

        self.orphans.lock().unwrap().clear();
    }

    pub fn add_orphan(&self, id: u64) {
        self.orphans.lock().unwrap().insert(id);
    }
}
