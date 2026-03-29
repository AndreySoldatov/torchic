use std::collections::{HashMap, HashSet};

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
        .lock()
        .unwrap()
        .map
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
        .lock()
        .unwrap()
        .map
        .get(&out.id())
        .unwrap()
        .clone();

    if lhs.requires_grad() {
        rt().grad_store
            .lock()
            .unwrap()
            .acc(lhs.id(), out_grad.clone());
    }
    if rhs.requires_grad() {
        rt().grad_store.lock().unwrap().acc(rhs.id(), out_grad);
    }
}

#[derive(Debug)]
pub(crate) struct GradStore {
    pub(crate) map: HashMap<u64, Tensor>,
}

impl GradStore {
    pub(crate) fn new() -> Self {
        Self {
            map: HashMap::new(),
        }
    }

    pub(crate) fn acc(&mut self, id: u64, t: Tensor) {
        self.map
            .entry(id)
            .and_modify(|e| *e = ops::add(e, &t).unwrap())
            .or_insert(t);
    }
}
