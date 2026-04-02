use std::{
    collections::{HashMap, HashSet},
    sync::Mutex,
};

use crate::{
    ops::{self, OpType, ReduceOpType},
    runtime::rt,
    tensor::Tensor,
};

#[derive(Debug)]
pub(crate) struct GradNode {
    pub(crate) op: OpType,
    pub(crate) parents: Vec<Tensor>,
}

fn topo_recursive(tensor: &Tensor, result: &mut Vec<Tensor>, visited: &mut HashSet<u64>) {
    if visited.insert(tensor.id()) {
        result.push(tensor.clone());
        if let Some(n) = &tensor.inner.grad_node {
            for parent in &n.parents {
                topo_recursive(parent, result, visited);
            }
        }
    }
}

fn topo(tensor: &Tensor) -> Vec<Tensor> {
    let mut result = vec![];
    let mut visited: HashSet<u64> = HashSet::new();

    topo_recursive(tensor, &mut result, &mut visited);

    return result;
}

pub(crate) fn backward(tensor: &Tensor) {
    let topo = topo(&tensor);

    rt().grad_store
        .map
        .lock()
        .unwrap()
        .insert(tensor.id(), Tensor::ones(tensor.shape(), false));

    for t in topo {
        if let Some(n) = &t.inner.grad_node {
            match &n.op {
                OpType::BinopEwizeType(btype) => match btype {
                    ops::BinopEwizeType::Add => add_backward(&t, &n.parents[0], &n.parents[1]),
                    ops::BinopEwizeType::Mul => mul_backward(&t, &n.parents[0], &n.parents[1]),
                },
                OpType::Reduce(red) => match red {
                    ReduceOpType::Sum => sum_backward(&t, &n.parents[0]),
                },
            }
        }
    }
}

fn add_backward(out: &Tensor, lhs: &Tensor, rhs: &Tensor) {
    let out_grad = rt()
        .grad_store
        .map
        .lock()
        .unwrap()
        .get(&out.id())
        .unwrap()
        .clone();

    if lhs.requires_grad() {
        rt().grad_store.acc(lhs.id(), &out_grad);
    }
    if rhs.requires_grad() {
        rt().grad_store.acc(rhs.id(), &out_grad);
    }
}

fn mul_backward(out: &Tensor, lhs: &Tensor, rhs: &Tensor) {
    let out_grad = rt()
        .grad_store
        .map
        .lock()
        .unwrap()
        .get(&out.id())
        .unwrap()
        .clone();

    if lhs.requires_grad() {
        rt().grad_store
            .acc(lhs.id(), &ops::mul(&out_grad, rhs, true).unwrap());
    }
    if rhs.requires_grad() {
        rt().grad_store
            .acc(rhs.id(), &ops::mul(&out_grad, lhs, true).unwrap());
    }
}

fn sum_backward(out: &Tensor, p: &Tensor) {
    if p.requires_grad() {
        let out_grad = rt()
            .grad_store
            .map
            .lock()
            .unwrap()
            .get(&out.id())
            .unwrap()
            .clone();

        let grad_scal = out_grad.to_vec()[0];

        rt().grad_store.acc(
            p.id(),
            &Tensor::new(p.shape(), &vec![grad_scal; p.numel()], false),
        );
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

    pub(crate) fn acc(&self, id: u64, t: &Tensor) {
        self.map
            .lock()
            .unwrap()
            .entry(id)
            .and_modify(|e| *e = ops::add(e, t, true).unwrap())
            .or_insert(t.clone());
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
