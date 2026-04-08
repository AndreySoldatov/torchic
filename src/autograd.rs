use std::{
    collections::{HashMap, HashSet},
    sync::Mutex,
};

use crate::{
    ops::{self, OpType, ReduceOpType, ScalarEwizeType},
    runtime::{no_grad, rt},
    tensor::Tensor,
};

#[derive(Debug)]
pub(crate) struct GradNode {
    pub(crate) op: OpType,
    pub(crate) parents: Vec<Tensor>,
    pub(crate) meta: Option<GradNodeMeta>,
}

#[derive(Debug)]
pub(crate) enum GradNodeMeta {
    Scalar(f32),
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
    let _ng = no_grad().unwrap();

    rt().grad_store
        .map
        .lock()
        .unwrap()
        .insert(tensor.id(), Tensor::ones(tensor.shape(), false));

    for t in topo {
        if let Some(n) = &t.inner.grad_node {
            let out_grad = rt()
                .grad_store
                .map
                .lock()
                .unwrap()
                .get(&t.id())
                .unwrap()
                .clone();

            match &n.op {
                OpType::BinopEwizeType(typ) => match typ {
                    ops::BinopEwizeType::Add => {
                        bin_add_backward(&out_grad, &n.parents[0], &n.parents[1])
                    }
                    ops::BinopEwizeType::Mul => {
                        bin_mul_backward(&out_grad, &n.parents[0], &n.parents[1])
                    }
                    ops::BinopEwizeType::Div => todo!(),
                    ops::BinopEwizeType::Sub => todo!(),
                },
                OpType::UnopEwizeType(typ) => match typ {
                    ops::UnopEwizeType::Relu => relu_backward(&out_grad, &n.parents[0]),
                    ops::UnopEwizeType::ReluBackward => {
                        panic!(
                            "Relu backward cannot be called from user code with grad calculation"
                        )
                    }
                    ops::UnopEwizeType::Sqrt => todo!(),
                },
                OpType::Reduce(typ) => match typ {
                    ReduceOpType::Sum => sum_backward(&out_grad, &n.parents[0]),
                    ReduceOpType::Max => todo!(), // Too much complexity with current architecture
                },
                OpType::Transpose => transpose_backward(&out_grad, &n.parents[0]),
                OpType::Matmul => matmul_backward(&out_grad, &n.parents[0], &n.parents[1]),
                OpType::ScalarEwize(typ) => match typ {
                    ScalarEwizeType::Mul => {
                        let GradNodeMeta::Scalar(s) = n.meta.as_ref().unwrap();
                        scal_mul_backward(&out_grad, &n.parents[0], *s)
                    }
                    ScalarEwizeType::Add => todo!(),
                },
                OpType::Outer => outer_backward(&out_grad, &n.parents[0], &n.parents[1]),
                OpType::CrossEntropyLoss => {
                    cross_entropy_loss_backward(&out_grad, &n.parents[0], &n.parents[1])
                }
            }
        }
    }
}

fn outer_backward(out_grad: &Tensor, lhs: &Tensor, rhs: &Tensor) {
    if lhs.requires_grad() {
        acc(lhs.id(), &ops::matmul(out_grad, rhs).unwrap());
    }
    if rhs.requires_grad() {
        acc(
            rhs.id(),
            &ops::matmul(&ops::transposed(out_grad).unwrap(), lhs).unwrap(),
        );
    }
}

fn scal_mul_backward(out_grad: &Tensor, p: &Tensor, s: f32) {
    if p.requires_grad() {
        acc(p.id(), &ops::mul_scalar(out_grad, s).unwrap());
    }
}

fn bin_add_backward(out_grad: &Tensor, lhs: &Tensor, rhs: &Tensor) {
    if lhs.requires_grad() {
        acc(lhs.id(), &out_grad);
    }
    if rhs.requires_grad() {
        acc(rhs.id(), &out_grad);
    }
}

fn bin_mul_backward(out_grad: &Tensor, lhs: &Tensor, rhs: &Tensor) {
    if lhs.requires_grad() {
        rt().grad_store
            .acc(lhs.id(), &ops::mul(&out_grad, rhs).unwrap());
    }
    if rhs.requires_grad() {
        rt().grad_store
            .acc(rhs.id(), &ops::mul(&out_grad, lhs).unwrap());
    }
}

fn sum_backward(out_grad: &Tensor, p: &Tensor) {
    if p.requires_grad() {
        assert!(out_grad.shape().len() == 1 && out_grad.shape()[0] == 1);
        let grad_scal = out_grad.readback()[0];

        acc(
            p.id(),
            &Tensor::new(p.shape(), &vec![grad_scal; p.numel()], false),
        );
    }
}

fn transpose_backward(out_grad: &Tensor, p: &Tensor) {
    if p.requires_grad() {
        rt().grad_store
            .acc(p.id(), &ops::transposed(out_grad).unwrap());
    }
}

fn matmul_backward(out_grad: &Tensor, lhs: &Tensor, rhs: &Tensor) {
    if lhs.requires_grad() {
        match (lhs.shape(), rhs.shape()) {
            ([_, _], [_, _]) | ([_], [_, _]) => {
                // A @ B or x @ B
                acc(
                    lhs.id(),
                    &ops::matmul(out_grad, &ops::transposed(rhs).unwrap()).unwrap(),
                );
            }
            ([_, _], [_]) => {
                // A @ y
                acc(lhs.id(), &ops::outer(out_grad, rhs).unwrap());
            }
            ([_], [_]) => {
                // x @ y -> scalar
                assert!(out_grad.shape().len() == 1 && out_grad.shape()[0] == 1);
                let scalar = out_grad.readback()[0];

                acc(lhs.id(), &ops::mul_scalar(rhs, scalar).unwrap());
            }
            _ => {}
        }
    }

    if rhs.requires_grad() {
        match (lhs.shape(), rhs.shape()) {
            ([_, _], [_, _]) | ([_, _], [_]) => {
                // A @ B or A @ y
                acc(
                    rhs.id(),
                    &ops::matmul(&ops::transposed(lhs).unwrap(), out_grad).unwrap(),
                );
            }
            ([_], [_, _]) => {
                // x @ B
                acc(rhs.id(), &ops::outer(lhs, out_grad).unwrap());
            }
            ([_], [_]) => {
                // x @ y -> scalar
                assert!(out_grad.shape().len() == 1 && out_grad.shape()[0] == 1);
                let scalar = out_grad.readback()[0];

                acc(rhs.id(), &ops::mul_scalar(lhs, scalar).unwrap());
            }
            _ => {}
        }
    }
}

fn relu_backward(out_grad: &Tensor, p: &Tensor) {
    if p.requires_grad() {
        acc(
            p.id(),
            &ops::mul(
                out_grad,
                &ops::dispatch_unop_ewize(p, ops::UnopEwizeType::ReluBackward).unwrap(),
            )
            .unwrap(),
        );
    }
}

fn cross_entropy_loss_backward(out_grad: &Tensor, logits: &Tensor, targets: &Tensor) {
    assert!(out_grad.shape().len() == 1 && out_grad.shape()[0] == 1);

    let out_grad_scalar = out_grad.readback()[0];
    let [batch, classes] = logits.shape() else {
        panic!("Cross entropy loss expects batched logits");
    };

    let logits_data = logits.readback();
    let targets_data = targets.readback();

    if logits.requires_grad() {
        let logits_grad = ops::cross_entropy_loss_backward_logits(
            &logits_data,
            &targets_data,
            *batch,
            *classes,
            out_grad_scalar,
        );
        acc(
            logits.id(),
            &Tensor::new(logits.shape(), &logits_grad, false),
        );
    }

    if targets.requires_grad() {
        let targets_grad = ops::cross_entropy_loss_backward_targets(
            &logits_data,
            *batch,
            *classes,
            out_grad_scalar,
        );
        acc(
            targets.id(),
            &Tensor::new(targets.shape(), &targets_grad, false),
        );
    }
}

fn acc(id: u64, t: &Tensor) {
    rt().grad_store.acc(id, t);
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
            .and_modify(|e| *e = ops::add(e, t).unwrap())
            .or_insert(t.clone());
    }

    pub(crate) fn cleanup(&self) {
        if self.orphans.lock().unwrap().is_empty() {
            return;
        }

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
