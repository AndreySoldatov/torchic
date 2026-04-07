use std::{num::NonZeroU64, sync::Arc};

use bytemuck::{Pod, Zeroable};
use strum_macros::AsRefStr;
use wgpu::BindGroupDescriptor;

use crate::{
    AsBindingResource,
    autograd::{GradNode, GradNodeMeta},
    kernel_registry::{KernelEntry, KernelKey},
    runtime::{Runtime, do_grad, rt},
    tensor::{DTYPE_SIZE, Tensor, TensorInner, get_tensor_id},
};

#[derive(Debug, Eq, Hash, PartialEq, Clone, AsRefStr)]
pub enum OpType {
    BinopEwizeType(BinopEwizeType),
    UnopEwizeType(UnopEwizeType),
    Reduce(ReduceOpType),
    Matmul,
    Transpose,
    ScalarEwize(ScalarEwizeType),
    Outer,
}

#[derive(Debug, Eq, Hash, PartialEq, Clone)]
pub enum BinopEwizeType {
    Add,
    Mul,
}

#[derive(Debug, Eq, Hash, PartialEq, Clone)]
pub enum ReduceOpType {
    Sum,
}

#[derive(Debug, Eq, Hash, PartialEq, Clone)]
pub enum UnopEwizeType {
    Relu,
    ReluBackward,
}

#[derive(Debug, Eq, Hash, PartialEq, Clone)]
pub enum ScalarEwizeType {
    Mul,
}

#[derive(Debug)]
pub enum TensorOpError {
    MismatchedShapes,
    NonMatrixTensor,
    EmptyTensor,
}

fn should_grad(grads: &[bool]) -> bool {
    grads.iter().any(|&v| v) && do_grad()
}

pub fn mul_scalar(t: &Tensor, s: f32) -> Result<Tensor, TensorOpError> {
    dispatch_scalar_ewize(t, ScalarEwizeType::Mul, s)
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct ScalarMeta {
    s: f32,
}

pub fn dispatch_scalar_ewize(
    t: &Tensor,
    typ: ScalarEwizeType,
    s: f32,
) -> Result<Tensor, TensorOpError> {
    let op = OpType::ScalarEwize(typ);
    let rt = rt();

    let kernel = rt
        .kernel_registry
        .lock()
        .unwrap()
        .get(&KernelKey::Op(op.clone()));

    let out_buf = rt.storage_buffer_alloc.request(t.bsize() as u64);

    let mut ma = rt.metadata_arena.lock().unwrap();
    let meta = ma.allocate(&bytemuck::bytes_of(&ScalarMeta { s })).unwrap();

    let bg = create_bg(
        op.as_ref(),
        &[t, &out_buf, &meta],
        kernel.bind_group_layout(),
    );

    dispatch_pass(
        op.as_ref(),
        kernel.pipeline(),
        &bg,
        ((t.numel().div_ceil(64) as u32).min(65535), 1, 1),
    );

    let requires_grad = should_grad(&[t.requires_grad()]);
    let grad_node = if requires_grad {
        Some(GradNode {
            op: op.clone(),
            parents: vec![t.clone()],
            meta: Some(GradNodeMeta::Scalar(s)),
        })
    } else {
        None
    };

    Ok(Tensor {
        inner: Arc::new(TensorInner {
            id: get_tensor_id(),
            buf: out_buf,
            shape: t.shape().to_vec(),
            requires_grad,
            grad_node,
        }),
    })
}

pub fn relu(t: &Tensor) -> Result<Tensor, TensorOpError> {
    dispatch_unop_ewize(t, UnopEwizeType::Relu)
}

pub fn dispatch_unop_ewize(t: &Tensor, typ: UnopEwizeType) -> Result<Tensor, TensorOpError> {
    let op = OpType::UnopEwizeType(typ);
    let rt = rt();

    let kernel = rt
        .kernel_registry
        .lock()
        .unwrap()
        .get(&KernelKey::Op(op.clone()));

    let out_buf = rt.storage_buffer_alloc.request(t.bsize() as u64);

    let bg = create_bg(op.as_ref(), &[t, &out_buf], kernel.bind_group_layout());

    dispatch_pass(
        op.as_ref(),
        kernel.pipeline(),
        &bg,
        ((t.numel().div_ceil(64) as u32).min(65535), 1, 1),
    );

    let requires_grad = should_grad(&[t.requires_grad()]);
    let grad_node = if requires_grad {
        Some(GradNode {
            op: op.clone(),
            parents: vec![t.clone()],
            meta: None,
        })
    } else {
        None
    };

    Ok(Tensor {
        inner: Arc::new(TensorInner {
            id: get_tensor_id(),
            buf: out_buf,
            shape: t.shape().to_vec(),
            requires_grad,
            grad_node,
        }),
    })
}

pub fn add(lhs: &Tensor, rhs: &Tensor) -> Result<Tensor, TensorOpError> {
    dispatch_binop_ewize(lhs, rhs, BinopEwizeType::Add)
}

pub fn mul(lhs: &Tensor, rhs: &Tensor) -> Result<Tensor, TensorOpError> {
    dispatch_binop_ewize(lhs, rhs, BinopEwizeType::Mul)
}

pub fn dispatch_binop_ewize(
    lhs: &Tensor,
    rhs: &Tensor,
    typ: BinopEwizeType,
) -> Result<Tensor, TensorOpError> {
    if lhs.shape() != rhs.shape() {
        return Err(TensorOpError::MismatchedShapes);
    }
    let op = OpType::BinopEwizeType(typ);

    let rt = rt();

    let kernel = rt
        .kernel_registry
        .lock()
        .unwrap()
        .get(&KernelKey::Op(op.clone()));

    let out_buf = rt.storage_buffer_alloc.request(lhs.bsize() as u64);

    let bg = create_bg(
        op.as_ref(),
        &[lhs, rhs, &out_buf],
        kernel.bind_group_layout(),
    );

    dispatch_pass(
        op.as_ref(),
        kernel.pipeline(),
        &bg,
        ((lhs.numel().div_ceil(64) as u32).min(65535), 1, 1),
    );

    let requires_grad = should_grad(&[lhs.requires_grad(), rhs.requires_grad()]);
    let grad_node = if requires_grad {
        Some(GradNode {
            op,
            parents: vec![lhs.clone(), rhs.clone()],
            meta: None,
        })
    } else {
        None
    };

    Ok(Tensor {
        inner: Arc::new(TensorInner {
            id: get_tensor_id(),
            buf: out_buf,
            shape: lhs.shape().to_vec(),
            requires_grad: requires_grad,
            grad_node: grad_node,
        }),
    })
}

pub fn sum(t: &Tensor) -> Result<Tensor, TensorOpError> {
    if t.numel() == 0 {
        return Err(TensorOpError::EmptyTensor);
    }

    let op = OpType::Reduce(ReduceOpType::Sum);

    let rt = rt();

    let kernel = rt
        .kernel_registry
        .lock()
        .unwrap()
        .get(&KernelKey::Op(op.clone()));

    let mut output_size = t.numel().div_ceil(256).min(65535);
    let mut inp_buf = rt
        .storage_buffer_alloc
        .request((output_size * DTYPE_SIZE) as u64);
    let mut out_buf;

    let bg = create_bg(op.as_ref(), &[t, &inp_buf], kernel.bind_group_layout());

    dispatch_pass(
        op.as_ref(),
        kernel.pipeline(),
        &bg,
        (output_size as u32, 1, 1),
    );

    while output_size > 1 {
        output_size = output_size.div_ceil(256).min(65535);
        out_buf = rt
            .storage_buffer_alloc
            .request((output_size * DTYPE_SIZE) as u64);

        let bg = create_bg(
            op.as_ref(),
            &[&inp_buf, &out_buf],
            kernel.bind_group_layout(),
        );

        dispatch_pass(
            op.as_ref(),
            kernel.pipeline(),
            &bg,
            (output_size as u32, 1, 1),
        );

        inp_buf = out_buf;
    }

    let requires_grad = should_grad(&[t.requires_grad()]);
    let grad_node = if requires_grad {
        Some(GradNode {
            op,
            parents: vec![t.clone()],
            meta: None,
        })
    } else {
        None
    };

    Ok(Tensor {
        inner: Arc::new(TensorInner {
            id: get_tensor_id(),
            buf: inp_buf,
            shape: vec![1],
            requires_grad: requires_grad,
            grad_node: grad_node,
        }),
    })
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct MatmulMeta {
    m: u32,
    n: u32,
    k: u32,
}

enum CollapseDim {
    None,
    M,
    N,
    Both,
}

fn to_matrix_shape(
    shape1: &[usize],
    shape2: &[usize],
) -> Result<(u32, u32, u32, CollapseDim), TensorOpError> {
    if shape1.len() == 0 || shape2.len() == 0 {
        return Err(TensorOpError::EmptyTensor);
    }

    if shape1.len() > 2 || shape2.len() > 2 {
        return Err(TensorOpError::NonMatrixTensor);
    }

    match (shape1, shape2) {
        ([m, k1], [k2, n]) if k1 == k2 => Ok((*m as u32, *n as u32, *k1 as u32, CollapseDim::None)),
        ([m, k1], [k2]) if k1 == k2 => Ok((*m as u32, 1, *k1 as u32, CollapseDim::N)),
        ([k1], [k2, n]) if k1 == k2 => Ok((1, *n as u32, *k1 as u32, CollapseDim::M)),
        ([k1], [k2]) if k1 == k2 => Ok((1, 1, *k1 as u32, CollapseDim::Both)),
        _ => Err(TensorOpError::MismatchedShapes),
    }
}

pub fn matmul(lhs: &Tensor, rhs: &Tensor) -> Result<Tensor, TensorOpError> {
    let (m, n, k, col) = to_matrix_shape(lhs.shape(), rhs.shape())?;

    let out_shape = match col {
        CollapseDim::None => vec![m as usize, n as usize],
        CollapseDim::N => vec![m as usize],
        CollapseDim::M => vec![n as usize],
        CollapseDim::Both => vec![1],
    };
    let bsize = m * n * (DTYPE_SIZE as u32);

    let op = OpType::Matmul;

    let rt = rt();

    let kernel = rt
        .kernel_registry
        .lock()
        .unwrap()
        .get(&KernelKey::Op(op.clone()));

    let out_buf = rt.storage_buffer_alloc.request(bsize as u64);

    let mut ma = rt.metadata_arena.lock().unwrap();
    let meta = ma
        .allocate(bytemuck::bytes_of(&MatmulMeta { m, n, k }))
        .unwrap();

    let bg = create_bg(
        op.as_ref(),
        &[lhs, rhs, &out_buf, &meta],
        kernel.bind_group_layout(),
    );

    dispatch_pass(
        op.as_ref(),
        kernel.pipeline(),
        &bg,
        (n.div_ceil(32), m.div_ceil(32), 1),
    );

    let requires_grad = should_grad(&[lhs.requires_grad(), rhs.requires_grad()]);
    let grad_node = if requires_grad {
        Some(GradNode {
            op: op,
            parents: vec![lhs.clone(), rhs.clone()],
            meta: None,
        })
    } else {
        None
    };

    Ok(Tensor {
        inner: Arc::new(TensorInner {
            id: get_tensor_id(),
            buf: out_buf,
            shape: out_shape,
            requires_grad,
            grad_node,
        }),
    })
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct MatrixMeta {
    m: u32,
    n: u32,
}

pub fn transposed(t: &Tensor) -> Result<Tensor, TensorOpError> {
    if t.shape().len() != 2 {
        return Err(TensorOpError::NonMatrixTensor);
    }
    let (m, n) = (t.shape()[0] as u32, t.shape()[1] as u32);

    let op = OpType::Transpose;

    let rt = rt();
    let kernel = rt
        .kernel_registry
        .lock()
        .unwrap()
        .get(&KernelKey::Op(op.clone()));

    let out_buf = rt.storage_buffer_alloc.request(t.bsize() as u64);

    let mut ma = rt.metadata_arena.lock().unwrap();
    let meta = ma
        .allocate(bytemuck::bytes_of(&MatrixMeta { m, n }))
        .unwrap();

    let bg = create_bg(
        op.as_ref(),
        &[t, &out_buf, &meta],
        kernel.bind_group_layout(),
    );

    dispatch_pass(
        op.as_ref(),
        kernel.pipeline(),
        &bg,
        (t.numel().div_ceil(64) as u32, 1, 1),
    );

    let requires_grad = should_grad(&[t.requires_grad()]);
    let grad_node = if requires_grad {
        Some(GradNode {
            op: op,
            parents: vec![t.clone()],
            meta: None,
        })
    } else {
        None
    };

    Ok(Tensor {
        inner: Arc::new(TensorInner {
            id: get_tensor_id(),
            buf: out_buf,
            shape: vec![n as usize, m as usize],
            requires_grad,
            grad_node,
        }),
    })
}

pub fn outer(lhs: &Tensor, rhs: &Tensor) -> Result<Tensor, TensorOpError> {
    if lhs.shape().len() != 1 || rhs.shape().len() != 1 {
        return Err(TensorOpError::MismatchedShapes);
    }
    let (m, n) = (lhs.shape()[0] as u32, rhs.shape()[0] as u32);

    let op = OpType::Outer;

    let rt = rt();
    let kernel = rt
        .kernel_registry
        .lock()
        .unwrap()
        .get(&KernelKey::Op(op.clone()));

    let out_buf = rt
        .storage_buffer_alloc
        .request((m * n * DTYPE_SIZE as u32) as u64);

    let mut ma = rt.metadata_arena.lock().unwrap();
    let meta = ma
        .allocate(&bytemuck::bytes_of(&MatrixMeta { m, n }))
        .unwrap();

    let bg = create_bg(
        op.as_ref(),
        &[lhs, rhs, &out_buf, &meta],
        kernel.bind_group_layout(),
    );

    dispatch_pass(
        op.as_ref(),
        kernel.pipeline(),
        &bg,
        (m.div_ceil(8), n.div_ceil(8), 1),
    );

    let requires_grad = should_grad(&[lhs.requires_grad(), rhs.requires_grad()]);
    let grad_node = if requires_grad {
        Some(GradNode {
            op,
            parents: vec![lhs.clone(), rhs.clone()],
            meta: None,
        })
    } else {
        None
    };

    Ok(Tensor {
        inner: Arc::new(TensorInner {
            id: get_tensor_id(),
            buf: out_buf,
            shape: vec![m as usize, n as usize],
            requires_grad,
            grad_node,
        }),
    })
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct HeMeta {
    fact: f32,
    seed: u32,
}

pub fn he_init(seed: u32, fan_in: u32, shape: &[usize], requires_grad: bool) -> Tensor {
    let rt = rt();
    let numel = shape.iter().product::<usize>();
    let out_buf = rt.storage_buffer_alloc.request((numel * DTYPE_SIZE) as u64);

    let kernel = rt.kernel_registry.lock().unwrap().get(&KernelKey::HeInit);

    let fact = (2.0 / (fan_in as f32)).sqrt();

    let mut ma = rt.metadata_arena.lock().unwrap();
    let meta = ma
        .allocate(&bytemuck::bytes_of(&HeMeta { fact, seed }))
        .unwrap();

    let bg = create_bg("he init", &[&meta, &out_buf], kernel.bind_group_layout());

    dispatch_pass(
        "he init",
        kernel.pipeline(),
        &bg,
        (numel.div_ceil(64).min(65535) as u32, 1, 1),
    );

    Tensor {
        inner: Arc::new(TensorInner {
            id: get_tensor_id(),
            buf: out_buf,
            shape: shape.to_vec(),
            requires_grad,
            grad_node: None,
        }),
    }
}

pub(crate) fn dispatch_pass(
    label_prefix: &str,
    pipeline: &wgpu::ComputePipeline,
    bg: &wgpu::BindGroup,
    wgs: (u32, u32, u32),
) {
    let rt = rt();
    let encoder_label = format!("{} encoder", label_prefix);
    let mut encoder = rt
        .ctx
        .device
        .create_command_encoder(&wgpu::wgt::CommandEncoderDescriptor {
            label: Some(&encoder_label),
        });

    let pass_label = format!("{} pass", label_prefix);
    let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
        label: Some(&pass_label),
        timestamp_writes: None,
    });

    compute_pass.set_pipeline(pipeline);
    compute_pass.set_bind_group(0, bg, &[]);
    compute_pass.dispatch_workgroups(wgs.0, wgs.1, wgs.2);

    drop(compute_pass);

    rt.ctx.queue.submit(Some(encoder.finish()));
}

pub(crate) fn create_bg(
    prefix: &str,
    entries: &[&dyn AsBindingResource],
    bgl: &wgpu::BindGroupLayout,
) -> wgpu::BindGroup {
    let rt = rt();
    let label = format!("{} bg", prefix);

    let bge: Vec<wgpu::BindGroupEntry<'_>> = entries
        .iter()
        .enumerate()
        .map(|(i, e)| wgpu::BindGroupEntry {
            binding: i as u32,
            resource: e.as_binding_resource(),
        })
        .collect();

    rt.ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some(&label),
        layout: bgl,
        entries: &bge,
    })
}
