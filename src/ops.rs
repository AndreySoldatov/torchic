use std::{num::NonZeroU64, sync::Arc};

use bytemuck::{Pod, Zeroable};
use wgpu::BindGroupDescriptor;

use crate::{
    autograd::GradNode,
    kernel_registry::KernelKey,
    runtime::{do_grad, rt},
    tensor::{DTYPE_SIZE, Tensor, TensorInner, get_tensor_id},
};

#[derive(Debug, Eq, Hash, PartialEq, Clone)]
pub enum OpType {
    BinopEwizeType(BinopEwizeType),
    Reduce(ReduceOpType),
    Matmul,
    Transpose,
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

#[derive(Debug)]
pub enum TensorOpError {
    MismatchedShapes,
    EmptyInput,
    NonMatrixTensor,
}

pub fn add(lhs: &Tensor, rhs: &Tensor) -> Result<Tensor, TensorOpError> {
    dispatch_binop_ewize(lhs, rhs, BinopEwizeType::Add)
}

pub fn mul(lhs: &Tensor, rhs: &Tensor) -> Result<Tensor, TensorOpError> {
    dispatch_binop_ewize(lhs, rhs, BinopEwizeType::Mul)
}

fn should_grad(grads: &[bool]) -> bool {
    grads.iter().any(|&v| v) && do_grad()
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
        .get(&KernelKey { op: op.clone() });

    let out_buf = rt.storage_buffer_alloc.request(lhs.bsize() as u64);

    let bg = rt.ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("binop_ewize bg"),
        layout: kernel.bind_group_layout(),
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: lhs.buf_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: rhs.buf_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: out_buf.binding(),
            },
        ],
    });

    let mut encoder = rt
        .ctx
        .device
        .create_command_encoder(&wgpu::wgt::CommandEncoderDescriptor {
            label: Some("binop_ewize command encoder"),
        });

    let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
        label: Some("binop_ewize pass"),
        timestamp_writes: None,
    });

    compute_pass.set_pipeline(kernel.pipeline());
    compute_pass.set_bind_group(0, &bg, &[]);
    compute_pass.dispatch_workgroups((lhs.numel().div_ceil(64) as u32).min(65535), 1, 1);

    drop(compute_pass);

    rt.ctx.queue.submit(Some(encoder.finish()));

    let requires_grad = should_grad(&[lhs.requires_grad(), rhs.requires_grad()]);
    let grad_node = if requires_grad {
        Some(GradNode {
            op,
            parents: vec![lhs.clone(), rhs.clone()],
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
        return Err(TensorOpError::EmptyInput);
    }

    let op = OpType::Reduce(ReduceOpType::Sum);

    let rt = rt();

    let kernel = rt
        .kernel_registry
        .lock()
        .unwrap()
        .get(&KernelKey { op: op.clone() });

    let mut output_size = t.numel().div_ceil(256).min(65535);
    let mut inp_buf = rt
        .storage_buffer_alloc
        .request((output_size * DTYPE_SIZE) as u64);
    let mut out_buf;

    let bg = rt.ctx.device.create_bind_group(&BindGroupDescriptor {
        label: Some("sum bg"),
        layout: kernel.bind_group_layout(),
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: t.buf_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: inp_buf.binding(),
            },
        ],
    });

    let mut encoder = rt
        .ctx
        .device
        .create_command_encoder(&wgpu::wgt::CommandEncoderDescriptor {
            label: Some("reduce command encoder"),
        });

    let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
        label: Some("reduce pass"),
        timestamp_writes: None,
    });

    compute_pass.set_pipeline(kernel.pipeline());
    compute_pass.set_bind_group(0, &bg, &[]);
    compute_pass.dispatch_workgroups(output_size as u32, 1, 1);

    drop(compute_pass);
    rt.ctx.queue.submit(Some(encoder.finish()));

    while output_size > 1 {
        output_size = output_size.div_ceil(256).min(65535);
        out_buf = rt
            .storage_buffer_alloc
            .request((output_size * DTYPE_SIZE) as u64);

        let bg = rt.ctx.device.create_bind_group(&BindGroupDescriptor {
            label: Some("sum bg"),
            layout: kernel.bind_group_layout(),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: inp_buf.binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: out_buf.binding(),
                },
            ],
        });

        let mut encoder =
            rt.ctx
                .device
                .create_command_encoder(&wgpu::wgt::CommandEncoderDescriptor {
                    label: Some("reduce command encoder"),
                });

        let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("reduce pass"),
            timestamp_writes: None,
        });

        compute_pass.set_pipeline(kernel.pipeline());
        compute_pass.set_bind_group(0, &bg, &[]);
        compute_pass.dispatch_workgroups(output_size as u32, 1, 1);

        drop(compute_pass);
        rt.ctx.queue.submit(Some(encoder.finish()));

        inp_buf = out_buf;
    }

    let requires_grad = should_grad(&[t.requires_grad()]);
    let grad_node = if requires_grad {
        Some(GradNode {
            op,
            parents: vec![t.clone()],
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

pub fn matmul(lhs: &Tensor, rhs: &Tensor) -> Result<Tensor, TensorOpError> {
    if lhs.shape().len() != 2 || rhs.shape().len() != 2 {
        return Err(TensorOpError::NonMatrixTensor);
    }

    if lhs.shape()[1] != rhs.shape()[0] {
        return Err(TensorOpError::MismatchedShapes);
    }

    let out_shape = vec![lhs.shape()[0], rhs.shape()[1]];
    let bsize = out_shape.iter().product::<usize>() * DTYPE_SIZE;

    let op = OpType::Matmul;

    let rt = rt();

    let kernel = rt
        .kernel_registry
        .lock()
        .unwrap()
        .get(&KernelKey { op: op.clone() });

    let out_buf = rt.storage_buffer_alloc.request(bsize as u64);

    let m = lhs.shape()[0] as u32;
    let n = rhs.shape()[1] as u32;
    let k = lhs.shape()[1] as u32;

    let mut ma = rt.metadata_arena.lock().unwrap();
    let meta = ma
        .allocate(bytemuck::bytes_of(&MatmulMeta { m, n, k }))
        .unwrap();

    let bg = rt.ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("matmul bg"),
        layout: kernel.bind_group_layout(),
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: lhs.buf_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: rhs.buf_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: out_buf.binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                    buffer: meta.buf,
                    offset: meta.offset,
                    size: Some(NonZeroU64::new(meta.size).unwrap()),
                }),
            },
        ],
    });

    let mut encoder = rt
        .ctx
        .device
        .create_command_encoder(&wgpu::wgt::CommandEncoderDescriptor {
            label: Some("matmul command encoder"),
        });

    let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
        label: Some("matmul pass"),
        timestamp_writes: None,
    });

    compute_pass.set_pipeline(kernel.pipeline());
    compute_pass.set_bind_group(0, &bg, &[]);
    compute_pass.dispatch_workgroups(n.div_ceil(32), m.div_ceil(32), 1);

    drop(compute_pass);

    rt.ctx.queue.submit(Some(encoder.finish()));

    let requires_grad = should_grad(&[lhs.requires_grad(), rhs.requires_grad()]);
    let grad_node = if requires_grad {
        Some(GradNode {
            op: op,
            parents: vec![lhs.clone(), rhs.clone()],
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
struct TransposeMeta {
    m: u32,
    n: u32,
}

pub fn transpose(t: &Tensor) -> Result<Tensor, TensorOpError> {
    if t.shape().len() != 2 {
        return Err(TensorOpError::NonMatrixTensor);
    }

    let op = OpType::Transpose;

    let rt = rt();
    let kernel = rt
        .kernel_registry
        .lock()
        .unwrap()
        .get(&KernelKey { op: op.clone() });

    let out_buf = rt.storage_buffer_alloc.request(t.bsize() as u64);

    let mut ma = rt.metadata_arena.lock().unwrap();
    let meta = ma
        .allocate(bytemuck::bytes_of(&TransposeMeta {
            m: t.shape()[0] as u32,
            n: t.shape()[1] as u32,
        }))
        .unwrap();

    let bg = rt.ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("transpose bg"),
        layout: kernel.bind_group_layout(),
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: t.buf_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: out_buf.binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                    buffer: meta.buf,
                    offset: meta.offset,
                    size: Some(NonZeroU64::new(meta.size).unwrap()),
                }),
            },
        ],
    });

    let mut encoder = rt
        .ctx
        .device
        .create_command_encoder(&wgpu::wgt::CommandEncoderDescriptor {
            label: Some("transpose command encoder"),
        });

    let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
        label: Some("transpose pass"),
        timestamp_writes: None,
    });

    compute_pass.set_pipeline(kernel.pipeline());
    compute_pass.set_bind_group(0, &bg, &[]);
    compute_pass.dispatch_workgroups(t.numel().div_ceil(64) as u32, 1, 1);

    drop(compute_pass);

    rt.ctx.queue.submit(Some(encoder.finish()));

    let requires_grad = should_grad(&[t.requires_grad()]);
    let grad_node = if requires_grad {
        Some(GradNode {
            op: OpType::Transpose,
            parents: vec![t.clone()],
        })
    } else {
        None
    };

    Ok(Tensor {
        inner: Arc::new(TensorInner {
            id: get_tensor_id(),
            buf: out_buf,
            shape: vec![t.shape()[1], t.shape()[0]],
            requires_grad,
            grad_node,
        }),
    })
}
