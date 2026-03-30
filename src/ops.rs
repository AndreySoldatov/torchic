use std::sync::Arc;

use crate::{
    autograd::{GradNode, create_grad_node},
    buffer_alloc::{BufferLease, usage_marker::Storage},
    kernel_registry::KernelKey,
    runtime::rt,
    tensor::{Tensor, TensorInner, get_tensor_id},
};

#[derive(Debug, Eq, Hash, PartialEq, Clone)]
pub enum OpType {
    BinopEwizeType(BinopEwizeType),
}

#[derive(Debug, Eq, Hash, PartialEq, Clone)]
pub enum BinopEwizeType {
    Add,
    Mul,
    Sub,
    Div,
}

#[derive(Debug)]
pub enum TensorOpError {
    MismatchedShapes,
}

pub fn add(lhs: &Tensor, rhs: &Tensor) -> Result<Tensor, TensorOpError> {
    let buf = dispatch_binop_ewize(lhs, rhs, BinopEwizeType::Add)?;

    Ok(Tensor {
        inner: Arc::new(TensorInner {
            id: get_tensor_id(),
            buf,
            shape: lhs.shape().to_vec(),
            requires_grad: lhs.requires_grad() || rhs.requires_grad(),
            grad_node: create_grad_node(lhs, rhs, OpType::BinopEwizeType(BinopEwizeType::Add)),
        }),
    })
}

pub fn dispatch_binop_ewize(
    lhs: &Tensor,
    rhs: &Tensor,
    typ: BinopEwizeType,
) -> Result<BufferLease<Storage>, TensorOpError> {
    if lhs.shape() != rhs.shape() {
        return Err(TensorOpError::MismatchedShapes);
    }

    let rt = rt();

    let kernel = rt.kernel_registry.lock().unwrap().get(&KernelKey {
        op: OpType::BinopEwizeType(typ),
    });

    let out_buf = rt.storage_buffer_alloc.request(lhs.bsize() as u64);

    let bg = rt.ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("add_e_c bg"),
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
            label: Some("add_e_c command encoder"),
        });

    let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
        label: Some("add_e_c pass"),
        timestamp_writes: None,
    });

    compute_pass.set_pipeline(kernel.pipeline());
    compute_pass.set_bind_group(0, &bg, &[]);
    compute_pass.dispatch_workgroups(
        (lhs.shape().iter().product::<usize>().div_ceil(64) as u32).min(65535),
        1,
        1,
    );

    drop(compute_pass);

    rt.ctx.queue.submit(Some(encoder.finish()));

    Ok(out_buf)
}
