use std::{collections::HashMap, sync::Arc};

use crate::{
    ops::{BinopEwizeType, OpType},
    runtime::WGPUContext,
};

#[derive(Debug)]
pub struct KernelEntry {
    pipeline: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
}

impl KernelEntry {
    pub fn bind_group_layout(&self) -> &wgpu::BindGroupLayout {
        &self.bind_group_layout
    }

    pub fn pipeline(&self) -> &wgpu::ComputePipeline {
        &self.pipeline
    }
}

#[derive(Debug, Eq, Hash, PartialEq, Clone)]
pub struct KernelKey {
    pub op: OpType,
}

#[derive(Debug)]
pub struct KernelRegistry {
    ctx: WGPUContext,
    map: HashMap<KernelKey, Arc<KernelEntry>>,
}

impl KernelRegistry {
    pub fn new(ctx: WGPUContext) -> Self {
        Self {
            ctx,
            map: HashMap::new(),
        }
    }

    fn load_with_source(&mut self, key: &KernelKey, src: &str) {
        let label = format!("{:?} shader", key.op);
        let shader = self
            .ctx
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some(&label),
                source: wgpu::ShaderSource::Wgsl(src.into()),
            });

        let bind_group_layout = op_to_bgl(&key.op, self.ctx.device.clone());

        let label = format!("{:?} pipeline layout", key.op);
        let pl = self
            .ctx
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some(&label),
                bind_group_layouts: &[Some(&bind_group_layout)],
                immediate_size: 0,
            });

        let label = format!("{:?} compute pipeline", key.op);
        let pipeline = self
            .ctx
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(&label),
                layout: Some(&pl),
                module: &shader,
                entry_point: Some("main"),
                compilation_options: Default::default(),
                cache: None,
            });

        self.map.insert(
            key.clone(),
            Arc::new(KernelEntry {
                pipeline,
                bind_group_layout,
            }),
        );
    }

    fn load_known(&mut self, key: &KernelKey) {
        match &key.op {
            OpType::BinopEwizeType(typ) => {
                let template_base = include_str!("shader_templates/binop_ewize/binop_ewize.wgsl");
                let mut variables = HashMap::new();
                match typ {
                    BinopEwizeType::Add => {
                        variables.insert("operation", "output[idx] = input1[idx] + input2[idx];");
                    }
                    BinopEwizeType::Mul => {
                        variables.insert("operation", "output[idx] = input1[idx] * input2[idx];");
                    }
                }
                let src = subst::substitute(template_base, &variables)
                    .expect("Shader template not substituted correcty!");
                self.load_with_source(key, &src);
            }
            OpType::Reduce(_) => {
                self.load_with_source(key, include_str!("shader_templates/sum.wgsl"));
            }
        }
    }

    pub fn get(&mut self, key: &KernelKey) -> Arc<KernelEntry> {
        if !self.map.contains_key(key) {
            self.load_known(key);
        }

        self.map.get(key).unwrap().clone()
    }
}

fn op_to_bgl(op: &OpType, device: Arc<wgpu::Device>) -> wgpu::BindGroupLayout {
    match op {
        OpType::BinopEwizeType(_) => binop_ewize_bgl(device),
        OpType::Reduce(_) => reduce_bgl(device),
    }
}

fn binop_ewize_bgl(device: Arc<wgpu::Device>) -> wgpu::BindGroupLayout {
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("binop ewize bgl"),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 2,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
        ],
    })
}

fn reduce_bgl(device: Arc<wgpu::Device>) -> wgpu::BindGroupLayout {
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("Reduce bgl"),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
        ],
    })
}
