use std::{collections::HashMap, sync::Arc};

use crate::{
    ops::{BinopEwizeType, OpType, ScalarEwizeType, UnopEwizeType},
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
pub enum KernelKey {
    Op(OpType),
    HeInit,
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
        let label = format!("{:?} shader", key);
        let shader = self
            .ctx
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some(&label),
                source: wgpu::ShaderSource::Wgsl(src.into()),
            });

        let bind_group_layout = kernel_key_to_bgl(&key, self.ctx.device.clone());

        let label = format!("{:?} pipeline layout", key);
        let pl = self
            .ctx
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some(&label),
                bind_group_layouts: &[Some(&bind_group_layout)],
                immediate_size: 0,
            });

        let label = format!("{:?} compute pipeline", key);
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
        match key {
            KernelKey::Op(OpType::BinopEwizeType(typ)) => {
                let template_base = include_str!("shader_templates/binop_ewize.wgsl");
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
            KernelKey::Op(OpType::UnopEwizeType(typ)) => {
                let template_base = include_str!("shader_templates/unop_ewize.wgsl");
                let mut variables = HashMap::new();
                match typ {
                    UnopEwizeType::Relu => {
                        variables.insert(
                            "operation",
                            "output[idx] = select(0.0, input[idx], input[idx] >= 0.0);",
                        );
                    }
                    UnopEwizeType::ReluBackward => {
                        variables.insert(
                            "operation",
                            "output[idx] = select(0.0, 1.0, input[idx] > 0.0);",
                        );
                    }
                }
                let src = subst::substitute(template_base, &variables)
                    .expect("Shader template not substituted correcty!");
                self.load_with_source(key, &src);
            }
            KernelKey::Op(OpType::Reduce(_)) => {
                self.load_with_source(key, include_str!("shader_templates/sum.wgsl"));
            }
            KernelKey::Op(OpType::Matmul) => {
                self.load_with_source(key, include_str!("shader_templates/matmul.wgsl"));
            }
            KernelKey::Op(OpType::Transpose) => {
                self.load_with_source(key, include_str!("shader_templates/transpose.wgsl"));
            }
            KernelKey::Op(OpType::ScalarEwize(typ)) => {
                let template_base = include_str!("shader_templates/scalar_ewize.wgsl");
                let mut variables = HashMap::new();
                match typ {
                    ScalarEwizeType::Mul => {
                        variables.insert("operation", "output[idx] = input[idx] * s;");
                    }
                }
                let src = subst::substitute(template_base, &variables)
                    .expect("Shader template not substituted correcty!");
                self.load_with_source(key, &src);
            }
            KernelKey::Op(OpType::Outer) => {
                self.load_with_source(key, include_str!("shader_templates/outer.wgsl"));
            }
            KernelKey::HeInit => {
                self.load_with_source(key, include_str!("shader_templates/normal.wgsl"));
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

fn kernel_key_to_bgl(key: &KernelKey, device: Arc<wgpu::Device>) -> wgpu::BindGroupLayout {
    let read_only_mask = match key {
        KernelKey::Op(OpType::BinopEwizeType(_)) => vec![true, true, false],
        KernelKey::Op(OpType::Reduce(_)) => vec![true, false],
        KernelKey::Op(OpType::Matmul) => vec![true, true, false, true],
        KernelKey::Op(OpType::Transpose) => vec![true, false, true],
        KernelKey::Op(OpType::UnopEwizeType(_)) => vec![true, false],
        KernelKey::Op(OpType::ScalarEwize(_)) => vec![true, false],
        KernelKey::Op(OpType::Outer) => vec![true, true, false, true],
        KernelKey::HeInit => vec![true, false],
    };

    let prefix = format!("{:?}", key);
    create_bgl(&prefix, &read_only_mask, device)
}

fn entry(binding: u32, read_only: bool) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn create_bgl(
    prefix: &str,
    read_only: &[bool],
    device: Arc<wgpu::Device>,
) -> wgpu::BindGroupLayout {
    let label = format!("{} bgl", prefix);
    let entries: Vec<wgpu::BindGroupLayoutEntry> = read_only
        .iter()
        .enumerate()
        .map(|(i, e)| entry(i as u32, *e))
        .collect();
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some(&label),
        entries: &entries,
    })
}
