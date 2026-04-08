use std::sync::atomic::AtomicU32;

use crate::{
    ops::{self, TensorOpError},
    runtime::{no_grad, rt},
    tensor::Tensor,
};

pub trait Model {
    fn params(&self) -> Vec<Tensor>;
}

static LAYER_COUNTER: AtomicU32 = AtomicU32::new(1);

pub(crate) fn get_layer_count() -> u32 {
    LAYER_COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
}

pub struct Linear {
    pub(crate) weights: Tensor,
    pub(crate) bias: Option<Tensor>,
}

impl Linear {
    pub fn new(in_dim: usize, out_dim: usize, bias: bool) -> Self {
        let (seed, _) = rt().seed.overflowing_mul(get_layer_count());
        let weights = ops::he_init(seed, in_dim as u32, &[in_dim, out_dim], true);

        let bias = if bias {
            Some(Tensor::new(&[out_dim], &vec![0.0; out_dim], true))
        } else {
            None
        };

        Self { weights, bias }
    }

    pub fn forward(&self, input: &Tensor) -> Result<Tensor, TensorOpError> {
        let out = input.matmul(&self.weights)?;

        match (&self.bias, input.shape()) {
            (None, _) => Ok(out),
            (Some(b), [_]) => out.add(b),
            (Some(b), [batch, _]) => {
                let ones = Tensor::ones(&[*batch], false);
                let bias_rows = ones.outer(b)?;
                out.add(&bias_rows)
            }
            _ => Err(TensorOpError::NonMatrixTensor),
        }
    }
}

impl Model for Linear {
    fn params(&self) -> Vec<Tensor> {
        let mut res = vec![self.weights.clone()];

        if let Some(b) = &self.bias {
            res.push(b.clone());
        }

        res
    }
}

pub struct MLP {
    layers: Vec<Linear>,
}

impl MLP {
    pub fn new(features: &[usize], bias: bool) -> Self {
        let mut layers = vec![];

        for pair in features.windows(2) {
            layers.push(Linear::new(pair[0], pair[1], bias));
        }

        Self { layers }
    }

    pub fn forward(&self, input: &Tensor) -> Result<Tensor, TensorOpError> {
        let mut out = input.clone();

        for (i, layer) in self.layers.iter().enumerate() {
            out = layer.forward(&out)?;

            if i < self.layers.len() - 1 {
                out = out.relu()?;
            }
        }

        Ok(out)
    }
}

impl Model for MLP {
    fn params(&self) -> Vec<Tensor> {
        let mut res = vec![];
        for l in &self.layers {
            res.append(&mut l.params());
        }
        res
    }
}

pub struct Adam {
    params: Vec<Tensor>,
    m: Vec<Tensor>,
    v: Vec<Tensor>,
    lr: f32,
    b1: f32,
    b2: f32,
    eps: f32,
    t: i32,
}

impl Adam {
    pub fn new<T: Model>(model: &T, lr: f32, b1: f32, b2: f32, eps: f32) -> Self {
        let params = model.params();

        let m: Vec<Tensor> = params
            .iter()
            .map(|p| Tensor::new(p.shape(), &vec![0.0; p.numel()], false))
            .collect();

        let v: Vec<Tensor> = params
            .iter()
            .map(|p| Tensor::new(p.shape(), &vec![0.0; p.numel()], false))
            .collect();

        Self {
            params,
            m,
            v,
            lr,
            b1,
            b2,
            eps,
            t: 0,
        }
    }

    pub fn zero_grad(&self) {
        for p in &self.params {
            p.zero_grad()
        }
    }

    pub fn step(&mut self) {
        let _ng = no_grad().unwrap();

        self.t += 1;

        for ((m, v), p) in self
            .m
            .iter_mut()
            .zip(self.v.iter_mut())
            .zip(self.params.iter_mut())
        {
            let grad = p.grad();
            if grad.is_none() {
                continue;
            }
            let grad = grad.unwrap();

            *m = m
                .mul_s(self.b1)
                .unwrap()
                .add(&grad.mul_s(1.0 - self.b1).unwrap())
                .unwrap();

            let m_hat = m.mul_s(1.0 / (1.0 - self.b1.powi(self.t))).unwrap();

            *v = v
                .mul_s(self.b2)
                .unwrap()
                .add(&grad.mul(&grad).unwrap().mul_s(1.0 - self.b2).unwrap())
                .unwrap();

            let v_hat = v.mul_s(1.0 / (1.0 - self.b2.powi(self.t))).unwrap();

            let update = m_hat
                .div(&v_hat.sqrt().unwrap().add_s(self.eps).unwrap())
                .unwrap()
                .mul_s(-1.0 * self.lr)
                .unwrap();

            p.assign(&p.add(&update).unwrap());
        }
    }
}
