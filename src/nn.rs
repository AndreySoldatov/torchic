use std::sync::atomic::AtomicU32;

use crate::{
    ops::{self, TensorOpError},
    runtime::rt,
    tensor::Tensor,
};

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
