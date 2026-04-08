use std::io::{self, Write};

use mnist::*;
use ndarray::prelude::*;
use torchic::{
    nn::{Adam, MLP},
    runtime::{WGPUContext, init_runtime, no_grad},
    tensor::Tensor,
};

struct DataLoader {
    trn_img: Vec<f32>,
    trn_lbl: Vec<f32>,
    tst_img: Vec<f32>,
    tst_lbl: Vec<f32>,
    trn_cur: usize,
    tst_cur: usize,
}

impl DataLoader {
    fn new() -> Self {
        let Mnist {
            trn_img,
            trn_lbl,
            tst_img,
            tst_lbl,
            ..
        } = MnistBuilder::new()
            .label_format_one_hot()
            .base_path("./")
            .training_images_filename("train-images.idx3-ubyte")
            .training_labels_filename("train-labels.idx1-ubyte")
            .test_images_filename("t10k-images.idx3-ubyte")
            .test_labels_filename("t10k-labels.idx1-ubyte")
            .finalize();

        Self {
            trn_img: trn_img.into_iter().map(|v| v as f32 / 256.0).collect(),
            trn_lbl: trn_lbl.into_iter().map(|v| v as f32).collect(),
            tst_img: tst_img.into_iter().map(|v| v as f32 / 256.0).collect(),
            tst_lbl: tst_lbl.into_iter().map(|v| v as f32).collect(),
            trn_cur: 0,
            tst_cur: 0,
        }
    }

    fn sample_training_batch(&mut self, batch_size: usize) -> (Tensor, Tensor) {
        if self.trn_cur * 784 + batch_size * 784 > self.trn_img.len() {
            self.trn_cur = 0;
        }

        let img_start = self.trn_cur * 784;
        let img_end = self.trn_cur * 784 + batch_size * 784;
        let img_tensor = Tensor::new(&[batch_size, 784], &self.trn_img[img_start..img_end], false);

        let lbl_start = self.trn_cur * 10;
        let lbl_end = self.trn_cur * 10 + batch_size * 10;
        let lbl_tensor = Tensor::new(&[batch_size, 10], &self.trn_lbl[lbl_start..lbl_end], false);

        self.trn_cur += batch_size;

        (img_tensor, lbl_tensor)
    }

    fn sample_test_batch(&mut self, batch_size: usize) -> (Tensor, Tensor) {
        if self.tst_cur * 784 + batch_size * 784 > self.tst_img.len() {
            self.tst_cur = 0;
        }

        let img_start = self.tst_cur * 784;
        let img_end = self.tst_cur * 784 + batch_size * 784;
        let img_tensor = Tensor::new(&[batch_size, 784], &self.tst_img[img_start..img_end], false);

        let lbl_start = self.tst_cur * 10;
        let lbl_end = self.tst_cur * 10 + batch_size * 10;
        let lbl_tensor = Tensor::new(&[batch_size, 10], &self.tst_lbl[lbl_start..lbl_end], false);

        self.tst_cur += batch_size;

        (img_tensor, lbl_tensor)
    }
}

fn prompt_adapters() -> wgpu::Adapter {
    let adapters = WGPUContext::list_adapters();

    for (i, adapter) in adapters.iter().enumerate() {
        let info = format!(
            "{} + {}",
            adapter.get_info().name,
            adapter.get_info().backend
        );
        println!("{}. {}", i + 1, info)
    }

    print!("Enter selected adapter number: ");
    let _ = io::stdout().flush();
    let mut input = String::new();
    io::stdin().read_line(&mut input).unwrap();
    let num: usize = input.trim().parse().unwrap();

    let adapter = adapters.into_iter().nth(num - 1).unwrap();
    return adapter;
}

fn draw_number(data: &[f32]) {
    for l in data.chunks(28) {
        for c in l {
            if *c > 0.5 {
                print!("#");
            } else {
                print!(" ");
            }
        }
        println!()
    }
}

fn main() {
    // let adapter = WGPUContext::list_adapters().into_iter().nth(0).unwrap();
    init_runtime(prompt_adapters(), 42);

    let mut loader = DataLoader::new();

    let batch_size = 240;
    let lr = 1e-3;

    let model = MLP::new(&[784, 256, 128, 10], true);
    let mut optimizer = Adam::new(&model, lr, 0.9, 0.999, 1e-8);

    for i in 0..1000 {
        let (img, lbl) = loader.sample_training_batch(batch_size);

        let output = model.forward(&img).unwrap();
        let loss = output.cross_entropy_loss(&lbl).unwrap();

        optimizer.zero_grad();
        loss.backward();
        optimizer.step();

        println!("i: {}, Loss: {}", i, loss.to_vec()[0]);
    }

    {
        let _ng = no_grad().unwrap();
        let (img, label) = loader.sample_test_batch(5);

        let preds = model.forward(&img).unwrap();

        let preds: Vec<usize> = preds
            .to_vec()
            .chunks(10)
            .map(|v| {
                v.iter()
                    .enumerate()
                    .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                    .unwrap()
                    .0
            })
            .collect();

        let exps: Vec<usize> = label
            .to_vec()
            .chunks(10)
            .map(|v| {
                v.iter()
                    .enumerate()
                    .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                    .unwrap()
                    .0
            })
            .collect();

        for ((pred, exp), num) in preds.iter().zip(&exps).zip(img.to_vec().chunks(784)) {
            draw_number(num);
            println!("Expected: {} | Predicted: {}", exp, pred);
        }
    }
}
