use std::io::{self, Write};

use mnist::*;
use torchic::{
    nn::{Adam, MLP},
    runtime::{RuntimeStats, WGPUContext, init_runtime, no_grad, stats},
    tensor::Tensor,
};

struct DataLoader {
    trn_img: Vec<f32>,
    trn_lbl: Vec<f32>,
    tst_img: Vec<f32>,
    tst_lbl: Vec<f32>,
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
            trn_img: trn_img.into_iter().map(|v| v as f32 / 255.0).collect(),
            trn_lbl: trn_lbl.into_iter().map(|v| v as f32).collect(),
            tst_img: tst_img.into_iter().map(|v| v as f32 / 255.0).collect(),
            tst_lbl: tst_lbl.into_iter().map(|v| v as f32).collect(),
        }
    }

    fn training_batches(&self, batch_size: usize) -> Vec<(Tensor, Tensor)> {
        let mut res = vec![];

        for (imgs, lbls) in self
            .trn_img
            .chunks(batch_size * 784)
            .zip(self.trn_lbl.chunks(batch_size * 10))
        {
            res.push((
                Tensor::new(&[imgs.len() / 784, 784], imgs, false),
                Tensor::new(&[lbls.len() / 10, 10], lbls, false),
            ));
        }

        res
    }

    fn test_batches(&self, batch_size: usize) -> Vec<(Tensor, Tensor)> {
        let mut res = vec![];

        for (imgs, lbls) in self
            .tst_img
            .chunks(batch_size * 784)
            .zip(self.tst_lbl.chunks(batch_size * 10))
        {
            res.push((
                Tensor::new(&[imgs.len() / 784, 784], imgs, false),
                Tensor::new(&[lbls.len() / 10, 10], lbls, false),
            ));
        }

        res
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

    adapters.into_iter().nth(num - 1).unwrap()
}

fn one_hot_batch_to_indices(data: &[f32]) -> Vec<usize> {
    data.chunks(10)
        .map(|c| {
            c.iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .unwrap()
                .0
        })
        .collect()
}

fn main() {
    init_runtime(prompt_adapters(), 42);

    let mut stat_vec = vec![];
    stat_vec.push(stats());

    let loader = DataLoader::new();

    let batch_size = 240;
    let train_batches = loader.training_batches(batch_size);
    let test_batches = loader.test_batches(batch_size);

    stat_vec.push(stats());

    let epochs = 10;
    let lr = 1e-3;

    let model = MLP::new(&[784, 256, 128, 10], true);
    let mut optimizer = Adam::new(&model, lr, 0.9, 0.999, 1e-8);

    stat_vec.push(stats());

    let mut epoch_times = vec![];
    let mut losses = vec![];
    for epoch in 0..epochs {
        let start = std::time::Instant::now();

        let mut total_loss = 0.0;

        for (img, lbl) in &train_batches {
            let output = model.forward(img).unwrap();
            let loss = output.cross_entropy_loss(lbl).unwrap();

            optimizer.zero_grad();
            loss.backward();
            optimizer.step();

            total_loss += loss.to_vec()[0];
        }

        epoch_times.push(start.elapsed().as_secs_f32());
        losses.push(total_loss / train_batches.len() as f32);
        println!(
            "Epoch: {}, Loss: {}, epoch time: {:.2}s",
            epoch + 1,
            losses[losses.len() - 1],
            epoch_times[epoch_times.len() - 1],
        );
        stat_vec.push(stats());
    }
    std::fs::write("./epoch_times.txt", format!("{:?}", epoch_times)).unwrap();
    std::fs::write("./loss.txt", format!("{:?}", losses)).unwrap();
    serde_json::to_writer_pretty(std::fs::File::create("stats.json").unwrap(), &stat_vec).unwrap();

    {
        // Evaluation
        let _ng = no_grad().unwrap();

        let mut correct = 0;
        let mut total = 0;

        for (img, lbl) in &test_batches {
            let output = model.forward(img).unwrap();

            let pred = one_hot_batch_to_indices(&output.to_vec());
            let exp = one_hot_batch_to_indices(&lbl.to_vec());

            total += exp.len();

            for (p, e) in pred.iter().zip(exp.iter()) {
                correct += (p == e) as usize;
            }
        }

        println!("Test Accuracy: {}%", correct as f32 / total as f32 * 100.0);
    }
}
