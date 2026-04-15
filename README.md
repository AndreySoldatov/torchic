# Torchic
A portable WebGPU backed tensor framework built with rust.

Torchic is an eager prototype tensor framework with WebGPU as a primary backend.

It supports multiple features:
* Automatic differentiation via custom autograd engine
* Buffer and metadata automatic caching and reuse
* Neural network module that defines useful primitives, like linear layers, `Adam` and `CrossEntropyLoss`.

# Example
This example shows how to train and evaluate MNIST character recognition model with torchic with MLP architecture:
```rust
use std::io::{self, Write};

use mnist::*;
use torchic::{
    nn::{Adam, MLP},
    runtime::{WGPUContext, init_runtime, no_grad},
    tensor::Tensor,
};

// Dataset loading functionality is not really relevant to the showcase, so we'll skip it
struct DataLoader {...}

// Print available GPU + backend and prompt the user to select one
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

// Convert the one-hot vector to the index of max element (but for batches)
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
    // Init runtime with selected adapter and random seed
    init_runtime(prompt_adapters(), 42);

    let loader = DataLoader::new();

    let batch_size = 240;
    let train_batches = loader.training_batches(batch_size);
    let test_batches = loader.test_batches(batch_size);

    let epochs = 5;
    let lr = 1e-3;

    // Create MLP model with two hidden layers
    let model = MLP::new(&[784, 256, 128, 10], true);
    // Create Adam optimizer
    let mut optimizer = Adam::new(&model, lr, 0.9, 0.999, 1e-8);

    // Training loop
    for epoch in 0..epochs {
        let mut total_loss = 0.0;

        for (img, lbl) in &train_batches {
            // Forward pass
            let output = model.forward(img).unwrap();
            let loss = output.cross_entropy_loss(lbl).unwrap();

            // Backward pass
            optimizer.zero_grad();
            loss.backward();
            optimizer.step();

            total_loss += loss.to_vec()[0];
        }

        println!(
            "Epoch: {}, Loss: {}",
            epoch + 1,
            total_loss / train_batches.len() as f32
        )
    }

    {
        // Evaluation stage. Notice the no_grad() guard. It is similar to torch.no_grad()
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
```