# Torchic: GPU Tensor library in rust with WebGPU backend

Torchic uses WebGPU through wgpu-rs crate to perform tensor computations on the GPU.

# Example

```rust
use torchic::{runtime, tensor};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    dotenv::dotenv().ok();
    env_logger::init();

    let runtime = runtime::GPURuntime::new().await?;

    // Matrix 4x2
    let t1 = tensor::Tensor::from_cpu_data(
        &runtime,
        &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        &[4, 2],
    ).unwrap();

    // Matrix 2x3
    let t2 = tensor::Tensor::from_cpu_data(
        &runtime, 
        &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 
        &[2, 3]
    ).unwrap();

    // Immutable matrix multiplication
    let t3 = t1.matmul(&t2, &runtime)?;

    assert_eq!(
        vec![
            9.0, 12.0, 15.0, 19.0, 26.0, 33.0, 29.0, 40.0, 51.0, 39.0, 54.0, 69.0
        ],
        t3.copy_to_cpu(&runtime)
    );

    Ok(())
}
```