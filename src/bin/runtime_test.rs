use torchic::{runtime, tensor};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    dotenv::dotenv().ok();
    env_logger::init();

    let runtime = runtime::GPURuntime::new().await?;

    let mut t1 = tensor::Tensor::from_cpu_with_shape(
        &runtime,
        &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        &[4, 2],
    )
    .unwrap();
    let t2 =
        tensor::Tensor::from_cpu_with_shape(&runtime, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3])
            .unwrap();

    let t3 = t1.matmul(&t2, &runtime)?;

    println!("t3: {:?}", t3.copy_to_cpu(&runtime));

    Ok(())
}
