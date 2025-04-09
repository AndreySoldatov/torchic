use torchic::{runtime, tensor};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    dotenv::dotenv().ok();
    env_logger::init();

    let runtime = runtime::GPURuntime::new().await?;

    let mut t1 = tensor::Tensor::new_rand(&runtime, &[4, 4, 4]);

    t1 = t1.round(&runtime);
    t1 = t1.exp(&runtime);

    t1 = t1.add(&t1, &runtime)?;

    println!("t1: {:?}", t1.copy_to_cpu(&runtime));

    Ok(())
}
