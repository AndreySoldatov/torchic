use log::info;
use std::time::Instant;

fn main() {
    dotenv::dotenv().ok();
    env_logger::init();

    let mut input_data: Vec<f32> = vec![];
    input_data.reserve(4_000_000);
    for i in 0..4_000_000 {
        input_data.push(i as f32);
    }

    let now = Instant::now();

    for i in 0..4_000_000 {
        input_data[i] = (input_data[i].sin()).sqrt();
    }

    info!(
        "Result: {:?}, Time taken: {:?}",
        input_data[0..10].to_vec(),
        now.elapsed()
    );
}
