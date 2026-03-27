use torchic::runtime::{WGPUContext, init_runtime};

fn main() {
    let adapter = WGPUContext::list_adapters().into_iter().nth(0).unwrap();
    init_runtime(adapter);
}
