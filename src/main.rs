use torchic::{
    runtime::{WGPUContext, dump_stats, init_runtime},
    tensor::Tensor,
};

fn main() {
    let adapter = WGPUContext::list_adapters().into_iter().nth(0).unwrap();
    init_runtime(adapter);

    let t1 = Tensor::ones(&[500_000_000], false);
    let res = t1.sum().unwrap();
    println!("{:?}", res.to_vec());
}
