use torchic::{
    runtime::{WGPUContext, dump_stats, init_runtime},
    tensor::Tensor,
};

fn main() {
    let adapter = WGPUContext::list_adapters().into_iter().nth(0).unwrap();
    init_runtime(adapter);

    let t1 = Tensor::new(&[4], &[1.0, 2.0, 3.0, 4.0], true);
    let t2 = Tensor::new(&[4], &[10.0, 10.0, 10.0, 10.0], true);
    let res = t1.mul(&t2).unwrap().sum().unwrap();
    res.backward();

    println!("{:?}", res.to_vec());
    println!("{:?}", t2.grad().unwrap().to_vec());
}
