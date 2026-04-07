use torchic::{
    runtime::{WGPUContext, init_runtime},
    tensor::Tensor,
};

fn main() {
    let adapter = WGPUContext::list_adapters().into_iter().nth(0).unwrap();
    init_runtime(adapter, 42);

    // let t1 = Tensor::ones(&[3], true);
    // let t2 = Tensor::new(&[2], &[2.0, 5.0], true);

    // let res = t1.outer(&t2).unwrap();
    // res.backward();

    // println!("{:?}, {:?}", res.to_vec(), res.shape());
    // println!("{:?}", t2.grad().unwrap().to_vec());
}
