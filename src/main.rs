use torchic::{
    runtime::{WGPUContext, init_runtime},
    tensor::Tensor,
};

fn main() {
    let adapter = WGPUContext::list_adapters().into_iter().nth(0).unwrap();
    init_runtime(adapter);

    let t1 = Tensor::new(
        &[3, 3],
        &[1.0, 2.0, 1.0, 0.0, 1.0, 0.0, 2.0, 3.0, 4.0],
        true,
    );
    let t2 = Tensor::new(&[3, 2], &[2.0, 5.0, 6.0, 7.0, 1.0, 8.0], true);

    let res = t1.matmul(&t2).unwrap();
    res.backward();

    println!("{:?}", res.to_vec());
    println!("{:?}", t2.grad().unwrap().to_vec());
}
