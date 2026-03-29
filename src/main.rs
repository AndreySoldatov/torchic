use torchic::{
    runtime::{WGPUContext, dump_stats, init_runtime},
    tensor::Tensor,
};

fn main() {
    let adapter = WGPUContext::list_adapters().into_iter().nth(0).unwrap();
    init_runtime(adapter);

    for i in 0..10000 {
        let t1 = Tensor::new(vec![2000000], &[1.0; 2000000], true);
        let t2 = Tensor::new(vec![2000000], &[1.0; 2000000], true);
        let res = t1.add(&t2).unwrap();
        res.backward();
        println!("{}", i);
    }
    dump_stats();
}
