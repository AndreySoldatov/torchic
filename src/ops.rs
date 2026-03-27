use crate::tensor::Tensor;

pub fn add(lhs: &Tensor, rhs: &Tensor) -> Tensor {
    todo!()
    // let requires_grad = self.0.requires_grad || other.0.requires_grad;
    // let bsize = self.0.shape.iter().product::<usize>() * DTYPE_SIZE;

    // let self_id = self.0.id;
    // let other_id = other.0.id;
    // let self_rg = self.0.requires_grad;
    // let other_rg = other.0.requires_grad;
    // let out_id = get_id();

    // let grad_fn: Option<Box<dyn Fn() + Send + Sync + 'static>> = if requires_grad {
    //     Some(Box::new(move || {
    //         let out_grad = GRAD_STORE.lock().unwrap().map.get(&out_id).unwrap().clone();
    //         if self_rg {
    //             GRAD_STORE.lock().unwrap().acc(self_id, out_grad.clone());
    //         }
    //         if other_rg {
    //             GRAD_STORE.lock().unwrap().acc(other_id, out_grad);
    //         }
    //     }))
    // } else {
    //     None
    // };

    // Self(Arc::new(TensorInner {
    //     id: out_id,
    //     buf: BUFFER_ALLOC.lock().unwrap().request(bsize),
    //     shape: self.0.shape.clone(),
    //     requires_grad,
    //     grad_fn,
    // }))
}
