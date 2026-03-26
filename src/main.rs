fn main() {
    let adapter = WGPURuntime::list_adapters().into_iter().nth(0).unwrap();
    init_runtime(adapter);
}
