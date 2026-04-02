# TODO: Keep `sum_backward()` on GPU

## Problem

`sum_backward()` currently calls `out_grad.to_vec()[0]` to read the scalar gradient back to CPU, then builds a full parent-gradient tensor on CPU.

This is correct, but it introduces a GPU-to-CPU synchronization point in the middle of backward.

## Goal

Replace the CPU readback path with a small compute kernel that broadcasts a scalar tensor of shape `[1]` into an output tensor of shape `p.shape()`.

## Proposed kernel

Create a dedicated WGSL compute kernel:

- binding 0: input scalar buffer (`array<f32>`, length 1)
- binding 1: output buffer (`array<f32>`, length `numel`)
- behavior: `output[idx] = input[0]`
- use the same grid-stride loop pattern as the existing elementwise/reduction shaders

Suggested shader shape:

```wgsl
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(64)
fn main(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) nw: vec3<u32>
) {
    let total_threads = nw.x * 64u;
    let v = input[0];
    var idx = gid.x;

    loop {
        if (idx >= arrayLength(&output)) { return; }
        output[idx] = v;
        idx += total_threads;
    }
}
```

## Integration plan

1. Add a new kernel/op for scalar broadcast/fill.
2. Dispatch it from `sum_backward()` using `out_grad` as input.
3. Return a tensor with shape `p.shape()` and accumulate it into `grad_store`.
4. Remove the `to_vec()` readback from `sum_backward()`.

## Notes

- This is a targeted fix for reduction backward, not a full general broadcasting system.
- A dedicated kernel is likely the simplest path with the current buffer ownership model.
- Longer term, this could be generalized into broader broadcast support for elementwise ops.
