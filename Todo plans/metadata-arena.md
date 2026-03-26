# Add a Transient Metadata Arena

## Summary
Introduce a dedicated metadata arena for transient kernel metadata such as shapes and strides. The arena will use a single long-lived GPU buffer with monotonic append allocation during a batch and explicit `reset()` at a safe runtime boundary. This avoids creating many small one-shot buffers while keeping lifetime management simple and correct.

## Key Changes
- Add a `MetadataArena` that owns one GPU buffer, a write cursor, total capacity, and a fixed conservative alignment used for every allocation.
- Use a linear reset model for v1, not a wraparound ring:
  - `alloc(bytes)` rounds the cursor up to the fixed alignment, reserves a contiguous region, writes metadata into the shared buffer, and returns a handle.
  - `reset()` rewinds the cursor to zero and makes the whole arena reusable.
  - No per-allocation free and no incremental reclaim within a batch.
- Define a lightweight metadata handle returned by allocation:
  - Include at minimum `offset` and `size`.
  - Offset is guaranteed to satisfy the arena's fixed alignment.
  - The handle is intended for dispatch/bind setup against one shared metadata buffer.
- Bind model:
  - Keep one shared metadata buffer in the runtime.
  - Per dispatch, pass the metadata handle's offset/range into bind setup instead of creating a new metadata buffer.
  - Callers should treat arena allocations as valid only until the next `reset()`.
- Overflow behavior:
  - If a batch exceeds arena capacity, fail fast rather than silently reallocating or falling back to per-dispatch buffers.
  - Prefer returning a typed error in normal paths and asserting in debug-only convenience paths if needed.
- Sizing/configuration:
  - Arena capacity is fixed at creation time and must cover worst-case metadata produced between two resets.
  - Add a constructor/config parameter for arena size so runtime setup owns the tradeoff explicitly.

## Implementation Notes
- Keep alignment internal to the arena API for v1. Callers request byte payloads, not alignment.
- Implement allocation as “align cursor, check capacity, write payload, advance cursor.”
- Do not over-design this into a general allocator yet; it is a transient scratch arena for command-stream metadata.
- Place `reset()` at an explicit runtime-owned safe point such as end-of-iteration or end-of-submission batch, after the GPU can no longer read metadata from the previous batch.
- Keep metadata packing centralized so kernels share one canonical encoding for shapes, strides, and similar descriptors.

## Test Plan
- Sequential allocations return monotonically increasing aligned offsets and correct sizes.
- `reset()` rewinds the arena and subsequent allocations reuse the buffer from offset zero.
- Capacity overflow fails fast when one batch exceeds configured arena size.
- Multiple metadata writes in one batch preserve exact payload contents at the expected offsets.
- Alignment is enforced for all returned handles, including after varied-size allocations.
- Runtime integration scenario: allocate metadata for several dispatches in one batch, bind via shared buffer offsets, then reset before the next batch.

## Assumptions
- Arena memory is reclaimed only by explicit batch-level `reset()`, not by fine-grained fence tracking.
- The shared metadata buffer is consumed through per-dispatch offsets/ranges, not via one-off temporary buffers.
- A fixed conservative alignment is sufficient for all metadata formats used in v1.
- The first implementation should optimize for simplicity and correctness; if peak batch metadata becomes too large later, evolve toward multi-chunk or ring semantics then.
