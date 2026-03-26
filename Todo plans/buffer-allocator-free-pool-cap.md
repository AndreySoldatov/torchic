# Add a Free-Pool Memory Cap with LRU Eviction

## Summary
Add a max-memory constraint as a cache limit for reusable buffers, not as a hard cap on active allocator usage. The allocator will continue to serve live requests even if currently checked-out or pending buffers are large; the limit only controls how many reclaimed buffers are retained in `free_pool`. Eviction will use LRU over free buffers, with recency defined as "last returned to pool."

## Key Changes
- Extend allocator state to track memory classes explicitly:
  - `live_bytes`: bytes currently checked out by active `BufferLease`s.
  - `pending_bytes`: bytes dropped by leases but not yet reclaimed into `free_pool`.
  - `free_bytes`: bytes currently cached in `free_pool`.
  - `max_free_bytes`: configurable cap for cached reusable memory.
  - `lru_clock`: monotonic counter used to stamp reclaimed buffers.
- Extend `BufferEntry` with cache metadata needed for eviction:
  - `last_free_epoch: u64` or equivalent, updated when a buffer enters `free_pool`.
  - No "last used" timestamp is needed on checkout; LRU is based on last return to the pool.
- Keep `free_pool` indexed by capacity for fast best-fit reuse, but add an eviction structure for free buffers:
  - Maintain a secondary ordering over free buffers by `(last_free_epoch, BufferId)` so the allocator can evict oldest cached buffers first.
  - Eviction removes buffers from both the LRU structure and the size-indexed `free_pool`, then deletes them from `store`.
- Update lifecycle/accounting rules:
  - `allocate_new_buffer(size)` increments `live_bytes`.
  - Reusing from `free_pool` decrements `free_bytes` and increments `live_bytes`.
  - `free(buf)` moves the buffer from live to pending: decrement `live_bytes`, increment `pending_bytes`.
  - `reclaim()` waits for safety, then moves pending buffers to `free_pool`: decrement `pending_bytes`, increment `free_bytes`, stamp `last_free_epoch`, insert into LRU, then trim to `max_free_bytes`.
- Add cache trimming in two places:
  - After `reclaim()` populates `free_pool`, evict oldest free buffers until `free_bytes <= max_free_bytes`.
  - Before allocating a brand-new backing buffer in `request()`, run the same trim logic so stale cached buffers are dropped before growing allocator-owned memory further.
  - Do not evict active or pending buffers. If no free buffers remain, allow the new allocation even if total allocator-owned bytes grow; the cap is only for cache retention.
- Public API/interface changes:
  - Add a constructor parameter or config struct so callers can set `max_free_bytes` per allocator instance.
  - Expose lightweight stats for observability, at minimum `live_bytes`, `pending_bytes`, `free_bytes`, and `store_len`.
  - Keep reclaim explicit and runtime-driven; do not hide `wait_indefinitely()` inside the normal fast path except for the existing explicit reclaim method.

## Implementation Notes
- Prefer "free-pool cap" naming over generic "max memory" in code/comments so the semantics are unambiguous.
- Implement removal from `free_pool` carefully because eviction is no longer "pop from one vec"; you need exact removal by `BufferId` from both indexes.
- Use a monotonic epoch counter instead of wall-clock time. This keeps ordering deterministic and avoids time APIs.
- Preserve the existing size-fit heuristic in `request()`. Eviction policy and reuse policy are separate concerns.

## Test Plan
- Reuse path: request, drop, reclaim, request same-size buffer, confirm reuse occurs and byte counters move correctly.
- Cache cap enforcement on reclaim: reclaim several buffers so `free_bytes` exceeds the cap, confirm oldest free buffers are evicted until under cap.
- Cache cap enforcement on allocation: populate `free_pool`, then issue a request that cannot reuse any cached buffer and would allocate new memory; confirm allocator trims old free buffers before creating a new backing buffer.
- LRU ordering: reclaim multiple buffers in sequence, reuse one of them, reclaim it again, then trigger trimming and confirm the oldest still-free entries are evicted first.
- Safety invariants: active and pending buffers are never evicted; reclaim still requires explicit call; double-free behavior remains unchanged.
- Edge cases: exact-cap boundary, zero cached bytes allowed, multiple buffers of same capacity, and trimming when a `free_pool` bucket becomes empty.

## Assumptions
- The cap is a cache-retention limit for `free_pool`, not a hard allocator-wide OOM limit.
- Reclaim is called explicitly by higher-level runtime code at safe synchronization points such as end-of-iteration.
- LRU recency is defined by "last returned to pool," not by checkout time or wall-clock time.
