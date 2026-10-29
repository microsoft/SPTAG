# Local SPFresh version tracking

[Architecture](../Architecture.md) | [Documentation index](../README.md)

The current `LocalVersionMap` uses the shared `ConcurrentHashMap` adapter.
With Folly enabled it uses Folly; without Folly it uses a locked standard map.
The non-Folly adapter returns read-only owning snapshots from `find`, `begin`,
and insertion results so reads do not dereference erased/rehash-invalidated
storage after releasing the operation lock. Range iteration is a point-in-time
copy, not a live mutable view. Ordinary lookups copy one entry.

Insert, erase, conditional assignment, and clear serialize through the map
mutex. Conditional updates check and assign under the same exclusive lock.
This fallback is not lock-free and may have different performance from Folly.
Tests cover insertion, replacement, conditional updates, snapshot lifetime, and
concurrent compare-and-swap.

Local version persistence records the count followed by ID/version pairs.
Use a matching ID width when saving/loading. Whole-index checkpoint consistency
still requires the calling workflow's synchronization; per-operation map
locking does not make an entire multi-step checkpoint atomic.

Earlier user documentation described a segmented atomic bitmap and a different
TBB-accessor implementation. That description does not match this branch's
`LocalVersionMap` and is not a current compatibility or performance claim.
