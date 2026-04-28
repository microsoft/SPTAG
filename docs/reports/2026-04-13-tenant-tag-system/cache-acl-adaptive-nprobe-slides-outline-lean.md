# Lean Slides Outline: Cache Scheduling, ACL Design, And Adaptive Nprobe

Date: 2026-04-14

This version is optimized for presentation. Keep each slide to 2-3 on-screen points and move formulas, implementation detail, and caveats into the talk track.

## Slide 1: Title

On slide:

- Multi-Tenant Filtered Search In SPANN
- Cache, ACL routing, adaptive dense search
- Snapshot: 2026-04-13

Talk track:

- This talk is about one serving problem: keep recall stable while tenant mixing and tag filtering make latency harder.

## Slide 2: Problem

On slide:

- Many tenants share one server
- Tag filters change the right search path
- Limited cache drives tail latency

Talk track:

- The target is not just good ANN recall. It is good recall under cache pressure and mixed-tenant traffic.

## Slide 3: System Layout

On slide:

- TenantIndexManager: load, cache, route
- HeadIndex in memory, postings on SSD
- Search flow: graph -> posting scan -> top-k

Talk track:

- The wrapper decides tenant lifecycle and filter routing; SPANN handles graph search and posting scan.

## Slide 4: Cache Scheduling

On slide:

- Lazy load per tenant
- Byte budget + eviction
- In-flight tenants stay pinned

Talk track:

- The important benchmark result is that random tenant mixing hurts p99 much more than recall.
- If you want one number on the slide, use: p99 84.85 ms -> 288.86 ms.

## Slide 5: Build-Time ACL Metadata

On slide:

- Rebuild posting-to-vector assignments from SSD
- Build exact tag stats and sparse side index
- Preserve head-only tag metadata

Talk track:

- This is what makes filtered serving exact rather than heuristic: routing stats and head metadata are computed from real postings.

## Slide 6: Query-Time Routing

On slide:

- Sparse if direct posting union is small
- Dense otherwise
- Runtime knobs control the boundary

Talk track:

- The key change is that sparse routing now uses exact posting fanout, not tenant-size-scaled heuristics.

## Slide 7: Adaptive Dense Budget

On slide:

- Base budget = max(internal result count, top-k)
- Lower selectivity => more postings
- Goal: keep filtered coverage stable

Talk track:

- Do not put the full formula on the slide.
- Say: the dense path increases posting budget from selectivity, target recall, and coverage needs.
- If needed, keep the exact formula in speaker notes or backup slides only.

## Slide 8: Cache Benchmark Result

On slide:

- Sequential: 31.87 ms avg, 0.9953 recall
- Random mix: 47.27 ms avg, 0.9945 recall
- Main gap is locality, not correctness

Talk track:

- The recall line is stable. The latency line is what moves, especially tail latency.

## Slide 9: Top-k Finding

On slide:

- Broad filters stay stable as k grows
- Team/project filters do not
- High-k work should focus on dense coverage

Talk track:

- Do not show the whole table unless the audience needs it. One sentence is enough: narrow filters are not yet top-k general.

## Slide 10: Recent Branch Work

On slide:

- Exact sparse routing by posting union
- Persistent search params across lazy load
- Formal benchmark runner in Tools/benchmarks

Talk track:

- If you need a fourth point, add: top-k-aware dense budget and copy-back fix.

## Slide 11: Current Gaps

On slide:

- Dense path PS gating is off
- Cache admission still uses rough size estimates
- BuildSignatures is still expensive

Talk track:

- This slide should feel like a short risk register, not a paragraph of caveats.

## Slide 12: Next Steps

On slide:

- Run cache-budget sweep
- Revisit dense-path PS gating
- Tune high-k filtered search

Talk track:

- The message is simple: cache first for serving, dense coverage first for high-k quality.

## Suggested Deck Cleanup Rules

- Keep each slide to at most 3 bullets.
- Keep each bullet under 10 words when possible.
- Keep only one key number block per result slide.
- Move formulas, parameter lists, and caveats into speaker notes.
- Put detailed tables in backup slides, not the main deck.