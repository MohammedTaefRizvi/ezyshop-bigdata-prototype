# Step 4: Test and Refinement Log (SLO d)

Testing combined three methods:

- **Live rehearsal** of the video demo (realtime producer + `--follow` stream), which found T-19 to T-23.
- **Automated tests:** `pytest`, 15 tests. Unit tests cover the log, text analytics, privacy,
  windows and watermark. Integration tests run the whole pipeline at `tiny` scale.
- **End-to-end runs** at `tiny` (≈ 50 s) and `small` (≈ 3 min) scale, using the generated
  sample datasets with deliberately injected defects.
- **Instrumented diagnosis:** profiling, per-batch watermark traces, and Lambda reconciliation
  used as a correctness oracle.

Every entry below happened during development. The "before" and "after" values are measured
outputs of the prototype, not estimates.

## Summary

| ID | Area | Type | Issue identified | Resolution | Before → After |
|---|---|---|---|---|---|
| T-01 | Data generation | Error | pandas 3 deprecations (`dayofyear`, `weekday`) would break on upgrade | Switched to `day_of_year`, `day_of_week` | Warnings → none |
| T-02 | Data generation | Scalability | Social-post volume was fixed (20/day), not tied to the scale profile, so `tiny` had unrealistic text volume | Volume scales with `orders_per_day` | 9,925 → 1,487 posts at tiny |
| T-03 | Silver validation | Accuracy | Price-outlier rule `(p <= 0) OR (p > 3×list)` let **NaN prices through** (NaN comparisons are False) | Rewritten in positive form: keep only `0 < p <= 3×list` | NaN prices accepted → quarantined |
| T-04 | Gold / ML | Error | DuckDB 2.0 removed `Relation.fetchone()`. Draining it on the shared connection then raised "unfinished result" | `duckdb.execute(...).fetchall()` | Crash → fixed |
| T-05 | Forecasting | Error | sklearn binning crashed when a feature was entirely missing in training (`lag52` with < 1 year of history at tiny scale) | `usable_features()` drops all-NaN/constant features automatically | Crash → runs at any scale |
| T-06 | Streaming | Accuracy | **Multi-topic watermark.** The small `orders` topic was fully read in batch 1, so its max event time pushed the global watermark to the end of the stream. Clickstream events were then dropped as "late". Caught by the Lambda reconciliation (batch had 32,908 events the stream lacked) | Watermark = **min** across topics (Spark's default multiple-watermark policy). The sparse `inventory` topic is excluded so it cannot stall progress | 32,908 → 16,546 wrongly dropped |
| T-07 | Streaming | Accuracy / bottleneck | **Partition skew.** Polling a fixed 500 records per partition let lightly filled partitions race ahead in event time | Backlog-proportional rate limiting (as Spark's `maxOffsetsPerTrigger` does) | 16,546 → 629 dropped |
| T-08 | Streaming | Accuracy | Cart abandonment was computed per window, but purchases often land in the next minute, which inflated abandonment and caused false alerts | Count purchases in the same or next window | 11 → 4 abandonment alerts (outage only) |
| T-09 | Streaming | Accuracy | TRENDING alert compared a product with the *average* product, so the permanently popular head product alerted every minute (31 alerts) and hid the real viral product | Per-product EWMA baseline (spike vs its own history) + warm-up of 3 real-traffic windows | 31 → 2–3 alerts; viral product detected at onset (13× baseline) |
| T-10 | Simulator | Accuracy | Simulated sessions lasted up to ~2.5 min, so out-of-orderness exceeded the 2-min watermark | Session event gaps 2–12 s (realistic); watermark choice documented as a tunable trade-off | 629 → 162 dropped (≈ the 0.4 % injected 8-min-late events) |
| T-11 | Streaming | Accuracy | At `small` scale, revenue completeness was only **85.8 %**: one partition ran 2–3 min ahead in event time even with byte-proportional polling, because partitions do not fill uniformly over time | **Event-time-aligned polling**: every partition is read only up to *(earliest head time + 60 s)*, the same idea as Apache Flink's watermark alignment. Late heads no longer anchor the frontier | Completeness 85.8 % → 99.3 %; drops = injected late events only |
| T-12 | Streaming | **Performance** | Profiling (`cProfile`): `Series.isin(dedup_state)` rebuilt an Arrow array of the *whole* dedup state on every micro-batch, which was 60 % of runtime and O(state) | O(batch) hash-set lookups | **11,953 → 43,205 events/s** (3.6×) |
| T-13 | Streaming | Accuracy | Live stock went negative (−477) under peak-load replay | Stock floored at 0, unmet units counted as back-orders; STOCK_OUT severity added | Negative stock → 0 + 908 back-ordered units reported |
| T-14 | Streaming | Usability | 151 LOW_STOCK alerts at small scale: products *already* below reorder point alerted on their first sale | Those products are on the batch reorder list; real-time alerts only fire on a threshold *crossing* | 151 → 40 alerts |
| T-15 | Forecasting | Accuracy | The first model met 90 % (90.9 %) but **did not beat** the naive (91.3 %) and moving-average baselines, so it added no value | Campaign-aware features (own, competing and last week's campaign days) + EWMA level + **level-ratio target** with absolute-error loss | 90.9 % → **93.2–93.6 %**, +1.9 pts over the best baseline |
| T-16 | Streaming | Accuracy | Reconciliation showed 30 events processed but missing from Gold: at end-of-stream the flush advanced the watermark from the *slowest* topic, so the last window was never emitted. Also hardened: the watermark could move backwards when a topic first appeared | Flush closes every open window; watermark made monotonic (high-water mark) | stream_events 39,467 → 39,497 = processed; batch-only events = late drops exactly |
| T-17 | Benchmark | Bottleneck | Vectorising aggregation gave **no** gain (175 k vs 181 k ev/s): JSON deserialisation dominates. At 200–300 k events, process start-up hid parallel gains | Benchmark at 1 M events; added a native multi-threaded columnar JSON scan (DuckDB) | Best 181 k → **0.35–0.96 M events/s** (battery vs mains power); parallel scaling visible 1→8 workers |
| T-18 | Dashboard | Error | Two pages crashed: pandas 3 removed `Series.reset_index(names=…)` | `rename_axis(...).reset_index()`; all 6 pages re-checked in the browser | 2 page errors → 0 |
| T-19 | Streaming (live rehearsal) | Accuracy | At live rates (~20× the replayed history) **every** popular product looked "trending" because the baseline was a raw view count | Trending = a product's *share* of all views vs its usual share (robust to site-wide surges such as Black Friday) | False trending alerts on popular products → only the genuinely viral product |
| T-20 | Streaming (live rehearsal) | Error | When live traffic stopped, the watermark stopped moving, so the last windows (including the checkout-outage minute) were never closed or alerted | Source-idleness handling (cf. Flink `withIdleness`): open windows close after a quiet period of one watermark length | Outage alert missing → emitted |
| T-21 | Streaming (live rehearsal) | Accuracy | The end-of-stream flush from T-16 pushed the watermark ~3 min **into the future** (window end + lateness), so a live demo started straight after `run_all.py` had 118,970 of 120,000 events dropped as late | Flush moves the watermark to exactly the last window end; replayed traffic now ends 2 min in the past | Live events dropped as late: 118,970 of 120,000 → 701 of 60,000 (the 0.4 % deliberately delayed + jitter) |
| T-22 | Simulator (live rehearsal) | Accuracy | In realtime mode, events inside a session were stamped up to ~60 s *ahead* of the clock, more disorder than a 30 s demo watermark tolerates | Live events are stamped when sent, as real devices do | With T-19–T-23: the viral product is flagged at 10x its usual view share, only the faulty product triggers a review burst, first alerts ~110 s after traffic starts |
| T-23 | Streaming (live rehearsal) | Accuracy | "≥ 3 negative reviews per minute" fired for many popular products at high traffic | Burst = a product holds ≥ 25 % of the minute's negative reviews (and ≥ 3) | Several false bursts per minute → faulty product only |

## Selected evidence

### T-06 / T-07 / T-11: watermark progression (per-batch trace, `small` scale)

Before alignment, partition 2 was consistently ahead of the other partitions (clickstream
partition max event times at batch 6: `17:51:51, 17:52:24, 17:53:32, 17:52:15, …`). That
difference was enough to drop 117 of 214 orders in one micro-batch. After event-time-aligned
polling, the only events dropped are the 0.4 % that the simulator deliberately delays by
8 minutes (mobile app offline). The batch layer then restores them exactly:

```
received 40,002 | duplicates 359 | processed 39,497 | late_dropped 146
reconciliation: events_only_in_batch 146 | stream_events 39,497 | revenue completeness 99.4 %
```

### T-12: profile before the fix (top of `cProfile`, 31 micro-batches)

```
 ncalls  cumtime  function
     31    4.235  streaming.py:process_batch
     62    2.545  pandas Series.isin          <- 60 % of processing time
     32    0.667  checkpoint (pickle state)
```

### T-17: throughput benchmark (1,000,000 events, 12-core laptop)

| Strategy | Workers | Events/s | Matches single-process result |
|---|---|---|---|
| naive row-at-a-time | 1 | ~180 k | n/a |
| vectorised micro-batch | 1 | ~170 k | n/a |
| parallel map-reduce | 1 / 2 / 4 / 8 | ~165 k / ~250 k / ~330 k / ~275 k | yes |
| native columnar engine (all cores) | 12 | ~0.35 M (battery) to ~0.96 M (mains) | yes |

Interpretation:

- Scaling is near-linear up to 4 workers.
- It flattens at 8 workers because the topic has 8 partitions (the unit of parallelism) and the
  laptop's memory and disk bandwidth are shared.
- In production, adding partitions and consumers on more machines removes that ceiling.

Exact numbers for the latest run are in `outputs/benchmark_results.json`.

## Automated test suite (final run)

```
tests/test_units.py     8 passed
  partitioner determinism & balance · per-key ordering & offset resume · consumer-group isolation
  aligned polling bounds event-time skew · sentiment/negation/HTML cleaning · pseudonym properties
  mixed timestamp parsing · window aggregates mergeable · dedup + watermark across a restart
tests/test_pipeline.py  7 passed
  quality rules enforced & quarantined · no direct identifiers after Bronze
  Gold joins structured+unstructured · models beat baselines · streaming + exact Lambda reconciliation
  GDPR erasure propagates through a Silver rebuild · parallel benchmark equals single-process result
15 passed
```

## Known limitations (not fixed, documented)

- **Synthetic data.** Model accuracies are measured on generated data with known structure.
  Real EzyShop data will be noisier; sentiment agreement in particular is optimistic, because
  the review text comes from templates.
- **Single-machine scale.** Throughput results show headroom, not proof of 10 k TPS sustained
  on a cluster. A load test on the target infrastructure (e.g. MSK + EMR) is still required,
  as Assessment 2 recommends.
- **Delivery semantics.** At-least-once delivery + idempotent upserts give effectively-once
  *window results*. Live stock counters are not transactional across a crash between output and
  checkpoint.
- **Scale-out code.** `scale_out/` (Kafka + Spark) is reference code and was not executed in
  this environment (no Docker/Java).
