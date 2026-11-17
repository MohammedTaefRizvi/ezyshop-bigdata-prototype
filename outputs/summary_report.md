# EzyShop Big Data Prototype - Run Summary

Scale profile: **small**  |  data as of **2026-10-31**  |  total pipeline time **162.7 s**

## Scorecard against Assessment 2 requirements

| Requirement (Assessment 2) | Target | Prototype result | Status |
|---|---|---|---|
| Peak throughput | 10,000 events/s | 1,580,334 events/s on one laptop (158.03x headroom) | MET |
| Real-time insight latency | <= 300 s | 181 s by design (window+watermark); p95 produce-to-process 2.18 s | MET |
| Demand-forecast accuracy | >= 90 % | 93.5% (1-WAPE, 12 unseen weeks; best baseline 91.2%) | MET |
| Personalisation improvement | +25 % | +204.5% precision@10 vs popularity | MET |
| Identify customers at risk (12 % churn problem) | - | ROC-AUC 0.9489 (recency-only baseline 0.9245); top-decile lift 4.24x | n/a |
| Data quality (veracity) | rules enforced | 5,369 records quarantined, 7,694 duplicates removed | MET |
| Unstructured text analytics validity | - | sentiment vs star-rating correlation 0.882 | n/a |
| GDPR (pseudonymisation, erasure) | supported | HMAC pseudonyms in Silver/Gold; erasure demo: recommendations + profile deleted; key on suppression list for next Silver rebuild | MET |
| Storage efficiency (volume) | - | raw 168.3 MB -> Bronze Parquet 17.4 MB (9.7x) | n/a |

## Real-time layer

- Events received 40,002; duplicates removed 359; late events dropped by the watermark 146; micro-batches 31.
- Alerts raised: {"CART_ABANDONMENT_SPIKE": 3, "LOW_STOCK": 41, "NEGATIVE_REVIEW_BURST": 4, "TRENDING_PRODUCT": 2}
- Lambda reconciliation: batch recomputation found 146 events the stream had to drop as late; stream revenue completeness 99.40% -> corrected view in gold/stream_metrics_reconciled.

## Pipeline step timings (s)

| Step | Seconds |
|---|---|
| bronze | 1.4 |
| silver | 16.1 |
| gold | 1.2 |
| ml_demand_forecast | 4.6 |
| inventory_planning | 0.1 |
| ml_recommendations | 7.8 |
| ml_churn | 1.8 |
| stream_produce | 1.8 |
| streaming | 2.3 |
| bronze_stream_archive | 0.4 |
| reconciliation | 0.5 |
| benchmark | 124.5 |
| gdpr | 0.1 |

Charts: `outputs/charts/`. Gold extracts: `outputs/gold_samples/`. Action lists: `reorder_list.csv`, `retention_campaign_targets.csv`, `realtime_alerts.csv`.
