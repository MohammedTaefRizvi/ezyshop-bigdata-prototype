# Step 1: Problem Definition (SLO b, e)

## Problem from Assessment 2

EzyShop is a fast-growing Australian e-commerce retailer selling electronics, apparel and home
goods. It runs on a single-cloud relational database. That database cannot keep up with about
1.5 TB of data per day (growing 20 % a year) or with peaks of 10,000 transactions per second.
Data is spread across siloed systems, so the business has no real-time view of customers.
Restocking decisions are slow and personalisation is poor. EzyShop lost 12 % of its customers
in six months.

## Refined Big Data challenge (one-paragraph summary)

EzyShop's business problem becomes one Big Data challenge: **real-time customer-behaviour and
demand intelligence on a single integrated platform**. The platform must ingest high-velocity
order, clickstream and inventory events at up to 10,000 events per second. It must combine them
with semi-structured supplier and web-log data and with unstructured review and social-media
text. It must turn all of this into actionable insight within five minutes: trending products,
checkout failures, review crises and stock depletion. Three predictive capabilities must be
rebuilt on the same data:

- churn prediction, to address the 12 % loss;
- personalised recommendations, with a target of +25 %;
- weekly demand forecasts with at least 90 % accuracy, to drive re-ordering.

The data-related challenges map directly onto the 5 Vs:

- **Volume:** a growing multi-terabyte history that must be stored cheaply and scanned
  efficiently.
- **Velocity:** streaming events that need windowing, back-pressure and tolerance of late or
  duplicate delivery.
- **Variety:** CSV, nested JSON and free text that must be conformed and joined.
- **Veracity:** missing keys, duplicates, malformed records, mixed timestamp formats and
  outliers that would corrupt models if left unchecked.
- **Value:** insight must be measurable against business targets.

All of this must protect personal data under GDPR and stay inside the AUD 1.5 M annual budget.

## How the prototype makes each challenge testable

| V | Concrete challenge in the prototype | Where addressed |
|---|---|---|
| Volume | 2 years of history (~323 k order lines, 441 k click events, 34 k reviews, 21 k posts; `medium` scale is 4x) stored as compressed, partitioned Parquet | `bronze.py`, `silver.py` |
| Velocity | 40 k live events replayed through an 8-partition log; 1-min windows; 2-min watermark; <= 5-min insight | `event_log.py`, `streaming.py` |
| Variety | CSV + nested JSON + JSON-lines clickstream + free text joined in Gold | `silver.py`, `text_analytics.py`, `gold.py` |
| Veracity | Injected defects: 1 % duplicates, missing IDs, negative quantities, price outliers, future timestamps, truncated JSON, epoch-ms vs ISO timestamps, inconsistent category labels, empty reviews | `silver.py` quarantine + report |
| Value | Forecast accuracy, recommendation lift, churn AUC and throughput measured against the Assessment 2 targets | `ml/*`, `benchmark.py`, `report.py` |
