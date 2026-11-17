"""End-to-end integration tests on a 'tiny' pipeline run."""
import json

import duckdb
import pandas as pd

from ezyshop import benchmark, silver, streaming
from ezyshop.privacy import erase_customer
from ezyshop.serving_store import ServingStore


def test_silver_enforces_quality_rules(pipeline_cfg):
    rep = json.loads((pipeline_cfg.output_dir / "data_quality_report.json").read_text())
    o = rep["tables"]["orders"]
    assert o["deduplicated"] > 0
    assert o["rejected"]["missing_customer_id"] > 0 and o["rejected"]["non_positive_quantity"] > 0
    assert rep["tables"]["clickstream"]["rejected"]["malformed_json"] > 0
    s = pipeline_cfg.silver_dir.as_posix()
    bad = duckdb.sql(f"SELECT COUNT(*) FROM read_parquet('{s}/orders/**/*.parquet') "
                     f"WHERE quantity <= 0 OR customer_key IS NULL").fetchall()[0][0]
    assert bad == 0
    dups = duckdb.sql(f"SELECT COUNT(*) - COUNT(DISTINCT (order_id, line_no)) "
                      f"FROM read_parquet('{s}/orders/**/*.parquet')").fetchall()[0][0]
    assert dups == 0
    assert (pipeline_cfg.silver_dir / "_quarantine" / "orders.parquet").exists()


def test_no_direct_identifiers_beyond_bronze(pipeline_cfg):
    cust = pd.read_parquet(pipeline_cfg.silver_dir / "customers" / "data.parquet")
    assert not {"email", "first_name", "last_name", "phone", "customer_id"} & set(cust.columns)
    for t in pipeline_cfg.gold_dir.glob("*.parquet"):
        assert "customer_id" not in pd.read_parquet(t).columns, t.name


def test_gold_integrates_structured_and_unstructured(pipeline_cfg):
    pp = pd.read_parquet(pipeline_cfg.gold_dir / "product_performance.parquet")
    assert {"revenue", "avg_sentiment", "top_complaint", "views_120d"} <= set(pp.columns)
    assert pp["avg_sentiment"].notna().mean() > 0.8
    c360 = pd.read_parquet(pipeline_cfg.gold_dir / "customer_360.parquet")
    assert {"monetary", "avg_review_sentiment", "sessions_30d"} <= set(c360.columns)


def test_models_beat_baselines(pipeline_cfg):
    m = pipeline_cfg.models_dir
    rec = json.loads((m / "recommender_metrics.json").read_text())
    assert rec["als"]["precision@10"] > rec["popularity_baseline"]["precision@10"]
    ch = json.loads((m / "churn_metrics.json").read_text())
    assert ch["roc_auc"] > 0.7
    fc = json.loads((m / "forecast_metrics.json").read_text())
    assert 0 < fc["model_accuracy"] <= 1
    inv = pd.read_parquet(pipeline_cfg.gold_dir / "inventory_status.parquet")
    assert (inv["reorder_point"] >= 0).all()


def test_streaming_and_lambda_reconciliation(pipeline_cfg):
    store = ServingStore(pipeline_cfg.serving_db)
    assert store.query("SELECT COUNT(*) n FROM live_metrics")[0]["n"] > 0
    summary = store.get_kv("stream_summary")
    store.close()
    assert summary["duplicates"] > 0 and summary["within_budget"]
    rc = streaming.reconcile(pipeline_cfg)
    # batch view = stream view + exactly the events the watermark dropped
    assert rc["events_only_in_batch"] == summary["late_dropped"]
    # every processed event must reach an emitted (finalised) window - no leaks
    assert rc["stream_events"] == summary["processed"]


def test_gdpr_erasure_propagates(pipeline_cfg):
    key = erase_customer(pipeline_cfg, "C000001")
    store = ServingStore(pipeline_cfg.serving_db)
    assert not store.query("SELECT 1 FROM recommendations WHERE customer_key=?", (key,))
    store.close()
    silver.run(pipeline_cfg)  # next batch rebuild
    s = pipeline_cfg.silver_dir.as_posix()
    n = duckdb.sql(f"SELECT COUNT(*) FROM read_parquet('{s}/orders/**/*.parquet') WHERE customer_key = '{key}'").fetchall()[0][0]
    assert n == 0


def test_parallel_benchmark_matches_single_process(pipeline_cfg):
    res = benchmark.run(pipeline_cfg, events=20_000, workers=[2])
    assert all(r.get("matches_single_process", True) for r in res["runs"])
