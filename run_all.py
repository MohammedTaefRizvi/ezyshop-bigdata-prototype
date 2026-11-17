"""One-command end-to-end run of the EzyShop Big Data prototype.

    python run_all.py                 # small scale (default, ~2-4 min)
    python run_all.py --scale tiny    # quick smoke run (~30 s)
    python run_all.py --scale medium  # 4x data volume
    python run_all.py --fresh         # regenerate source data first

Steps: generate sources -> Bronze -> Silver -> Gold -> ML (forecast,
inventory, recommendations, churn) -> live stream (produce, process,
archive, reconcile) -> benchmark -> GDPR erasure demo -> report.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from ezyshop import bronze, gold, silver  # noqa: E402
from ezyshop import benchmark, producer, report, streaming  # noqa: E402
from ezyshop.config import Config  # noqa: E402
from ezyshop.data_generator import SourceGenerator  # noqa: E402
from ezyshop.ml import churn, forecast, recommender  # noqa: E402
from ezyshop.privacy import erase_customer, pseudonymise  # noqa: E402
from ezyshop.serving_store import ServingStore  # noqa: E402


def reset_stream(cfg: Config) -> None:
    for p in [cfg.eventlog_dir, cfg.gold_dir / "stream_metrics"] + list(cfg.bronze_dir.glob("stream_*")):
        if p.exists():
            shutil.rmtree(p)
    store = ServingStore(cfg.serving_db)
    for t in ["live_metrics", "alerts", "live_stock"]:
        store.clear(t)
    store.close()


def gdpr_demo(cfg: Config) -> dict:
    """Erase one customer and verify they disappear from the serving layer."""
    store = ServingStore(cfg.serving_db)
    row = store.query("SELECT customer_key FROM recommendations LIMIT 1")
    store.close()
    if not row:
        return {"verified": "skipped"}
    # find the raw id behind that pseudonym (only possible with the salt, i.e. inside the trust boundary)
    import pandas as pd
    ids = pd.read_csv(cfg.raw_dir / "customers.csv", usecols=["customer_id"])["customer_id"].unique()
    target = next(c for c in ids if pseudonymise(c, cfg.pii_salt) == row[0]["customer_key"])
    key = erase_customer(cfg, target)
    store = ServingStore(cfg.serving_db)
    gone = not store.query("SELECT 1 FROM recommendations WHERE customer_key=?", (key,))
    store.close()
    return {"customer_key": key, "removed_from_serving": gone,
            "verified": "recommendations + profile deleted; key on suppression list for next Silver rebuild" if gone else "FAILED"}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scale", choices=["tiny", "small", "medium"], default=None)
    ap.add_argument("--fresh", action="store_true", help="regenerate synthetic source data")
    ap.add_argument("--stream-events", type=int, default=40_000)
    ap.add_argument("--bench-events", type=int, default=1_000_000)
    ap.add_argument("--skip-benchmark", action="store_true")
    a = ap.parse_args()

    cfg = Config.from_env(scale=a.scale)
    cfg.ensure_dirs()
    marker = cfg.raw_dir / "_scale.txt"
    log: dict = {"scale": cfg.scale, "timings": {}}
    t_all = time.perf_counter()

    def step(name, fn, *args, **kw):
        print(f"\n=== {name} ===", flush=True)
        t = time.perf_counter()
        res = fn(*args, **kw)
        log["timings"][name] = round(time.perf_counter() - t, 1)
        log[name] = res
        brief = {k: v for k, v in (res or {}).items() if not isinstance(v, (dict, list))} if isinstance(res, dict) else res
        print(json.dumps(brief, indent=1, default=str)[:900], flush=True)
        return res

    if a.fresh or not marker.exists() or marker.read_text().strip() != cfg.scale:
        if cfg.base_dir.exists():
            shutil.rmtree(cfg.base_dir)
        cfg.ensure_dirs()
        step("generate_sources", SourceGenerator(cfg).run)
        marker.write_text(cfg.scale)
    step("bronze", bronze.run, cfg)
    step("silver", silver.run, cfg)
    step("gold", gold.run, cfg)
    step("ml_demand_forecast", forecast.run, cfg)
    step("inventory_planning", forecast.build_inventory_status, cfg)
    step("ml_recommendations", recommender.run, cfg)
    step("ml_churn", churn.run, cfg)
    reset_stream(cfg)
    step("stream_produce", producer.run, cfg, events=a.stream_events)
    sp = streaming.StreamProcessor(cfg)
    step("streaming", sp.run)
    sp.store.close()
    step("bronze_stream_archive", bronze.archive_stream, cfg)
    step("reconciliation", streaming.reconcile, cfg)
    if not a.skip_benchmark:
        res = step("benchmark", benchmark.run, cfg, a.bench_events)
        (cfg.output_dir / "benchmark_results.json").write_text(json.dumps(res, indent=2))
    step("gdpr", gdpr_demo, cfg)
    log["total_seconds"] = round(time.perf_counter() - t_all, 1)
    step("report", report.run, cfg, log)
    (cfg.output_dir / "pipeline_run_log.json").write_text(json.dumps(log, indent=2, default=str))
    print("\n" + (cfg.output_dir / "summary_report.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
