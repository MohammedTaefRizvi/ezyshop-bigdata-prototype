"""Throughput benchmark for the stream-processing core (Assessment 2 target:
10,000 transactions/second at peak).

Three implementations of the same windowed aggregation, all including JSON
deserialisation of the raw event-log records:

1. naive      - one Python dict update per event (row-at-a-time consumer)
2. vectorised - micro-batch -> columnar DataFrame -> group-by (one core)
3. parallel   - map-reduce: one worker per *partition* computes partial
                aggregates, which are merged by summation (data parallelism,
                the execution model of Spark/Kafka consumer groups)

The parallel result is checked to be identical to the single-process result,
which proves the aggregates are correctly mergeable.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timedelta
from pathlib import Path

import duckdb
import pandas as pd

from .config import Config
from .event_log import EventLog


def _load_lines(path: str) -> list[str]:
    with open(path, encoding="utf-8") as f:
        return f.readlines()


def naive(lines: list[str], window_s: int) -> dict:
    out: dict = defaultdict(lambda: defaultdict(float))
    for line in lines:
        v = json.loads(line)["value"]
        ts = datetime.fromisoformat(v["event_time"])
        w = ts.replace(second=(ts.second // window_s) * window_s if window_s < 60 else 0, microsecond=0)
        agg = out[w]
        agg["events"] += 1
        agg[v["event_type"]] += 1
        if v["event_type"] == "order":
            agg["revenue"] += v["total"]
    return out


def vectorised_frame(lines: list[str], window_s: int) -> pd.DataFrame:
    vals = [json.loads(l)["value"] for l in lines]
    df = pd.DataFrame.from_records(vals, columns=["event_type", "event_time", "total"])
    df["event_time"] = pd.to_datetime(df["event_time"], format="ISO8601")
    w = df["event_time"].dt.floor(f"{window_s}s")
    et = df["event_type"]
    return pd.DataFrame({"events": 1, "orders": (et == "order").astype(int),
                         "product_views": (et == "product_view").astype(int),
                         "revenue": pd.to_numeric(df["total"], errors="coerce").fillna(0).where(et == "order", 0)}
                        ).groupby(w).sum()


def duckdb_columnar(glob: str, window_s: int) -> pd.DataFrame:
    """Native, multi-threaded columnar JSON scan + aggregation (C++ engine)."""
    con = duckdb.connect()
    df = con.execute(f"""
        SELECT time_bucket(INTERVAL '{window_s} seconds', CAST(value.event_time AS TIMESTAMP)) AS event_time,
               COUNT(*) AS events,
               COUNT(*) FILTER (WHERE value.event_type = 'order') AS orders,
               COUNT(*) FILTER (WHERE value.event_type = 'product_view') AS product_views,
               COALESCE(SUM(CAST(value.total AS DOUBLE)) FILTER (WHERE value.event_type = 'order'), 0) AS revenue
        FROM read_json('{glob}', format = 'newline_delimited', union_by_name = true, sample_size = -1)
        GROUP BY 1""").df()
    con.close()
    df = df.set_index("event_time")
    df.index = df.index.astype("datetime64[ns]")
    return df[["events", "orders", "product_views", "revenue"]].astype({"events": "int64", "orders": "int64",
                                                                        "product_views": "int64"})


def partition_task(args: tuple[str, int]) -> pd.DataFrame:
    path, window_s = args
    return vectorised_frame(_load_lines(path), window_s)


def run(cfg: Config, events: int = 200_000, workers: list[int] | None = None) -> dict:
    from .producer import LiveSimulator

    workers = workers or sorted({1, 2, 4, min(8, os.cpu_count() or 4)})
    tmp = Path(tempfile.mkdtemp(prefix="ezy_bench_"))
    try:
        log = EventLog(tmp, cfg.num_partitions)
        sim = LiveSimulator(cfg, seed=5)
        start = datetime.now() - timedelta(hours=1)
        buf = []
        for topic, key, ev in sim.generate(events, start, 3600):
            buf.append((key, ev))
            if len(buf) >= 20_000:
                log.produce_batch("bench", buf)
                buf = []
        log.produce_batch("bench", buf)
        files = [str(tmp / "bench" / f"partition-{p:04d}.jsonl") for p in range(cfg.num_partitions)]
        all_lines = [l for f in files for l in _load_lines(f)]
        n = len(all_lines)
        w = cfg.window_seconds
        res: dict = {"events": n, "partitions": cfg.num_partitions, "cpu_count": os.cpu_count(), "runs": []}

        t = time.perf_counter()
        naive(all_lines, w)
        dt = time.perf_counter() - t
        res["runs"].append({"strategy": "naive row-at-a-time", "workers": 1, "seconds": round(dt, 3),
                            "events_per_sec": round(n / dt)})

        t = time.perf_counter()
        ref = vectorised_frame(all_lines, w)
        dt = time.perf_counter() - t
        res["runs"].append({"strategy": "vectorised micro-batch", "workers": 1, "seconds": round(dt, 3),
                            "events_per_sec": round(n / dt)})

        for k in workers:
            with ProcessPoolExecutor(max_workers=k) as ex:
                list(ex.map(partition_task, [(files[0], w)]))  # warm the pool (process start-up excluded)
                t = time.perf_counter()
                parts = list(ex.map(partition_task, [(f, w) for f in files]))
                merged = pd.concat(parts).groupby(level=0).sum()
                dt = time.perf_counter() - t
            ok = merged.sort_index().round(2).equals(ref.sort_index().round(2))
            res["runs"].append({"strategy": "parallel map-reduce", "workers": k, "seconds": round(dt, 3),
                                "events_per_sec": round(n / dt), "matches_single_process": bool(ok)})
        t = time.perf_counter()
        col = duckdb_columnar((tmp / "bench" / "*.jsonl").as_posix(), w)
        dt = time.perf_counter() - t
        ok = col.sort_index().round(2).values.tolist() == ref.sort_index().round(2).values.tolist()
        res["runs"].append({"strategy": "native columnar engine (DuckDB, all cores)", "workers": os.cpu_count(),
                            "seconds": round(dt, 3), "events_per_sec": round(n / dt), "matches_single_process": bool(ok)})
        best = max(r["events_per_sec"] for r in res["runs"])
        res["best_events_per_sec"] = best
        res["target_tps"] = cfg.target_tps
        res["headroom_vs_target"] = round(best / cfg.target_tps, 2)
        res["note"] = ("Single-machine numbers. Production scales horizontally by adding partitions and consumers; "
                       "each event at 10k TPS is ~1.7 KB (Assessment 2), i.e. ~17 MB/s ingest.")
        return res
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="Benchmark stream-processing throughput")
    ap.add_argument("--events", type=int, default=200_000)
    a = ap.parse_args()
    cfg = Config.from_env()
    out = run(cfg, a.events)
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    (cfg.output_dir / "benchmark_results.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
