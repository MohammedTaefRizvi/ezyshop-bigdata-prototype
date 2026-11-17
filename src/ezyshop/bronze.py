"""Bronze layer: land every source *as-is* in compressed columnar Parquet.

Principles (medallion architecture, Armbrust et al., 2021):
* schema-on-read - everything is stored as strings / raw JSON so a bad record
  can never break ingestion and can always be re-processed later;
* lineage columns ``_source_file``, ``_ingested_at``, ``_batch_id``;
* idempotent - each source file is keyed by its checksum, so re-running the
  ingestion does not create duplicates;
* the stream archiver reads the event log through its *own* consumer group,
  independent from the real-time job (Lambda batch path).
"""
from __future__ import annotations

import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from .config import Config
from .event_log import Consumer, EventLog

STREAM_TOPICS = ["clickstream", "orders", "inventory", "reviews"]


def _checksum(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:12]


def _write(df: pd.DataFrame, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pandas(df, preserve_index=False), out, compression="zstd")


def _lineage(df: pd.DataFrame, src: Path, batch_id: str) -> pd.DataFrame:
    df["_source_file"] = src.name
    df["_ingested_at"] = datetime.now(timezone.utc).isoformat()
    df["_batch_id"] = batch_id
    return df


def ingest_file(cfg: Config, src: Path, table: str, partition: str | None = None) -> dict:
    batch_id = _checksum(src)
    sub = f"{partition}/" if partition else ""
    out = cfg.bronze_dir / table / f"{sub}batch_{batch_id}.parquet"
    if out.exists():
        return {"table": table, "file": src.name, "rows": 0, "skipped": True}
    if src.suffix == ".csv":
        df = pd.read_csv(src, dtype=str, keep_default_na=False)
    elif src.suffix == ".jsonl":
        with open(src, encoding="utf-8") as f:
            df = pd.DataFrame({"raw": [ln.rstrip("\n") for ln in f if ln.strip()]})
    elif src.suffix == ".json":
        df = pd.DataFrame({"raw": [json.dumps(r) for r in json.loads(src.read_text(encoding="utf-8"))]})
    else:
        raise ValueError(src)
    # remove stale batches of the same partition (source file was regenerated)
    for old in (cfg.bronze_dir / table / sub).glob("batch_*.parquet") if (cfg.bronze_dir / table / sub).exists() else []:
        old.unlink()
    _write(_lineage(df, src, batch_id), out)
    return {"table": table, "file": src.name, "rows": len(df), "skipped": False}


def archive_stream(cfg: Config) -> dict:
    """Drain event-log topics into Bronze (consumer group 'bronze-archiver')."""
    log = EventLog(cfg.eventlog_dir, cfg.num_partitions)
    consumer = Consumer(log, "bronze-archiver", STREAM_TOPICS)
    counts = {}
    while True:
        batch = consumer.poll(50_000)
        n = sum(len(v) for v in batch.values())
        if n == 0:
            break
        stamp = int(time.time() * 1000)
        for topic, recs in batch.items():
            if not recs:
                continue
            df = pd.DataFrame({"raw": [json.dumps(r["value"]) for r in recs],
                               "key": [r["key"] for r in recs],
                               "partition": [r["partition"] for r in recs],
                               "offset": [r["offset"] for r in recs],
                               "produced_at": [r["produced_at"] for r in recs]})
            df["_ingested_at"] = datetime.now(timezone.utc).isoformat()
            _write(df, cfg.bronze_dir / f"stream_{topic}" / f"part-{stamp}.parquet")
            counts[topic] = counts.get(topic, 0) + len(recs)
        consumer.commit()   # commit only after the batch is durably written
    return counts


def run(cfg: Config) -> dict:
    cfg.ensure_dirs()
    t0 = time.perf_counter()
    raw = cfg.raw_dir
    results = []
    for name in ["customers", "products", "orders", "campaigns", "inventory_snapshot"]:
        results.append(ingest_file(cfg, raw / f"{name}.csv", name))
    results.append(ingest_file(cfg, raw / "suppliers.json", "suppliers"))
    results.append(ingest_file(cfg, raw / "reviews.jsonl", "reviews"))
    results.append(ingest_file(cfg, raw / "social_posts.jsonl", "social_posts"))
    for f in sorted((raw / "clickstream").glob("*.jsonl")):
        results.append(ingest_file(cfg, f, "clickstream", partition=f"event_date={f.stem}"))
    raw_bytes = sum(p.stat().st_size for p in raw.rglob("*") if p.is_file())
    bronze_bytes = sum(p.stat().st_size for p in cfg.bronze_dir.rglob("*.parquet"))
    summary: dict = {}
    for r in results:
        s = summary.setdefault(r["table"], {"files": 0, "rows_ingested": 0, "files_skipped": 0})
        s["files"] += 1
        s["rows_ingested"] += r["rows"]
        s["files_skipped"] += int(r["skipped"])
    return {"tables": summary, "raw_mb": round(raw_bytes / 1e6, 1), "bronze_mb": round(bronze_bytes / 1e6, 1),
            "compression_ratio": round(raw_bytes / max(bronze_bytes, 1), 1),
            "seconds": round(time.perf_counter() - t0, 2)}


if __name__ == "__main__":
    print(json.dumps(run(Config.from_env()), indent=2))
