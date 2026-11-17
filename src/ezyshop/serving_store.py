"""Low-latency serving store (stand-in for Cassandra / DynamoDB).

Holds the key-value lookups the website and dashboards need in milliseconds:
per-customer recommendations, live streaming metrics, alerts and live stock.
SQLite in WAL mode allows one writer (the streaming job) and many concurrent
readers (dashboard, API) - the same access pattern as the production store.
All writes are idempotent upserts keyed on a primary key, so replaying a
micro-batch never double counts.
"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS recommendations (customer_key TEXT PRIMARY KEY, items TEXT, model_version TEXT, updated_at TEXT);
CREATE TABLE IF NOT EXISTS customer_profile (customer_key TEXT PRIMARY KEY, profile TEXT, updated_at TEXT);
CREATE TABLE IF NOT EXISTS live_metrics (
    window_start TEXT PRIMARY KEY, window_end TEXT, events INTEGER, sessions INTEGER, page_views INTEGER,
    product_views INTEGER, add_to_carts INTEGER, checkouts INTEGER, purchases INTEGER, orders INTEGER,
    units INTEGER, revenue REAL, reviews INTEGER, avg_review_sentiment REAL, negative_reviews INTEGER,
    cart_abandonment_rate REAL, conversion_rate REAL, p95_latency_s REAL, updated_at TEXT);
CREATE TABLE IF NOT EXISTS alerts (alert_id TEXT PRIMARY KEY, alert_type TEXT, severity TEXT, window_start TEXT,
    entity TEXT, message TEXT, created_at TEXT);
CREATE TABLE IF NOT EXISTS live_stock (product_id TEXT PRIMARY KEY, stock INTEGER, reorder_point REAL, updated_at TEXT);
CREATE TABLE IF NOT EXISTS kv (key TEXT PRIMARY KEY, value TEXT);
CREATE TABLE IF NOT EXISTS gdpr_suppression (customer_key TEXT PRIMARY KEY, requested_at TEXT);
"""


class ServingStore:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.path, timeout=30, check_same_thread=False)
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA synchronous=NORMAL")
        self.conn.executescript(SCHEMA)
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()

    def upsert_many(self, table: str, rows: list[dict]) -> int:
        if not rows:
            return 0
        cols = list(rows[0].keys())
        sql = f"INSERT OR REPLACE INTO {table} ({','.join(cols)}) VALUES ({','.join('?' * len(cols))})"
        self.conn.executemany(sql, [tuple(r[c] for c in cols) for r in rows])
        self.conn.commit()
        return len(rows)

    def query(self, sql: str, params: tuple = ()) -> list[dict]:
        cur = self.conn.execute(sql, params)
        names = [d[0] for d in cur.description]
        return [dict(zip(names, r)) for r in cur.fetchall()]

    def put_kv(self, key: str, value) -> None:
        self.upsert_many("kv", [{"key": key, "value": json.dumps(value, default=str)}])

    def get_kv(self, key: str, default=None):
        rows = self.query("SELECT value FROM kv WHERE key=?", (key,))
        return json.loads(rows[0]["value"]) if rows else default

    def clear(self, table: str) -> None:
        self.conn.execute(f"DELETE FROM {table}")
        self.conn.commit()
