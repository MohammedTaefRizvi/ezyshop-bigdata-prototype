"""Speed layer: micro-batch stream processing over the event log.

Implements the Structured-Streaming concepts the production design relies on
(Zaharia et al., 2016; Spark Structured Streaming):

* micro-batches pulled from all partitions by consumer group 'stream-analytics';
* event-time tumbling windows (60 s) with a watermark (120 s allowed lateness):
  late data within the watermark is merged, data older than the watermark is
  dropped *and counted* (the batch layer corrects it - see ``reconcile``);
* de-duplication on ``event_id`` with state bounded by the watermark;
* stateful live stock per product + rules that turn metrics into actions
  (alerts) - the "promotion opportunities within five minutes" requirement;
* checkpointing: state and offsets are persisted together after outputs are
  written, so a restart resumes without loss (at-least-once + idempotent
  upserts = effectively-once results).

Insight latency by design = window (60 s) + watermark (120 s) + poll (<=1 s)
~= 3 min, inside the 5-minute budget in Assessment 2 (Table A6).
"""
from __future__ import annotations

import argparse
import json
import pickle
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from . import text_analytics as ta
from .config import Config
from .event_log import Consumer, EventLog
from .privacy import pseudonymise_series
from .serving_store import ServingStore

TOPICS = ["clickstream", "orders", "reviews", "inventory"]
GROUP = "stream-analytics"


def records_to_frame(batch: dict[str, list[dict]]) -> pd.DataFrame:
    rows = []
    for topic, recs in batch.items():
        for r in recs:
            v = r["value"]
            v["_topic"] = topic
            v["_produced_at"] = r["produced_at"]
            rows.append(v)
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame.from_records(rows)
    for col in ["product_id", "session_id", "customer_id", "total", "lines", "text", "rating", "quantity"]:
        if col not in df:
            df[col] = None
    df["event_time"] = pd.to_datetime(df["event_time"], format="ISO8601")
    return df


def aggregate_events(df: pd.DataFrame, window_seconds: int = 60) -> pd.DataFrame:
    """Vectorised, *mergeable* per-window aggregates (sums/counts only), so
    partial results from parallel partitions can be combined (map-reduce)."""
    w = df["event_time"].dt.floor(f"{window_seconds}s")
    et = df["event_type"]
    agg = pd.DataFrame({
        "events": 1,
        "page_views": (et == "page_view").astype(int),
        "product_views": (et == "product_view").astype(int),
        "add_to_carts": (et == "add_to_cart").astype(int),
        "checkouts": (et == "checkout").astype(int),
        "orders": (et == "order").astype(int),
        "revenue": pd.to_numeric(df["total"], errors="coerce").fillna(0.0).where(et == "order", 0.0),
        "reviews": (et == "review").astype(int),
    }).groupby(w).sum()
    agg.index.name = "window_start"
    return agg


class StreamProcessor:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.log = EventLog(cfg.eventlog_dir, cfg.num_partitions)
        self.consumer = Consumer(self.log, GROUP, TOPICS)
        self.store = ServingStore(cfg.serving_db)
        self.state_path = cfg.eventlog_dir / "_state" / f"{GROUP}.pkl"
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        self.window = pd.Timedelta(seconds=cfg.window_seconds)
        self.lateness = pd.Timedelta(seconds=cfg.watermark_seconds)
        if self.state_path.exists():
            self.s = pickle.loads(self.state_path.read_bytes())
        else:
            self.s = {"windows": {}, "seen": {}, "topic_max": {}, "finalized": set(),
                      "stock": self._initial_stock(), "low_stock_alerted": set(), "backorders": Counter(),
                      "view_baseline": {}, "windows_seen": 0,
                      "stats": Counter(), "latencies": [], "batch_times": []}
        self.reorder = self._reorder_points()
        if not self.s.get("alerts_primed"):
            # products already at/below their reorder point are on the batch
            # reorder list (gold.inventory_status); real-time alerts are for
            # products that *cross* the threshold while the stream is running
            self.s["low_stock_alerted"] |= {p for p, rp in self.reorder.items() if self.s["stock"].get(p, 0) <= rp}
            self.s["alerts_primed"] = True

    # ------------------------------------------------------------ bootstrap
    def _initial_stock(self) -> dict[str, float]:
        p = self.cfg.gold_dir / "inventory_status.parquet"
        if not p.exists():
            return {}
        inv = pd.read_parquet(p, columns=["product_id", "on_hand"])
        return dict(zip(inv["product_id"], inv["on_hand"].astype(float)))

    def _reorder_points(self) -> dict[str, float]:
        p = self.cfg.gold_dir / "inventory_status.parquet"
        if not p.exists():
            return {}
        inv = pd.read_parquet(p, columns=["product_id", "reorder_point"])
        return dict(zip(inv["product_id"], inv["reorder_point"].astype(float)))

    @property
    def watermark(self):
        """Global watermark = MIN over input topics of (max event time - lateness).
        Using the max would let a small, fully-consumed topic (orders) push the
        watermark past a large topic still being read (clickstream) and drop
        its events as 'late' - the same reason Spark defaults to the 'min'
        multiple-watermark policy."""
        tm = self.s["topic_max"]
        if not tm:
            return None
        # watermarks must be monotonic: when a topic delivers its first events
        # later, the min over topics would move *backwards* and let events into
        # windows that were already finalised and emitted (found in test T-15)
        cand = min(tm.values()) - self.lateness
        prev = self.s.get("wm_high")
        if prev is None or cand > prev:
            self.s["wm_high"] = cand
        return self.s["wm_high"]

    # ----------------------------------------------------------- one batch
    def process_batch(self, df: pd.DataFrame) -> dict:
        st, stats = self.s, self.s["stats"]
        t_start = time.time()
        n_in = len(df)
        stats["received"] += n_in
        # 1) de-duplicate (within batch and against state)
        df = df.drop_duplicates("event_id")
        # O(batch) hash lookups; Series.isin(state) rebuilt an array of the whole
        # state every batch and was 60 % of runtime (profiling, test log T-07)
        seen = st["seen"]
        df = df[[e not in seen for e in df["event_id"].values]]
        stats["duplicates"] += n_in - len(df)
        # 2) watermark: drop events older than the watermark
        wm = self.watermark
        if wm is not None:
            late = df["event_time"] < wm
            stats["late_dropped"] += int(late.sum())
            df = df[~late]
        if df.empty:
            return {"processed": 0}
        st["seen"].update(dict(zip(df["event_id"], df["event_time"])))
        for topic, tmax in df.groupby("_topic")["event_time"].max().items():
            if topic == "inventory":
                continue  # sparse topic: must not hold back the global watermark
            st["topic_max"][topic] = max(st["topic_max"].get(topic, tmax), tmax)

        # 3) pseudonymise customer ids before anything is stored
        df = df.assign(customer_key=pseudonymise_series(df["customer_id"], self.cfg.pii_salt))
        df["window_start"] = df["event_time"].dt.floor(f"{self.cfg.window_seconds}s")

        # 4) windowed aggregates (mergeable sums) + distinct sets
        agg = aggregate_events(df, self.cfg.window_seconds)
        for ws, row in agg.iterrows():
            w = st["windows"].setdefault(ws, {"sums": Counter(), "sessions": set(), "cart_sessions": set(),
                                              "purchase_sessions": set(), "views": Counter(), "neg_reviews": Counter(),
                                              "sentiments": [], "units": 0, "latencies": []})
            w["sums"].update(row.to_dict())
        for ws, g in df.groupby("window_start"):
            w = st["windows"][ws]
            w["sessions"].update(g["session_id"].dropna())
            w["cart_sessions"].update(g.loc[g["event_type"] == "add_to_cart", "session_id"])
            w["purchase_sessions"].update(g.loc[g["event_type"] == "order", "session_id"])
            w["views"].update(g.loc[g["event_type"] == "product_view", "product_id"])
            w["latencies"].extend((t_start - g["_produced_at"]).clip(lower=0).round(3).tolist())
            for line_list in g.loc[g["event_type"] == "order", "lines"]:
                w["units"] += sum(l["quantity"] for l in line_list)
            rv = g[g["event_type"] == "review"]
            for pid, text in zip(rv["product_id"], rv["text"]):
                score = ta.sentiment(ta.clean_text(text))     # unstructured text scored in-stream
                w["sentiments"].append(score)
                if score <= -0.25:
                    w["neg_reviews"][pid] += 1

        # 5) stateful live inventory
        alerts = []
        touched = set()
        inv_events = df[df["event_type"].isin(["order", "restock"])].sort_values("event_time")
        for etype, ts, pid0, qty0, lines in zip(inv_events["event_type"], inv_events["event_time"],
                                                inv_events["product_id"], inv_events["quantity"], inv_events["lines"]):
            if etype == "restock":
                st["stock"][pid0] = st["stock"].get(pid0, 0.0) + qty0
                st["low_stock_alerted"].discard(pid0)
                touched.add(pid0)
                continue
            for l in lines:
                pid = l["product_id"]
                have = st["stock"].get(pid, 0.0)
                if have < l["quantity"]:
                    st["backorders"][pid] += l["quantity"] - max(have, 0)
                st["stock"][pid] = max(0.0, have - l["quantity"])
                touched.add(pid)
                rp = self.reorder.get(pid)
                if rp is not None and st["stock"][pid] <= rp and pid not in st["low_stock_alerted"]:
                    st["low_stock_alerted"].add(pid)
                    kind, sev = ("STOCK_OUT", "critical") if st["stock"][pid] <= 0 else ("LOW_STOCK", "high")
                    alerts.append(self._alert(kind, sev, str(ts.floor("min")), pid,
                                              f"Live stock {st['stock'][pid]:.0f} at/below reorder point {rp:.0f} - raise purchase order"))
        now = datetime.now(timezone.utc).isoformat()
        self.store.upsert_many("live_stock", [{"product_id": p, "stock": int(st["stock"][p]),
                                               "reorder_point": self.reorder.get(p), "updated_at": now} for p in touched])

        # 6) emit: provisional rows for open windows, final rows + alerts for closed ones
        rows, final_rows = [], []
        for ws in sorted(set(df["window_start"])):
            rows.append(self._window_row(ws))
        wm = self.watermark
        for ws in sorted(st["windows"]):
            if ws + self.window <= wm and ws not in st["finalized"]:
                r = self._window_row(ws)
                final_rows.append(r)
                alerts.extend(self._window_alerts(ws, r))
                st["finalized"].add(ws)
        self.store.upsert_many("live_metrics", rows + final_rows)
        self.store.upsert_many("alerts", alerts)
        self._append_gold(final_rows)

        # 7) bound state: forget dedup ids and window details behind the watermark
        horizon = wm - pd.Timedelta(minutes=10)
        st["seen"] = {k: v for k, v in st["seen"].items() if v >= horizon}
        for ws in [w for w in st["windows"] if w < horizon and w in st["finalized"]]:
            del st["windows"][ws]

        stats["processed"] += len(df)
        st["latencies"].extend(df["_produced_at"].map(lambda p: t_start - p).clip(lower=0).tolist()[-2000:])
        st["latencies"] = st["latencies"][-20000:]
        st["batch_times"].append((len(df), time.time() - t_start))
        return {"processed": len(df), "alerts": len(alerts), "windows_finalized": len(final_rows)}

    def _alert(self, kind, severity, ws, entity, msg):
        return {"alert_id": f"{kind}:{ws}:{entity}", "alert_type": kind, "severity": severity, "window_start": ws,
                "entity": entity, "message": msg, "created_at": datetime.now(timezone.utc).isoformat()}

    def _window_row(self, ws) -> dict:
        w = self.s["windows"][ws]
        s = w["sums"]
        carts = len(w["cart_sessions"])
        # a cart added near the end of a window is often paid for in the next one
        nxt = self.s["windows"].get(ws + self.window)
        paid = w["purchase_sessions"] | (nxt["purchase_sessions"] if nxt else set())
        buys = len(paid & w["cart_sessions"])
        lat = np.array(w["latencies"]) if w["latencies"] else np.array([0.0])
        return {"window_start": str(ws), "window_end": str(ws + self.window), "events": int(s["events"]),
                "sessions": len(w["sessions"]), "page_views": int(s["page_views"]), "product_views": int(s["product_views"]),
                "add_to_carts": int(s["add_to_carts"]), "checkouts": int(s["checkouts"]), "purchases": int(s["orders"]),
                "orders": int(s["orders"]), "units": int(w["units"]), "revenue": round(float(s["revenue"]), 2),
                "reviews": int(s["reviews"]),
                "avg_review_sentiment": round(float(np.mean(w["sentiments"])), 3) if w["sentiments"] else None,
                "negative_reviews": int(sum(w["neg_reviews"].values())),
                "cart_abandonment_rate": round(1 - buys / carts, 3) if carts else None,
                "conversion_rate": round(int(s["orders"]) / max(len(w["sessions"]), 1), 3),
                "p95_latency_s": round(float(np.percentile(lat, 95)), 3),
                "updated_at": datetime.now(timezone.utc).isoformat()}

    def _window_alerts(self, ws, row) -> list[dict]:
        w = self.s["windows"][ws]
        out = []
        # trending = a product's SHARE of all product views jumps vs its own
        # recent share (EWMA). Shares, not raw counts: a site-wide surge (Black
        # Friday) must not flag every popular product (found in live rehearsal, T-19)
        base = self.s["view_baseline"]
        total = sum(w["views"].values())
        if self.s["windows_seen"] >= 3 and total:
            for pid, v in w["views"].most_common(3):
                share, b = v / total, base.get(pid, 0.0)
                if v >= 10 and share >= 0.02 and share >= 3 * max(b, 0.005):
                    out.append(self._alert("TRENDING_PRODUCT", "opportunity", str(ws), pid,
                                           f"{share:.0%} of product views ({v} in 1 min) vs usual {b:.1%} "
                                           f"({share / max(b, 0.005):.0f}x) - push promotion, check stock"))
        if row["events"] >= 50 and total:   # only learn baselines from windows with real traffic
            for pid in set(base) | set(w["views"]):
                base[pid] = 0.8 * base.get(pid, 0.0) + 0.2 * w["views"].get(pid, 0) / total
            self.s["windows_seen"] += 1
        # a burst = one product takes a large SHARE of the minute's negative
        # reviews; a fixed count misfires as traffic grows (T-23)
        neg_total = sum(w["neg_reviews"].values())
        for pid, n in w["neg_reviews"].items():
            if n >= 3 and n >= 0.25 * neg_total:
                out.append(self._alert("NEGATIVE_REVIEW_BURST", "high", str(ws), pid,
                                       f"{n} negative reviews in 1 min - check product batch, pause ads"))
        carts = len(w["cart_sessions"])
        if carts >= 15 and (row["cart_abandonment_rate"] or 0) >= 0.7:
            out.append(self._alert("CART_ABANDONMENT_SPIKE", "high", str(ws), "checkout",
                                   f"Abandonment {row['cart_abandonment_rate']:.0%} across {carts} carts - check payment service"))
        return out

    def _append_gold(self, rows: list[dict]) -> None:
        if not rows:
            return
        d = self.cfg.gold_dir / "stream_metrics"
        d.mkdir(parents=True, exist_ok=True)
        pq.write_table(pa.Table.from_pylist(rows), d / f"part-{int(time.time() * 1e6)}.parquet")

    def checkpoint(self) -> None:
        tmp = self.state_path.with_suffix(".tmp")
        tmp.write_bytes(pickle.dumps(self.s))
        tmp.replace(self.state_path)
        self.consumer.commit()

    def flush(self) -> None:
        """End of a bounded stream: close all open windows."""
        if self.s["windows"]:
            # end of a bounded input: advance the watermark past *every* open
            # window (advancing per-topic maxima is not enough when the slowest
            # topic ends earlier - the last window was never emitted, test T-16)
            # The watermark moves to exactly the end of the last window - never
            # further, or live data arriving afterwards would be judged late (T-21)
            end = max(self.s["windows"]) + self.window
            self.s["wm_high"] = max(self.s.get("wm_high") or end, end)
            self.process_batch_empty_finalize()

    def process_batch_empty_finalize(self) -> None:
        st, wm = self.s, self.watermark
        final_rows, alerts = [], []
        for ws in sorted(st["windows"]):
            if ws + self.window <= wm and ws not in st["finalized"]:
                r = self._window_row(ws)
                final_rows.append(r)
                alerts.extend(self._window_alerts(ws, r))
                st["finalized"].add(ws)
        self.store.upsert_many("live_metrics", final_rows)
        self.store.upsert_many("alerts", alerts)
        self._append_gold(final_rows)

    def summary(self) -> dict:
        st = self.s
        lat = np.array(st["latencies"]) if st["latencies"] else np.array([0.0])
        n = sum(b[0] for b in st["batch_times"])
        secs = sum(b[1] for b in st["batch_times"])
        out = {**{k: int(v) for k, v in st["stats"].items()},
               "backordered_units": int(sum(st["backorders"].values())),
               "micro_batches": len(st["batch_times"]),
               "processing_events_per_sec": round(n / max(secs, 1e-9)),
               "produce_to_process_latency_p50_s": round(float(np.percentile(lat, 50)), 2),
               "produce_to_process_latency_p95_s": round(float(np.percentile(lat, 95)), 2),
               "design_insight_latency_s": self.cfg.window_seconds + self.cfg.watermark_seconds + 1,
               "latency_budget_s": self.cfg.latency_budget_seconds,
               "alerts_by_type": {r["alert_type"]: r["n"] for r in
                                  self.store.query("SELECT alert_type, COUNT(*) n FROM alerts GROUP BY 1")}}
        out["within_budget"] = out["design_insight_latency_s"] <= self.cfg.latency_budget_seconds
        return out

    def run(self, follow: bool = False, idle_exit_s: float | None = None) -> dict:
        last_data = time.time()
        while True:
            wm = self.watermark
            batch = self.consumer.poll_aligned(step_seconds=self.cfg.window_seconds,
                                               not_before=None if wm is None else wm.to_pydatetime())
            df = records_to_frame(batch)
            if not df.empty:
                info = self.process_batch(df)
                self.checkpoint()
                last_data = time.time()
                if follow:
                    print(f"[stream] {datetime.now():%H:%M:%S} processed={info['processed']} "
                          f"alerts={info.get('alerts', 0)} lag={sum(self.consumer.lag().values())}", flush=True)
            elif not follow:
                break
            else:
                idle = time.time() - last_data
                # source idleness (cf. Flink withIdleness): with no new data the
                # watermark cannot advance, so close open windows after a quiet
                # period instead of holding their results back forever (T-20)
                if idle > self.cfg.watermark_seconds and any(w not in self.s["finalized"] for w in self.s["windows"]):
                    self.flush()
                    self.checkpoint()
                    print(f"[stream] {datetime.now():%H:%M:%S} idle {idle:.0f}s - open windows closed", flush=True)
                if idle_exit_s and idle > idle_exit_s:
                    break
                time.sleep(1.0)
        if not follow:
            self.flush()
            self.checkpoint()
        result = self.summary()
        self.store.put_kv("stream_summary", result)
        return result


def reconcile(cfg: Config) -> dict:
    """Lambda batch path: recompute window totals from the *complete* event log
    (no watermark) and compare with what the speed layer emitted. Differences
    are exactly the late events the stream had to drop; the batch view is
    written to Gold as the corrected record of truth."""
    log = EventLog(cfg.eventlog_dir, cfg.num_partitions)
    recs = {t: log.read_all(t) for t in TOPICS}
    df = records_to_frame(recs)
    if df.empty:
        return {}
    n_raw = len(df)
    df = df.drop_duplicates("event_id")
    batch = aggregate_events(df, cfg.window_seconds)
    d = cfg.gold_dir / "stream_metrics"
    stream = pd.concat([pq.read_table(f).to_pandas() for f in d.glob("*.parquet")]) if d.exists() else pd.DataFrame()
    stream = stream.drop_duplicates("window_start", keep="last")
    stream["window_start"] = pd.to_datetime(stream["window_start"])
    stream = stream.set_index("window_start")
    st_cols = stream[["events", "orders", "revenue"]].add_suffix("_stream")
    j = batch.join(st_cols, how="left").fillna(0)
    j["events_diff"] = j["events"] - j["events_stream"]
    j["revenue_diff"] = (j["revenue"] - j["revenue_stream"]).round(2)
    out = j.reset_index()
    out["window_start"] = out["window_start"].astype(str)
    out.to_parquet(cfg.gold_dir / "stream_metrics_reconciled.parquet", index=False)
    res = {"raw_records": n_raw, "unique_events": len(df), "duplicates_removed": n_raw - len(df),
           "batch_events": int(batch["events"].sum()), "stream_events": int(stream["events"].sum()),
           "events_only_in_batch": int(j["events_diff"].sum()),
           "batch_revenue": round(float(batch["revenue"].sum()), 2),
           "stream_revenue": round(float(stream["revenue"].sum()), 2),
           "windows_with_corrections": int((j["events_diff"] != 0).sum()), "windows": len(j)}
    res["stream_revenue_completeness"] = round(res["stream_revenue"] / max(res["batch_revenue"], 1e-9), 4)
    return res


def main() -> None:
    ap = argparse.ArgumentParser(description="Run the real-time stream processor")
    ap.add_argument("--follow", action="store_true", help="keep running and poll for new events")
    ap.add_argument("--reconcile", action="store_true", help="run the batch reconciliation afterwards")
    ap.add_argument("--watermark-seconds", type=int, default=None,
                    help="allowed lateness (default 120). A smaller value (e.g. 30) makes alerts appear sooner in a "
                         "live demo, at the cost of dropping more out-of-order events")
    a = ap.parse_args()
    cfg = Config.from_env()
    if a.watermark_seconds is not None:
        cfg.watermark_seconds = a.watermark_seconds
    sp = StreamProcessor(cfg)
    try:
        print(json.dumps(sp.run(follow=a.follow), indent=2, default=str))
    except KeyboardInterrupt:
        sp.checkpoint()
        print("stopped - state checkpointed")
    if a.reconcile:
        print(json.dumps(reconcile(cfg), indent=2))


if __name__ == "__main__":
    main()
