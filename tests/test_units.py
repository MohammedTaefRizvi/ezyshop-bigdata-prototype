"""Unit tests for the core Big Data mechanics."""
from datetime import datetime, timedelta

import pandas as pd

from ezyshop import text_analytics as ta
from ezyshop.event_log import Consumer, EventLog, stable_partition
from ezyshop.privacy import pseudonymise
from ezyshop.silver import parse_ts
from ezyshop.streaming import StreamProcessor, aggregate_events, records_to_frame


# ------------------------------------------------------------------ event log
def test_partitioner_is_deterministic_and_spread():
    parts = [stable_partition(f"C{i:06d}", 8) for i in range(4000)]
    assert parts == [stable_partition(f"C{i:06d}", 8) for i in range(4000)]
    counts = pd.Series(parts).value_counts()
    assert len(counts) == 8 and counts.min() > 400  # roughly balanced


def test_per_key_ordering_and_offset_resume(tmp_path):
    log = EventLog(tmp_path, 4)
    log.produce_batch("t", [("cust-1", {"n": i, "event_time": f"2026-01-01T00:00:{i:02d}"}) for i in range(20)])
    c = Consumer(log, "g", ["t"])
    first = c.poll(max_records_per_partition=5)["t"]
    assert [r["value"]["n"] for r in first] == [0, 1, 2, 3, 4]  # same key -> same partition, in order
    c.commit()
    c2 = Consumer(log, "g", ["t"])                                 # restart resumes from committed offset
    assert [r["value"]["n"] for r in c2.poll(5)["t"]] == [5, 6, 7, 8, 9]
    other = Consumer(log, "other-group", ["t"])                    # independent consumer group
    assert other.poll(1)["t"][0]["value"]["n"] == 0


def test_aligned_poll_keeps_partitions_in_step(tmp_path):
    log = EventLog(tmp_path, 2)
    t0 = datetime(2026, 1, 1)
    # partition A is dense, partition B sparse: count-based polling would let B race ahead in time
    recs = [(k, {"event_time": (t0 + timedelta(seconds=s)).isoformat()})
            for k, step in (("a", 1), ("b", 30)) for s in range(0, 600, step)]
    log.produce_batch("t", recs)
    c = Consumer(log, "g", ["t"])
    batch = c.poll_aligned(step_seconds=60)["t"]
    times = [datetime.fromisoformat(r["value"]["event_time"]) for r in batch]
    assert max(times) <= t0 + timedelta(seconds=60)


# -------------------------------------------------------------- text analytics
def test_sentiment_polarity_negation_and_cleaning():
    assert ta.sentiment("Excellent quality, love it") > 0.5
    assert ta.sentiment("Terrible, broke after a week") < -0.5
    assert ta.sentiment("not good") < 0
    assert ta.clean_text("Great<br/>value &amp; fast") == "Great value & fast"
    assert "delivery" in ta.aspects("Delivery was late")
    assert ta.detect_subcategory("my new laptop is great") == "Laptops"


# --------------------------------------------------------------------- privacy
def test_pseudonymisation_is_deterministic_keyed_and_opaque():
    a = pseudonymise("C000001", "salt-1")
    assert a == pseudonymise("C000001", "salt-1")
    assert a != pseudonymise("C000001", "salt-2")
    assert "C000001" not in a and len(a) == 16
    assert pseudonymise(None, "s") is None


def test_parse_ts_handles_mixed_formats():
    s = pd.Series(["2026-01-02T03:04:05", "1767323045000", "garbage"])
    out = parse_ts(s)
    assert out.iloc[0] == pd.Timestamp("2026-01-02 03:04:05")
    assert out.iloc[1].year == 2026
    assert pd.isna(out.iloc[2])


# ------------------------------------------------------------------- streaming
def _ev(i, t, kind="page_view", **kw):
    return {"event_id": f"e{i}", "event_type": kind, "event_time": t.isoformat(), "session_id": f"s{i}",
            "customer_id": "C1", **kw}


def test_window_aggregates_are_mergeable():
    t0 = datetime(2026, 1, 1, 10, 0, 5)
    vals = [_ev(i, t0 + timedelta(seconds=i * 7), "order" if i % 5 == 0 else "page_view", total=10.0) for i in range(40)]
    df = records_to_frame({"x": [{"value": dict(v), "produced_at": 0} for v in vals]})
    whole = aggregate_events(df)
    parts = pd.concat([aggregate_events(df.iloc[:17]), aggregate_events(df.iloc[17:])]).groupby(level=0).sum()
    assert whole.equals(parts.loc[whole.index])


def test_stream_dedup_and_watermark(tmp_cfg):
    log = EventLog(tmp_cfg.eventlog_dir, 2)
    t0 = datetime(2026, 1, 1, 10, 0, 0)
    first = [("C1", _ev(i, t0 + timedelta(seconds=10 * i))) for i in range(60)]          # 10 minutes
    dup = [("C1", _ev(3, t0 + timedelta(seconds=30)))]                                       # duplicate id
    late = [("C1", _ev(999, t0 + timedelta(seconds=5)))]                                     # 10 min late
    log.produce_batch("clickstream", first)
    sp = StreamProcessor(tmp_cfg)
    sp.run()
    log.produce_batch("clickstream", dup + late)
    sp2 = StreamProcessor(tmp_cfg)   # restart from checkpoint
    sp2.run()
    stats = sp2.s["stats"]
    assert stats["processed"] == 60
    assert stats["duplicates"] == 1
    assert stats["late_dropped"] == 1
    sp.store.close(); sp2.store.close()
