"""A file-backed, partitioned, append-only event log that reproduces the
semantics of Apache Kafka that matter to the architecture (Kreps et al., 2011):

* topics split into N partitions; records routed by ``hash(key) % N`` so all
  events for one customer stay in order on one partition;
* immutable, offset-addressed records that can be replayed;
* independent consumer groups, each with its own committed offsets, so the
  streaming (speed) layer and the Bronze archiver (batch layer) read the same
  stream at their own pace - the "shock absorber" from Assessment 2.

Layout::

    eventlog/<topic>/meta.json
    eventlog/<topic>/partition-0003.jsonl
    eventlog/_offsets/<group>/<topic>.json   {"3": {"offset": 120, "pos": 48211}}

In production this class is replaced by a Kafka/MSK/Kinesis client; the
producer and consumer call sites keep the same shape.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path


def stable_partition(key: str | None, num_partitions: int) -> int:
    """Deterministic partitioner (Python's hash() is salted per process)."""
    if key is None:
        key = ""
    return int.from_bytes(hashlib.md5(key.encode("utf-8")).digest()[:4], "big") % num_partitions


class EventLog:
    def __init__(self, root: Path, num_partitions: int = 8):
        self.root = Path(root)
        self.default_partitions = num_partitions
        self.root.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------- topics
    def topic_dir(self, topic: str) -> Path:
        return self.root / topic

    def create_topic(self, topic: str, partitions: int | None = None) -> int:
        d = self.topic_dir(topic)
        meta = d / "meta.json"
        if meta.exists():
            return json.loads(meta.read_text())["partitions"]
        n = partitions or self.default_partitions
        d.mkdir(parents=True, exist_ok=True)
        for p in range(n):
            (d / f"partition-{p:04d}.jsonl").touch()
        meta.write_text(json.dumps({"topic": topic, "partitions": n, "created": time.time()}))
        return n

    def partitions(self, topic: str) -> int:
        meta = self.topic_dir(topic) / "meta.json"
        if not meta.exists():
            return 0
        return json.loads(meta.read_text())["partitions"]

    def topics(self) -> list[str]:
        return sorted(p.name for p in self.root.iterdir() if (p / "meta.json").exists())

    def _pfile(self, topic: str, partition: int) -> Path:
        return self.topic_dir(topic) / f"partition-{partition:04d}.jsonl"

    # ------------------------------------------------------------ producer
    def produce_batch(self, topic: str, records: list[tuple[str | None, dict]]) -> int:
        """Append (key, value) records. Batching amortises file I/O exactly as
        Kafka's producer ``linger.ms``/``batch.size`` do."""
        n = self.create_topic(topic)
        buckets: dict[int, list[str]] = {}
        now = time.time()
        for key, value in records:
            p = stable_partition(key, n)
            buckets.setdefault(p, []).append(json.dumps({"key": key, "produced_at": now, "value": value},
                                                        separators=(",", ":")))
        for p, lines in buckets.items():
            with open(self._pfile(topic, p), "a", encoding="utf-8") as f:
                f.write("\n".join(lines) + "\n")
        return len(records)

    def produce(self, topic: str, key: str | None, value: dict) -> None:
        self.produce_batch(topic, [(key, value)])

    def end_offsets(self, topic: str) -> dict[int, int]:
        out = {}
        for p in range(self.partitions(topic)):
            with open(self._pfile(topic, p), "rb") as f:
                out[p] = sum(1 for _ in f)
        return out

    def read_all(self, topic: str) -> list[dict]:
        """Full replay of a topic (used by the batch/reconciliation path)."""
        out = []
        for p in range(self.partitions(topic)):
            with open(self._pfile(topic, p), "r", encoding="utf-8") as f:
                for line in f:
                    if line.endswith("\n"):
                        rec = json.loads(line)
                        rec["partition"] = p
                        out.append(rec)
        return out


class Consumer:
    """Consumer-group member that reads assigned partitions from committed
    offsets. ``commit()`` persists offsets atomically (write-then-rename)."""

    def __init__(self, log: EventLog, group: str, topics: list[str], partitions: list[int] | None = None):
        self.log = log
        self.group = group
        self.topics = topics
        self.assigned = partitions
        self.state: dict[str, dict[str, dict[str, int]]] = {}
        for t in topics:
            self.log.create_topic(t)
            path = self._offset_path(t)
            self.state[t] = json.loads(path.read_text()) if path.exists() else {}

    def _offset_path(self, topic: str) -> Path:
        d = self.log.root / "_offsets" / self.group
        d.mkdir(parents=True, exist_ok=True)
        return d / f"{topic}.json"

    def poll(self, max_records_per_partition: int = 5000, max_bytes: int | None = None) -> dict[str, list[dict]]:
        """Read the next micro-batch.

        ``max_bytes`` enables *backlog-proportional* rate limiting (what Spark's
        Kafka source does with ``maxOffsetsPerTrigger``): the budget is split
        across every topic-partition in proportion to its unread backlog, so all
        partitions advance through event time at the same pace. A fixed
        per-partition record count lets lighter partitions race ahead, which
        pushes the watermark forward and makes the slower partitions' events
        look 'late' (found during testing - see docs/03_test_log.md)."""
        out: dict[str, list[dict]] = {}
        budgets: dict[tuple[str, int], float] = {}
        if max_bytes:
            backlog = {}
            for t in self.topics:
                for p in (self.assigned if self.assigned is not None else range(self.log.partitions(t))):
                    pos = self.state[t].get(str(p), {"pos": 0})["pos"]
                    backlog[(t, p)] = max(0, os.path.getsize(self.log._pfile(t, p)) - pos)
            total = sum(backlog.values())
            if total == 0:
                return {t: [] for t in self.topics}
            scale = min(1.0, max_bytes / total)
            budgets = {k: v * scale for k, v in backlog.items()}
        for t in self.topics:
            recs: list[dict] = []
            parts = self.assigned if self.assigned is not None else range(self.log.partitions(t))
            for p in parts:
                st = self.state[t].setdefault(str(p), {"offset": 0, "pos": 0})
                byte_budget = budgets.get((t, p)) if max_bytes else None
                with open(self.log._pfile(t, p), "rb") as f:
                    f.seek(st["pos"])
                    n, used = 0, 0
                    while (n < max_records_per_partition) if byte_budget is None else (used < byte_budget):
                        line = f.readline()
                        if not line or not line.endswith(b"\n"):
                            break  # end of log or a record still being written
                        rec = json.loads(line)
                        rec["partition"], rec["offset"] = p, st["offset"]
                        recs.append(rec)
                        st["offset"] += 1
                        st["pos"] += len(line)
                        used += len(line)
                        n += 1
            out[t] = recs
        return out

    def _head_time(self, t: str, p: int, time_field: str) -> str | None:
        st = self.state[t].get(str(p), {"pos": 0})
        with open(self.log._pfile(t, p), "rb") as f:
            f.seek(st["pos"])
            line = f.readline()
        if not line or not line.endswith(b"\n"):
            return None
        return json.loads(line)["value"].get(time_field)

    def poll_aligned(self, step_seconds: float = 60.0, time_field: str = "event_time",
                     max_records_per_partition: int = 50_000, not_before=None) -> dict[str, list[dict]]:
        """Event-time-aligned micro-batch (cf. Apache Flink *watermark alignment*).

        Every partition is read only up to a common frontier = (earliest head
        event time across all partitions) + ``step_seconds``. Partitions that
        are ahead in event time therefore wait for slower ones, so no
        partition can push the watermark past events another partition has not
        delivered yet. Discovered as necessary during testing: byte- or
        count-based rate limits let unevenly filled partitions drift apart by
        minutes during catch-up/replay (see docs/03_test_log.md)."""
        from datetime import datetime, timedelta

        heads = {}
        for t in self.topics:
            for p in (self.assigned if self.assigned is not None else range(self.log.partitions(t))):
                h = self._head_time(t, p, time_field)
                if h is not None:
                    heads[(t, p)] = datetime.fromisoformat(h)
        if not heads:
            return {t: [] for t in self.topics}
        # a head older than the caller's watermark is already late; it must not
        # anchor the frontier (that would produce a run of tiny batches)
        floor = min(heads.values())
        if not_before is not None:
            floor = max(floor, not_before)
        frontier = floor + timedelta(seconds=step_seconds)
        out: dict[str, list[dict]] = {t: [] for t in self.topics}
        for (t, p) in heads:
            st = self.state[t].setdefault(str(p), {"offset": 0, "pos": 0})
            with open(self.log._pfile(t, p), "rb") as f:
                f.seek(st["pos"])
                n = 0
                while n < max_records_per_partition:
                    line = f.readline()
                    if not line or not line.endswith(b"\n"):
                        break
                    rec = json.loads(line)
                    ts = rec["value"].get(time_field)
                    if ts is not None and datetime.fromisoformat(ts) > frontier:
                        break  # this partition is ahead: let the others catch up
                    rec["partition"], rec["offset"] = p, st["offset"]
                    out[t].append(rec)
                    st["offset"] += 1
                    st["pos"] += len(line)
                    n += 1
        return out

    def commit(self) -> None:
        for t in self.topics:
            path = self._offset_path(t)
            tmp = path.with_suffix(".tmp")
            tmp.write_text(json.dumps(self.state[t]))
            os.replace(tmp, path)

    def lag(self) -> dict[str, int]:
        out = {}
        for t in self.topics:
            ends = self.log.end_offsets(t)
            out[t] = sum(ends[p] - self.state[t].get(str(p), {"offset": 0})["offset"] for p in ends)
        return out
