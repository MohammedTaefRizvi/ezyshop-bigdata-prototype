"""Replays the prototype's local event log into a real Kafka cluster, so the
same simulated traffic can drive spark_streaming_job.py.

    pip install kafka-python
    python scale_out/kafka_bridge.py --bootstrap localhost:9092

Reference code for the scale-out path (not executed in the development
environment). Keys are preserved, so Kafka's partitioner keeps per-customer
ordering exactly as the local log does.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ezyshop.config import Config  # noqa: E402
from ezyshop.event_log import EventLog  # noqa: E402


def main() -> None:
    from kafka import KafkaProducer  # optional dependency

    ap = argparse.ArgumentParser()
    ap.add_argument("--bootstrap", default="localhost:9092")
    a = ap.parse_args()
    cfg = Config.from_env()
    log = EventLog(cfg.eventlog_dir)
    prod = KafkaProducer(bootstrap_servers=a.bootstrap, acks="all", linger_ms=20, compression_type="gzip",
                         key_serializer=lambda k: (k or "").encode(), value_serializer=lambda v: json.dumps(v).encode())
    n = 0
    for topic in log.topics():
        for rec in log.read_all(topic):
            prod.send(topic, key=rec["key"], value=rec["value"])
            n += 1
    prod.flush()
    print(f"published {n} records")


if __name__ == "__main__":
    main()
