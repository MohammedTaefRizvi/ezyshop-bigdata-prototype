"""Live event producer - simulates EzyShop's website, app, review form and
warehouse systems publishing events to the event log (Kafka analogue).

Scenarios baked into the stream so the real-time layer has something to find:
* a product goes viral on social media (view spike)      -> TRENDING_PRODUCT
* a faulty batch triggers a burst of 1-star reviews      -> NEGATIVE_REVIEW_BURST
* a checkout outage makes carts get abandoned            -> CART_ABANDONMENT_SPIKE
* stock depletes as orders flow                          -> LOW_STOCK
plus at-least-once duplicates and late-arriving events (mobile app offline)
to exercise de-duplication, watermarks and Lambda reconciliation.

Usage::

    python -m ezyshop.producer --events 40000              # fast replay (30 simulated minutes)
    python -m ezyshop.producer --realtime --rate 200       # live demo, event time = now
"""
from __future__ import annotations

import argparse
import json
import time
import uuid
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from .config import Config
from .data_generator import NEG, POS
from .event_log import EventLog


class LiveSimulator:
    def __init__(self, cfg: Config, seed: int = 99):
        self.rng = np.random.default_rng(seed)
        prods = pd.read_csv(cfg.raw_dir / "products.csv", dtype=str)
        prods["price"] = pd.to_numeric(prods["list_price"].str.replace("$", "", regex=False), errors="coerce")
        prods = prods.dropna(subset=["price"])
        self.products = prods["product_id"].values
        self.prices = dict(zip(prods["product_id"], prods["price"]))
        self.customers = pd.read_csv(cfg.raw_dir / "customers.csv", usecols=["customer_id"])["customer_id"].unique()
        pop = self.rng.pareto(1.5, len(self.products)) + 0.1
        self.pop = pop / pop.sum()
        # the scenario actors
        cheap = prods.sort_values("price")["product_id"].values
        self.viral = str(cheap[len(cheap) // 3])
        self.faulty = str(cheap[len(cheap) // 2])

    def _review_text(self, rating: int) -> str:
        rng = self.rng
        if rating >= 4:
            return ". ".join(rng.choice(POS[a]) for a in ["general", "quality"])
        return ". ".join(rng.choice(NEG[a]) for a in ["quality", "general"])

    def generate(self, n_events: int, start: datetime, span_s: float, realtime: bool = False):
        """Yield (topic, key, event) tuples in event-time order."""
        rng = self.rng
        produced = 0
        while produced < n_events:
            frac = produced / max(n_events, 1)
            t = datetime.now() if realtime else start + timedelta(seconds=frac * span_s)
            minute = (t - start).total_seconds() / 60 if not realtime else (frac * span_s) / 60
            viral_on = 8 <= minute < 18
            outage_on = 20 <= minute < 24
            burst_on = 13 <= minute < 16
            cust = str(rng.choice(self.customers)) if rng.random() < 0.7 else None
            sess = f"LS{uuid.uuid4().hex[:12]}"
            key = cust or sess
            evs = []

            def ev(kind, **kw):
                nonlocal t
                t = t + timedelta(seconds=float(rng.uniform(2, 12)))
                return {"event_id": uuid.uuid4().hex, "event_type": kind, "event_time": t.isoformat(timespec="milliseconds"),
                        "session_id": sess, "customer_id": cust, **kw}

            evs.append(("clickstream", key, ev("page_view", device=str(rng.choice(["mobile", "desktop", "tablet"])))))
            views = list(rng.choice(self.products, size=int(rng.integers(1, 5)), p=self.pop))
            if viral_on and rng.random() < 0.35:
                views.insert(0, self.viral)
            for pid in views:
                evs.append(("clickstream", key, ev("product_view", product_id=str(pid))))
            if rng.random() < 0.45:
                pid = str(views[0])
                evs.append(("clickstream", key, ev("add_to_cart", product_id=pid)))
                buy_p = 0.15 if outage_on else 0.62
                if rng.random() < buy_p:
                    lines = [{"product_id": pid, "quantity": 1, "unit_price": round(self.prices.get(pid, 50.0), 2)}]
                    if rng.random() < 0.3:
                        extra = str(rng.choice(self.products, p=self.pop))
                        lines.append({"product_id": extra, "quantity": 1, "unit_price": round(self.prices.get(extra, 50.0), 2)})
                    evs.append(("clickstream", key, ev("checkout")))
                    oid = f"LO{uuid.uuid4().hex[:10]}"
                    evs.append(("orders", key, ev("order", order_id=oid, lines=lines,
                                                  total=round(sum(l["unit_price"] * l["quantity"] for l in lines), 2))))
            if burst_on and rng.random() < 0.06:
                evs.append(("reviews", self.faulty, ev("review", product_id=self.faulty, rating=1,
                                                        text=self._review_text(1))))
            elif rng.random() < 0.03:
                pid = str(rng.choice(self.products, p=self.pop))
                r = int(rng.choice([1, 2, 3, 4, 5], p=[0.07, 0.08, 0.15, 0.3, 0.4]))
                evs.append(("reviews", pid, ev("review", product_id=pid, rating=r, text=self._review_text(r))))
            if rng.random() < 0.004:
                pid = str(rng.choice(self.products))
                evs.append(("inventory", pid, ev("restock", product_id=pid, quantity=int(rng.integers(20, 100)))))
            for topic, k, e in evs:
                if realtime:
                    # live devices send an event when it happens: stamp it now,
                    # never with a future time (T-22)
                    e["event_time"] = datetime.now().isoformat(timespec="milliseconds")
                r = rng.random()
                if r < 0.004:                       # offline mobile app: very late (beyond watermark)
                    e["event_time"] = (datetime.fromisoformat(e["event_time"]) - timedelta(minutes=8)).isoformat(timespec="milliseconds")
                elif r < 0.01:                      # slightly late (within allowed lateness)
                    e["event_time"] = (datetime.fromisoformat(e["event_time"]) - timedelta(seconds=50)).isoformat(timespec="milliseconds")
                yield topic, k, e
                produced += 1
                if rng.random() < 0.01:             # at-least-once delivery duplicate
                    yield topic, k, dict(e)
                    produced += 1


def run(cfg: Config, events: int = 40_000, realtime: bool = False, rate: float = 0.0,
        span_minutes: float = 30.0, start: datetime | None = None) -> dict:
    log = EventLog(cfg.eventlog_dir, cfg.num_partitions)
    sim = LiveSimulator(cfg)
    # replayed traffic ends ~2 minutes in the past (sessions run up to ~1 min past
    # their start), so a live demo started afterwards is never "older" than it
    start = start or datetime.now().replace(microsecond=0) - timedelta(minutes=span_minutes + 2)
    batch: dict[str, list] = {}
    t0 = time.perf_counter()
    n = 0
    flush_every = max(1, int(rate)) if rate else 5000
    for topic, key, ev in sim.generate(events, start, span_minutes * 60, realtime=realtime):
        batch.setdefault(topic, []).append((key, ev))
        n += 1
        if n % flush_every == 0:
            for tp, recs in batch.items():
                log.produce_batch(tp, recs)
            batch = {}
            if rate:
                target = n / rate
                elapsed = time.perf_counter() - t0
                if target > elapsed:
                    time.sleep(target - elapsed)
    for tp, recs in batch.items():
        log.produce_batch(tp, recs)
    secs = time.perf_counter() - t0
    return {"events_produced": n, "seconds": round(secs, 2), "events_per_sec": round(n / max(secs, 1e-9)),
            "viral_product": sim.viral, "faulty_product": sim.faulty}


def main() -> None:
    ap = argparse.ArgumentParser(description="Publish simulated live events to the event log")
    ap.add_argument("--events", type=int, default=40_000)
    ap.add_argument("--realtime", action="store_true", help="event time = wall clock (use with --rate)")
    ap.add_argument("--rate", type=float, default=0.0, help="events per second (0 = as fast as possible)")
    ap.add_argument("--span-minutes", type=float, default=30.0, help="simulated duration in replay mode")
    a = ap.parse_args()
    print(json.dumps(run(Config.from_env(), a.events, a.realtime, a.rate, a.span_minutes), indent=2))


if __name__ == "__main__":
    main()
