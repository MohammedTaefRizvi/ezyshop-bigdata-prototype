"""Central configuration for the EzyShop Big Data prototype.

Every module receives a ``Config`` so that the same code can run against the
default ``data/`` folder, a temporary folder (tests) or a larger scale profile.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Scale profiles. "small" is the default demo (~230k order lines, ~400k
# clickstream events). "medium" is ~4x larger and is used to show scaling.
SCALES: dict[str, dict] = {
    "tiny": dict(customers=400, products=60, days=420, orders_per_day=30, click_days=120),
    "small": dict(customers=5000, products=400, days=730, orders_per_day=220, click_days=120),
    "medium": dict(customers=20000, products=1200, days=730, orders_per_day=900, click_days=120),
}


@dataclass
class Config:
    base_dir: Path = PROJECT_ROOT / "data"
    output_dir: Path = PROJECT_ROOT / "outputs"
    scale: str = "small"
    seed: int = 42
    num_partitions: int = 8            # partitions per event-log topic (Kafka analogue)
    end_date: str = "2026-10-31"        # last day of historical data
    pii_salt: str = os.environ.get("EZYSHOP_PII_SALT", "dev-only-salt-change-me")
    window_seconds: int = 60            # tumbling window for streaming metrics
    watermark_seconds: int = 120        # allowed lateness for streaming events
    latency_budget_seconds: int = 300   # 5-minute insight requirement (Assessment 2)
    target_tps: int = 10_000            # peak throughput requirement (Assessment 2)

    # ------------------------------------------------------------------ paths
    @property
    def raw_dir(self) -> Path:
        return self.base_dir / "raw"

    @property
    def bronze_dir(self) -> Path:
        return self.base_dir / "lakehouse" / "bronze"

    @property
    def silver_dir(self) -> Path:
        return self.base_dir / "lakehouse" / "silver"

    @property
    def gold_dir(self) -> Path:
        return self.base_dir / "lakehouse" / "gold"

    @property
    def eventlog_dir(self) -> Path:
        return self.base_dir / "eventlog"

    @property
    def models_dir(self) -> Path:
        return self.base_dir / "models"

    @property
    def serving_db(self) -> Path:
        return self.base_dir / "serving" / "serving.db"

    @property
    def params(self) -> dict:
        return SCALES[self.scale]

    def ensure_dirs(self) -> None:
        for p in (self.raw_dir, self.bronze_dir, self.silver_dir, self.gold_dir,
                  self.eventlog_dir, self.models_dir, self.serving_db.parent, self.output_dir):
            p.mkdir(parents=True, exist_ok=True)

    @classmethod
    def from_env(cls, **overrides) -> "Config":
        kwargs: dict = {}
        if os.environ.get("EZYSHOP_DATA_DIR"):
            kwargs["base_dir"] = Path(os.environ["EZYSHOP_DATA_DIR"])
        if os.environ.get("EZYSHOP_OUTPUT_DIR"):
            kwargs["output_dir"] = Path(os.environ["EZYSHOP_OUTPUT_DIR"])
        if os.environ.get("EZYSHOP_SCALE"):
            kwargs["scale"] = os.environ["EZYSHOP_SCALE"]
        kwargs.update({k: v for k, v in overrides.items() if v is not None})
        return cls(**kwargs)
