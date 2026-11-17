import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ezyshop.config import Config  # noqa: E402


@pytest.fixture
def tmp_cfg(tmp_path) -> Config:
    cfg = Config(base_dir=tmp_path / "data", output_dir=tmp_path / "out", scale="tiny")
    cfg.ensure_dirs()
    return cfg


@pytest.fixture(scope="session")
def pipeline_cfg(tmp_path_factory) -> Config:
    """Runs the whole pipeline once at 'tiny' scale for the integration tests."""
    from ezyshop import bronze, gold, producer, silver, streaming
    from ezyshop.data_generator import SourceGenerator
    from ezyshop.ml import churn, forecast, recommender

    base = tmp_path_factory.mktemp("pipeline")
    cfg = Config(base_dir=base / "data", output_dir=base / "out", scale="tiny")
    cfg.ensure_dirs()
    SourceGenerator(cfg).run()
    bronze.run(cfg)
    silver.run(cfg)
    gold.run(cfg)
    forecast.run(cfg)
    forecast.build_inventory_status(cfg)
    recommender.run(cfg)
    churn.run(cfg)
    producer.run(cfg, events=8000)
    sp = streaming.StreamProcessor(cfg)
    sp.run()
    sp.store.close()
    return cfg
