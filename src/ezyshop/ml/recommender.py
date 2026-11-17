"""Personalised product recommendations with implicit-feedback matrix
factorisation (ALS; Hu, Koren & Volinsky, 2008; Koren et al., 2009).

Signals: purchases (strong) and product views from the clickstream (weak) -
i.e. structured + semi-structured data. Confidence c = 1 + alpha*log(1+w).

Evaluation follows the Assessment 2 commitment to *measure* the 25 %
personalisation-improvement target rather than assume it: a temporal split
(train before the cut-off, test on purchases in the last 60 days) and
precision@10 / recall@10 / hit-rate@10 / NDCG@10 against a popularity
baseline (what a non-personalised storefront shows). The relative lift in
precision@10 is the KPI reported against the +25 % target.

ALS is embarrassingly parallel per user/item, which is why Spark MLlib
ships it; this NumPy implementation keeps the same maths for a laptop.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone

import duckdb
import numpy as np
import pandas as pd

from ..config import Config
from ..serving_store import ServingStore

TEST_DAYS = 60
K = 10


def interactions(cfg: Config, until: str | None = None) -> pd.DataFrame:
    s = cfg.silver_dir.as_posix()
    cond_o = f"AND order_ts < TIMESTAMP '{until}'" if until else ""
    cond_c = f"AND event_ts < TIMESTAMP '{until}'" if until else ""
    return duckdb.sql(f"""
        SELECT customer_key, product_id, SUM(w) AS w FROM (
            SELECT customer_key, product_id, 1.0 * quantity AS w FROM read_parquet('{s}/orders/**/*.parquet')
            WHERE status = 'completed' {cond_o}
            UNION ALL
            SELECT customer_key, product_id, 0.25 AS w FROM read_parquet('{s}/clickstream/**/*.parquet')
            WHERE event_type = 'product_view' AND customer_key IS NOT NULL AND product_id IS NOT NULL {cond_c})
        GROUP BY ALL""").df()


class ALS:
    def __init__(self, factors: int = 24, reg: float = 0.1, alpha: float = 15.0, iterations: int = 12, seed: int = 7):
        self.f, self.reg, self.alpha, self.iters, self.seed = factors, reg, alpha, iterations, seed

    def fit(self, df: pd.DataFrame) -> "ALS":
        self.users = pd.Index(df["customer_key"].unique())
        self.items = pd.Index(df["product_id"].unique())
        u = self.users.get_indexer(df["customer_key"])
        i = self.items.get_indexer(df["product_id"])
        conf = 1 + self.alpha * np.log1p(df["w"].values)
        self.user_rows = self._group(u, i, conf, len(self.users))
        self.item_rows = self._group(i, u, conf, len(self.items))
        rng = np.random.default_rng(self.seed)
        self.U = rng.normal(0, 0.1, (len(self.users), self.f))
        self.V = rng.normal(0, 0.1, (len(self.items), self.f))
        for _ in range(self.iters):
            self.U = self._solve(self.V, self.user_rows)
            self.V = self._solve(self.U, self.item_rows)
        return self

    @staticmethod
    def _group(a, b, c, n):
        order = np.argsort(a, kind="stable")
        a, b, c = a[order], b[order], c[order]
        bounds = np.searchsorted(a, np.arange(n + 1))
        return [(b[bounds[k]:bounds[k + 1]], c[bounds[k]:bounds[k + 1]]) for k in range(n)]

    def _solve(self, Y: np.ndarray, rows) -> np.ndarray:
        YtY = Y.T @ Y
        lam = self.reg * np.eye(self.f)
        X = np.zeros((len(rows), self.f))
        for k, (idx, c) in enumerate(rows):
            if len(idx) == 0:
                continue
            Yk = Y[idx]
            A = YtY + (Yk.T * (c - 1)) @ Yk + lam
            X[k] = np.linalg.solve(A, (Yk.T * c).sum(axis=1))
        return X

    def recommend(self, user_idx: np.ndarray, exclude: dict[int, set[int]], k: int = K) -> np.ndarray:
        scores = self.U[user_idx] @ self.V.T
        for r, u in enumerate(user_idx):
            ex = list(exclude.get(u, ()))
            if ex:
                scores[r, ex] = -np.inf
        return np.argsort(-scores, axis=1)[:, :k]


def _metrics(recs: list[list[str]], truth: list[set[str]]) -> dict:
    p, r, h, n = [], [], [], []
    disc = 1 / np.log2(np.arange(2, K + 2))
    for rec, t in zip(recs, truth):
        hits = np.array([x in t for x in rec[:K]], dtype=float)
        p.append(hits.sum() / K)
        r.append(hits.sum() / len(t))
        h.append(float(hits.any()))
        idcg = disc[: min(len(t), K)].sum()
        n.append((hits * disc[: len(hits)]).sum() / idcg)
    return {"precision@10": round(float(np.mean(p)), 4), "recall@10": round(float(np.mean(r)), 4),
            "hit_rate@10": round(float(np.mean(h)), 4), "ndcg@10": round(float(np.mean(n)), 4)}


def evaluate(cfg: Config) -> dict:
    cutoff = (pd.Timestamp(cfg.end_date) - pd.Timedelta(days=TEST_DAYS - 1)).strftime("%Y-%m-%d")
    train = interactions(cfg, until=cutoff)
    s = cfg.silver_dir.as_posix()
    test = duckdb.sql(f"""SELECT DISTINCT customer_key, product_id FROM read_parquet('{s}/orders/**/*.parquet')
                          WHERE status = 'completed' AND order_ts >= TIMESTAMP '{cutoff}'""").df()
    model = ALS().fit(train)
    bought = train[train["w"] >= 1].copy()   # only purchases are excluded from recommendations
    excl: dict[int, set[int]] = {}
    for u, i in zip(model.users.get_indexer(bought["customer_key"]), model.items.get_indexer(bought["product_id"])):
        excl.setdefault(u, set()).add(i)
    # ground truth: *new* items bought in the test window by known users
    test = test[test["customer_key"].isin(model.users) & test["product_id"].isin(model.items)]
    test["u"] = model.users.get_indexer(test["customer_key"])
    test["i"] = model.items.get_indexer(test["product_id"])
    test = test[[i not in excl.get(u, ()) for u, i in zip(test["u"], test["i"])]]
    truth_by_u = test.groupby("u")["product_id"].apply(set)
    users = truth_by_u.index.values
    top = model.recommend(users, excl)
    als_recs = [[model.items[j] for j in row] for row in top]
    # popularity baseline (same exclusions)
    pop = train[train["w"] >= 1].groupby("product_id")["w"].sum().sort_values(ascending=False).index
    pop_idx = model.items.get_indexer(pop)
    pop_recs = []
    for u in users:
        ex = excl.get(u, set())
        pop_recs.append([model.items[j] for j in pop_idx if j not in ex][:K])
    truth = list(truth_by_u.values)
    als_m, pop_m = _metrics(als_recs, truth), _metrics(pop_recs, truth)
    lift = (als_m["precision@10"] - pop_m["precision@10"]) / max(pop_m["precision@10"], 1e-9)
    return {"train_interactions": int(len(train)), "test_users": int(len(users)), "cutoff": cutoff,
            "als": als_m, "popularity_baseline": pop_m, "precision_lift_vs_baseline": round(lift, 4),
            "target_lift": 0.25, "meets_target": bool(lift >= 0.25)}


def train_and_publish(cfg: Config) -> dict:
    """Fit on all history, write top-10 per customer to Gold + serving store."""
    data = interactions(cfg)
    model = ALS().fit(data)
    bought = data[data["w"] >= 1]
    excl: dict[int, set[int]] = {}
    for u, i in zip(model.users.get_indexer(bought["customer_key"]), model.items.get_indexer(bought["product_id"])):
        excl.setdefault(u, set()).add(i)
    users = np.arange(len(model.users))
    rows, recs = [], []
    version = datetime.now(timezone.utc).strftime("als-%Y%m%d%H%M")
    now = datetime.now(timezone.utc).isoformat()
    for start in range(0, len(users), 2000):
        chunk = users[start:start + 2000]
        top = model.recommend(chunk, excl)
        for u, row in zip(chunk, top):
            items = [model.items[j] for j in row]
            recs.append({"customer_key": model.users[u], "items": json.dumps(items), "model_version": version,
                         "updated_at": now})
            rows.extend({"customer_key": model.users[u], "rank": r + 1, "product_id": p} for r, p in enumerate(items))
    pd.DataFrame(rows).to_parquet(cfg.gold_dir / "recommendations.parquet", index=False)
    cfg.models_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cfg.models_dir / "als_model.npz", U=model.U, V=model.V,
                        users=np.array(model.users, dtype=str), items=np.array(model.items, dtype=str))
    store = ServingStore(cfg.serving_db)
    try:
        store.clear("recommendations")
        store.upsert_many("recommendations", recs)
    finally:
        store.close()
    return {"customers_scored": len(recs), "model_version": version}


def run(cfg: Config) -> dict:
    metrics = evaluate(cfg)
    metrics.update(train_and_publish(cfg))
    (cfg.models_dir / "recommender_metrics.json").write_text(json.dumps(metrics, indent=2))
    return metrics
