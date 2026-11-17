"""Customer churn prediction (Assessment 2: EzyShop lost 12 % of customers in
six months; at-risk customers feed retention campaigns).

Point-in-time design to avoid leakage: features are computed *as of* a
cut-off date T using only data before T; the label is "no order in the 90
days after T". The same feature SQL is then run as of today to score the
current base. Features combine structured purchases (RFM, discounts,
returns), semi-structured engagement (sessions/carts in last 30 days) and
unstructured review sentiment.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone

import duckdb
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split

from ..config import Config
from ..serving_store import ServingStore

HORIZON_DAYS = 90
FEATURES = ["recency_days", "tenure_days", "orders_365d", "orders_90d", "orders_prev_90d", "order_trend",
            "monetary_365d", "avg_order_value", "n_categories", "avg_discount", "return_rate",
            "review_count", "avg_review_sentiment", "negative_review_share",
            "sessions_30d", "cart_sessions_30d", "product_views_30d"]


def features_as_of(cfg: Config, as_of: str) -> pd.DataFrame:
    s = cfg.silver_dir.as_posix()
    return duckdb.sql(f"""
        WITH o AS (
            SELECT customer_key,
                   DATE_DIFF('day', MAX(order_ts), TIMESTAMP '{as_of}') AS recency_days,
                   COUNT(DISTINCT order_id) FILTER (WHERE order_ts > TIMESTAMP '{as_of}' - INTERVAL 365 DAY) AS orders_365d,
                   COUNT(DISTINCT order_id) FILTER (WHERE order_ts > TIMESTAMP '{as_of}' - INTERVAL 90 DAY) AS orders_90d,
                   COUNT(DISTINCT order_id) FILTER (WHERE order_ts <= TIMESTAMP '{as_of}' - INTERVAL 90 DAY
                                                    AND order_ts > TIMESTAMP '{as_of}' - INTERVAL 180 DAY) AS orders_prev_90d,
                   SUM(line_revenue) FILTER (WHERE order_ts > TIMESTAMP '{as_of}' - INTERVAL 365 DAY) AS monetary_365d,
                   SUM(line_revenue) / COUNT(DISTINCT order_id) AS avg_order_value,
                   COUNT(DISTINCT category) AS n_categories, AVG(discount) AS avg_discount,
                   AVG(CASE WHEN status = 'returned' THEN 1.0 ELSE 0.0 END) AS return_rate
            FROM read_parquet('{s}/orders/**/*.parquet')
            WHERE order_ts < TIMESTAMP '{as_of}' AND status <> 'cancelled' GROUP BY 1),
        r AS (SELECT customer_key, COUNT(*) AS review_count, AVG(sentiment) AS avg_review_sentiment,
                     AVG(CASE WHEN sentiment_label = 'negative' THEN 1.0 ELSE 0.0 END) AS negative_review_share
              FROM read_parquet('{s}/reviews/*.parquet') WHERE review_ts < TIMESTAMP '{as_of}' GROUP BY 1),
        c AS (SELECT customer_key, COUNT(DISTINCT session_id) AS sessions_30d,
                     COUNT(DISTINCT session_id) FILTER (WHERE event_type = 'add_to_cart') AS cart_sessions_30d,
                     COUNT(*) FILTER (WHERE event_type = 'product_view') AS product_views_30d
              FROM read_parquet('{s}/clickstream/**/*.parquet', hive_partitioning=true)
              WHERE customer_key IS NOT NULL AND event_ts < TIMESTAMP '{as_of}'
                AND event_ts >= TIMESTAMP '{as_of}' - INTERVAL 30 DAY GROUP BY 1)
        SELECT cu.customer_key, cu.segment,
               DATE_DIFF('day', cu.signup_date, TIMESTAMP '{as_of}') AS tenure_days,
               o.* EXCLUDE (customer_key), o.orders_90d - o.orders_prev_90d AS order_trend,
               COALESCE(r.review_count, 0) AS review_count, r.avg_review_sentiment, r.negative_review_share,
               COALESCE(c.sessions_30d, 0) AS sessions_30d, COALESCE(c.cart_sessions_30d, 0) AS cart_sessions_30d,
               COALESCE(c.product_views_30d, 0) AS product_views_30d
        FROM read_parquet('{s}/customers/*.parquet') cu
        JOIN o USING (customer_key) LEFT JOIN r USING (customer_key) LEFT JOIN c USING (customer_key)
        WHERE o.orders_365d > 0 AND cu.signup_date <= TIMESTAMP '{as_of}' - INTERVAL 30 DAY""").df()


def labels(cfg: Config, start: str, end: str) -> set[str]:
    s = cfg.silver_dir.as_posix()
    df = duckdb.sql(f"""SELECT DISTINCT customer_key FROM read_parquet('{s}/orders/**/*.parquet')
                        WHERE order_ts >= TIMESTAMP '{start}' AND order_ts <= TIMESTAMP '{end}'
                        AND status <> 'cancelled'""").df()
    return set(df["customer_key"])


def run(cfg: Config) -> dict:
    end = pd.Timestamp(cfg.end_date) + pd.Timedelta(hours=23, minutes=59, seconds=59)
    cutoff = (end - pd.Timedelta(days=HORIZON_DAYS)).normalize()
    train_df = features_as_of(cfg, str(cutoff))
    active_after = labels(cfg, str(cutoff), str(end))
    train_df["churned"] = (~train_df["customer_key"].isin(active_after)).astype(int)

    X, y = train_df[FEATURES], train_df["churned"]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, stratify=y, random_state=7)
    model = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=15,
                                           min_samples_leaf=20, l2_regularization=1.0, random_state=7)
    model.fit(X_tr, y_tr)
    p = model.predict_proba(X_te)[:, 1]
    # baseline: a recency-only rule (what a simple "days since last order" report gives)
    base = LogisticRegression().fit(X_tr[["recency_days"]], y_tr)
    pb = base.predict_proba(X_te[["recency_days"]])[:, 1]

    top = np.argsort(-p)[: max(1, len(p) // 10)]
    imp = permutation_importance(model, X_te, y_te, scoring="roc_auc", n_repeats=5, random_state=7)
    importance = sorted(zip(FEATURES, imp.importances_mean), key=lambda t: -t[1])[:8]
    metrics = {
        "as_of_cutoff": str(cutoff.date()), "horizon_days": HORIZON_DAYS,
        "customers_labelled": int(len(train_df)), "churn_rate": round(float(y.mean()), 4),
        "roc_auc": round(float(roc_auc_score(y_te, p)), 4),
        "pr_auc": round(float(average_precision_score(y_te, p)), 4),
        "baseline_recency_only_roc_auc": round(float(roc_auc_score(y_te, pb)), 4),
        "top_decile_precision": round(float(y_te.values[top].mean()), 4),
        "top_decile_lift": round(float(y_te.values[top].mean() / y_te.mean()), 2),
        "top_features": [{"feature": f, "auc_drop": round(float(v), 4)} for f, v in importance],
    }

    # ---- refit on all labelled data, score today's customer base
    final = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=15,
                                           min_samples_leaf=20, l2_regularization=1.0, random_state=7).fit(X, y)
    now_df = features_as_of(cfg, str(end))
    now_df["churn_probability"] = final.predict_proba(now_df[FEATURES])[:, 1].round(4)
    now_df["risk_band"] = pd.cut(now_df["churn_probability"], [-0.01, 0.3, 0.6, 1.0], labels=["Low", "Medium", "High"]).astype(str)
    value_rank = now_df["monetary_365d"].rank(pct=True)
    now_df["value_tier"] = np.where(value_rank >= 0.8, "Top 20%", np.where(value_rank >= 0.5, "Mid", "Low"))
    now_df["recommended_action"] = np.select(
        [(now_df["risk_band"] == "High") & (now_df["value_tier"] == "Top 20%"),
         now_df["risk_band"] == "High", now_df["risk_band"] == "Medium"],
        ["VIP win-back: personal offer + service call", "Win-back e-mail with personalised recommendations",
         "Nurture: targeted content / loyalty points"], "No action")
    out_cols = ["customer_key", "segment", "churn_probability", "risk_band", "value_tier", "recommended_action"] + FEATURES
    now_df[out_cols].to_parquet(cfg.gold_dir / "churn_scores.parquet", index=False)
    metrics["scored_customers"] = int(len(now_df))
    metrics["high_risk_customers"] = int((now_df["risk_band"] == "High").sum())

    joblib.dump(final, cfg.models_dir / "churn_model.joblib")
    (cfg.models_dir / "churn_metrics.json").write_text(json.dumps(metrics, indent=2))

    # customer profiles for the serving API
    store = ServingStore(cfg.serving_db)
    try:
        ts = datetime.now(timezone.utc).isoformat()
        store.upsert_many("customer_profile", [
            {"customer_key": r.customer_key,
             "profile": json.dumps({"segment": r.segment, "churn_probability": float(r.churn_probability),
                                    "risk_band": r.risk_band, "value_tier": r.value_tier,
                                    "recommended_action": r.recommended_action,
                                    "recency_days": int(r.recency_days), "orders_365d": int(r.orders_365d)}),
             "updated_at": ts} for r in now_df.itertuples()])
    finally:
        store.close()
    return metrics
