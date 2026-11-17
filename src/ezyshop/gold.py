"""Gold layer: business-ready tables built with SQL over the Silver Parquet
lakehouse using DuckDB.

DuckDB is a vectorised, parallel, columnar SQL engine that reads Parquet
in place (predicate + projection pushdown, Hive partition pruning). It plays
the role that Spark SQL / Redshift Spectrum play in the production design,
and the SQL below is ANSI and ports to Spark SQL almost unchanged.

The key *variety* outcome is ``product_performance`` and ``customer_360``:
structured sales, semi-structured clickstream and unstructured review text
joined into one view.
"""
from __future__ import annotations

import json
import time

import duckdb

from .config import Config


def connect(cfg: Config) -> duckdb.DuckDBPyConnection:
    con = duckdb.connect()
    s = cfg.silver_dir.as_posix()
    views = {
        "orders": f"read_parquet('{s}/orders/**/*.parquet', hive_partitioning=true)",
        "clickstream": f"read_parquet('{s}/clickstream/**/*.parquet', hive_partitioning=true)",
    }
    for t in ["customers", "products", "reviews", "social_posts", "inventory", "suppliers", "campaigns"]:
        views[t] = f"read_parquet('{s}/{t}/*.parquet')"
    for name, src in views.items():
        con.execute(f"CREATE OR REPLACE VIEW {name} AS SELECT * FROM {src}")
    return con


def gold_view(con: duckdb.DuckDBPyConnection, cfg: Config, name: str) -> None:
    con.execute(f"CREATE OR REPLACE VIEW {name} AS SELECT * FROM read_parquet('{(cfg.gold_dir / name).as_posix()}.parquet')")


GOLD_SQL = {
    # --------------------------------------------------------------- sales
    "daily_sales": """
        SELECT order_date, category, subcategory,
               COUNT(DISTINCT order_id) AS orders, SUM(quantity) AS units,
               ROUND(SUM(line_revenue), 2) AS revenue, COUNT(DISTINCT customer_key) AS customers
        FROM orders WHERE status <> 'cancelled'
        GROUP BY ALL ORDER BY order_date, category, subcategory""",
    # ------------------------------------------------- funnel (semi-structured)
    "funnel_daily": """
        SELECT CAST(event_date AS DATE) AS event_date,
               COUNT(DISTINCT session_id) AS sessions,
               COUNT(*) FILTER (WHERE event_type = 'product_view') AS product_views,
               COUNT(DISTINCT session_id) FILTER (WHERE event_type = 'add_to_cart') AS cart_sessions,
               COUNT(DISTINCT session_id) FILTER (WHERE event_type = 'purchase') AS purchase_sessions,
               ROUND(1 - COUNT(DISTINCT session_id) FILTER (WHERE event_type = 'purchase')
                     / NULLIF(COUNT(DISTINCT session_id) FILTER (WHERE event_type = 'add_to_cart'), 0), 4) AS cart_abandonment_rate,
               ROUND(COUNT(DISTINCT session_id) FILTER (WHERE event_type = 'purchase')
                     / NULLIF(COUNT(DISTINCT session_id), 0), 4) AS conversion_rate,
               COUNT(*) FILTER (WHERE device_type = 'mobile') / COUNT(*) AS mobile_share
        FROM clickstream GROUP BY 1 ORDER BY 1""",
    # ---------------------- structured + unstructured integration per product
    "product_performance": """
        WITH s AS (
            SELECT product_id, SUM(quantity) FILTER (WHERE status <> 'cancelled') AS units,
                   ROUND(SUM(line_revenue) FILTER (WHERE status <> 'cancelled'), 2) AS revenue,
                   AVG(CASE WHEN status = 'returned' THEN 1.0 ELSE 0.0 END) AS return_rate
            FROM orders GROUP BY 1),
        r AS (
            SELECT product_id, COUNT(*) AS review_count, ROUND(AVG(rating_filled), 2) AS avg_rating,
                   ROUND(AVG(sentiment), 3) AS avg_sentiment,
                   ROUND(AVG(CASE WHEN sentiment_label = 'negative' THEN 1.0 ELSE 0.0 END), 3) AS negative_share
            FROM reviews GROUP BY 1),
        a AS (
            SELECT product_id, aspect, COUNT(*) AS n,
                   ROW_NUMBER() OVER (PARTITION BY product_id ORDER BY COUNT(*) DESC) AS rk
            FROM (SELECT product_id, UNNEST(string_split(aspects, ',')) AS aspect
                  FROM reviews WHERE sentiment_label = 'negative' AND aspects <> '')
            GROUP BY 1, 2),
        v AS (SELECT product_id, COUNT(*) AS views_120d FROM clickstream
              WHERE event_type = 'product_view' AND product_id IS NOT NULL GROUP BY 1)
        SELECT p.product_id, p.product_name, p.category, p.subcategory, p.list_price,
               COALESCE(s.units, 0) AS units, COALESCE(s.revenue, 0) AS revenue,
               ROUND(COALESCE(s.return_rate, 0), 3) AS return_rate,
               COALESCE(r.review_count, 0) AS review_count, r.avg_rating, r.avg_sentiment, r.negative_share,
               a.aspect AS top_complaint, COALESCE(v.views_120d, 0) AS views_120d
        FROM products p LEFT JOIN s USING (product_id) LEFT JOIN r USING (product_id)
        LEFT JOIN a ON a.product_id = p.product_id AND a.rk = 1
        LEFT JOIN v ON v.product_id = p.product_id
        ORDER BY revenue DESC""",
    # -------------------------------------------------------- customer 360
    "customer_360": """
        WITH o AS (
            SELECT customer_key, COUNT(DISTINCT order_id) AS frequency,
                   ROUND(SUM(line_revenue), 2) AS monetary, MAX(order_ts) AS last_order_ts, MIN(order_ts) AS first_order_ts,
                   MODE(category) AS favourite_category, AVG(discount) AS avg_discount
            FROM orders WHERE status <> 'cancelled' GROUP BY 1),
        r AS (SELECT customer_key, COUNT(*) AS review_count, ROUND(AVG(sentiment), 3) AS avg_review_sentiment
              FROM reviews GROUP BY 1),
        c AS (SELECT customer_key,
                     COUNT(DISTINCT session_id) FILTER (WHERE event_ts > TIMESTAMP '{as_of}' - INTERVAL 30 DAY) AS sessions_30d,
                     COUNT(DISTINCT session_id) FILTER (WHERE event_type = 'add_to_cart'
                                                       AND event_ts > TIMESTAMP '{as_of}' - INTERVAL 30 DAY) AS cart_sessions_30d
              FROM clickstream WHERE customer_key IS NOT NULL GROUP BY 1)
        SELECT cu.customer_key, cu.segment, cu.city, cu.age_band, cu.marketing_opt_in,
               DATE_DIFF('day', cu.signup_date, TIMESTAMP '{as_of}') AS tenure_days,
               DATE_DIFF('day', o.last_order_ts, TIMESTAMP '{as_of}') AS recency_days,
               COALESCE(o.frequency, 0) AS frequency, COALESCE(o.monetary, 0) AS monetary,
               ROUND(o.monetary / NULLIF(o.frequency, 0), 2) AS avg_order_value,
               o.favourite_category, ROUND(o.avg_discount, 3) AS avg_discount,
               COALESCE(r.review_count, 0) AS review_count, r.avg_review_sentiment,
               COALESCE(c.sessions_30d, 0) AS sessions_30d, COALESCE(c.cart_sessions_30d, 0) AS cart_sessions_30d,
               NTILE(5) OVER (ORDER BY DATE_DIFF('day', o.last_order_ts, TIMESTAMP '{as_of}') DESC NULLS FIRST) AS r_score,
               NTILE(5) OVER (ORDER BY COALESCE(o.frequency, 0)) AS f_score,
               NTILE(5) OVER (ORDER BY COALESCE(o.monetary, 0)) AS m_score
        FROM customers cu LEFT JOIN o USING (customer_key) LEFT JOIN r USING (customer_key)
        LEFT JOIN c USING (customer_key)""",
    # ----------------------------------------------- voice of the customer
    "category_sentiment_daily": """
        SELECT CAST(ts AS DATE) AS day, category, source, COUNT(*) AS mentions,
               ROUND(AVG(sentiment), 3) AS avg_sentiment,
               ROUND(AVG(CASE WHEN sentiment_label = 'negative' THEN 1.0 ELSE 0.0 END), 3) AS negative_share
        FROM (SELECT s.post_ts AS ts, s.category, 'social' AS source, s.sentiment, s.sentiment_label FROM social_posts s
              UNION ALL
              SELECT r.review_ts, p.category, 'review', r.sentiment, r.sentiment_label
              FROM reviews r JOIN products p USING (product_id))
        GROUP BY ALL ORDER BY 1, 2, 3""",
    # ---------------------------------------------------- marketing ROI
    "campaign_performance": """
        WITH d AS (SELECT order_date, category, SUM(line_revenue) AS revenue FROM orders
                   WHERE status <> 'cancelled' GROUP BY 1, 2)
        SELECT c.campaign_id, c.campaign_name, c.channel, c.target_category, c.start_date, c.end_date, c.spend_aud,
               DATE_DIFF('day', c.start_date, c.end_date) + 1 AS days,
               ROUND((SELECT SUM(revenue) FROM d WHERE d.category = c.target_category
                      AND d.order_date BETWEEN c.start_date AND c.end_date), 2) AS revenue_during,
               ROUND((SELECT AVG(revenue) FROM d WHERE d.category = c.target_category
                      AND d.order_date BETWEEN c.start_date - INTERVAL 28 DAY AND c.start_date - INTERVAL 1 DAY)
                     * (DATE_DIFF('day', c.start_date, c.end_date) + 1), 2) AS baseline_revenue
        FROM campaigns c ORDER BY c.start_date""",
}


def run(cfg: Config) -> dict:
    t0 = time.perf_counter()
    cfg.gold_dir.mkdir(parents=True, exist_ok=True)
    con = connect(cfg)
    as_of = f"{cfg.end_date} 23:59:59"
    stats = {}
    for name, sql in GOLD_SQL.items():
        s = time.perf_counter()
        out = (cfg.gold_dir / f"{name}.parquet").as_posix()
        con.execute(f"COPY ({sql.format(as_of=as_of)}) TO '{out}' (FORMAT PARQUET, COMPRESSION ZSTD)")
        rows = con.execute(f"SELECT COUNT(*) FROM read_parquet('{out}')").fetchone()[0]
        stats[name] = {"rows": rows, "seconds": round(time.perf_counter() - s, 3)}
    # campaign uplift / ROI computed on top of the table above
    out = (cfg.gold_dir / "campaign_performance.parquet").as_posix()
    con.execute(f"""COPY (SELECT *, ROUND(revenue_during - baseline_revenue, 2) AS incremental_revenue,
                          ROUND((revenue_during - baseline_revenue) / spend_aud, 2) AS roi
                          FROM read_parquet('{out}')) TO '{out}.tmp' (FORMAT PARQUET)""")
    import os
    os.replace(out + ".tmp", out)
    con.close()
    return {"tables": stats, "seconds": round(time.perf_counter() - t0, 2)}


if __name__ == "__main__":
    print(json.dumps(run(Config.from_env()), indent=2, default=str))
