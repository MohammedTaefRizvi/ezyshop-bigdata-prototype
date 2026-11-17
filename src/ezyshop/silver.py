"""Silver layer: validated, de-duplicated, typed, conformed and pseudonymised.

This is where the *veracity* challenge is addressed. Every rule is explicit
and every rejected record is written to ``silver/_quarantine/<table>`` with a
reason, so data stewards can fix sources instead of silently losing data.
Unstructured text is turned into structured features here (sentiment,
aspects, category), which is what allows reviews/social to be joined with
sales in the Gold layer.
"""
from __future__ import annotations

import json
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq

from . import text_analytics as ta
from .config import Config
from .privacy import pseudonymise_series, suppressed_keys

CATEGORY_MAP = {"electronics": "Electronics", "apparel": "Apparel", "home": "Home", "home & living": "Home"}


def read_bronze(cfg: Config, table: str) -> pd.DataFrame:
    path = cfg.bronze_dir / table
    files = sorted(path.rglob("*.parquet")) if path.exists() else []
    if not files:
        return pd.DataFrame()
    return pd.concat([pq.read_table(f).to_pandas() for f in files], ignore_index=True)


def write_silver(cfg: Config, df: pd.DataFrame, table: str, partition_cols: list[str] | None = None) -> None:
    out = cfg.silver_dir / table
    if out.exists():
        shutil.rmtree(out) if out.is_dir() else out.unlink()
    tbl = pa.Table.from_pandas(df, preserve_index=False)
    if partition_cols:
        ds.write_dataset(tbl, out, format="parquet", partitioning=partition_cols, partitioning_flavor="hive",
                         existing_data_behavior="overwrite_or_ignore",
                         file_options=ds.ParquetFileFormat().make_write_options(compression="zstd"))
    else:
        out.mkdir(parents=True, exist_ok=True)
        pq.write_table(tbl, out / "data.parquet", compression="zstd")


class QualityTracker:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.report: dict[str, dict] = {}
        self.quarantine: dict[str, list[pd.DataFrame]] = {}

    def reject(self, table: str, df: pd.DataFrame, mask, reason: str) -> pd.DataFrame:
        mask = np.asarray(mask, dtype=bool)
        bad = df[mask]
        if len(bad):
            q = bad.copy()
            q["_reject_reason"] = reason
            self.quarantine.setdefault(table, []).append(q.astype(str))
        self.report.setdefault(table, {"input": 0, "output": 0, "rejected": {}, "deduplicated": 0})
        self.report[table]["rejected"][reason] = int(mask.sum())
        return df[~mask]

    def dedup(self, table: str, df: pd.DataFrame, subset: list[str]) -> pd.DataFrame:
        before = len(df)
        df = df.drop_duplicates(subset=subset, keep="first")
        self.report.setdefault(table, {"input": 0, "output": 0, "rejected": {}, "deduplicated": 0})
        self.report[table]["deduplicated"] += before - len(df)
        return df

    def io(self, table: str, n_in: int, n_out: int) -> None:
        self.report.setdefault(table, {"input": 0, "output": 0, "rejected": {}, "deduplicated": 0})
        self.report[table]["input"] = n_in
        self.report[table]["output"] = n_out

    def flush(self) -> None:
        qdir = self.cfg.silver_dir / "_quarantine"
        if qdir.exists():
            shutil.rmtree(qdir)
        qdir.mkdir(parents=True, exist_ok=True)
        for table, frames in self.quarantine.items():
            pq.write_table(pa.Table.from_pandas(pd.concat(frames, ignore_index=True), preserve_index=False),
                           qdir / f"{table}.parquet")


def parse_ts(s: pd.Series) -> pd.Series:
    """Accept ISO-8601 strings and epoch milliseconds (mixed source formats)."""
    s = s.astype("string")
    is_epoch = s.str.fullmatch(r"\d{12,14}").fillna(False)
    out = pd.to_datetime(s.where(~is_epoch), errors="coerce", format="ISO8601")
    if is_epoch.any():
        out = out.where(~is_epoch, pd.to_datetime(pd.to_numeric(s.where(is_epoch)), unit="ms", errors="coerce"))
    return out.astype("datetime64[us]")


# ---------------------------------------------------------------- per table
def build_customers(cfg: Config, q: QualityTracker, suppressed: set[str]) -> pd.DataFrame:
    df = read_bronze(cfg, "customers")
    n_in = len(df)
    df = q.dedup("customers", df, ["customer_id"])
    signup = pd.to_datetime(df["signup_date"], format="%Y-%m-%d", errors="coerce")
    alt = pd.to_datetime(df["signup_date"], format="%d/%m/%Y", errors="coerce")  # AU format fallback
    df["signup_date"] = signup.fillna(alt)
    df = q.reject("customers", df, df["signup_date"].isna(), "unparseable_signup_date")
    df["customer_key"] = pseudonymise_series(df["customer_id"], cfg.pii_salt)
    df = q.reject("customers", df, df["customer_key"].isin(suppressed), "gdpr_erasure")
    age = pd.to_numeric(df["age"], errors="coerce")
    df["age_band"] = pd.cut(age, [0, 24, 34, 44, 54, 64, 200], labels=["18-24", "25-34", "35-44", "45-54", "55-64", "65+"]).astype(str)
    df["marketing_opt_in"] = df["marketing_opt_in"].str.lower().eq("true")
    # data minimisation: names, e-mail, phone and exact age are dropped
    out = df[["customer_key", "city", "age_band", "segment", "signup_date", "marketing_opt_in"]].copy()
    q.io("customers", n_in, len(out))
    return out


def build_products(cfg: Config, q: QualityTracker) -> pd.DataFrame:
    df = read_bronze(cfg, "products")
    n_in = len(df)
    df = q.dedup("products", df, ["product_id"])
    df["category"] = df["category"].str.strip().str.lower().map(CATEGORY_MAP)
    df = q.reject("products", df, df["category"].isna(), "unknown_category")
    df["list_price"] = pd.to_numeric(df["list_price"].str.replace("$", "", regex=False), errors="coerce")
    df = q.reject("products", df, ~(df["list_price"] > 0), "invalid_price")
    df["launch_date"] = pd.to_datetime(df["launch_date"], errors="coerce")
    out = df[["product_id", "product_name", "category", "subcategory", "list_price", "supplier_id", "launch_date"]].copy()
    q.io("products", n_in, len(out))
    return out


def build_orders(cfg: Config, q: QualityTracker, products: pd.DataFrame, suppressed: set[str], as_of) -> pd.DataFrame:
    df = read_bronze(cfg, "orders")
    n_in = len(df)
    df = q.dedup("orders", df, ["order_id", "line_no"])
    df["order_ts"] = parse_ts(df["order_ts"])
    df["quantity"] = pd.to_numeric(df["quantity"], errors="coerce")
    df["unit_price"] = pd.to_numeric(df["unit_price"], errors="coerce")
    df["discount"] = pd.to_numeric(df["discount"], errors="coerce").fillna(0.0)
    df["line_no"] = pd.to_numeric(df["line_no"], errors="coerce").astype("Int64")
    df = q.reject("orders", df, df["customer_id"].eq("") | df["customer_id"].isna(), "missing_customer_id")
    df = q.reject("orders", df, df["order_ts"].isna(), "unparseable_timestamp")
    df = q.reject("orders", df, df["order_ts"] > as_of + pd.Timedelta(days=1), "future_timestamp")
    df = q.reject("orders", df, ~(df["quantity"] > 0), "non_positive_quantity")
    price = df["product_id"].map(products.set_index("product_id")["list_price"])
    df = q.reject("orders", df, price.isna(), "unknown_product")
    price = df["product_id"].map(products.set_index("product_id")["list_price"])
    df = q.reject("orders", df, ~((df["unit_price"] > 0) & (df["unit_price"] <= 3 * price)), "price_outlier")
    df["customer_key"] = pseudonymise_series(df["customer_id"], cfg.pii_salt)
    df = q.reject("orders", df, df["customer_key"].isin(suppressed), "gdpr_erasure")
    df = df.merge(products[["product_id", "category", "subcategory"]], on="product_id", how="left")
    df["line_revenue"] = (df["quantity"] * df["unit_price"]).round(2)
    df["order_date"] = df["order_ts"].dt.date.astype("datetime64[us]")
    df["order_month"] = df["order_ts"].dt.strftime("%Y-%m")
    out = df[["order_id", "line_no", "order_ts", "order_date", "order_month", "customer_key", "product_id",
              "category", "subcategory", "quantity", "unit_price", "discount", "line_revenue", "channel", "status"]]
    q.io("orders", n_in, len(out))
    return out.reset_index(drop=True)


def build_clickstream(cfg: Config, q: QualityTracker, suppressed: set[str]) -> pd.DataFrame:
    df = read_bronze(cfg, "clickstream")
    n_in = len(df)
    parsed, ok = [], []
    for raw in df["raw"].values:
        try:
            parsed.append(json.loads(raw))
            ok.append(True)
        except json.JSONDecodeError:
            parsed.append({})
            ok.append(False)
    ok = np.array(ok)
    q.reject("clickstream", df, ~ok, "malformed_json")
    ev = pd.json_normalize([p for p, k in zip(parsed, ok) if k], sep="_")  # flattens nested device/geo/page
    ev = q.dedup("clickstream", ev, ["event_id"])
    ev["event_ts"] = parse_ts(ev["event_ts"].astype(str))
    ev = q.reject("clickstream", ev, ev["event_ts"].isna(), "unparseable_timestamp")
    ev["customer_key"] = pseudonymise_series(ev["customer_id"], cfg.pii_salt)
    ev = q.reject("clickstream", ev, ev["customer_key"].isin(suppressed), "gdpr_erasure")
    ev["event_date"] = ev["event_ts"].dt.strftime("%Y-%m-%d")
    if "order_id" not in ev:
        ev["order_id"] = None
    out = ev[["event_id", "event_type", "event_ts", "event_date", "session_id", "customer_key", "product_id",
              "order_id", "page_url", "page_referrer", "device_type", "device_os", "geo_city"]]
    q.io("clickstream", n_in, len(out))
    return out.reset_index(drop=True)


def _text_features(texts: pd.Series) -> pd.DataFrame:
    return pd.DataFrame([ta.analyse(t) for t in texts.values], index=texts.index)


def build_reviews(cfg: Config, q: QualityTracker, suppressed: set[str]) -> pd.DataFrame:
    raw = read_bronze(cfg, "reviews")
    n_in = len(raw)
    df = pd.DataFrame([json.loads(r) for r in raw["raw"].values])
    df = q.dedup("reviews", df, ["review_id"])
    df["review_ts"] = parse_ts(df["review_ts"])
    feats = _text_features(df["title"].fillna("") + ". " + df["body"].fillna(""))
    df = pd.concat([df, feats], axis=1)
    df = q.reject("reviews", df, df["body"].fillna("").str.strip().eq(""), "empty_text")
    df["customer_key"] = pseudonymise_series(df["customer_id"], cfg.pii_salt)
    df = q.reject("reviews", df, df["customer_key"].isin(suppressed), "gdpr_erasure")
    df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
    df["rating_missing"] = df["rating"].isna()
    # impute a star rating from text when the customer left only free text
    df["rating_filled"] = df["rating"].fillna((3 + 2 * df["sentiment"]).round().clip(1, 5))
    out = df[["review_id", "order_id", "product_id", "customer_key", "review_ts", "rating", "rating_filled",
              "rating_missing", "title", "clean_text", "sentiment", "sentiment_label", "aspects"]]
    q.io("reviews", n_in, len(out))
    return out.reset_index(drop=True)


def build_social(cfg: Config, q: QualityTracker) -> pd.DataFrame:
    raw = read_bronze(cfg, "social_posts")
    n_in = len(raw)
    df = pd.DataFrame([json.loads(r) for r in raw["raw"].values])
    df = q.dedup("social_posts", df, ["post_id"])
    df["post_ts"] = parse_ts(df["post_ts"])
    df = pd.concat([df, _text_features(df["text"])], axis=1)
    df["subcategory"] = [ta.detect_subcategory(t) for t in df["clean_text"].values]
    df["category"] = df["subcategory"].map(ta.SUBCAT_TO_CATEGORY)
    df = q.reject("social_posts", df, df["subcategory"].isna(), "no_product_mention")
    # the author handle is personal data and is not needed for analytics
    out = df[["post_id", "platform", "post_ts", "clean_text", "sentiment", "sentiment_label", "aspects",
              "subcategory", "category", "likes", "shares"]]
    q.io("social_posts", n_in, len(out))
    return out.reset_index(drop=True)


def build_inventory(cfg: Config, q: QualityTracker) -> pd.DataFrame:
    df = read_bronze(cfg, "inventory_snapshot")
    n_in = len(df)
    df["on_hand"] = pd.to_numeric(df["on_hand"], errors="coerce")
    df["on_order"] = pd.to_numeric(df["on_order"], errors="coerce").fillna(0)
    df = q.reject("inventory", df, df["on_hand"].isna() | (df["on_hand"] < 0), "invalid_on_hand")
    df["snapshot_ts"] = parse_ts(df["snapshot_ts"])
    out = df[["snapshot_ts", "warehouse", "product_id", "on_hand", "on_order"]]
    q.io("inventory", n_in, len(out))
    return out.reset_index(drop=True)


def build_suppliers(cfg: Config, q: QualityTracker) -> pd.DataFrame:
    raw = read_bronze(cfg, "suppliers")
    df = pd.json_normalize([json.loads(r) for r in raw["raw"].values], sep="_")
    if "lead_time_days" not in df:
        df["lead_time_days"] = np.nan
    df["lead_time_imputed"] = df["lead_time_days"].isna()
    # impute missing lead time with the median of suppliers in the same country group
    df["overseas"] = df["country"] != "Australia"
    df["lead_time_days"] = df.groupby("overseas")["lead_time_days"].transform(lambda s: s.fillna(s.median()))
    q.io("suppliers", len(raw), len(df))
    q.report["suppliers"]["imputed_lead_time"] = int(df["lead_time_imputed"].sum())
    return df[["supplier_id", "name", "country", "overseas", "lead_time_days", "lead_time_imputed", "reliability"]]


def build_campaigns(cfg: Config, q: QualityTracker) -> pd.DataFrame:
    df = read_bronze(cfg, "campaigns")
    df["campaign_id"] = pd.to_numeric(df["campaign_id"])
    df["start_date"] = pd.to_datetime(df["start_date"])
    df["end_date"] = pd.to_datetime(df["end_date"])
    df["spend_aud"] = pd.to_numeric(df["spend_aud"])
    q.io("campaigns", len(df), len(df))
    return df[["campaign_id", "campaign_name", "channel", "target_category", "start_date", "end_date", "spend_aud"]]


def run(cfg: Config) -> dict:
    t0 = time.perf_counter()
    q = QualityTracker(cfg)
    suppressed = suppressed_keys(cfg)
    as_of = pd.Timestamp(cfg.end_date)
    timings = {}

    def step(name, fn, *args, partition_cols=None):
        s = time.perf_counter()
        df = fn(cfg, q, *args)
        write_silver(cfg, df, name, partition_cols)
        timings[name] = round(time.perf_counter() - s, 2)
        return df

    step("customers", build_customers, suppressed)
    products = step("products", build_products)
    step("orders", build_orders, products, suppressed, as_of, partition_cols=["order_month"])
    step("clickstream", build_clickstream, suppressed, partition_cols=["event_date"])
    reviews = step("reviews", build_reviews, suppressed)
    step("social_posts", build_social)
    step("inventory", build_inventory)
    step("suppliers", build_suppliers)
    step("campaigns", build_campaigns)
    q.flush()

    # validate the unstructured analytics against a structured signal
    rated = reviews.dropna(subset=["rating"])
    agree = ((rated["rating"] >= 4) & (rated["sentiment_label"] == "positive")) | \
            ((rated["rating"] <= 2) & (rated["sentiment_label"] == "negative")) | \
            ((rated["rating"] == 3) & (rated["sentiment_label"] == "neutral"))
    polar = rated[rated["rating"] != 3]
    polar_agree = ((polar["rating"] >= 4) == (polar["sentiment"] > 0)).mean()
    report = {
        "tables": q.report,
        "sentiment_validation": {
            "reviews_with_rating": int(len(rated)),
            "exact_label_agreement": round(float(agree.mean()), 3),
            "polarity_agreement_excl_3_star": round(float(polar_agree), 3),
            "corr_sentiment_vs_rating": round(float(rated["sentiment"].corr(rated["rating"])), 3),
        },
        "gdpr_suppressed_customers": len(suppressed),
        "timings_s": timings,
        "seconds": round(time.perf_counter() - t0, 2),
    }
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    (cfg.output_dir / "data_quality_report.json").write_text(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    print(json.dumps(run(Config.from_env()), indent=2))
