"""EzyShop Big Data dashboard (Streamlit).

    streamlit run dashboard/app.py

Pages follow the stakeholder views in Assessment 2: management KPIs/ROI,
marketing real-time, customer & personalisation, operations (forecast +
inventory), voice of the customer, and the data platform itself.
"""
from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path

import duckdb
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from ezyshop.config import Config  # noqa: E402

cfg = Config.from_env()
G = cfg.gold_dir
st.set_page_config(page_title="EzyShop Big Data Platform", page_icon="🛒", layout="wide")


# ----------------------------------------------------------------- data access
@st.cache_data(ttl=60)
def gold(name: str) -> pd.DataFrame:
    p = G / f"{name}.parquet"
    if p.is_dir():
        return duckdb.sql(f"SELECT * FROM read_parquet('{p.as_posix()}/*.parquet')").df()
    return pd.read_parquet(p) if p.exists() else pd.DataFrame()


def serving(sql: str) -> pd.DataFrame:
    if not cfg.serving_db.exists():
        return pd.DataFrame()
    con = sqlite3.connect(f"file:{cfg.serving_db.as_posix()}?mode=ro", uri=True, timeout=10)
    try:
        return pd.read_sql_query(sql, con)
    finally:
        con.close()


def jload(p: Path) -> dict:
    return json.loads(p.read_text()) if p.exists() else {}


if not (G / "daily_sales.parquet").exists():
    st.error("No Gold data found. Run `python run_all.py` first.")
    st.stop()

fm = jload(cfg.models_dir / "forecast_metrics.json")
rm = jload(cfg.models_dir / "recommender_metrics.json")
cm = jload(cfg.models_dir / "churn_metrics.json")
bm = jload(cfg.output_dir / "benchmark_results.json")
dq = jload(cfg.output_dir / "data_quality_report.json")

st.sidebar.title("🛒 EzyShop")
st.sidebar.caption("Big Data lakehouse prototype · DAT801A Assessment 3")
page = st.sidebar.radio("View", ["Executive overview", "Live operations (real-time)", "Customers & personalisation",
                                 "Demand & inventory", "Voice of the customer", "Data platform & quality"])
st.sidebar.markdown("---")
st.sidebar.caption(f"Data as of **{cfg.end_date}** · scale **{cfg.scale}**")


def kpi(col, label, value, delta=None, help=None):
    col.metric(label, value, delta, help=help)


# ======================================================================= pages
if page == "Executive overview":
    st.title("Executive overview")
    ds = gold("daily_sales")
    ds["order_date"] = pd.to_datetime(ds["order_date"])
    last30 = ds[ds.order_date > ds.order_date.max() - pd.Timedelta(days=30)]
    prev30 = ds[(ds.order_date <= ds.order_date.max() - pd.Timedelta(days=30)) &
                (ds.order_date > ds.order_date.max() - pd.Timedelta(days=60))]
    c = st.columns(4)
    r1, r0 = last30.revenue.sum(), prev30.revenue.sum()
    kpi(c[0], "Revenue (30 days)", f"A${r1 / 1e6:,.2f}M", f"{(r1 / r0 - 1):+.1%} vs prior 30d")
    kpi(c[1], "Orders (30 days)", f"{last30.orders.sum():,}")
    cs = gold("churn_scores")
    kpi(c[2], "Customers at high churn risk", f"{(cs.risk_band == 'High').sum():,}",
        f"of {len(cs):,} active", help="Gradient-boosted churn model, 90-day horizon")
    kpi(c[3], "Products to reorder now", f"{(gold('inventory_status').action != 'OK').sum():,}")

    st.subheader("Targets from Assessment 2 — measured, not assumed")
    t = st.columns(4)
    if bm:
        kpi(t[0], "Throughput (target 10k ev/s)", f"{bm['best_events_per_sec']:,}/s", f"{bm['headroom_vs_target']}x headroom")
    if fm:
        kpi(t[1], "Forecast accuracy (target 90%)", f"{fm['model_accuracy']:.1%}",
            f"{fm['model_accuracy'] - max(fm['baseline_naive_accuracy'], fm['baseline_moving_avg4_accuracy']):+.1%} vs best baseline")
    if rm:
        kpi(t[2], "Personalisation lift (target +25%)", f"{rm['precision_lift_vs_baseline']:+.0%}",
            "precision@10 vs popularity")
    kpi(t[3], "Insight latency (budget 300 s)", f"{cfg.window_seconds + cfg.watermark_seconds + 1} s", "window + watermark")

    wk = ds.groupby([pd.Grouper(key="order_date", freq="W-MON"), "category"], as_index=False)["revenue"].sum()
    st.plotly_chart(px.area(wk, x="order_date", y="revenue", color="category",
                            title="Weekly revenue by category (Black Friday, Christmas and EOFY peaks visible)"),
                    width="stretch")
    cp = gold("campaign_performance")
    if len(cp):
        st.subheader("Marketing ROI by campaign")
        st.dataframe(cp[["campaign_name", "channel", "target_category", "start_date", "spend_aud", "revenue_during",
                         "baseline_revenue", "incremental_revenue", "roi"]].sort_values("roi", ascending=False),
                     hide_index=True, width="stretch")

elif page == "Live operations (real-time)":
    st.title("Live operations — real-time stream")
    st.caption("Speed layer: event log → 60 s windows with 120 s watermark → serving store. "
               "Run `python -m ezyshop.streaming --follow --watermark-seconds 30` and `python -m ezyshop.producer --realtime --rate 500 --events 60000` to watch it move.")

    @st.fragment(run_every="3s")
    def live():
        lm = serving("SELECT * FROM live_metrics ORDER BY window_start")
        al = serving("SELECT * FROM alerts ORDER BY window_start DESC LIMIT 200")
        summ = serving("SELECT value FROM kv WHERE key='stream_summary'")
        if lm.empty:
            st.info("No streaming data yet.")
            return
        lm["window_start"] = pd.to_datetime(lm["window_start"])
        lm = lm[lm["events"] >= 5]
        # show the most recent 45 minutes only (older runs would squash the axis)
        lm = lm[lm["window_start"] >= lm["window_start"].max() - pd.Timedelta(minutes=45)]
        last = lm.iloc[-1]
        c = st.columns(5)
        c[0].metric("Revenue (last window)", f"A${last.revenue:,.0f}")
        c[1].metric("Sessions (last window)", f"{int(last.sessions):,}")
        c[2].metric("Conversion", f"{last.conversion_rate:.1%}")
        c[3].metric("Cart abandonment", "-" if pd.isna(last.cart_abandonment_rate) else f"{last.cart_abandonment_rate:.0%}")
        c[4].metric("Open alerts", f"{len(al):,}")
        fig = go.Figure()
        fig.add_bar(x=lm.window_start, y=lm.revenue, name="revenue / min (A$)", marker_color="#2563eb")
        fig.add_scatter(x=lm.window_start, y=lm.cart_abandonment_rate, name="cart abandonment", yaxis="y2",
                        line=dict(color="#ef4444", width=2))
        fig.update_layout(title="Per-minute revenue and cart abandonment",
                          yaxis2=dict(overlaying="y", side="right", range=[0, 1], tickformat=".0%"), height=380,
                          legend=dict(orientation="h"))
        st.plotly_chart(fig, width="stretch")
        a, b = st.columns([3, 2])
        with a:
            st.subheader("Actionable alerts")
            if not al.empty:
                order = {"critical": 0, "high": 1, "opportunity": 2}
                al["sev"] = al.severity.map(order)
                st.dataframe(al.sort_values(["sev", "window_start"], ascending=[True, False])[
                    ["window_start", "alert_type", "severity", "entity", "message"]], hide_index=True, width="stretch", height=360)
        with b:
            st.subheader("Alert mix")
            if not al.empty:
                st.plotly_chart(px.bar(al.alert_type.value_counts().reset_index(), x="count", y="alert_type",
                                       orientation="h"), width="stretch")
            if not summ.empty:
                s = json.loads(summ.value.iloc[0])
                st.caption(f"Processed {s.get('processed', 0):,} events · duplicates removed {s.get('duplicates', 0):,} · "
                           f"late (beyond watermark) {s.get('late_dropped', 0):,} · p95 produce→process "
                           f"{s.get('produce_to_process_latency_p95_s', 0)} s")
        stock = serving("SELECT * FROM live_stock WHERE reorder_point IS NOT NULL ORDER BY stock - reorder_point LIMIT 15")
        if not stock.empty:
            st.subheader("Live stock closest to reorder point")
            st.dataframe(stock, hide_index=True, width="stretch")

    live()
    rec = gold("stream_metrics_reconciled")
    if len(rec):
        with st.expander("Lambda reconciliation: speed layer vs batch recomputation"):
            st.caption("The batch path replays the full event log without a watermark; differences are late events the "
                       "stream had to drop. The batch view is the corrected record of truth.")
            st.dataframe(rec[rec.events_diff != 0], hide_index=True, width="stretch")

elif page == "Customers & personalisation":
    st.title("Customers & personalisation")
    cs = gold("churn_scores")
    c = st.columns(4)
    kpi(c[0], "Churn model ROC-AUC", f"{cm.get('roc_auc', 0):.3f}", f"recency-only baseline {cm.get('baseline_recency_only_roc_auc', 0):.3f}")
    kpi(c[1], "Top-decile lift", f"{cm.get('top_decile_lift', 0)}x")
    kpi(c[2], "Recommender precision@10", f"{rm.get('als', {}).get('precision@10', 0):.3f}",
        f"{rm.get('precision_lift_vs_baseline', 0):+.0%} vs popularity")
    kpi(c[3], "Hit-rate@10", f"{rm.get('als', {}).get('hit_rate@10', 0):.1%}")
    a, b = st.columns(2)
    a.plotly_chart(px.histogram(cs, x="churn_probability", color="risk_band", nbins=40,
                                title="Churn-risk distribution (active customers)",
                                color_discrete_map={"Low": "#10b981", "Medium": "#f59e0b", "High": "#ef4444"}), width="stretch")
    tf = pd.DataFrame(cm.get("top_features", []))
    if len(tf):
        b.plotly_chart(px.bar(tf.sort_values("auc_drop"), x="auc_drop", y="feature", orientation="h",
                              title="What drives churn (permutation importance)"), width="stretch")
    st.plotly_chart(px.box(cs, x="segment", y="churn_probability", color="segment", title="Churn risk by segment"),
                    width="stretch")
    st.subheader("Retention campaign list — high risk, high value")
    st.dataframe(cs[cs.recommended_action.str.startswith("VIP")].sort_values("churn_probability", ascending=False)
                 [["customer_key", "segment", "churn_probability", "value_tier", "recency_days", "orders_365d",
                   "sessions_30d", "avg_review_sentiment", "recommended_action"]].head(100), hide_index=True, width="stretch")
    st.subheader("Look up a customer (pseudonymised key)")
    key = st.selectbox("customer_key", cs.sort_values("churn_probability", ascending=False).customer_key.head(300))
    recs = gold("recommendations")
    pp = gold("product_performance")
    r = recs[recs.customer_key == key].merge(pp[["product_id", "product_name", "category", "list_price", "avg_rating"]],
                                             on="product_id", how="left")
    x, y = st.columns([1, 2])
    x.json(cs[cs.customer_key == key][["segment", "churn_probability", "risk_band", "value_tier",
                                       "recommended_action"]].iloc[0].to_dict())
    y.dataframe(r[["rank", "product_id", "product_name", "category", "list_price", "avg_rating"]], hide_index=True, width="stretch")

elif page == "Demand & inventory":
    st.title("Demand forecasting & inventory")
    c = st.columns(4)
    kpi(c[0], "Forecast accuracy (1-WAPE)", f"{fm.get('model_accuracy', 0):.1%}", "target 90%")
    kpi(c[1], "Naive baseline", f"{fm.get('baseline_naive_accuracy', 0):.1%}")
    kpi(c[2], "4-week moving average", f"{fm.get('baseline_moving_avg4_accuracy', 0):.1%}")
    kpi(c[3], "Seasonal naive", f"{fm.get('baseline_seasonal_naive_accuracy', 0):.1%}")
    fc = gold("demand_forecast")
    sub = st.selectbox("Sub-category", ["All"] + sorted(fc.subcategory.unique()))
    d = fc if sub == "All" else fc[fc.subcategory == sub]
    d = d.groupby(["week_start", "kind"], as_index=False)[["units", "forecast"]].sum(min_count=1)
    fig = go.Figure()
    h = d[d.kind == "history"]
    fig.add_scatter(x=h.week_start, y=h.units, name="actual", line=dict(color="#334155"))
    ho = d[d.kind == "holdout"]
    fig.add_scatter(x=ho.week_start, y=ho.forecast, name="forecast (unseen weeks)", line=dict(color="#2563eb", dash="dash"))
    fu = d[d.kind == "future"]
    fig.add_scatter(x=fu.week_start, y=fu.forecast, name="next 6 weeks", line=dict(color="#f59e0b", width=3))
    fig.update_layout(title=f"Weekly units — {sub}", height=420, legend=dict(orientation="h"))
    st.plotly_chart(fig, width="stretch")
    if fm.get("accuracy_by_subcategory"):
        acc = pd.Series(fm["accuracy_by_subcategory"]).rename("accuracy").rename_axis("subcategory").reset_index()
        st.plotly_chart(px.bar(acc, x="subcategory", y="accuracy", title="Accuracy by sub-category (holdout)",
                               range_y=[0.5, 1]).add_hline(y=0.9, line_dash="dot"), width="stretch")
    inv = gold("inventory_status")
    st.subheader("Reorder recommendations (forecast-driven reorder point = demand × lead time + safety stock)")
    st.dataframe(inv[inv.action != "OK"][["product_id", "product_name", "subcategory", "on_hand", "on_order",
                                          "daily_demand", "lead_time_days", "safety_stock", "reorder_point",
                                          "days_of_cover", "suggested_order_qty", "action"]].round(1),
                 hide_index=True, width="stretch")

elif page == "Voice of the customer":
    st.title("Voice of the customer — unstructured data")
    pp = gold("product_performance")
    sd = gold("category_sentiment_daily")
    sd["day"] = pd.to_datetime(sd["day"])
    wk = sd.groupby([pd.Grouper(key="day", freq="W-MON"), "category", "source"], as_index=False).agg(
        avg_sentiment=("avg_sentiment", "mean"), mentions=("mentions", "sum"))
    st.plotly_chart(px.line(wk, x="day", y="avg_sentiment", color="category", line_dash="source",
                            title="Weekly sentiment by category — reviews vs social media"), width="stretch")
    a, b = st.columns(2)
    a.plotly_chart(px.scatter(pp[pp.review_count > 5], x="avg_sentiment", y="return_rate", size="revenue", color="category",
                              hover_name="product_name",
                              title="Review sentiment vs return rate (bubble = revenue): structured × unstructured"),
                   width="stretch")
    comp = pp.dropna(subset=["top_complaint"]).top_complaint.value_counts().reset_index()
    b.plotly_chart(px.bar(comp, x="top_complaint", y="count", title="Most common complaint aspect per product"),
                   width="stretch")
    st.subheader("Products needing attention (high negative share)")
    st.dataframe(pp[pp.review_count >= 10].sort_values("negative_share", ascending=False)
                 [["product_id", "product_name", "category", "revenue", "avg_rating", "avg_sentiment", "negative_share",
                   "top_complaint", "return_rate"]].head(25), hide_index=True, width="stretch")

else:
    st.title("Data platform & quality")
    st.image(str(ROOT / "docs" / "architecture.png"), caption="Lakehouse architecture implemented by the prototype")
    if dq:
        rows = [{"table": t, "input": v["input"], "output": v["output"], "duplicates_removed": v["deduplicated"],
                 **{f"rejected: {k}": n for k, n in v["rejected"].items() if n}} for t, v in dq["tables"].items()]
        st.subheader("Silver-layer data quality (veracity)")
        st.dataframe(pd.DataFrame(rows).fillna(0), hide_index=True, width="stretch")
        st.caption(f"Sentiment validation vs star ratings: {dq['sentiment_validation']}")
    if bm:
        st.subheader("Throughput benchmark")
        runs = pd.DataFrame(bm["runs"])
        runs["label"] = runs.strategy + " (" + runs.workers.astype(str) + "w)"
        st.plotly_chart(px.bar(runs, x="events_per_sec", y="label", orientation="h",
                               title=f"{bm['events']:,} events — target 10,000/s").add_vline(x=10_000, line_dash="dot"),
                        width="stretch")
    st.subheader("Lakehouse layers")
    layer_rows = []
    for layer in ["bronze", "silver", "gold"]:
        d = cfg.base_dir / "lakehouse" / layer
        for t in sorted(p for p in d.iterdir() if not p.name.startswith("_")) if d.exists() else []:
            files = list(t.rglob("*.parquet")) if t.is_dir() else [t]
            layer_rows.append({"layer": layer, "table": t.stem, "files": len(files),
                               "size_mb": round(sum(f.stat().st_size for f in files) / 1e6, 2)})
    st.dataframe(pd.DataFrame(layer_rows), hide_index=True, width="stretch")
    log = jload(cfg.output_dir / "pipeline_run_log.json")
    if log.get("timings"):
        st.subheader("Last pipeline run — step timings (s)")
        st.plotly_chart(px.bar(pd.Series(log["timings"]).rename("seconds").rename_axis("step").reset_index(), x="step", y="seconds"),
                        width="stretch")
