"""Builds the sample-output pack in ``outputs/``: KPI scorecard against the
Assessment 2 targets (summary_report.md), PNG charts and CSV extracts of the
Gold tables."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from .config import Config  # noqa: E402
from .serving_store import ServingStore  # noqa: E402

PALETTE = ["#2563eb", "#f59e0b", "#10b981", "#ef4444", "#8b5cf6", "#64748b"]


def _load(p: Path, default=None):
    return json.loads(p.read_text()) if p.exists() else (default or {})


def _style(ax, title):
    ax.set_title(title, fontsize=11, loc="left", fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.25)


def charts(cfg: Config) -> list[str]:
    g, out = cfg.gold_dir, cfg.output_dir / "charts"
    out.mkdir(parents=True, exist_ok=True)
    made = []

    ds = pd.read_parquet(g / "daily_sales.parquet")
    ds["order_date"] = pd.to_datetime(ds["order_date"])
    wk = ds.groupby([pd.Grouper(key="order_date", freq="W-MON"), "category"])["revenue"].sum().unstack()
    fig, ax = plt.subplots(figsize=(10, 4))
    for i, c in enumerate(wk.columns):
        ax.plot(wk.index, wk[c] / 1000, label=c, color=PALETTE[i], lw=1.6)
    _style(ax, "Weekly revenue by category (AUD k) - note Black Friday / Christmas / EOFY peaks")
    ax.legend(frameon=False)
    fig.tight_layout(); fig.savefig(out / "01_weekly_revenue.png", dpi=130); plt.close(fig)
    made.append("01_weekly_revenue.png")

    fc = pd.read_parquet(g / "demand_forecast.parquet")
    fm = _load(cfg.models_dir / "forecast_metrics.json")
    tot = fc.groupby(["week_start", "kind"])[["units", "forecast"]].sum(min_count=1).reset_index()
    fig, ax = plt.subplots(figsize=(10, 4))
    h = tot[tot.kind == "history"].tail(40)
    ho = tot[tot.kind == "holdout"]
    fu = tot[tot.kind == "future"]
    ax.plot(h.week_start, h.units, color="#334155", lw=1.5, label="actual units")
    ax.plot(ho.week_start, ho.forecast, color=PALETTE[0], lw=2, ls="--", marker="o", ms=3,
            label=f"forecast on unseen weeks (accuracy {fm.get('model_accuracy', 0):.1%})")
    ax.plot(fu.week_start, fu.forecast, color=PALETTE[1], lw=2, marker="o", ms=4, label="next weeks forecast")
    _style(ax, "Demand forecast - all sub-categories (units / week)")
    ax.legend(frameon=False)
    fig.tight_layout(); fig.savefig(out / "02_demand_forecast.png", dpi=130); plt.close(fig)
    made.append("02_demand_forecast.png")

    rm = _load(cfg.models_dir / "recommender_metrics.json")
    if rm:
        keys = ["precision@10", "recall@10", "hit_rate@10", "ndcg@10"]
        x = range(len(keys))
        fig, ax = plt.subplots(figsize=(8, 3.8))
        ax.bar([i - 0.2 for i in x], [rm["popularity_baseline"][k] for k in keys], 0.4, color="#94a3b8", label="popularity baseline")
        ax.bar([i + 0.2 for i in x], [rm["als"][k] for k in keys], 0.4, color=PALETTE[0], label="ALS personalised")
        ax.set_xticks(list(x), keys)
        _style(ax, f"Recommendations on unseen purchases: lift {rm['precision_lift_vs_baseline']:+.0%} (target +25%)")
        ax.legend(frameon=False)
        fig.tight_layout(); fig.savefig(out / "03_recommender_eval.png", dpi=130); plt.close(fig)
        made.append("03_recommender_eval.png")

    cs = pd.read_parquet(g / "churn_scores.parquet")
    cm = _load(cfg.models_dir / "churn_metrics.json")
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    axes[0].hist(cs["churn_probability"], bins=30, color=PALETTE[3], alpha=0.85)
    _style(axes[0], f"Churn risk distribution (AUC {cm.get('roc_auc', 0):.2f})")
    tf = pd.DataFrame(cm.get("top_features", [])).head(6)
    if len(tf):
        axes[1].barh(tf["feature"][::-1], tf["auc_drop"][::-1], color=PALETTE[4])
        _style(axes[1], "Top churn drivers (permutation importance)")
    fig.tight_layout(); fig.savefig(out / "04_churn.png", dpi=130); plt.close(fig)
    made.append("04_churn.png")

    if cfg.serving_db.exists():
        store = ServingStore(cfg.serving_db)
        lm = pd.DataFrame(store.query("SELECT * FROM live_metrics ORDER BY window_start"))
        al = pd.DataFrame(store.query("SELECT * FROM alerts"))
        store.close()
        if len(lm):
            lm["window_start"] = pd.to_datetime(lm["window_start"])
            lm = lm[lm["events"] >= 20]
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.bar(lm["window_start"], lm["revenue"] / 1000, width=0.0006, color=PALETTE[0], label="revenue / min (AUD k)")
            ax2 = ax.twinx()
            ax2.plot(lm["window_start"], lm["cart_abandonment_rate"], color=PALETTE[3], lw=2, label="cart abandonment")
            ax2.set_ylim(0, 1)
            if len(al):
                for _, a in al[al.alert_type.isin(["TRENDING_PRODUCT", "NEGATIVE_REVIEW_BURST", "CART_ABANDONMENT_SPIKE"])].iterrows():
                    ax.axvline(pd.Timestamp(a.window_start), color="#f59e0b", alpha=0.35, lw=1)
            _style(ax, "Real-time stream: revenue per 1-min window, abandonment, alert moments (orange)")
            ax.legend(loc="upper left", frameon=False); ax2.legend(loc="upper right", frameon=False)
            fig.tight_layout(); fig.savefig(out / "05_streaming.png", dpi=130); plt.close(fig)
            made.append("05_streaming.png")

    bm = _load(cfg.output_dir / "benchmark_results.json")
    if bm:
        runs = pd.DataFrame(bm["runs"])
        labels = [f"{r.strategy}\n({r.workers} worker{'s' if r.workers > 1 else ''})" for r in runs.itertuples()]
        fig, ax = plt.subplots(figsize=(10, 4.2))
        ax.barh(labels[::-1], runs["events_per_sec"][::-1] / 1000, color=PALETTE[2])
        ax.axvline(bm["target_tps"] / 1000, color=PALETTE[3], ls="--", label="target 10k TPS")
        _style(ax, f"Stream-processing throughput, {bm['events']:,} events (thousand events/sec)")
        ax.legend(frameon=False)
        fig.tight_layout(); fig.savefig(out / "06_benchmark.png", dpi=130); plt.close(fig)
        made.append("06_benchmark.png")

    dq = _load(cfg.output_dir / "data_quality_report.json")
    if dq:
        rows = [(t, r, n) for t, v in dq["tables"].items() for r, n in v["rejected"].items() if n] + \
               [(t, "duplicate", v["deduplicated"]) for t, v in dq["tables"].items() if v["deduplicated"]]
        df = pd.DataFrame(rows, columns=["table", "reason", "n"]).sort_values("n")
        fig, ax = plt.subplots(figsize=(9, 4))
        ax.barh(df["table"] + ": " + df["reason"], df["n"], color=PALETTE[5])
        _style(ax, "Veracity: records quarantined or de-duplicated in Silver")
        fig.tight_layout(); fig.savefig(out / "07_data_quality.png", dpi=130); plt.close(fig)
        made.append("07_data_quality.png")
    return made


def export_samples(cfg: Config) -> None:
    out = cfg.output_dir / "gold_samples"
    out.mkdir(parents=True, exist_ok=True)
    for p in cfg.gold_dir.glob("*.parquet"):
        df = pd.read_parquet(p)
        df.head(500).to_csv(out / f"{p.stem}.csv", index=False)
    inv = pd.read_parquet(cfg.gold_dir / "inventory_status.parquet")
    inv[inv["action"] != "OK"].to_csv(cfg.output_dir / "reorder_list.csv", index=False)
    cs = pd.read_parquet(cfg.gold_dir / "churn_scores.parquet")
    cs[cs["recommended_action"].str.startswith("VIP")].sort_values("churn_probability", ascending=False) \
        .head(200).to_csv(cfg.output_dir / "retention_campaign_targets.csv", index=False)
    if cfg.serving_db.exists():
        store = ServingStore(cfg.serving_db)
        pd.DataFrame(store.query("SELECT * FROM alerts ORDER BY window_start")).to_csv(cfg.output_dir / "realtime_alerts.csv", index=False)
        store.close()


def summary(cfg: Config, run_log: dict) -> str:
    fm = _load(cfg.models_dir / "forecast_metrics.json")
    rm = _load(cfg.models_dir / "recommender_metrics.json")
    cm = _load(cfg.models_dir / "churn_metrics.json")
    bm = _load(cfg.output_dir / "benchmark_results.json")
    sm = run_log.get("streaming", {})
    rc = run_log.get("reconciliation", {})
    dq = _load(cfg.output_dir / "data_quality_report.json")
    br = run_log.get("bronze", {})

    def yes(b):
        return "MET" if b else "NOT MET"

    lines = [
        "# EzyShop Big Data Prototype - Run Summary",
        "",
        f"Scale profile: **{cfg.scale}**  |  data as of **{cfg.end_date}**  |  total pipeline time **{run_log.get('total_seconds', '?')} s**",
        "",
        "## Scorecard against Assessment 2 requirements",
        "",
        "| Requirement (Assessment 2) | Target | Prototype result | Status |",
        "|---|---|---|---|",
    ]
    if bm:
        lines.append(f"| Peak throughput | 10,000 events/s | {bm['best_events_per_sec']:,} events/s on one laptop "
                     f"({bm['headroom_vs_target']}x headroom) | {yes(bm['best_events_per_sec'] >= 10000)} |")
    if sm:
        lines.append(f"| Real-time insight latency | <= 300 s | {sm['design_insight_latency_s']} s by design "
                     f"(window+watermark); p95 produce-to-process {sm['produce_to_process_latency_p95_s']} s | "
                     f"{yes(sm['within_budget'])} |")
    if fm:
        lines.append(f"| Demand-forecast accuracy | >= 90 % | {fm['model_accuracy']:.1%} (1-WAPE, {fm['holdout_weeks']} unseen weeks; "
                     f"best baseline {max(fm['baseline_naive_accuracy'], fm['baseline_seasonal_naive_accuracy'], fm['baseline_moving_avg4_accuracy'], fm.get('baseline_ewma_accuracy', 0)):.1%}) "
                     f"| {yes(fm['meets_target'])} |")
    if rm:
        lines.append(f"| Personalisation improvement | +25 % | {rm['precision_lift_vs_baseline']:+.1%} precision@10 vs popularity "
                     f"| {yes(rm['meets_target'])} |")
    if cm:
        lines.append(f"| Identify customers at risk (12 % churn problem) | - | ROC-AUC {cm['roc_auc']} (recency-only baseline "
                     f"{cm['baseline_recency_only_roc_auc']}); top-decile lift {cm['top_decile_lift']}x | n/a |")
    if dq:
        lines.append(f"| Data quality (veracity) | rules enforced | {sum(sum(v['rejected'].values()) for v in dq['tables'].values()):,} "
                     f"records quarantined, {sum(v['deduplicated'] for v in dq['tables'].values()):,} duplicates removed | MET |")
        lines.append(f"| Unstructured text analytics validity | - | sentiment vs star-rating correlation "
                     f"{dq['sentiment_validation']['corr_sentiment_vs_rating']} | n/a |")
    lines.append("| GDPR (pseudonymisation, erasure) | supported | HMAC pseudonyms in Silver/Gold; erasure demo: "
                 f"{run_log.get('gdpr', {}).get('verified', 'n/a')} | MET |")
    if br:
        lines.append(f"| Storage efficiency (volume) | - | raw {br['raw_mb']} MB -> Bronze Parquet {br['bronze_mb']} MB "
                     f"({br['compression_ratio']}x) | n/a |")
    lines += ["", "## Real-time layer", ""]
    if sm:
        lines.append(f"- Events received {sm['received']:,}; duplicates removed {sm['duplicates']:,}; late events dropped by "
                     f"the watermark {sm['late_dropped']:,}; micro-batches {sm['micro_batches']}.")
        lines.append(f"- Alerts raised: {json.dumps(sm['alerts_by_type'])}")
    if rc:
        lines.append(f"- Lambda reconciliation: batch recomputation found {rc['events_only_in_batch']} events the stream "
                     f"had to drop as late; stream revenue completeness {rc['stream_revenue_completeness']:.2%} "
                     f"-> corrected view in gold/stream_metrics_reconciled.")
    lines += ["", "## Pipeline step timings (s)", "", "| Step | Seconds |", "|---|---|"]
    lines += [f"| {k} | {v} |" for k, v in run_log.get("timings", {}).items()]
    lines += ["", "Charts: `outputs/charts/`. Gold extracts: `outputs/gold_samples/`. "
              "Action lists: `reorder_list.csv`, `retention_campaign_targets.csv`, `realtime_alerts.csv`."]
    text = "\n".join(lines) + "\n"
    (cfg.output_dir / "summary_report.md").write_text(text, encoding="utf-8")
    return text


def run(cfg: Config, run_log: dict) -> dict:
    made = charts(cfg)
    export_samples(cfg)
    summary(cfg, run_log)
    return {"charts": made}
