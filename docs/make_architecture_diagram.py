"""Renders docs/architecture.png (run: python docs/make_architecture_diagram.py)."""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

OUT = Path(__file__).resolve().parent / "architecture.png"

COLS = [
    ("1  SOURCES", "#e2e8f0", [
        ("Orders / CRM / products", "CSV - structured"),
        ("Inventory, campaigns", "CSV - structured"),
        ("Suppliers", "nested JSON - semi-str."),
        ("Clickstream", "JSON lines - semi-str."),
        ("Reviews, social posts", "free text - unstructured"),
        ("Live web/app/WMS events", "real-time producer"),
    ]),
    ("2  INGESTION", "#dbeafe", [
        ("Partitioned event log", "key-hash partitions, offsets,\nconsumer groups  [Kafka / MSK]"),
        ("Batch file ingestion", "checksum-idempotent loads\n[Glue / Debezium CDC]"),
    ]),
    ("3  STORAGE  (lakehouse)", "#dcfce7", [
        ("BRONZE", "raw, schema-on-read Parquet\n+ lineage  [S3 + Delta]"),
        ("SILVER", "validated, deduplicated,\npseudonymised, text -> features"),
        ("GOLD", "business tables, features,\nforecasts, scores"),
        ("Quarantine", "rejected rows + reason"),
    ]),
    ("4  PROCESSING", "#fef9c3", [
        ("Batch path", "DuckDB SQL over Parquet\n[Spark SQL / EMR / Glue]"),
        ("Speed path", "micro-batch windows, watermark,\ndedup, state  [Spark Streaming]"),
        ("Reconciliation", "batch recompute corrects\nlate data (Lambda)"),
    ]),
    ("5  MACHINE LEARNING", "#fce7f3", [
        ("Demand forecast", "global GBM, level-ratio\n[MLlib / SageMaker]"),
        ("Recommendations", "implicit ALS (MF)\n+ live-session re-rank"),
        ("Churn prediction", "point-in-time GBM\nRFM + clicks + sentiment"),
    ]),
    ("6  SERVING & USERS", "#ede9fe", [
        ("Serving store", "key-value, idempotent upserts\n[Cassandra / DynamoDB]"),
        ("REST API", "recommendations, profiles,\nlive metrics, alerts"),
        ("Dashboards", "management, marketing (live),\noperations  [QuickSight / Power BI]"),
        ("Action lists", "reorder list, retention\ntargets, real-time alerts"),
    ]),
]


def main() -> None:
    fig, ax = plt.subplots(figsize=(19, 9.2))
    ax.set_xlim(0, 19)
    ax.set_ylim(0, 9.2)
    ax.axis("off")
    col_w, gap, x0 = 2.75, 0.42, 0.25
    centers = []
    for i, (title, colour, boxes) in enumerate(COLS):
        x = x0 + i * (col_w + gap)
        ax.add_patch(FancyBboxPatch((x, 0.55), col_w, 7.75, boxstyle="round,pad=0.02,rounding_size=0.15",
                                    fc=colour, ec="#94a3b8", lw=1))
        ax.text(x + col_w / 2, 8.0, title, ha="center", va="center", fontsize=11, fontweight="bold", color="#0f172a")
        n = len(boxes)
        h = min(1.25, (7.0 - 0.15 * (n - 1)) / n)
        top = 7.6
        for j, (name, sub) in enumerate(boxes):
            y = top - (j + 1) * h - j * 0.15
            ax.add_patch(FancyBboxPatch((x + 0.12, y), col_w - 0.24, h, boxstyle="round,pad=0.01,rounding_size=0.08",
                                        fc="white", ec="#64748b", lw=0.8))
            ax.text(x + col_w / 2, y + h * 0.68, name, ha="center", va="center", fontsize=9.2, fontweight="bold")
            ax.text(x + col_w / 2, y + h * 0.3, sub, ha="center", va="center", fontsize=7.4, color="#334155")
        centers.append(x)
    for i in range(len(COLS) - 1):
        xa = centers[i] + col_w
        xb = centers[i + 1]
        ax.add_patch(FancyArrowPatch((xa + 0.02, 4.4), (xb - 0.02, 4.4), arrowstyle="-|>", mutation_scale=22,
                                     color="#1e293b", lw=2))
    ax.text(9.5, 8.95, "EzyShop Big Data Lakehouse - prototype implementation (production equivalents in [brackets])",
            ha="center", fontsize=14, fontweight="bold")
    ax.text(9.5, 0.2, "Cross-cutting: GDPR pseudonymisation & erasure  |  data-quality rules & quarantine  |  "
                      "checkpointed offsets/state  |  run log, test suite & benchmark", ha="center", fontsize=9.5,
            color="#334155")
    fig.savefig(OUT, dpi=140, bbox_inches="tight")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
