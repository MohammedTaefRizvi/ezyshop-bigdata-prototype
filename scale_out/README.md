# Scale-out path (Kafka + Spark)

The runnable prototype in `../src` implements the architecture with laptop-friendly
components, each mapped to its production counterpart:

| Prototype (runs anywhere, `pip install`) | Production (this folder / AWS) |
|---|---|
| `event_log.py` partitioned log + consumer groups | Apache Kafka / Amazon MSK / Kinesis |
| Parquet medallion folders | Amazon S3 + Delta Lake |
| DuckDB SQL (`gold.py`) | Spark SQL on EMR / AWS Glue, Redshift |
| `streaming.py` micro-batches, watermark, dedup | Spark Structured Streaming (`spark_streaming_job.py`) |
| NumPy ALS / scikit-learn GBMs | Spark MLlib ALS / SageMaker |
| SQLite serving store | Cassandra / DynamoDB |
| Streamlit | QuickSight / Power BI |

Files here:

- `docker-compose.yml`: single-broker Kafka (KRaft) + Spark master and 2 workers.
- `kafka_bridge.py`: replays the local event log into Kafka, keeping the keys.
- `spark_streaming_job.py`: the windowed, watermarked, de-duplicated stream job.
- `spark_batch_job.py`: Bronze to Silver validation and a Gold aggregate.

> **Status:** these files are the reference code for the production path. They were
> **not executed** during development because the build machine had no Docker or Java.
> All results in `outputs/` come from the Python prototype. Run them yourself to
> verify, as described in `docker-compose.yml`.
