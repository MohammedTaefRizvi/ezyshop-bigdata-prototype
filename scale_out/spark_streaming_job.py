"""Spark Structured Streaming equivalent of src/ezyshop/streaming.py.

Same semantics as the prototype: event-time 1-minute tumbling windows,
2-minute watermark, de-duplication on event_id within the watermark,
checkpointed offsets/state. Reference implementation for the scale-out path
(not executed in the development environment - see scale_out/README.md).
"""
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import (ArrayType, DoubleType, IntegerType, StringType, StructField, StructType)

BOOTSTRAP = "kafka:19092"
OUT = "/data/lakehouse/gold/stream_metrics_spark"
CHECKPOINT = "/data/checkpoints/stream_metrics_spark"

line = StructType([StructField("product_id", StringType()), StructField("quantity", IntegerType()),
                   StructField("unit_price", DoubleType())])
schema = StructType([
    StructField("event_id", StringType()), StructField("event_type", StringType()),
    StructField("event_time", StringType()), StructField("session_id", StringType()),
    StructField("customer_id", StringType()), StructField("product_id", StringType()),
    StructField("total", DoubleType()), StructField("lines", ArrayType(line)), StructField("text", StringType()),
])

spark = SparkSession.builder.appName("ezyshop-stream-analytics").getOrCreate()
spark.conf.set("spark.sql.streaming.multipleWatermarkPolicy", "min")  # same policy as the prototype

events = (spark.readStream.format("kafka")
          .option("kafka.bootstrap.servers", BOOTSTRAP)
          .option("subscribe", "clickstream,orders,reviews,inventory")
          .option("startingOffsets", "earliest")
          .option("maxOffsetsPerTrigger", 200_000)      # backpressure: bounded micro-batches
          .load()
          .select(F.from_json(F.col("value").cast("string"), schema).alias("e"))
          .select("e.*")
          .withColumn("event_time", F.to_timestamp("event_time"))
          # pseudonymise before anything is persisted (GDPR)
          .withColumn("customer_key", F.substring(F.sha2(F.concat(F.lit("salt"), F.col("customer_id")), 256), 1, 16))
          .drop("customer_id"))

deduped = events.withWatermark("event_time", "2 minutes").dropDuplicates(["event_id", "event_time"])

metrics = (deduped.groupBy(F.window("event_time", "1 minute"))
           .agg(F.count("*").alias("events"),
                F.approx_count_distinct("session_id").alias("sessions"),   # HyperLogLog sketch
                F.sum(F.when(F.col("event_type") == "product_view", 1).otherwise(0)).alias("product_views"),
                F.sum(F.when(F.col("event_type") == "add_to_cart", 1).otherwise(0)).alias("add_to_carts"),
                F.sum(F.when(F.col("event_type") == "order", 1).otherwise(0)).alias("orders"),
                F.sum(F.when(F.col("event_type") == "order", F.col("total")).otherwise(0.0)).alias("revenue"))
           .select(F.col("window.start").alias("window_start"), F.col("window.end").alias("window_end"), "*")
           .drop("window"))

query = (metrics.writeStream.outputMode("append")      # emitted once the watermark closes the window
         .format("parquet").option("path", OUT).option("checkpointLocation", CHECKPOINT)
         .trigger(processingTime="10 seconds").start())
query.awaitTermination()
