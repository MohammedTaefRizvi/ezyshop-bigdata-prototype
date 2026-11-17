"""Spark batch equivalent of the Silver orders build + a Gold aggregate
(src/ezyshop/silver.py, src/ezyshop/gold.py). Reference implementation for
the scale-out path; not executed in the development environment.

    spark-submit /jobs/spark_batch_job.py
"""
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

SALT = "dev-only-salt-change-me"
spark = SparkSession.builder.appName("ezyshop-batch").getOrCreate()

products = (spark.read.option("header", True).csv("/data/raw/products.csv")
            .withColumn("category", F.initcap(F.trim(F.regexp_replace("category", " & Living", ""))))
            .withColumn("list_price", F.regexp_replace("list_price", r"\$", "").cast("double")))

orders = (spark.read.option("header", True).csv("/data/raw/orders.csv")
          .dropDuplicates(["order_id", "line_no"])
          .withColumn("order_ts", F.to_timestamp("order_ts"))
          .withColumn("quantity", F.col("quantity").cast("int"))
          .withColumn("unit_price", F.col("unit_price").cast("double")))

valid = (orders.join(products.select("product_id", "category", "subcategory", "list_price"), "product_id")
         .where(F.col("customer_id").isNotNull() & (F.col("quantity") > 0) & (F.col("unit_price") > 0)
                & (F.col("unit_price") <= 3 * F.col("list_price")) & (F.col("order_ts") <= F.current_timestamp()))
         .withColumn("customer_key", F.substring(F.sha2(F.concat(F.lit(SALT), "customer_id"), 256), 1, 16))
         .drop("customer_id")
         .withColumn("line_revenue", F.round(F.col("quantity") * F.col("unit_price"), 2))
         .withColumn("order_month", F.date_format("order_ts", "yyyy-MM")))

(valid.repartition("order_month").write.mode("overwrite").partitionBy("order_month")
 .parquet("/data/lakehouse/silver_spark/orders"))

daily = (valid.where(F.col("status") != "cancelled")
         .groupBy(F.to_date("order_ts").alias("order_date"), "category", "subcategory")
         .agg(F.countDistinct("order_id").alias("orders"), F.sum("quantity").alias("units"),
              F.round(F.sum("line_revenue"), 2).alias("revenue")))
daily.write.mode("overwrite").parquet("/data/lakehouse/gold_spark/daily_sales")
spark.stop()
