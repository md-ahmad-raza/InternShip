import argparse
import os
import tempfile
from urllib.parse import urlparse

import requests
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, hour, avg, count, sum as spark_sum, min as spark_min, max as spark_max, dayofweek


def build_spark_session(app_name: str = "BigDataAnalysis_NYC_Taxi"):
    return SparkSession.builder \
        .appName(app_name) \
        .config("spark.executor.memory", "4g") \
        .config("spark.sql.shuffle.partitions", "200") \
        .getOrCreate()


def download_remote_parquet(url: str) -> str:
    parsed = urlparse(url)
    suffix = os.path.splitext(parsed.path)[1] or ".parquet"
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    temp_file.close()

    print(f"Downloading remote file to local temp file: {temp_file.name}")
    with requests.get(url, stream=True) as response:
        response.raise_for_status()
        with open(temp_file.name, "wb") as out_file:
            for chunk in response.iter_content(chunk_size=16_384):
                if chunk:
                    out_file.write(chunk)

    return temp_file.name


def get_local_input_path(input_path: str) -> tuple[str, bool]:
    parsed = urlparse(input_path)
    if parsed.scheme in {"http", "https"}:
        local_path = download_remote_parquet(input_path)
        return local_path, True
    return input_path, False


def main(input_path: str, output_path: str | None):
    spark = build_spark_session()
    spark.sparkContext.setLogLevel("ERROR")

    local_input_path, downloaded_temp = get_local_input_path(input_path)
    print(f"Loading dataset from: {local_input_path}")
    df = spark.read.parquet(local_input_path)

    print("== Dataset schema ==")
    df.printSchema()

    print("== Sample rows ==")
    df.select("tpep_pickup_datetime", "fare_amount", "tip_amount", "total_amount") \
        .limit(5) \
        .show(truncate=False)

    print("== Summary metrics ==")
    df = df.filter(col("fare_amount") > 0)
    df.cache()

    total_rides = df.count()
    revenue_summary = df.agg(
        spark_sum("fare_amount").alias("total_fare_amount"),
        spark_sum("tip_amount").alias("total_tip_amount"),
        spark_sum("total_amount").alias("total_amount")
    ).collect()[0]

    print(f"Total rides analyzed: {total_rides:,}")
    print(f"Total fare revenue: ${revenue_summary['total_fare_amount']:.2f}")
    print(f"Total tip revenue: ${revenue_summary['total_tip_amount']:.2f}")
    print(f"Total passenger charges: ${revenue_summary['total_amount']:.2f}")

    print("== Average fares ==")
    avg_stats = df.select(
        avg("fare_amount").alias("average_fare"),
        avg("tip_amount").alias("average_tip"),
        avg("total_amount").alias("average_total")
    ).collect()[0]
    print(f"Average fare amount: ${avg_stats['average_fare']:.2f}")
    print(f"Average tip amount: ${avg_stats['average_tip']:.2f}")
    print(f"Average total paid: ${avg_stats['average_total']:.2f}")

    print("== Hourly analysis ==")
    hourly_df = df.withColumn("pickup_hour", hour(col("tpep_pickup_datetime"))) \
        .groupBy("pickup_hour") \
        .agg(
            avg("fare_amount").alias("avg_fare"),
            count("fare_amount").alias("trip_count"),
            spark_sum("fare_amount").alias("total_fare")
        ) \
        .orderBy("pickup_hour")

    hourly_df.show(24)

    busiest_hour = hourly_df.orderBy(col("trip_count").desc()).limit(1)
    highest_avg_fare_hour = hourly_df.orderBy(col("avg_fare").desc()).limit(1)

    print("== Key insights ==")
    busiest_hour.show(1)
    highest_avg_fare_hour.show(1)

    print("== Weekday vs weekend insight ==")
    weekday_df = df.withColumn("day_of_week", dayofweek(col("tpep_pickup_datetime"))) \
        .withColumn("is_weekend", (col("day_of_week") >= 6).cast("int")) \
        .groupBy("is_weekend") \
        .agg(
            count("fare_amount").alias("trip_count"),
            avg("fare_amount").alias("avg_fare")
        )
    weekday_df.show()

    if output_path:
        print(f"Writing hourly analysis to: {output_path}")
        hourly_df.coalesce(1).write.mode("overwrite").parquet(output_path)

    spark.stop()

    if downloaded_temp and os.path.exists(local_input_path):
        try:
            os.remove(local_input_path)
            print(f"Removed temporary download: {local_input_path}")
        except OSError as exc:
            print(f"Warning: could not remove temporary file {local_input_path}: {exc}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Big data analysis for NYC taxi trip dataset")
    parser.add_argument(
        "--input",
        default="https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2023-01.parquet",
        help="Input Parquet file path or URL"
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional output path to save aggregated results"
    )
    args = parser.parse_args()
    main(args.input, args.output)
