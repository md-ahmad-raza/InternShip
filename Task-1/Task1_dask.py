import argparse
import os
import tempfile
from urllib.parse import urlparse

import requests
import dask.dataframe as dd
from dask.distributed import Client

# Download a remote file to a local temporary path.
# This helper is used when a remote Parquet URL cannot be read directly by Dask.
def download_file(url: str) -> str:
    parsed = urlparse(url)
    suffix = os.path.splitext(parsed.path)[1] or ".parquet"
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    temp_file.close()

    with requests.get(url, stream=True) as response:
        response.raise_for_status()
        with open(temp_file.name, "wb") as out_file:
            for chunk in response.iter_content(chunk_size=16_384):
                if chunk:
                    out_file.write(chunk)

    return temp_file.name


# Load a dataset from either a local path or a remote URL.
# Supports Parquet and CSV/CSV.GZ formats.
def load_dataset(input_path: str):
    parsed = urlparse(input_path)
    is_remote = parsed.scheme in {"http", "https"}
    temp_file_path = None

    if is_remote:
        try:
            # Try to read remote Parquet directly with Dask.
            return dd.read_parquet(input_path), None
        except Exception as exc:
            # If remote read fails, download locally and retry.
            print(f"Remote read failed, downloading file locally: {exc}")
            temp_file_path = download_file(input_path)
            print(f"Downloaded remote file to {temp_file_path}")
            return dd.read_parquet(temp_file_path), temp_file_path

    if os.path.exists(input_path):
        if input_path.lower().endswith(".parquet"):
            return dd.read_parquet(input_path), None

        if input_path.lower().endswith(".csv") or input_path.lower().endswith(".csv.gz"):
            return dd.read_csv(input_path, parse_dates=["tpep_pickup_datetime"]), None

    raise ValueError("Unsupported input format. Use a local or remote .parquet or .csv file.")


# Main analysis workflow.
# Handles connecting to Dask, loading data, validation, metrics, and optional output.
def main(input_path: str, output_path: str | None, force_csv: bool = False):
    client = Client()
    print(f"Starting Dask client: {client}")

    print(f"Loading dataset from: {input_path}")
    df, temp_file_path = load_dataset(input_path)

    print("== Dataset schema ==")
    print(df.dtypes)

    # Validate required columns exist before proceeding.
    if "tpep_pickup_datetime" not in df.columns or "fare_amount" not in df.columns:
        raise KeyError("Input dataset must contain tpep_pickup_datetime and fare_amount columns.")

    # Remove invalid rows and persist the filtered DataFrame in memory.
    df = df[df["fare_amount"] > 0]
    df = df.persist()

    print("== Summary metrics ==")
    total_rides = df.shape[0].compute()

    revenue_summary = df[["fare_amount", "tip_amount", "total_amount"]].sum().compute()
    print(f"Total rides analyzed: {total_rides:,}")
    print(f"Total fare revenue: ${revenue_summary['fare_amount']:.2f}")
    print(f"Total tip revenue: ${revenue_summary['tip_amount']:.2f}")
    print(f"Total passenger charges: ${revenue_summary['total_amount']:.2f}")

    print("== Average fares ==")
    avg_stats = df[["fare_amount", "tip_amount", "total_amount"]].mean().compute()
    print(f"Average fare amount: ${avg_stats['fare_amount']:.2f}")
    print(f"Average tip amount: ${avg_stats['tip_amount']:.2f}")
    print(f"Average total paid: ${avg_stats['total_amount']:.2f}")

    print("== Hourly analysis ==")
    # Extract pickup hour for hourly aggregation.
    df = df.assign(pickup_hour=df["tpep_pickup_datetime"].dt.hour)
    hourly = df.groupby("pickup_hour").agg(
        avg_fare=("fare_amount", "mean"),
        trip_count=("fare_amount", "count"),
        total_fare=("fare_amount", "sum")
    ).reset_index().compute().sort_values("pickup_hour")
    print(hourly)

    busiest_hour = hourly.loc[hourly["trip_count"].idxmax()]
    highest_avg_fare_hour = hourly.loc[hourly["avg_fare"].idxmax()]

    print("== Key insights ==")
    print(f"Busiest hour: {int(busiest_hour['pickup_hour'])} with {int(busiest_hour['trip_count']):,} trips")
    print(f"Highest average fare hour: {int(highest_avg_fare_hour['pickup_hour'])} with ${highest_avg_fare_hour['avg_fare']:.2f} average fare")

    print("== Weekday vs weekend insight ==")
    # Create weekday/weekend indicator to compare ride patterns.
    df = df.assign(
        day_of_week=df["tpep_pickup_datetime"].dt.dayofweek,
        is_weekend=(df["tpep_pickup_datetime"].dt.dayofweek >= 5)
    )
    weekend_stats = df.groupby("is_weekend").agg(
        trip_count=("fare_amount", "count"),
        avg_fare=("fare_amount", "mean")
    ).reset_index().compute()
    weekend_stats["is_weekend"] = weekend_stats["is_weekend"].map({False: "weekday", True: "weekend"})
    print(weekend_stats)

    if output_path:
        print(f"Writing hourly analysis to: {output_path}")
        _, ext = os.path.splitext(output_path)
        if force_csv or ext.lower() == ".csv":
            # Save the aggregated hourly results as a single CSV file.
            hourly.to_csv(output_path, index=False, single_file=True)
        else:
            # Save the aggregated results as Parquet.
            dd.from_pandas(hourly, npartitions=1).to_parquet(output_path, engine="pyarrow", write_index=False)

    if temp_file_path and os.path.exists(temp_file_path):
        try:
            os.remove(temp_file_path)
            print(f"Removed temporary file: {temp_file_path}")
        except OSError as exc:
            print(f"Warning: could not remove temporary file {temp_file_path}: {exc}")

    client.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dask big-data analysis for NYC taxi trip dataset")
    parser.add_argument(
        "--input",
        default="https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2023-01.parquet",
        help="Input Parquet or CSV file path or URL"
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional output path to save aggregated results"
    )
    parser.add_argument(
        "--force-csv",
        action="store_true",
        help="Write output as CSV regardless of output extension"
    )
    args = parser.parse_args()
    main(args.input, args.output, args.force_csv)
