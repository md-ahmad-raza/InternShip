import argparse
import tempfile
from urllib.parse import urlparse

import pandas as pd
import requests
import dask.dataframe as dd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Default remote dataset URL used when no input path is provided.
DEFAULT_URL = "https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2023-01.parquet"


def download_remote_file(url: str) -> str:
    # Download a remote Parquet file into a temporary local file.
    parsed = urlparse(url)
    suffix = "" if parsed.path.endswith(".parquet") else ".parquet"
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    temp_file.close()

    print(f"Downloading remote dataset to temporary file: {temp_file.name}")
    with requests.get(url, stream=True) as response:
        response.raise_for_status()
        with open(temp_file.name, "wb") as out_file:
            for chunk in response.iter_content(chunk_size=16_384):
                if chunk:
                    out_file.write(chunk)

    return temp_file.name


def load_data(input_path: str, sample_size: int) -> pd.DataFrame:
    # Load dataset from a local or remote parquet source.
    # If remote read fails, download the file and read it locally.
    parsed = urlparse(input_path)
    is_remote = parsed.scheme in {"http", "https"}

    if is_remote:
        print("Loading remote dataset with Dask...")
        try:
            df = dd.read_parquet(input_path, columns=[
                "tpep_pickup_datetime",
                "VendorID",
                "passenger_count",
                "trip_distance",
                "RatecodeID",
                "store_and_fwd_flag",
                "payment_type",
                "fare_amount",
                "tip_amount",
                "total_amount",
            ])
            df = df.dropna(subset=["fare_amount", "tip_amount", "trip_distance"])
            print(f"Sampling {sample_size} rows from the remote dataset...")
            sampled = df.head(sample_size, compute=True)
            return sampled
        except Exception as exc:
            print(f"Remote read failed: {exc}")
            print("Falling back to downloading the remote Parquet file locally...")
            local_path = download_remote_file(input_path)
            df = pd.read_parquet(local_path, columns=[
                "tpep_pickup_datetime",
                "VendorID",
                "passenger_count",
                "trip_distance",
                "RatecodeID",
                "store_and_fwd_flag",
                "payment_type",
                "fare_amount",
                "tip_amount",
                "total_amount",
            ])
            return df.sample(n=min(sample_size, len(df)), random_state=42)

    print("Loading local dataset...")
    df = pd.read_parquet(input_path)
    return df.sample(n=min(sample_size, len(df)), random_state=42)


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    # Convert raw columns into model-ready features.
    df = df.copy()
    df["tpep_pickup_datetime"] = pd.to_datetime(df["tpep_pickup_datetime"], errors="coerce")
    df = df[df["tpep_pickup_datetime"].notna()]
    df = df[df["fare_amount"] > 0]

    # Extract time-based features from the pickup timestamp.
    df["pickup_hour"] = df["tpep_pickup_datetime"].dt.hour
    df["pickup_dayofweek"] = df["tpep_pickup_datetime"].dt.dayofweek

    # Create the target label indicating whether the tip rate is high.
    df["tip_rate"] = df["tip_amount"] / df["fare_amount"]
    df["high_tip"] = (df["tip_rate"] >= 0.25).astype(int)

    # Filter out rows with invalid passenger count or distance.
    df = df[df["trip_distance"] >= 0]
    df = df[df["passenger_count"] > 0]

    return df


def create_pipeline(numeric_features, categorical_features):
    # Build a preprocessing and classification pipeline.
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)),
        ]
    )

    return pipeline


def evaluate_model(model, X_test, y_test):
    # Evaluate the trained model using several performance metrics.
    y_pred = model.predict(X_test)
    y_score = model.predict_proba(X_test)[:, 1]

    print("\n=== Model evaluation ===")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(f"Precision: {precision_score(y_test, y_pred):.4f}")
    print(f"Recall: {recall_score(y_test, y_pred):.4f}")
    print(f"F1-score: {f1_score(y_test, y_pred):.4f}")
    print(f"ROC AUC: {roc_auc_score(y_test, y_score):.4f}")
    print("\nClassification report:")
    print(classification_report(y_test, y_pred, digits=4))


def main(input_path: str, sample_size: int):
    # Orchestrate the data loading, feature creation, training, and evaluation steps.
    print("=== Predictive Analysis: Taxi Tip Classification ===")
    raw_df = load_data(input_path, sample_size)
    df = build_features(raw_df)

    features = [
        "passenger_count",
        "trip_distance",
        "fare_amount",
        "total_amount",
        "pickup_hour",
        "pickup_dayofweek",
        "VendorID",
        "RatecodeID",
        "store_and_fwd_flag",
        "payment_type",
    ]

    df = df.dropna(subset=features + ["high_tip"])
    X = df[features]
    y = df["high_tip"]

    numeric_features = [
        "passenger_count",
        "trip_distance",
        "fare_amount",
        "total_amount",
        "pickup_hour",
        "pickup_dayofweek",
    ]
    categorical_features = ["VendorID", "RatecodeID", "store_and_fwd_flag", "payment_type"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    pipeline = create_pipeline(numeric_features, categorical_features)
    print("Training model...")
    pipeline.fit(X_train, y_train)

    evaluate_model(pipeline, X_test, y_test)

    print("\nFeature categories used:")
    print(f"Numeric features: {numeric_features}")
    print(f"Categorical features: {categorical_features}")
    print("\nPrediction target: high_tip = tip_rate >= 25%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Task-2: Predictive analysis using taxi trip data")
    parser.add_argument(
        "--input",
        default=DEFAULT_URL,
        help="Path or URL for the taxi trip parquet dataset"
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=10000,
        help="Number of rows to sample for training and evaluation"
    )
    args = parser.parse_args()
    main(args.input, args.sample_size)
