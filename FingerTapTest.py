import argparse
import os
import sys
import pandas as pd
import numpy as np
import pickle
from collections import Counter

from sklearn.metrics import classification_report, accuracy_score


DEFAULT_FEATURES = [
    "Tap Count",
    "Time",
    "Distance (cm)",
    "Speed (cm/s)",
    "StartX",
    "StartY",
]


def extract_start_xy(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure StartX and StartY exist. If only `Start Position` is present,
    attempt to parse it (formats like "(314, 90)" or "314,90")."""
    if "StartX" in df.columns and "StartY" in df.columns:
        return df

    if "Start Position" in df.columns:

        def _parse(pos):
            try:
                s = str(pos).strip().strip('"').strip("'")
                s = s.strip("()")
                parts = [p.strip() for p in s.split(",")]
                if len(parts) >= 2:
                    # allow float coordinates but cast to int
                    return int(float(parts[0])), int(float(parts[1]))
            except Exception:
                pass
            return (np.nan, np.nan)

        xy = df["Start Position"].apply(
            lambda p: pd.Series(_parse(p), index=["StartX", "StartY"])
        )
        df = pd.concat([df.reset_index(drop=True), xy.reset_index(drop=True)], axis=1)
    return df


def load_model(model_path: str):
    if not os.path.exists(model_path):
        print(f"ERROR: Model file not found: {model_path}")
        sys.exit(1)
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    return model


def load_data(input_path: str) -> pd.DataFrame:
    if not os.path.exists(input_path):
        print(f"ERROR: Input file not found: {input_path}")
        sys.exit(1)
    ext = os.path.splitext(input_path)[1].lower()
    if ext in [".xls", ".xlsx"]:
        df = pd.read_excel(input_path)
    else:
        df = pd.read_csv(input_path)
    return df


def infer_expected_features(model, fallback=DEFAULT_FEATURES):
    try:
        feats = list(model.feature_names_in_)
        if not isinstance(feats, list):
            feats = list(feats)
        print(f"Using feature names from model.feature_names_in_: {feats}")
        return feats
    except Exception:
        print(
            f"model.feature_names_in_ not available — falling back to default features: {fallback}"
        )
        return fallback


def coerce_and_impute(df: pd.DataFrame, cols):
    """Convert columns to numeric and fill small number of NaNs with mean."""
    for c in cols:
        if c not in df.columns:
            continue
        df[c] = pd.to_numeric(df[c], errors="coerce")
    # report NaNs
    nan_counts = df[cols].isnull().sum()
    total = len(df)
    for c, n in nan_counts.items():
        if n > 0:
            print(f"Column '{c}' has {n}/{total} missing after coercion.")
    # Impute numeric columns with column mean (only for those present)
    present_cols = [c for c in cols if c in df.columns]
    if present_cols:
        means = df[present_cols].mean()
        df[present_cols] = df[present_cols].fillna(means)
    return df


def main(args):
    model = load_model(args.model)
    df = load_data(args.input)
    orig_columns = df.columns.tolist()

    # Extract StartX/StartY if needed
    df = extract_start_xy(df)

    # Determine expected feature set
    expected_features = infer_expected_features(model)

    # Ensure expected features are present (try to be helpful)
    missing = [f for f in expected_features if f not in df.columns]
    if missing:
        print(f"The following required features are missing from input: {missing}")
        # common attempt: maybe user has 'Distance (pixels)' not 'Distance (cm)'
        if "Distance (cm)" in expected_features and "Distance (pixels)" in df.columns:
            print(
                "Found 'Distance (pixels)'. You may need to convert pixels->cm before predicting."
            )
        print(
            "Please provide a CSV/XLSX with the required columns or re-train the model with feature names matching the input."
        )
        sys.exit(1)

    # Coerce to numeric and impute
    df = coerce_and_impute(df, expected_features)

    # Prepare feature matrix in the same order the model expects
    X = df[expected_features].copy()

    # Predict
    try:
        preds = model.predict(X)
    except Exception as e:
        print("Model prediction failed:", e)
        sys.exit(1)

    df["Predicted_Severity"] = preds

    # Optionally add probabilities
    if args.probs:
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X)
            classes = list(model.classes_)
            # add columns prob_<class>
            for i, cls in enumerate(classes):
                df[f"prob_{cls}"] = proba[:, i]
            df["Predicted_Confidence"] = proba.max(axis=1)
        else:
            print("Model has no predict_proba method — skipping probability columns.")

    # If ground truth present, print evaluation
    if "SeverityLevel" in df.columns:
        try:
            y_true = pd.to_numeric(df["SeverityLevel"], errors="coerce")
            acc = (
                accuracy_score(y_true, df["Predicted_Severity"])
                if y_true.notna().all()
                else None
            )
            if acc is not None:
                print(f"\nEvaluation on provided labels — Accuracy: {acc*100:.2f}%")
                print(
                    classification_report(
                        y_true, df["Predicted_Severity"], zero_division=0
                    )
                )
            else:
                print(
                    "SeverityLevel column exists but contains missing values — skipping evaluation."
                )
        except Exception as e:
            print("Could not compute evaluation metrics:", e)

    # Save output (CSV or XLSX depending on extension)
    out_ext = os.path.splitext(args.output)[1].lower()
    if out_ext in [".xls", ".xlsx"]:
        df.to_excel(args.output, index=False)
    else:
        df.to_csv(args.output, index=False)

    print(f"Predictions saved to: {args.output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Predict SeverityLevel from saved model and input CSV/XLSX"
    )
    parser.add_argument("--model", required=True, help="Path to .pkl model file")
    parser.add_argument("--input", required=True, help="Path to input CSV or XLSX file")
    parser.add_argument(
        "--output", required=True, help="Path to output file (CSV or XLSX)"
    )
    parser.add_argument(
        "--probs",
        action="store_true",
        help="Include predicted class probabilities (if model supports it)",
    )

    args = parser.parse_args()
    main(args)
