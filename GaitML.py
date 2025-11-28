import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline


def train_gait_classifier(file_path, target_column="severity_level"):
    """
    Trains a classifier for gait severity levels (0, 1, 2, 3) handling class imbalance.
    """

    # 1. Load Data
    print(f"Loading data from {file_path}...")
    try:
        # low_memory=False prevents warnings about mixed types during initial read
        df = pd.read_csv(file_path, low_memory=False)
    except FileNotFoundError:
        print("Error: CSV file not found. Please ensure 'gait_data.csv' exists.")
        return

    # Basic Data Checks
    if target_column not in df.columns:
        print(f"Error: Target column '{target_column}' not found in CSV.")
        # Attempt to find the column if capitalization differs
        potential_matches = [
            col
            for col in df.columns
            if col.lower() == target_column.lower().replace("_", " ")
        ]
        if potential_matches:
            print(f"Did you mean '{potential_matches[0]}'? Using that instead.")
            target_column = potential_matches[0]
        else:
            return

    # --- CRITICAL FIX: CLEANING DIRTY DATA ---
    print("Cleaning data...")
    initial_shape = df.shape

    # 1. Clean Target
    # Force target column to numeric, turning non-numbers (like text headers) into NaN
    df[target_column] = pd.to_numeric(df[target_column], errors="coerce")
    # Drop rows where target is NaN (we can't train without a label)
    df = df.dropna(subset=[target_column])
    # Ensure target is integer for classification
    df[target_column] = df[target_column].astype(int)

    print(
        f"Dataset shape: {df.shape} (dropped {initial_shape[0] - df.shape[0]} bad rows)"
    )
    print("Class distribution before resampling:")
    print(df[target_column].value_counts().sort_index())

    # 2. Separate Features and Target
    X = df.drop(columns=[target_column])
    y = df[target_column]

    # 3. Smart Feature Cleaning (Drop bad COLUMNS, not rows)
    # We identify columns that are likely identifiers or timestamps (strings)
    # and drop them if they don't convert to numbers well.

    cols_to_drop = []
    for col in X.columns:
        # Try converting to numeric
        numeric_series = pd.to_numeric(X[col], errors="coerce")

        # If more than 50% of the data becomes NaN after conversion, it's likely a string column (timestamp/ID)
        # invalid for this model
        nan_ratio = numeric_series.isna().mean()

        if nan_ratio > 0.5:
            print(f"Dropping column '{col}' (appears to be non-numeric/text)")
            cols_to_drop.append(col)
        else:
            # Update the column to the numeric version (NaNs will be handled by Imputer later)
            X[col] = numeric_series

    # Drop the identified bad columns
    if cols_to_drop:
        X = X.drop(columns=cols_to_drop)

    print(f"Final shape after cleaning columns: {X.shape}")
    print(f"Features being used: {list(X.columns)}")

    if X.shape[1] == 0:
        print("Error: No valid numeric features found. Check your CSV format.")
        return

    # Split data FIRST
    # We do NOT drop rows with NaN features here. The pipeline's Imputer will handle them.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 3. Build Pipeline
    # We use a Pipeline to ensure scaling and SMOTE happen in the correct order
    # (SMOTE only on training data, Scaling fitted on train applied to test)

    pipeline = ImbPipeline(
        [
            (
                "imputer",
                SimpleImputer(strategy="median"),
            ),  # Handle missing values in features
            ("scaler", StandardScaler()),  # Scale features
            (
                "smote",
                SMOTE(random_state=42, k_neighbors=3),
            ),  # Oversample minority classes
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=100,
                    random_state=42,
                    class_weight="balanced_subsample",  # Extra safety for imbalance
                ),
            ),
        ]
    )

    # 4. Train Model
    print("\nTraining model with SMOTE and Random Forest...")
    pipeline.fit(X_train, y_train)

    # 5. Predictions & Evaluation
    print("Evaluating on Test Set...")
    y_pred = pipeline.predict(X_test)

    # Calculate metrics
    print("\n--- Classification Report ---")
    print(classification_report(y_test, y_pred))

    # Confusion Matrix Visualization
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=[0, 1, 2, 3],
        yticklabels=[0, 1, 2, 3],
    )
    plt.title("Confusion Matrix (Severity Levels)")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

    # Feature Importance (Extracting from pipeline)
    # We need to access the classifier step specifically
    rf_model = pipeline.named_steps["classifier"]
    importances = rf_model.feature_importances_
    feature_names = X.columns

    # Create DataFrame for plotting
    feat_imp_df = pd.DataFrame({"Feature": feature_names, "Importance": importances})
    feat_imp_df = feat_imp_df.sort_values(by="Importance", ascending=False).head(10)

    print("\n--- Top 10 Important Features ---")
    print(feat_imp_df)


if __name__ == "__main__":
    # Ensure you have installed: pip install pandas scikit-learn imbalanced-learn seaborn matplotlib

    # Replace with your actual csv filename
    csv_filename = "NeuroAIClinicML/gait_data.csv"

    # Run the training function
    train_gait_classifier(csv_filename, target_column="severity_level")
