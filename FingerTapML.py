# =============================================================================
# IMPORTS
# =============================================================================
import pandas as pd
from collections import Counter
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    balanced_accuracy_score,
)
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import pickle
import numpy as np

print("--- Starting Model Training Pipeline ---")

# =============================================================================
# CONFIGURATION
# =============================================================================
# The path to your Excel data file.
DATA_FILE_PATH = "NeuroAIClinicML/FTAgam.xlsx"
# The name for the final saved model file.
MODEL_FILENAME = "final_severity_predictor.pkl"

# =============================================================================
# STEP 1: LOAD DATA
# =============================================================================
try:
    df = pd.read_excel(DATA_FILE_PATH)
    print(f"✅ Data successfully loaded from '{DATA_FILE_PATH}'.")
except FileNotFoundError:
    print(
        f"❌ ERROR: The file '{DATA_FILE_PATH}' was not found. Please check the file name."
    )
    exit()

# =============================================================================
# STEP 2: PREPARE AND CLEAN DATA
# =============================================================================
# --- Feature Engineering: Create StartX and StartY ---
if "Start Position" in df.columns:
    df["StartX"] = df["Start Position"].apply(
        lambda pos: int(str(pos).strip("()").split(",")[0])
    )
    df["StartY"] = df["Start Position"].apply(
        lambda pos: int(str(pos).strip("()").split(",")[1])
    )
    print("✅ 'StartX' and 'StartY' columns created.")

# --- Verify Classes ---
if "SeverityLevel" in df.columns:
    print(f"✅ Found severity classes: {sorted(df['SeverityLevel'].unique())}")

# --- Impute Missing Values (NaNs) with Per-Class Means ---
features_to_clean = [
    "Tap Count",
    "Time",
    "Distance (cm)",
    "Speed (cm/s)",
    "StartX",
    "StartY",
]
existing_features = [col for col in features_to_clean if col in df.columns]

print("\n--- Data Cleaning ---")
print("Missing values BEFORE imputation:")
print(df[existing_features].isnull().sum())

# Impute missing values using per-class mean (more representative for imbalanced data)
for feature in existing_features:
    if df[feature].isnull().sum() > 0:
        class_means = df.groupby("SeverityLevel")[feature].mean()
        df[feature] = df.groupby("SeverityLevel")[feature].transform(
            lambda x: x.fillna(
                class_means[x.name]
                if x.name in class_means.index
                else df[feature].mean()
            )
        )
        # Fallback: fill remaining NaNs with global mean
        df[feature].fillna(df[feature].mean(), inplace=True)

print("\nMissing values AFTER imputation:")
print(df[existing_features].isnull().sum())
print("-" * 25)

# =============================================================================
# STEP 3: DEFINE FEATURES (X) AND TARGET (y)
# =============================================================================
feature_columns = [
    "Tap Count",
    "Time",
    "Distance (cm)",
    "Speed (cm/s)",
    "StartX",
    "StartY",
]
target_column = "SeverityLevel"

X = df[feature_columns]
y = df[target_column]

# =============================================================================
# STEP 4: SPLIT DATA AND APPLY SMOTE
# =============================================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

print(f"\nOriginal training set distribution: {Counter(y_train)}")

# Scale features for better model performance
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Ensure at least 2 samples per class before SMOTE
class_counts = Counter(y_train)
if any(v < 2 for v in class_counts.values()):
    print("⚠️ Some classes have too few samples for SMOTE. Training may be unstable.")

# Apply SMOTE with controlled sampling strategy (oversample minority classes to 80% of majority)
smote = SMOTE(sampling_strategy={3: 50, 4: 50}, random_state=42, k_neighbors=3)
try:
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)
    print(f"Resampled training set distribution: {Counter(y_train_resampled)}\n")
except Exception as e:
    print(f"⚠️ SMOTE failed: {e}. Using original training set.\n")
    X_train_resampled, y_train_resampled = X_train_scaled, y_train

# =============================================================================
# STEP 5: TRAIN THE MODEL
# =============================================================================
model = RandomForestClassifier(
    n_estimators=500,
    max_depth=15,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1,
    class_weight="balanced_subsample",
)

print("--- Training the model on SMOTE-balanced and scaled data... ---")
model.fit(X_train_resampled, y_train_resampled)
print("✅ Model training complete!")

# Perform cross-validation to evaluate robustness
cv_scores = cross_val_score(
    model, X_train_resampled, y_train_resampled, cv=5, scoring="balanced_accuracy"
)
print(
    f"Cross-validation Balanced Accuracy (mean ± std): {cv_scores.mean():.4f} ± {cv_scores.std():.4f}"
)

# =============================================================================
# STEP 6: EVALUATE THE MODEL
# =============================================================================
y_pred = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
balanced_accuracy = balanced_accuracy_score(y_test, y_pred)

print("\n--- Model Evaluation ---")
print(f"Model Accuracy: {accuracy * 100:.2f}%")
print(f"Balanced Accuracy (per-class average): {balanced_accuracy * 100:.2f}%\n")
print("Classification Report:")
print(classification_report(y_test, y_pred, zero_division=0))

# Feature importance
print("\n--- Feature Importance ---")
feature_importance = pd.DataFrame(
    {"Feature": feature_columns, "Importance": model.feature_importances_}
).sort_values("Importance", ascending=False)
print(feature_importance)

# =============================================================================
# STEP 7: SAVE THE FINAL MODEL AND SCALER
# =============================================================================
with open(MODEL_FILENAME, "wb") as file:
    pickle.dump(model, file)

# Save the scaler for use during prediction
scaler_filename = "feature_scaler.pkl"
with open(scaler_filename, "wb") as file:
    pickle.dump(scaler, file)

print(f"\n✅ Final model has been saved to '{MODEL_FILENAME}'")
print(f"✅ Feature scaler has been saved to '{scaler_filename}'")
print("--- Pipeline Finished ---")
