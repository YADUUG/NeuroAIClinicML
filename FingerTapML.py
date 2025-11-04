# =============================================================================
# IMPORTS
# =============================================================================
import pandas as pd
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from imblearn.over_sampling import SMOTE
import pickle

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

# --- Impute Missing Values (NaNs) ---
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

# Numeric features: fill with mean
df[existing_features] = df[existing_features].fillna(df[existing_features].mean())

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

# Ensure at least 2 samples per class before SMOTE
class_counts = Counter(y_train)
if any(v < 2 for v in class_counts.values()):
    print("⚠️ Some classes have too few samples for SMOTE. Training may be unstable.")

# Apply SMOTE
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
print(f"Resampled training set distribution: {Counter(y_train_resampled)}\n")

# =============================================================================
# STEP 5: TRAIN THE MODEL
# =============================================================================
model = RandomForestClassifier(
    n_estimators=150, random_state=42, n_jobs=-1, class_weight="balanced"
)

print("--- Training the model on SMOTE-balanced data... ---")
model.fit(X_train_resampled, y_train_resampled)
print("✅ Model training complete!")

# =============================================================================
# STEP 6: EVALUATE THE MODEL
# =============================================================================
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("\n--- Model Evaluation ---")
print(f"Model Accuracy: {accuracy * 100:.2f}%\n")
print("Classification Report:")
print(classification_report(y_test, y_pred, zero_division=0))

# =============================================================================
# STEP 7: SAVE THE FINAL MODEL
# =============================================================================
with open(MODEL_FILENAME, "wb") as file:
    pickle.dump(model, file)

print(f"\n✅ Final model has been saved to '{MODEL_FILENAME}'")
print("--- Pipeline Finished ---")
