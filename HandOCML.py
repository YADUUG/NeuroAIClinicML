import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# 1. Load the Data from CSV
# Replace 'dataset.csv' with your actual file name
df = pd.read_csv("NeuroAIClinicML/HandOC.csv")

# 2. Preprocessing
# Convert 'Timestamp' into numeric features (Hour and Minute)
# This is necessary because Random Forest cannot process raw date strings.
# Adjust the format string ('%d-%m-%Y %H:%M') if your date format is different.
df["Timestamp"] = pd.to_datetime(df["Timestamp"], format="%d-%m-%Y %H:%M")
df["Hour"] = df["Timestamp"].dt.hour
df["Minute"] = df["Timestamp"].dt.minute

# Drop the original timestamp column after extraction
df = df.drop("Timestamp", axis=1)

# 3. Separate Features (X) and Target (y)
X = df.drop("Severity Level", axis=1)
y = df["Severity Level"]

# 4. Split the Data
# 'stratify=y' ensures that the train and test sets have the same proportion
# of class labels as the input dataset (crucial for imbalanced data).
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 5. Initialize the Classifier with Class Balancing
# class_weight='balanced' automatically adjusts weights inversely proportional
# to class frequencies in the input data.
rf_classifier = RandomForestClassifier(
    n_estimators=100,
    class_weight="balanced",  # <--- Balances uneven classes
    random_state=42,
)

# 6. Train the Model
rf_classifier.fit(X_train, y_train)

# 7. Evaluate
y_pred = rf_classifier.predict(X_test)

print("--- Confusion Matrix ---")
print(confusion_matrix(y_test, y_pred))

print("\n--- Classification Report ---")
# zero_division=0 prevents errors if the test set is too small to contain all classes
print(classification_report(y_test, y_pred, zero_division=0))
