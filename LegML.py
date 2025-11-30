import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LinearRegression
from scipy.stats import variation
import matplotlib.pyplot as plt
import io
import os


class GaitMLAnalyzer:
    def __init__(self, csv_data):
        """
        Initializes the ML Analyzer with the cleaned CSV data.
        """
        # Load data from bytes or file path
        if isinstance(csv_data, bytes):
            self.df = pd.read_csv(io.BytesIO(csv_data))
        else:
            self.df = pd.read_csv(csv_data)

        # Ensure data types are numeric and handle errors
        self.df["Cycle Interval (s)"] = pd.to_numeric(
            self.df["Cycle Interval (s)"], errors="coerce"
        )
        self.df["Timestamp (s)"] = pd.to_numeric(
            self.df["Timestamp (s)"], errors="coerce"
        )

        # Filter out the first event of each leg/session (often 0.000 or noise)
        self.df = self.df[self.df["Cycle Interval (s)"] > 0.01]

    def extract_clinical_features(self):
        """
        FEATURE ENGINEERING:
        Transforms raw time-series data into a single vector of clinical biomarkers.
        Returns a DataFrame where one row = one leg's performance summary.
        """
        features_list = []

        # Group by Leg and Event (e.g., Left Tap, Right Tap)
        groups = self.df.groupby(["Leg", "Event"])

        for (leg, event_type), group in groups:
            intervals = group["Cycle Interval (s)"].values
            timestamps = group["Timestamp (s)"].values

            if len(intervals) < 2:
                continue

            # 1. BASIC STATISTICS
            mean_interval = np.mean(intervals)
            median_interval = np.median(intervals)

            # 2. CADENCE (Taps per Minute)
            # Formula: 60 seconds / average time between taps
            cadence_hz = 1.0 / mean_interval if mean_interval > 0 else 0
            taps_per_minute = cadence_hz * 60

            # 3. ARRHYTHMIA SCORE (Coefficient of Variation)
            # Clinical Standard: CV > 6-7% often indicates motor control issues (e.g., Ataxia/Parkinson's)
            # Formula: Standard Deviation / Mean
            cv_score = variation(intervals) if mean_interval > 0 else 0

            # 4. FATIGUE/BRADYKINESIA SLOPE (Linear Regression)
            # We fit a line to the intervals over time.
            # Positive Slope = Intervals getting longer = Slowing Down (Fatigue)
            # Negative Slope = Speeding up (Hastening/Festination)
            slope = 0
            if len(intervals) > 2:
                X = timestamps.reshape(-1, 1)
                y = intervals.reshape(-1, 1)
                reg = LinearRegression().fit(X, y)
                slope = reg.coef_[0][0]

            # 5. FREEZING OF GAIT (FOG) INDEX
            # Count intervals that are drastically longer (> 2.5x) than the median.
            # These represent sudden stops or blocks in movement.
            freeze_threshold = median_interval * 2.5
            freeze_count = np.sum(intervals > freeze_threshold)

            features_list.append(
                {
                    "Leg": leg,
                    "Event": event_type,
                    "Total_Events": len(intervals),
                    "Mean_Interval_sec": round(mean_interval, 3),
                    "Cadence_TPM": round(taps_per_minute, 1),
                    "Arrhythmia_CV": round(cv_score, 4),  # Critical ML Feature
                    "Fatigue_Slope": round(slope, 5),  # Critical ML Feature
                    "Freezing_Events": int(freeze_count),  # Critical ML Feature
                }
            )

        return pd.DataFrame(features_list)

    def run_anomaly_detection(self):
        """
        UNSUPERVISED LEARNING (Isolation Forest):
        Detects individual taps that are statistical outliers compared to the
        local rhythm of that specific leg. These highlight micro-freezes or spasms.
        """
        results = {}

        groups = self.df.groupby(["Leg", "Event"])

        for (leg, event_type), group in groups:
            # Need enough data points to detect outliers statistically
            if len(group) < 5:
                continue

            # Prepare data: Cycle Interval is the main feature
            X = group[["Cycle Interval (s)"]].values

            # Isolation Forest Model
            # contamination=0.05 implies we expect ~5% of data might be anomalous
            iso_forest = IsolationForest(contamination=0.05, random_state=42)
            preds = iso_forest.fit_predict(X)

            # -1 indicates anomaly, 1 indicates normal
            anomalies = group[preds == -1]

            if not anomalies.empty:
                results[f"{leg} {event_type}"] = anomalies[
                    ["Timestamp (s)", "Cycle Interval (s)", "Velocity"]
                ]

        return results

    def heuristic_clinical_score(self, features_df):
        """
        RULE-BASED CLASSIFIER:
        Assigns a Severity Score (0-4) based on the extracted ML features.
        This acts as an immediate clinical proxy before training a supervised model.
        """
        scores = []

        for index, row in features_df.iterrows():
            score = 0
            reasons = []

            # Rule 1: Rhythm Consistency (CV)
            # Healthy gait CV is usually < 3-4%
            if row["Arrhythmia_CV"] < 0.05:
                reasons.append("Steady Rhythm")
            elif 0.05 <= row["Arrhythmia_CV"] < 0.10:
                score += 1
                reasons.append("Mild Irregularity")
            else:
                score += 2
                reasons.append("Significant Arrhythmia")

            # Rule 2: Fatigue (Slope)
            # If intervals increase by > 10ms per second
            if row["Fatigue_Slope"] > 0.01:
                score += 1
                reasons.append("Detected Fatigue/Slowing")
            elif row["Fatigue_Slope"] < -0.01:
                score += 1
                reasons.append("Detected Hasting/Festination")

            # Rule 3: Freezing
            if row["Freezing_Events"] > 0:
                score += 2
                reasons.append(f"Detected {row['Freezing_Events']} Freezes")

            # Cap score at 4
            score = min(score, 4)

            scores.append(
                {
                    "Leg": row["Leg"],
                    "Predicted_Severity": score,
                    "Primary_Factors": ", ".join(reasons),
                }
            )

        return pd.DataFrame(scores)


# --- Standalone Script Execution ---
def analyze_gait_data(csv_path, output_dir="."):
    """
    Runs the ML analysis on a CSV file, prints the report to console,
    and saves the rhythm charts to disk.
    """
    print(f"--- Processing {csv_path} ---")
    analyzer = GaitMLAnalyzer(csv_path)

    # 1. Extract Features
    features = analyzer.extract_clinical_features()

    if features.empty:
        print("Error: Not enough data events to generate ML report.")
        return

    # 2. Calculate Heuristic Score
    scores = analyzer.heuristic_clinical_score(features)

    # Merge for display
    full_report = pd.merge(features, scores, on="Leg")

    print("\n=== 1. CLINICAL BIOMARKERS ===")
    print(full_report.to_string(index=False))

    # 3. Generate Rhythm Plots (Saved to file)
    print("\n=== 2. GENERATING RHYTHM PLOTS ===")
    legs = features["Leg"].unique()

    for idx, leg in enumerate(legs):
        leg_data = analyzer.df[analyzer.df["Leg"] == leg]

        if not leg_data.empty:
            plt.figure(figsize=(10, 6))

            # Scatter Plot: Time vs Interval
            plt.scatter(
                leg_data["Timestamp (s)"],
                leg_data["Cycle Interval (s)"],
                alpha=0.6,
                color="#1f77b4",
                label="Tap Events",
            )

            # Trendline
            if len(leg_data) > 1:
                z = np.polyfit(
                    leg_data["Timestamp (s)"], leg_data["Cycle Interval (s)"], 1
                )
                p = np.poly1d(z)
                plt.plot(
                    leg_data["Timestamp (s)"],
                    p(leg_data["Timestamp (s)"]),
                    "r--",
                    alpha=0.8,
                    label="Fatigue Trend",
                )

            plt.title(f"{leg} Leg Rhythm Analysis")
            plt.xlabel("Time (s)")
            plt.ylabel("Tap Interval (s)")
            plt.grid(True, alpha=0.3)
            plt.legend()

            # Save plot
            filename = os.path.join(output_dir, f"{leg}_leg_rhythm.png")
            plt.savefig(filename)
            plt.close()
            print(f"Saved plot to: {filename}")

    # 4. Anomaly Report
    print("\n=== 3. ANOMALY DETECTION (Potential Freezing) ===")
    anomalies = analyzer.run_anomaly_detection()

    if anomalies:
        for key, df_anom in anomalies.items():
            print(f"\n[!] Anomalies detected in {key}:")
            print(df_anom.to_string(index=False))
    else:
        print("No statistical anomalies detected.")


if __name__ == "__main__":
    # Example usage: Replace 'data.csv' with your actual file path
    analyze_gait_data("path_to_your_generated_csv.csv")
    pass
