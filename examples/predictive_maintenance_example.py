"""Predictive maintenance example.

This example demonstrates how to use the predictive maintenance features
for equipment health monitoring and failure prediction.
"""

import numpy as np
import pandas as pd

from anomaly_detection_toolkit.predictive_maintenance import (
    AlertSystem,
    FailureClassifier,
    FeatureExtractor,
    PredictiveMaintenanceSystem,
    RULEstimator,
)
from anomaly_detection_toolkit.statistical import ZScoreDetector


def generate_degradation_data(n_samples: int = 500, noise_level: float = 0.1) -> pd.DataFrame:
    """Generate synthetic degradation data simulating equipment failure."""
    np.random.seed(42)

    # Simulate degradation: values increase over time until failure
    degradation = np.linspace(0, 1, n_samples) ** 2  # Quadratic degradation
    noise = np.random.randn(n_samples) * noise_level

    # Add some anomalies (sudden spikes)
    anomaly_indices = [100, 200, 300, 400]
    for idx in anomaly_indices:
        degradation[idx : idx + 5] += 0.3

    values = degradation + noise

    # Create DataFrame with timestamp
    dates = pd.date_range("2024-01-01", periods=n_samples, freq="H")
    df = pd.DataFrame({"timestamp": dates, "sensor_value": values})

    return df


def main():
    """Run predictive maintenance example."""
    print("=" * 70)
    print("PREDICTIVE MAINTENANCE EXAMPLE")
    print("=" * 70)

    # Generate synthetic data
    print("\n1. Generating synthetic degradation data...")
    data = generate_degradation_data(n_samples=500)
    print(f"   Generated {len(data)} samples")
    print(f"   Data range: {data['sensor_value'].min():.3f} to {data['sensor_value'].max():.3f}")

    # Split into training and testing
    split_idx = int(len(data) * 0.7)
    train_data = data.iloc[:split_idx]
    test_data = data.iloc[split_idx:]

    print(f"\n   Training samples: {len(train_data)}")
    print(f"   Test samples: {len(test_data)}")

    # Extract features
    print("\n2. Extracting features...")
    feature_extractor = FeatureExtractor(
        rolling_windows=[5, 10, 20, 50],
        frequency_features=True,
        change_detection=True,
    )

    train_features = feature_extractor.extract(train_data["sensor_value"])

    print(f"   Extracted {len(train_features.columns)} features")
    print(f"   Feature examples: {list(train_features.columns[:5])}...")

    # Create RUL labels (time until failure)
    # Assume failure occurs when degradation > 0.9
    failure_threshold = 0.9
    train_rul = np.maximum(0, (failure_threshold - train_data["sensor_value"].values) * 100)

    # Create failure labels (1 if degradation > 0.8, else 0)
    train_failure = (train_data["sensor_value"].values > 0.8).astype(int)

    # Train RUL estimator
    print("\n3. Training RUL estimator...")
    rul_estimator = RULEstimator(method="regression", n_estimators=50, random_state=42)
    rul_estimator.fit(train_features, train_rul)
    print("   RUL estimator trained")

    # Train failure classifier
    print("\n4. Training failure classifier...")
    failure_classifier = FailureClassifier(n_estimators=50, random_state=42)
    failure_classifier.fit(train_features, train_failure)
    print("   Failure classifier trained")

    # Set up alert system
    print("\n5. Setting up alert system...")
    thresholds = {
        "sensor_value_rolling_mean_10": {
            "warning": 0.5,
            "critical": 0.7,
            "failure": 0.9,
        },
        "sensor_value_rolling_std_10": {
            "warning": 0.15,
            "critical": 0.25,
        },
    }

    escalation_rules = {
        "warning": {"min_count": 3},  # Escalate after 3 warnings
        "critical": {"min_count": 2},  # Escalate after 2 critical alerts
    }

    alert_system = AlertSystem(thresholds=thresholds, escalation_rules=escalation_rules)
    print("   Alert system configured")

    # Set up anomaly detector
    print("\n6. Setting up anomaly detector...")
    anomaly_detector = ZScoreDetector(n_std=2.5)
    anomaly_detector.fit(train_data["sensor_value"].values.reshape(-1, 1))
    print("   Anomaly detector trained")

    # Create predictive maintenance system
    print("\n7. Creating predictive maintenance system...")
    pm_system = PredictiveMaintenanceSystem(
        feature_extractor=feature_extractor,
        rul_estimator=rul_estimator,
        failure_classifier=failure_classifier,
        alert_system=alert_system,
        anomaly_detector=anomaly_detector,
    )
    print("   System ready")

    # Process test data
    print("\n8. Processing test data...")
    results_list = []
    for idx, row in test_data.iterrows():
        # Use last 50 points for context
        start_idx = max(0, test_data.index.get_loc(idx) - 50)
        window_data = test_data.iloc[start_idx : test_data.index.get_loc(idx) + 1]["sensor_value"]

        result = pm_system.process(
            window_data,
            timestamp=row["timestamp"],
            asset_id="EQUIPMENT_001",
            return_features=False,
        )
        results_list.append(result)

    # Analyze results
    print("\n9. Results Summary:")
    print("-" * 70)

    rul_predictions = [r["rul"] for r in results_list if r.get("rul") is not None]
    failure_probs = [
        r["failure_probability"] for r in results_list if r.get("failure_probability") is not None
    ]
    all_alerts = [alert for r in results_list for alert in r["alerts"]]

    if rul_predictions:
        print("   RUL Predictions:")
        print(f"     Mean RUL: {np.mean(rul_predictions):.2f} hours")
        print(f"     Min RUL: {np.min(rul_predictions):.2f} hours")
        print(f"     Max RUL: {np.max(rul_predictions):.2f} hours")

    if failure_probs:
        print("\n   Failure Probabilities:")
        print(f"     Mean: {np.mean(failure_probs):.3f}")
        print(f"     Max: {np.max(failure_probs):.3f}")
        high_risk = sum(1 for p in failure_probs if p > 0.7)
        print(f"     High risk samples (>0.7): {high_risk}")

    print(f"\n   Alerts Generated: {len(all_alerts)}")
    if all_alerts:
        alert_levels = {}
        for alert in all_alerts:
            level = alert.level.value
            alert_levels[level] = alert_levels.get(level, 0) + 1

        for level, count in alert_levels.items():
            print(f"     {level.upper()}: {count}")

        # Show recent critical alerts
        critical_alerts = [a for a in all_alerts if a.level.value == "critical"]
        if critical_alerts:
            print("\n   Recent Critical Alerts:")
            for alert in critical_alerts[-5:]:
                print(f"     [{alert.timestamp}] {alert.message}")

    # Anomaly detection results
    anomaly_scores = [
        r["anomaly_score"] for r in results_list if r.get("anomaly_score") is not None
    ]
    anomaly_predictions = [
        r["anomaly_prediction"] for r in results_list if r.get("anomaly_prediction") is not None
    ]

    if anomaly_predictions:
        anomalies_detected = sum(1 for p in anomaly_predictions if p == -1)
        print("\n   Anomaly Detection:")
        print(f"     Anomalies detected: {anomalies_detected}")
        if anomaly_scores:
            print(f"     Mean anomaly score: {np.mean(anomaly_scores):.3f}")

    print("\n" + "=" * 70)
    print("Example completed successfully!")
    print("=" * 70)


if __name__ == "__main__":
    main()
