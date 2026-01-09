"""Real-time Predictive Maintenance Example.

This example demonstrates:
- Real-time data ingestion for streaming sensor data
- Continuous monitoring and processing
- Dashboard visualization
- Alert generation and tracking
"""

from datetime import datetime

import numpy as np
import pandas as pd

from anomaly_detection_toolkit.predictive_maintenance import (
    AlertSystem,
    DashboardVisualizer,
    FailureClassifier,
    FeatureExtractor,
    PredictiveMaintenanceSystem,
    RealTimeIngestion,
    RULEstimator,
)
from anomaly_detection_toolkit.statistical import ZScoreDetector


def simulate_sensor_stream(
    asset_id: str, n_points: int = 200, degradation_rate: float = 0.01
) -> pd.DataFrame:
    """Simulate streaming sensor data with degradation."""
    np.random.seed(42)

    timestamps = pd.date_range(start=datetime.now(), periods=n_points, freq="1min")

    # Simulate degradation: values increase over time
    base_value = 50.0
    trend = np.linspace(0, degradation_rate * n_points, n_points)
    noise = np.random.randn(n_points) * 2

    sensor_values = base_value + trend + noise

    df = pd.DataFrame(
        {"timestamp": timestamps, "sensor_value": sensor_values, "asset_id": asset_id}
    )

    return df


def main():
    """Run real-time predictive maintenance example."""
    print("=" * 70)
    print("REAL-TIME PREDICTIVE MAINTENANCE EXAMPLE")
    print("=" * 70)

    # Step 1: Set up predictive maintenance system
    print("\n1. Setting up predictive maintenance system...")

    # Feature extractor
    feature_extractor = FeatureExtractor(
        rolling_windows=[5, 10, 20],
        frequency_features=False,  # Skip for real-time (computationally expensive)
        change_detection=True,
    )

    # RUL estimator (needs to be trained first)
    rul_estimator = RULEstimator(method="regression", n_estimators=50, random_state=42)

    # Failure classifier (needs to be trained first)
    failure_classifier = FailureClassifier(n_estimators=50, random_state=42)

    # Alert system
    thresholds = {
        "sensor_value": {
            "warning": 60.0,
            "critical": 70.0,
            "failure": 80.0,
        }
    }
    alert_system = AlertSystem(thresholds=thresholds)

    # Anomaly detector
    anomaly_detector = ZScoreDetector(n_std=2.5)

    # Create PM system
    pm_system = PredictiveMaintenanceSystem(
        feature_extractor=feature_extractor,
        rul_estimator=rul_estimator,
        failure_classifier=failure_classifier,
        alert_system=alert_system,
        anomaly_detector=anomaly_detector,
    )

    print("   PM system configured")

    # Step 2: Train models on historical data
    print("\n2. Training models on historical data...")

    # Generate training data
    train_data = simulate_sensor_stream("ASSET_001", n_points=500, degradation_rate=0.005)

    # Extract features
    train_features = feature_extractor.extract(train_data["sensor_value"])

    # Create RUL labels (simplified: assume failure at high values)
    train_rul = np.maximum(0, 100 - (train_data["sensor_value"].values - 50) * 10)
    train_failure = (train_data["sensor_value"].values > 70).astype(int)

    # Fit anomaly detector
    anomaly_detector.fit(train_data["sensor_value"].values.reshape(-1, 1))

    # Train RUL estimator
    rul_estimator.fit(train_features, train_rul)
    print("   RUL estimator trained")

    # Train failure classifier
    failure_classifier.fit(train_features, train_failure)
    print("   Failure classifier trained")

    # Step 3: Set up real-time ingestion
    print("\n3. Setting up real-time ingestion system...")
    ingestion = RealTimeIngestion(
        pm_system=pm_system,
        window_size=50,  # Process every 50 data points
    )
    print("   Real-time ingestion ready")

    # Step 4: Simulate real-time data stream
    print("\n4. Simulating real-time data stream...")
    stream_data = simulate_sensor_stream("ASSET_001", n_points=200, degradation_rate=0.01)

    results_count = 0
    for idx, row in stream_data.iterrows():
        result = ingestion.ingest(
            data=row["sensor_value"],
            asset_id=row["asset_id"],
            timestamp=row["timestamp"],
            sensor_name="sensor_value",
        )

        if "rul" in result:
            results_count += 1
            if results_count % 10 == 0:
                print(f"   Processed {results_count} windows...")
                if result.get("alerts"):
                    print(f"     Alerts: {len(result['alerts'])}")

    print(f"   Total windows processed: {results_count}")

    # Step 5: Get results history
    print("\n5. Retrieving results history...")
    all_assets = ingestion.get_all_assets()
    print(f"   Monitoring {len(all_assets)} asset(s)")

    results_history = {}
    for asset_id in all_assets:
        results_history[asset_id] = ingestion.get_latest_results(asset_id, n=1000)
        print(f"   {asset_id}: {len(results_history[asset_id])} results")

    # Step 6: Create dashboards
    print("\n6. Creating dashboards...")
    visualizer = DashboardVisualizer(figsize=(16, 12))

    # Detailed dashboard
    try:
        visualizer.create_dashboard(
            results_history=results_history,
            save_path="pm_dashboard.png",
        )
        print("   Detailed dashboard created: pm_dashboard.png")
    except ImportError:
        print("   Matplotlib not available, skipping dashboard creation")

    # Summary dashboard
    try:
        visualizer.create_summary_dashboard(
            results_history=results_history,
            save_path="pm_summary_dashboard.png",
        )
        print("   Summary dashboard created: pm_summary_dashboard.png")
    except ImportError:
        print("   Matplotlib not available, skipping summary dashboard")

    # Step 7: Analyze results
    print("\n7. Results Summary:")
    print("-" * 70)

    for asset_id in all_assets:
        results = results_history[asset_id]
        if not results:
            continue

        latest = results[-1]
        print(f"\n   {asset_id}:")
        if latest.get("rul") is not None:
            print(f"     Current RUL: {latest['rul']:.2f}")
        if latest.get("failure_probability") is not None:
            print(f"     Failure Probability: {latest['failure_probability']:.3f}")
        if latest.get("anomaly_score") is not None:
            print(f"     Anomaly Score: {latest['anomaly_score']:.3f}")

        # Count alerts
        total_alerts = sum(len(r.get("alerts", [])) for r in results)
        if total_alerts > 0:
            print(f"     Total Alerts: {total_alerts}")
            # Count by level
            alert_levels = {}
            for r in results:
                for alert in r.get("alerts", []):
                    level = alert.level.value
                    alert_levels[level] = alert_levels.get(level, 0) + 1
            for level, count in alert_levels.items():
                print(f"       {level.upper()}: {count}")

    print("\n" + "=" * 70)
    print("Real-time monitoring example completed!")
    print("=" * 70)

    return {
        "ingestion": ingestion,
        "results_history": results_history,
        "visualizer": visualizer,
    }


if __name__ == "__main__":
    results = main()
