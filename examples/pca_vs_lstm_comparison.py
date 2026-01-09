"""PCA vs LSTM Comparison for Turbomachinery Anomaly Detection.

This example demonstrates the comparison between PCA-based and LSTM sequence models
for detecting early drift in rotating equipment, as described in the article outline.
"""

from datetime import datetime, timedelta
from typing import Tuple

import numpy as np
import pandas as pd

from anomaly_detection_toolkit.autoencoders import LSTMAutoencoderDetector
from anomaly_detection_toolkit.evaluation import compare_detectors, evaluate_detector
from anomaly_detection_toolkit.pca_detector import PCADetector
from anomaly_detection_toolkit.visualization import (
    plot_comparison_metrics,
    plot_pca_boundary,
    plot_reconstruction_error,
    plot_sensor_drift,
)


def generate_turbomachinery_data(
    n_samples: int = 2000,
    n_sensors: int = 5,
    failure_point: int = 1500,
    drift_rate: float = 0.01,
    noise_level: float = 0.5,
) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Generate synthetic turbomachinery sensor data with gradual drift leading to failure.

    Parameters
    ----------
    n_samples : int, default=2000
        Number of time steps.
    n_sensors : int, default=5
        Number of sensor channels.
    failure_point : int, default=1500
        Time step where failure occurs.
    drift_rate : float, default=0.01
        Rate of sensor drift before failure.
    noise_level : float, default=0.5
        Standard deviation of noise.

    Returns
    -------
    df : DataFrame
        Sensor data with timestamps.
    y_true : ndarray
        True anomaly labels (1 for anomaly, 0 for normal).
    """
    np.random.seed(42)

    # Generate timestamps
    start_time = datetime(2024, 1, 1)
    timestamps = [start_time + timedelta(hours=i) for i in range(n_samples)]

    # Base healthy operation values
    base_values = np.random.uniform(50, 100, n_sensors)

    # Generate sensor readings
    sensor_data = []
    y_true = np.zeros(n_samples)

    for i in range(n_samples):
        # Normal operation
        if i < failure_point - 100:
            # Healthy operation with small variations
            values = base_values + np.random.randn(n_sensors) * noise_level
        # Early drift phase
        elif i < failure_point:
            # Gradual drift
            drift_factor = (i - (failure_point - 100)) / 100.0
            drift = np.random.uniform(-drift_rate, drift_rate, n_sensors) * drift_factor * 10
            values = base_values + drift + np.random.randn(n_sensors) * noise_level * 1.5
            y_true[i] = 1  # Mark as anomaly during drift
        # Failure phase
        else:
            # Significant deviation
            failure_deviation = np.random.uniform(5, 15, n_sensors)
            values = base_values + failure_deviation + np.random.randn(n_sensors) * noise_level * 2
            y_true[i] = 1  # Mark as anomaly

        sensor_data.append(values)

    # Create DataFrame
    sensor_names = [f"sensor_{i+1}" for i in range(n_sensors)]
    df = pd.DataFrame(sensor_data, columns=sensor_names)
    df["timestamp"] = timestamps

    return df, y_true


def main():
    """Run PCA vs LSTM comparison example."""
    print("=" * 70)
    print("PCA vs LSTM COMPARISON FOR TURBOMACHINERY ANOMALY DETECTION")
    print("=" * 70)

    # Step 1: Generate synthetic turbomachinery data
    print("\n1. Generating synthetic turbomachinery sensor data...")
    df, y_true = generate_turbomachinery_data(
        n_samples=2000,
        n_sensors=5,
        failure_point=1500,
        drift_rate=0.01,
    )

    sensor_cols = [col for col in df.columns if col.startswith("sensor_")]
    X = df[sensor_cols].values
    timestamps = df["timestamp"].values

    print(f"   Generated {len(df)} samples with {len(sensor_cols)} sensors")
    print(f"   Anomaly rate: {y_true.sum() / len(y_true):.2%}")

    # Split into train (healthy) and test
    train_size = 1000
    X_train = X[:train_size]
    X_test = X[train_size:]
    y_test = y_true[train_size:]
    timestamps_test = timestamps[train_size:]

    print(f"\n   Training samples (healthy): {len(X_train)}")
    print(f"   Test samples: {len(X_test)}")
    print(f"   Test anomaly rate: {y_test.sum() / len(y_test):.2%}")

    # Step 2: Train PCA detector
    print("\n2. Training PCA-based detector...")
    pca_detector = PCADetector(
        n_components=0.95,
        score_method="reconstruction",
        contamination=0.05,
        random_state=42,
    )
    pca_detector.fit(X_train)
    print(f"   PCA components: {pca_detector.get_n_components()}")
    print(f"   Explained variance: {pca_detector.get_explained_variance_ratio().sum():.2%}")

    # Step 3: Train LSTM detector
    print("\n3. Training LSTM sequence model...")
    try:
        lstm_detector = LSTMAutoencoderDetector(
            window_size=20,
            lstm_units=[32, 16],
            epochs=50,
            batch_size=32,
            threshold_std=3.0,
            random_state=42,
        )

        # Prepare time series for LSTM (flatten multi-sensor data)
        ts_train = X_train.flatten()
        ts_test = X_test.flatten()

        lstm_detector.fit(ts_train)
        print("   LSTM autoencoder trained")
    except ImportError:
        print("   TensorFlow/Keras not available, skipping LSTM")
        lstm_detector = None

    # Step 4: Evaluate detectors
    print("\n4. Evaluating detectors...")

    # Evaluate PCA
    pca_scores = pca_detector.score_samples(X_test)
    pca_metrics = evaluate_detector(
        pca_detector, X_test, y_test, scores=pca_scores, timestamps=np.arange(len(X_test))
    )

    print("\n   PCA Detector Metrics:")
    print(f"     Accuracy:  {pca_metrics['accuracy']:.4f}")
    print(f"     Precision: {pca_metrics['precision']:.4f}")
    print(f"     Recall:   {pca_metrics['recall']:.4f}")
    print(f"     F1 Score: {pca_metrics['f1']:.4f}")
    if "mean_lead_time" in pca_metrics:
        print(f"     Mean Lead Time: {pca_metrics['mean_lead_time']:.2f}")

    # Evaluate LSTM if available
    if lstm_detector is not None:
        lstm_scores = lstm_detector.score_samples(ts_test.reshape(-1, 1))
        # Convert LSTM predictions to match test data shape
        lstm_predictions = lstm_detector.predict(ts_test.reshape(-1, 1))

        # For comparison, we need to align LSTM predictions with test samples
        # LSTM works on windows, so we'll use the last prediction for each sample
        n_windows = len(ts_test) - lstm_detector.window_size + 1
        lstm_scores_aligned = np.zeros(len(X_test))
        lstm_predictions_aligned = np.ones(len(X_test))

        # Map window-based predictions to samples
        for i in range(min(n_windows, len(X_test))):
            idx = min(i + lstm_detector.window_size - 1, len(X_test) - 1)
            lstm_scores_aligned[idx] = lstm_scores[i]
            lstm_predictions_aligned[idx] = lstm_predictions[i]

        # Create a simple wrapper for evaluation
        class LSTMWrapper:
            def predict(self, X):
                return lstm_predictions_aligned

            def score_samples(self, X):
                return lstm_scores_aligned

        lstm_wrapper = LSTMWrapper()
        lstm_metrics = evaluate_detector(
            lstm_wrapper,
            X_test,
            y_test,
            scores=lstm_scores_aligned,
            timestamps=np.arange(len(X_test)),
        )

        print("\n   LSTM Detector Metrics:")
        print(f"     Accuracy:  {lstm_metrics['accuracy']:.4f}")
        print(f"     Precision: {lstm_metrics['precision']:.4f}")
        print(f"     Recall:   {lstm_metrics['recall']:.4f}")
        print(f"     F1 Score: {lstm_metrics['f1']:.4f}")
        if "mean_lead_time" in lstm_metrics:
            print(f"     Mean Lead Time: {lstm_metrics['mean_lead_time']:.2f}")

    # Step 5: Side-by-side comparison
    print("\n5. Creating comparison...")
    detectors = {"PCA": pca_detector}
    if lstm_detector is not None:
        detectors["LSTM"] = lstm_wrapper

    comparison_df = compare_detectors(detectors, X_test, y_test, timestamps=np.arange(len(X_test)))
    print("\n   Comparison Results:")
    print(
        comparison_df[["detector", "accuracy", "precision", "recall", "f1"]].to_string(index=False)
    )

    # Step 6: Create visualizations
    print("\n6. Creating visualizations...")
    try:
        # Sensor drift with anomalies
        plot_sensor_drift(
            X_test[:, 0],  # First sensor
            pca_detector.predict(X_test),
            timestamps=timestamps_test,
            save_path="sensor_drift_pca.png",
        )

        # PCA boundary visualization
        plot_pca_boundary(
            pca_detector,
            X_test,
            y_true=y_test,
            save_path="pca_boundary.png",
        )

        # Reconstruction error over time
        plot_reconstruction_error(
            pca_detector,
            X_test,
            y_true=y_test,
            timestamps=timestamps_test,
            save_path="pca_reconstruction_error.png",
        )

        # Comparison metrics chart
        plot_comparison_metrics(
            comparison_df,
            metrics=["precision", "recall", "f1"],
            save_path="comparison_metrics.png",
        )

        print("   Visualizations saved:")
        print("     - sensor_drift_pca.png")
        print("     - pca_boundary.png")
        print("     - pca_reconstruction_error.png")
        print("     - comparison_metrics.png")

    except ImportError:
        print("   Matplotlib not available, skipping visualizations")

    # Step 7: Integration strategy discussion
    print("\n7. Integration Strategy:")
    print("-" * 70)
    print("   When to use PCA:")
    print("     - Limited training data available")
    print("     - Need fast, interpretable results")
    print("     - Linear relationships in sensor data")
    print("     - Real-time processing requirements")
    print("\n   When to use LSTM:")
    print("     - Rich temporal patterns in data")
    print("     - Sufficient training data available")
    print("     - Nonlinear relationships important")
    print("     - Can tolerate longer processing time")
    print("\n   Ensemble Approach:")
    print("     - Combine both methods for robustness")
    print("     - Use PCA for fast screening, LSTM for confirmation")
    print("     - Reduce false positives through agreement")

    print("\n" + "=" * 70)
    print("Comparison completed successfully!")
    print("=" * 70)

    return {
        "pca_detector": pca_detector,
        "lstm_detector": lstm_detector if lstm_detector is not None else None,
        "comparison_df": comparison_df,
        "pca_metrics": pca_metrics,
        "lstm_metrics": lstm_metrics if lstm_detector is not None else None,
    }


if __name__ == "__main__":
    results = main()
