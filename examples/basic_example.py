"""Basic example using the anomaly detection toolkit."""

import numpy as np

from anomaly_detection_toolkit import IsolationForestDetector, VotingEnsemble, ZScoreDetector


def main():
    """Run basic anomaly detection example."""
    print("=" * 70)
    print("ANOMALY DETECTION TOOLKIT - BASIC EXAMPLE")
    print("=" * 70)

    # Generate synthetic data with anomalies
    np.random.seed(42)
    n_samples = 1000
    n_features = 3

    # Normal data
    X_normal = np.random.randn(n_samples, n_features)

    # Inject anomalies
    X_anomalies = X_normal.copy()
    anomaly_indices = [100, 200, 300, 400, 500]
    X_anomalies[anomaly_indices] += 5  # Shift anomalies away from normal cluster

    print(f"\nGenerated {n_samples} samples with {len(anomaly_indices)} injected anomalies")
    print(f"Anomaly indices: {anomaly_indices}\n")

    # Method 1: Z-score detection (univariate)
    print("1. Z-Score Detection (on first feature):")
    print("-" * 70)
    z_detector = ZScoreDetector(n_std=3.0, random_state=42)
    z_detector.fit(X_anomalies[:, 0])
    z_predictions, z_scores = z_detector.fit_predict(X_anomalies[:, 0])
    z_anomalies = np.where(z_predictions == -1)[0]
    print(f"   Detected {len(z_anomalies)} anomalies")
    print(f"   Anomaly indices: {z_anomalies[:10].tolist()}...")

    # Method 2: Isolation Forest (multivariate)
    print("\n2. Isolation Forest Detection (multivariate):")
    print("-" * 70)
    iso_detector = IsolationForestDetector(contamination=0.05, n_estimators=200, random_state=42)
    iso_detector.fit(X_anomalies)
    iso_predictions, iso_scores = iso_detector.fit_predict(X_anomalies)
    iso_anomalies = np.where(iso_predictions == -1)[0]
    print(f"   Detected {len(iso_anomalies)} anomalies")
    print(f"   Anomaly indices: {iso_anomalies[:10].tolist()}...")

    # Method 3: Ensemble
    print("\n3. Ensemble Detection (Voting):")
    print("-" * 70)
    from anomaly_detection_toolkit import LOFDetector, RobustCovarianceDetector

    detectors = [
        IsolationForestDetector(contamination=0.05, random_state=42),
        LOFDetector(contamination=0.05, n_neighbors=20),
        RobustCovarianceDetector(contamination=0.05, random_state=42),
    ]

    ensemble = VotingEnsemble(detectors, voting_threshold=2)
    ensemble.fit(X_anomalies)
    ensemble_predictions, ensemble_scores = ensemble.fit_predict(X_anomalies)
    ensemble_anomalies = np.where(ensemble_predictions == -1)[0]
    print(f"   Detected {len(ensemble_anomalies)} anomalies (2+ detectors agree)")
    print(f"   Anomaly indices: {ensemble_anomalies[:10].tolist()}...")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"True anomalies:      {len(anomaly_indices)} at indices {anomaly_indices}")
    print(f"Z-Score detected:    {len(z_anomalies)} anomalies")
    print(f"Isolation Forest:    {len(iso_anomalies)} anomalies")
    print(f"Ensemble (2+ votes): {len(ensemble_anomalies)} anomalies")

    # Calculate precision (how many detected anomalies match true anomalies)
    true_set = set(anomaly_indices)
    detected_sets = {
        "Z-Score": set(z_anomalies),
        "Isolation Forest": set(iso_anomalies),
        "Ensemble": set(ensemble_anomalies),
    }

    print("\nDetection Precision (overlap with true anomalies):")
    for method, detected in detected_sets.items():
        if detected:
            overlap = len(true_set & detected)
            precision = overlap / len(detected) * 100
            print(f"   {method:20s}: {overlap}/{len(detected)} ({precision:.1f}%)")


if __name__ == "__main__":
    main()
