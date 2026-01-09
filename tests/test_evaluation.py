"""Tests for evaluation utilities."""

import numpy as np
import pandas as pd

from anomaly_detection_toolkit.evaluation import (
    calculate_confusion_matrix_metrics,
    calculate_lead_time,
    compare_detectors,
    evaluate_detector,
)
from anomaly_detection_toolkit.pca_detector import PCADetector
from anomaly_detection_toolkit.statistical import ZScoreDetector


class TestEvaluation:
    """Tests for evaluation utilities."""

    def test_evaluate_detector(self):
        """Test detector evaluation."""
        detector = ZScoreDetector(n_std=2.0, random_state=42)
        X = np.random.randn(100, 1)
        y_true = np.zeros(100)
        y_true[10:15] = 1  # Inject anomalies

        detector.fit(X)
        metrics = evaluate_detector(detector, X, y_true)

        assert "accuracy" in metrics
        assert "precision" in metrics
        assert "recall" in metrics
        assert "f1" in metrics
        assert all(
            0 <= v <= 1
            for v in [metrics["accuracy"], metrics["precision"], metrics["recall"], metrics["f1"]]
        )

    def test_evaluate_detector_with_scores(self):
        """Test evaluation with pre-computed scores."""
        detector = ZScoreDetector(n_std=2.0, random_state=42)
        X = np.random.randn(100, 1)
        y_true = np.zeros(100)
        y_true[10:15] = 1

        detector.fit(X)
        scores = detector.score_samples(X)
        metrics = evaluate_detector(detector, X, y_true, scores=scores)

        assert "roc_auc" in metrics
        assert 0 <= metrics["roc_auc"] <= 1

    def test_calculate_lead_time(self):
        """Test lead time calculation."""
        # Create scenario: anomalies detected at indices 5, 10, 15
        # True events at indices 8, 12, 18
        predictions = np.ones(20)
        predictions[[5, 10, 15]] = -1  # Detections

        y_true = np.zeros(20)
        y_true[[8, 12, 18]] = 1  # Events

        timestamps = np.arange(20)
        metrics = calculate_lead_time(predictions, y_true, timestamps)

        assert "mean_lead_time" in metrics
        assert "median_lead_time" in metrics
        assert "early_detections" in metrics
        assert "late_detections" in metrics

    def test_calculate_lead_time_no_events(self):
        """Test lead time with no events."""
        predictions = np.ones(100)
        y_true = np.zeros(100)

        metrics = calculate_lead_time(predictions, y_true)

        assert metrics["mean_lead_time"] == 0.0
        assert metrics["early_detections"] == 0

    def test_calculate_confusion_matrix_metrics(self):
        """Test confusion matrix calculation."""
        predictions = np.array([1, 1, -1, -1, 1, -1])
        y_true = np.array([1, 0, 1, 0, 0, 1])

        metrics = calculate_confusion_matrix_metrics(predictions, y_true)

        assert "true_positives" in metrics
        assert "true_negatives" in metrics
        assert "false_positives" in metrics
        assert "false_negatives" in metrics
        assert all(v >= 0 for v in metrics.values())

    def test_compare_detectors(self):
        """Test detector comparison."""
        X = np.random.randn(100, 5)
        y_true = np.zeros(100)
        y_true[10:20] = 1  # Inject anomalies

        detectors = {
            "PCA": PCADetector(n_components=0.95, random_state=42),
            "ZScore": ZScoreDetector(n_std=2.0, random_state=42),
        }

        # Fit detectors
        detectors["PCA"].fit(X)
        detectors["ZScore"].fit(X[:, 0])

        # Compare (using first feature for ZScore)
        comparison_df = compare_detectors(detectors, X, y_true)

        assert isinstance(comparison_df, pd.DataFrame)
        assert len(comparison_df) == len(detectors)
        assert "detector" in comparison_df.columns
        assert "accuracy" in comparison_df.columns
        assert "precision" in comparison_df.columns
        assert "recall" in comparison_df.columns
        assert "f1" in comparison_df.columns
