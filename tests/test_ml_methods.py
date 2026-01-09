"""Tests for machine learning anomaly detection methods."""

import numpy as np

from anomaly_detection_toolkit import IsolationForestDetector, LOFDetector, RobustCovarianceDetector


class TestIsolationForestDetector:
    """Test Isolation Forest detector."""

    def test_fit_predict(self):
        """Test fit and predict."""
        np.random.seed(42)
        X = np.random.randn(1000, 3)
        X[100:105] += 5  # Inject anomalies

        detector = IsolationForestDetector(contamination=0.05, random_state=42)
        detector.fit(X)
        predictions = detector.predict(X)

        assert len(predictions) == len(X)
        assert np.all(np.isin(predictions, [-1, 1]))

    def test_score_samples(self):
        """Test score_samples method."""
        X = np.random.randn(100, 3)
        detector = IsolationForestDetector(contamination=0.05, random_state=42)
        detector.fit(X)
        scores = detector.score_samples(X)

        assert len(scores) == len(X)


class TestLOFDetector:
    """Test LOF detector."""

    def test_fit_predict(self):
        """Test fit and predict."""
        np.random.seed(42)
        X = np.random.randn(1000, 3)
        X[100:105] += 5  # Inject anomalies

        detector = LOFDetector(contamination=0.05, n_neighbors=20)
        detector.fit(X)
        predictions = detector.predict(X)

        assert len(predictions) == len(X)
        assert np.all(np.isin(predictions, [-1, 1]))

    def test_score_samples(self):
        """Test score_samples method."""
        X = np.random.randn(100, 3)
        detector = LOFDetector(contamination=0.05, n_neighbors=20)
        detector.fit(X)
        scores = detector.score_samples(X)

        assert len(scores) == len(X)


class TestRobustCovarianceDetector:
    """Test Robust Covariance detector."""

    def test_fit_predict(self):
        """Test fit and predict."""
        np.random.seed(42)
        X = np.random.randn(1000, 3)
        X[100:105] += 5  # Inject anomalies

        detector = RobustCovarianceDetector(contamination=0.05, random_state=42)
        detector.fit(X)
        predictions = detector.predict(X)

        assert len(predictions) == len(X)
        assert np.all(np.isin(predictions, [-1, 1]))

    def test_score_samples(self):
        """Test score_samples method."""
        X = np.random.randn(100, 3)
        detector = RobustCovarianceDetector(contamination=0.05, random_state=42)
        detector.fit(X)
        scores = detector.score_samples(X)

        assert len(scores) == len(X)
