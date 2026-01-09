"""Tests for PCA-based anomaly detector."""

import numpy as np
import pandas as pd
import pytest

from anomaly_detection_toolkit.pca_detector import PCADetector


class TestPCADetector:
    """Tests for PCADetector."""

    def test_fit_predict(self):
        """Test PCA detector fit and predict."""
        detector = PCADetector(n_components=0.95, random_state=42)
        n_samples = 200
        n_features = 5

        # Generate training data
        X_train = np.random.randn(n_samples, n_features)

        # Generate test data with some anomalies
        X_test = np.random.randn(50, n_features)
        X_test[10:15] += 5  # Inject anomalies

        detector.fit(X_train)
        predictions = detector.predict(X_test)

        assert len(predictions) == len(X_test)
        assert all(p in [-1, 1] for p in predictions)
        assert np.any(predictions == -1)  # Should detect some anomalies

    def test_score_samples_reconstruction(self):
        """Test reconstruction error scoring."""
        detector = PCADetector(score_method="reconstruction", random_state=42)
        X = np.random.randn(100, 5)
        detector.fit(X)

        scores = detector.score_samples(X)
        assert len(scores) == len(X)
        assert all(s >= 0 for s in scores)  # Reconstruction error is non-negative

    def test_score_samples_mahalanobis(self):
        """Test Mahalanobis distance scoring."""
        detector = PCADetector(score_method="mahalanobis", random_state=42)
        X = np.random.randn(100, 5)
        detector.fit(X)

        scores = detector.score_samples(X)
        assert len(scores) == len(X)
        assert all(s >= 0 for s in scores)  # Distance is non-negative

    def test_score_samples_both(self):
        """Test combined scoring method."""
        detector = PCADetector(score_method="both", random_state=42)
        X = np.random.randn(100, 5)
        detector.fit(X)

        scores = detector.score_samples(X)
        assert len(scores) == len(X)
        assert all(0 <= s <= 1 for s in scores)  # Normalized between 0 and 1

    def test_n_components_float(self):
        """Test PCA with float n_components (variance ratio)."""
        detector = PCADetector(n_components=0.9, random_state=42)
        X = np.random.randn(100, 10)
        detector.fit(X)

        n_components = detector.get_n_components()
        assert n_components <= 10
        assert n_components > 0

    def test_n_components_int(self):
        """Test PCA with integer n_components."""
        detector = PCADetector(n_components=3, random_state=42)
        X = np.random.randn(100, 10)
        detector.fit(X)

        n_components = detector.get_n_components()
        assert n_components == 3

    def test_explained_variance_ratio(self):
        """Test explained variance ratio."""
        detector = PCADetector(n_components=0.95, random_state=42)
        X = np.random.randn(100, 5)
        detector.fit(X)

        variance_ratio = detector.get_explained_variance_ratio()
        assert len(variance_ratio) == detector.get_n_components()
        assert all(0 <= v <= 1 for v in variance_ratio)
        assert variance_ratio.sum() >= 0.95  # Should explain at least 95% variance

    def test_transform(self):
        """Test transform to PC space."""
        detector = PCADetector(n_components=3, random_state=42)
        X = np.random.randn(100, 10)
        detector.fit(X)

        X_transformed = detector.transform(X)
        assert X_transformed.shape == (100, 3)

    def test_dataframe_input(self):
        """Test with DataFrame input."""
        detector = PCADetector(random_state=42)
        df = pd.DataFrame(np.random.randn(100, 5), columns=[f"feature_{i}" for i in range(5)])
        detector.fit(df)

        predictions = detector.predict(df)
        assert len(predictions) == len(df)

    def test_series_input(self):
        """Test with Series input."""
        detector = PCADetector(random_state=42)
        series = pd.Series(np.random.randn(100))
        detector.fit(series)

        predictions = detector.predict(series)
        assert len(predictions) == len(series)

    def test_contamination_threshold(self):
        """Test that contamination parameter affects threshold."""
        detector1 = PCADetector(contamination=0.1, random_state=42)
        detector2 = PCADetector(contamination=0.2, random_state=42)

        X = np.random.randn(100, 5)
        detector1.fit(X)
        detector2.fit(X)

        # Higher contamination should result in lower threshold (more anomalies detected)
        assert detector1.threshold_ >= detector2.threshold_

    def test_invalid_score_method(self):
        """Test error for invalid score method."""
        detector = PCADetector(score_method="invalid", random_state=42)
        X = np.random.randn(100, 5)

        with pytest.raises(ValueError, match="Unknown score_method"):
            detector.fit(X)
