"""Tests for statistical anomaly detection methods."""

import numpy as np
import pandas as pd

from anomaly_detection_toolkit import IQROutlierDetector, SeasonalBaselineDetector, ZScoreDetector


class TestZScoreDetector:
    """Test Z-score detector."""

    def test_fit_predict(self):
        """Test fit and predict."""
        # Generate data with anomalies
        np.random.seed(42)
        data = np.random.randn(1000)
        data[100:105] += 5  # Inject anomalies

        detector = ZScoreDetector(n_std=3.0)
        detector.fit(data)
        predictions = detector.predict(data)

        assert len(predictions) == len(data)
        assert np.all(np.isin(predictions, [-1, 1]))
        assert (predictions == -1).sum() > 0  # Should detect some anomalies

    def test_score_samples(self):
        """Test score_samples method."""
        data = np.random.randn(100)
        detector = ZScoreDetector(n_std=3.0)
        detector.fit(data)
        scores = detector.score_samples(data)

        assert len(scores) == len(data)
        assert np.all(scores >= 0)

    def test_fit_predict_univariate(self):
        """Test fit_predict method."""
        data = np.random.randn(100)
        detector = ZScoreDetector(n_std=3.0)
        predictions, scores = detector.fit_predict(data)

        assert len(predictions) == len(data)
        assert len(scores) == len(data)


class TestIQROutlierDetector:
    """Test IQR outlier detector."""

    def test_fit_predict(self):
        """Test fit and predict."""
        data = np.random.randn(1000)
        data[100:105] += 10  # Inject outliers

        detector = IQROutlierDetector(factor=1.5)
        detector.fit(data)
        predictions = detector.predict(data)

        assert len(predictions) == len(data)
        assert np.all(np.isin(predictions, [-1, 1]))

    def test_score_samples(self):
        """Test score_samples method."""
        data = np.random.randn(100)
        detector = IQROutlierDetector(factor=1.5)
        detector.fit(data)
        scores = detector.score_samples(data)

        assert len(scores) == len(data)
        assert np.all(np.isin(scores, [0.0, 1.0]))


class TestSeasonalBaselineDetector:
    """Test seasonal baseline detector."""

    def test_fit_predict(self):
        """Test fit and predict with weekly seasonality."""
        dates = pd.date_range("2020-01-01", periods=365, freq="D")
        values = np.sin(2 * np.pi * np.arange(365) / 7) * 10 + 50 + np.random.randn(365) * 2
        values[100:105] += 10  # Inject anomalies

        df = pd.DataFrame({"date": dates, "value": values})

        detector = SeasonalBaselineDetector(seasonality="week", threshold_sigma=2.5)
        detector.fit(df, date_col="date", value_col="value")
        predictions = detector.predict(df, date_col="date", value_col="value")

        assert len(predictions) == len(df)
        assert np.all(np.isin(predictions, [-1, 1]))

    def test_score_samples(self):
        """Test score_samples method."""
        dates = pd.date_range("2020-01-01", periods=100, freq="D")
        values = np.sin(2 * np.pi * np.arange(100) / 7) * 10 + 50

        df = pd.DataFrame({"date": dates, "value": values})

        detector = SeasonalBaselineDetector(seasonality="week")
        detector.fit(df, date_col="date", value_col="value")
        scores = detector.score_samples(df, date_col="date", value_col="value")

        assert len(scores) == len(df)
        assert np.all(scores >= 0)
