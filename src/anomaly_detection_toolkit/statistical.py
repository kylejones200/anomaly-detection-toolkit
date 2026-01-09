"""Statistical methods for anomaly detection."""

from typing import Optional, Union

import numpy as np
import pandas as pd

from .base import BaseDetector


class StatisticalDetector(BaseDetector):
    """Base class for statistical anomaly detectors."""

    def __init__(self, random_state: Optional[int] = None):
        """Initialize the statistical detector."""
        super().__init__(random_state)
        self.threshold: Optional[float] = None

    def fit(self, X: Union[np.ndarray, pd.DataFrame, pd.Series]):
        """Fit the detector (no training needed for statistical methods)."""
        pass

    def predict(self, X: Union[np.ndarray, pd.DataFrame, pd.Series]) -> np.ndarray:
        """Predict anomalies."""
        if self.threshold is None:
            raise ValueError("Detector must be fitted before prediction.")
        scores = self.score_samples(X)
        predictions = np.where(scores > self.threshold, -1, 1)
        return predictions

    def score_samples(self, X: Union[np.ndarray, pd.DataFrame, pd.Series]) -> np.ndarray:
        """Compute anomaly scores."""
        raise NotImplementedError


class ZScoreDetector(StatisticalDetector):
    """
    Z-score based anomaly detector.

    Flags points that are more than n_std standard deviations from the mean.

    Parameters
    ----------
    n_std : float, default=3.0
        Number of standard deviations for threshold.
    random_state : int, optional
        Random state for reproducibility.
    """

    def __init__(self, n_std: float = 3.0, random_state: Optional[int] = None):
        super().__init__(random_state)
        self.n_std = n_std
        self.mean_ = None
        self.std_ = None
        self.threshold = None

    def fit(self, X: Union[np.ndarray, pd.DataFrame, pd.Series]):
        """
        Fit the detector by computing mean and standard deviation.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        """
        X = self._validate_input(X)
        self.mean_ = np.mean(X, axis=0)
        self.std_ = np.std(X, axis=0)
        # Avoid division by zero
        self.std_ = np.where(self.std_ == 0, 1.0, self.std_)
        self.threshold = self.n_std

    def score_samples(self, X: Union[np.ndarray, pd.DataFrame, pd.Series]) -> np.ndarray:
        """
        Compute Z-scores for samples.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Data to score.

        Returns
        -------
        scores : ndarray of shape (n_samples,)
            Maximum absolute Z-score across features for each sample.
        """
        if self.mean_ is None or self.std_ is None:
            raise ValueError("Detector must be fitted before scoring.")

        X = self._validate_input(X)
        z_scores = np.abs((X - self.mean_) / self.std_)
        # Return maximum Z-score across features
        if z_scores.ndim > 1:
            return np.max(z_scores, axis=1)
        return z_scores.flatten()


class IQROutlierDetector(StatisticalDetector):
    """
    Interquartile Range (IQR) based outlier detector.

    Flags points outside Q1 - 1.5*IQR and Q3 + 1.5*IQR.

    Parameters
    ----------
    factor : float, default=1.5
        IQR multiplier for outlier threshold.
    random_state : int, optional
        Random state for reproducibility.
    """

    def __init__(self, factor: float = 1.5, random_state: Optional[int] = None):
        super().__init__(random_state)
        self.factor = factor
        self.q1_ = None
        self.q3_ = None
        self.iqr_ = None

    def fit(self, X: Union[np.ndarray, pd.DataFrame, pd.Series]):
        """
        Fit the detector by computing quartiles.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        """
        X = self._validate_input(X)
        self.q1_ = np.percentile(X, 25, axis=0)
        self.q3_ = np.percentile(X, 75, axis=0)
        self.iqr_ = self.q3_ - self.q1_
        # Avoid division by zero
        self.iqr_ = np.where(self.iqr_ == 0, 1.0, self.iqr_)
        self.threshold = 1.0

    def score_samples(self, X: Union[np.ndarray, pd.DataFrame, pd.Series]) -> np.ndarray:
        """
        Compute IQR-based scores for samples.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Data to score.

        Returns
        -------
        scores : ndarray of shape (n_samples,)
            Outlier scores. 1.0 indicates outlier, 0.0 indicates normal.
        """
        if self.q1_ is None or self.q3_ is None:
            raise ValueError("Detector must be fitted before scoring.")

        X = self._validate_input(X)

        # Check if outside bounds for any feature
        lower_bound = self.q1_ - self.factor * self.iqr_
        upper_bound = self.q3_ + self.factor * self.iqr_

        outlier_mask = (X < lower_bound) | (X > upper_bound)
        if outlier_mask.ndim > 1:
            scores = np.any(outlier_mask, axis=1).astype(float)
        else:
            scores = outlier_mask.astype(float)

        return scores

    def predict(self, X: Union[np.ndarray, pd.DataFrame, pd.Series]) -> np.ndarray:
        """Predict anomalies."""
        scores = self.score_samples(X)
        predictions = np.where(scores >= self.threshold, -1, 1)
        return predictions


class SeasonalBaselineDetector(BaseDetector):
    """
    Seasonal baseline anomaly detector for time series.

    Calculates seasonal baselines (e.g., weekly, monthly) and flags
    points that deviate significantly from expected seasonal patterns.

    Parameters
    ----------
    seasonality : str, default='week'
        Seasonality to use. Options: 'week', 'month', 'day', 'hour'.
    threshold_sigma : float, default=2.5
        Number of standard deviations for threshold.
    random_state : int, optional
        Random state for reproducibility.
    """

    def __init__(
        self,
        seasonality: str = "week",
        threshold_sigma: float = 2.5,
        random_state: Optional[int] = None,
    ):
        super().__init__(random_state)
        self.seasonality = seasonality
        self.threshold_sigma = threshold_sigma
        self.baseline_ = None
        self.seasonal_stats_ = None

    def _get_seasonal_key(self, dates: pd.Series) -> pd.Series:
        """Extract seasonal key from dates based on seasonality type."""
        seasonality_map = {
            "week": lambda d: d.dt.isocalendar().week,
            "month": lambda d: d.dt.month,
            "day": lambda d: d.dt.dayofyear,
            "hour": lambda d: d.dt.hour,
        }
        if self.seasonality not in seasonality_map:
            raise ValueError(f"Unknown seasonality: {self.seasonality}")
        return seasonality_map[self.seasonality](dates)

    def fit(self, data: pd.DataFrame, date_col: str = "date", value_col: str = "value"):
        """
        Fit the detector by computing seasonal baselines.

        Parameters
        ----------
        data : pd.DataFrame
            DataFrame with date and value columns.
        date_col : str, default='date'
            Name of the date column.
        value_col : str, default='value'
            Name of the value column.
        """
        df = data.copy()
        df[date_col] = pd.to_datetime(df[date_col])
        df["seasonal_key"] = self._get_seasonal_key(df[date_col])

        # Compute seasonal statistics
        seasonal_stats = df.groupby("seasonal_key").agg({value_col: ["mean", "std"]}).reset_index()
        seasonal_stats.columns = ["seasonal_key", "mean", "std"]
        seasonal_stats["std"] = seasonal_stats["std"].fillna(0)
        seasonal_stats["std"] = seasonal_stats["std"].replace(0, 1.0)
        self.seasonal_stats_ = seasonal_stats

    def predict(
        self, data: pd.DataFrame, date_col: str = "date", value_col: str = "value"
    ) -> np.ndarray:
        """
        Predict anomalies in the data.

        Parameters
        ----------
        data : pd.DataFrame
            DataFrame with date and value columns.
        date_col : str, default='date'
            Name of the date column.
        value_col : str, default='value'
            Name of the value column.

        Returns
        -------
        predictions : ndarray of shape (n_samples,)
            Anomaly predictions. -1 for anomalies, 1 for normal.
        """
        scores = self.score_samples(data, date_col, value_col)
        predictions = np.where(scores > self.threshold_sigma, -1, 1)
        return predictions

    def score_samples(
        self, data: pd.DataFrame, date_col: str = "date", value_col: str = "value"
    ) -> np.ndarray:
        """
        Compute Z-scores relative to seasonal baseline.

        Parameters
        ----------
        data : pd.DataFrame
            DataFrame with date and value columns.
        date_col : str, default='date'
            Name of the date column.
        value_col : str, default='value'
            Name of the value column.

        Returns
        -------
        scores : ndarray of shape (n_samples,)
            Z-scores relative to seasonal baseline.
        """
        if self.seasonal_stats_ is None:
            raise ValueError("Detector must be fitted before scoring.")

        df = data.copy()
        df[date_col] = pd.to_datetime(df[date_col])
        df["seasonal_key"] = self._get_seasonal_key(df[date_col])

        # Merge with seasonal stats - vectorized pandas operation
        if self.seasonal_stats_ is None:
            raise ValueError("Detector must be fitted before scoring.")
        df = df.merge(self.seasonal_stats_, on="seasonal_key", how="left")

        # Compute Z-scores - vectorized
        z_scores = np.abs((df[value_col] - df["mean"]) / df["std"])
        return z_scores.fillna(0).values
