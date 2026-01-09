"""Machine learning methods for anomaly detection."""

from typing import Optional, Union

import numpy as np
import pandas as pd
from sklearn.covariance import EllipticEnvelope
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler

from .base import BaseDetector


class IsolationForestDetector(BaseDetector):
    """
    Isolation Forest anomaly detector.

    Isolation Forest is an ensemble method that isolates anomalies
    by randomly selecting features and splitting values.

    Parameters
    ----------
    contamination : float, default=0.05
        Expected proportion of outliers in the data.
    n_estimators : int, default=200
        Number of base estimators.
    random_state : int, optional
        Random state for reproducibility.
    n_jobs : int, default=-1
        Number of jobs to run in parallel.
    """

    def __init__(
        self,
        contamination: float = 0.05,
        n_estimators: int = 200,
        random_state: Optional[int] = None,
        n_jobs: int = -1,
    ):
        super().__init__(random_state)
        self.contamination = contamination
        self.n_estimators = n_estimators
        self.n_jobs = n_jobs
        self.model_ = None
        self.scaler_ = StandardScaler()

    def fit(self, X: Union[np.ndarray, pd.DataFrame, pd.Series]):
        """
        Fit the Isolation Forest detector.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        """
        X = self._validate_input(X)
        X_scaled = self.scaler_.fit_transform(X)

        self.model_ = IsolationForest(
            contamination=self.contamination,
            n_estimators=self.n_estimators,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
        )
        self.model_.fit(X_scaled)

    def predict(self, X: Union[np.ndarray, pd.DataFrame, pd.Series]) -> np.ndarray:
        """
        Predict anomalies in the data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Data to predict.

        Returns
        -------
        predictions : ndarray of shape (n_samples,)
            Anomaly predictions. -1 for anomalies, 1 for normal.
        """
        if self.model_ is None:
            raise ValueError("Detector must be fitted before prediction.")

        X = self._validate_input(X)
        X_scaled = self.scaler_.transform(X)
        return self.model_.predict(X_scaled)

    def score_samples(self, X: Union[np.ndarray, pd.DataFrame, pd.Series]) -> np.ndarray:
        """
        Compute anomaly scores (negative decision function).

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Data to score.

        Returns
        -------
        scores : ndarray of shape (n_samples,)
            Anomaly scores. Higher values indicate more anomalous samples.
        """
        if self.model_ is None:
            raise ValueError("Detector must be fitted before scoring.")

        X = self._validate_input(X)
        X_scaled = self.scaler_.transform(X)
        scores = self.model_.decision_function(X_scaled)
        # Invert so higher scores = more anomalous
        return -scores


class LOFDetector(BaseDetector):
    """
    Local Outlier Factor (LOF) anomaly detector.

    LOF measures the local deviation of density of a given sample
    with respect to its neighbors.

    Parameters
    ----------
    contamination : float, default=0.05
        Expected proportion of outliers in the data.
    n_neighbors : int, default=20
        Number of neighbors to use.
    random_state : int, optional
        Random state for reproducibility.
    n_jobs : int, default=-1
        Number of jobs to run in parallel.
    """

    def __init__(
        self,
        contamination: float = 0.05,
        n_neighbors: int = 20,
        random_state: Optional[int] = None,
        n_jobs: int = -1,
    ):
        super().__init__(random_state)
        self.contamination = contamination
        self.n_neighbors = n_neighbors
        self.n_jobs = n_jobs
        self.model_ = None
        self.scaler_ = StandardScaler()

    def fit(self, X: Union[np.ndarray, pd.DataFrame, pd.Series]):
        """
        Fit the LOF detector.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        """
        X = self._validate_input(X)
        X_scaled = self.scaler_.fit_transform(X)

        # Set novelty=True to enable predict() method for new data
        self.model_ = LocalOutlierFactor(
            contamination=self.contamination,
            n_neighbors=self.n_neighbors,
            n_jobs=self.n_jobs,
            novelty=True,
        )
        self.model_.fit(X_scaled)

    def predict(self, X: Union[np.ndarray, pd.DataFrame, pd.Series]) -> np.ndarray:
        """
        Predict anomalies in the data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Data to predict.

        Returns
        -------
        predictions : ndarray of shape (n_samples,)
            Anomaly predictions. -1 for anomalies, 1 for normal.
        """
        if self.model_ is None:
            raise ValueError("Detector must be fitted before prediction.")

        X = self._validate_input(X)
        X_scaled = self.scaler_.transform(X)
        return self.model_.predict(X_scaled)

    def score_samples(self, X: Union[np.ndarray, pd.DataFrame, pd.Series]) -> np.ndarray:
        """
        Compute LOF scores (negative outlier factor).

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Data to score.

        Returns
        -------
        scores : ndarray of shape (n_samples,)
            Anomaly scores. Higher values indicate more anomalous samples.
        """
        if self.model_ is None:
            raise ValueError("Detector must be fitted before scoring.")

        X = self._validate_input(X)
        X_scaled = self.scaler_.transform(X)

        # LOF model must be refitted for scoring new data
        model = LocalOutlierFactor(
            contamination=self.contamination, n_neighbors=self.n_neighbors, n_jobs=self.n_jobs
        )
        model.fit(X_scaled)
        scores = model.negative_outlier_factor_
        # Invert so higher scores = more anomalous
        return -scores


class RobustCovarianceDetector(BaseDetector):
    """
    Robust Covariance (Elliptic Envelope) anomaly detector.

    Assumes that the data is Gaussian distributed and fits an
    elliptic envelope to the data.

    Parameters
    ----------
    contamination : float, default=0.05
        Expected proportion of outliers in the data.
    support_fraction : float, default=0.8
        Proportion of points to be used as support.
    random_state : int, optional
        Random state for reproducibility.
    """

    def __init__(
        self,
        contamination: float = 0.05,
        support_fraction: float = 0.8,
        random_state: Optional[int] = None,
    ):
        super().__init__(random_state)
        self.contamination = contamination
        self.support_fraction = support_fraction
        self.model_ = None
        self.scaler_ = StandardScaler()

    def fit(self, X: Union[np.ndarray, pd.DataFrame, pd.Series]):
        """
        Fit the Robust Covariance detector.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        """
        X = self._validate_input(X)
        X_scaled = self.scaler_.fit_transform(X)

        self.model_ = EllipticEnvelope(
            contamination=self.contamination,
            support_fraction=self.support_fraction,
            random_state=self.random_state,
        )
        self.model_.fit(X_scaled)

    def predict(self, X: Union[np.ndarray, pd.DataFrame, pd.Series]) -> np.ndarray:
        """
        Predict anomalies in the data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Data to predict.

        Returns
        -------
        predictions : ndarray of shape (n_samples,)
            Anomaly predictions. -1 for anomalies, 1 for normal.
        """
        if self.model_ is None:
            raise ValueError("Detector must be fitted before prediction.")

        X = self._validate_input(X)
        X_scaled = self.scaler_.transform(X)
        return self.model_.predict(X_scaled)

    def score_samples(self, X: Union[np.ndarray, pd.DataFrame, pd.Series]) -> np.ndarray:
        """
        Compute Mahalanobis distances (negative so higher = more anomalous).

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Data to score.

        Returns
        -------
        scores : ndarray of shape (n_samples,)
            Anomaly scores. Higher values indicate more anomalous samples.
        """
        if self.model_ is None:
            raise ValueError("Detector must be fitted before scoring.")

        X = self._validate_input(X)
        X_scaled = self.scaler_.transform(X)
        scores = self.model_.decision_function(X_scaled)
        # Invert so higher scores = more anomalous
        return -scores
