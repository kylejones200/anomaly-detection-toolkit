"""Base classes for anomaly detectors."""

from abc import ABC, abstractmethod
from typing import Optional, Tuple, Union

import numpy as np
import pandas as pd


class BaseDetector(ABC):
    """Base class for all anomaly detectors."""

    def __init__(self, random_state: Optional[int] = None):
        """
        Initialize the detector.

        Parameters
        ----------
        random_state : int, optional
            Random state for reproducibility.
        """
        self.random_state = random_state
        if random_state is not None:
            np.random.seed(random_state)

    @abstractmethod
    def fit(self, X: Union[np.ndarray, pd.DataFrame, pd.Series]):
        """
        Fit the detector on the data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        """
        pass

    @abstractmethod
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
        pass

    @abstractmethod
    def score_samples(self, X: Union[np.ndarray, pd.DataFrame, pd.Series]) -> np.ndarray:
        """
        Compute anomaly scores for samples.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Data to score.

        Returns
        -------
        scores : ndarray of shape (n_samples,)
            Anomaly scores. Higher values indicate more anomalous samples.
        """
        pass

    def fit_predict(
        self, X: Union[np.ndarray, pd.DataFrame, pd.Series]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Fit the detector and predict anomalies.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Data to fit and predict.

        Returns
        -------
        predictions : ndarray of shape (n_samples,)
            Anomaly predictions. -1 for anomalies, 1 for normal.
        scores : ndarray of shape (n_samples,)
            Anomaly scores.
        """
        self.fit(X)
        predictions = self.predict(X)
        scores = self.score_samples(X)
        return predictions, scores

    def _validate_input(self, X: Union[np.ndarray, pd.DataFrame, pd.Series]) -> np.ndarray:
        """Convert input to numpy array."""
        if isinstance(X, pd.Series):
            X = X.values.reshape(-1, 1)
        elif isinstance(X, pd.DataFrame):
            X = X.values
        elif isinstance(X, np.ndarray):
            if X.ndim == 1:
                X = X.reshape(-1, 1)
        else:
            # Try to convert to numpy array
            X = np.asarray(X)
            if X.ndim == 1:
                X = X.reshape(-1, 1)
        return X
