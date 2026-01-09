"""Ensemble methods for anomaly detection."""

from typing import Callable, List, Optional, Union

import numpy as np
import pandas as pd

from .base import BaseDetector


class VotingEnsemble(BaseDetector):
    """
    Voting ensemble that combines predictions from multiple detectors.

    An anomaly is flagged if at least `voting_threshold` detectors agree.

    Parameters
    ----------
    detectors : list of BaseDetector
        List of anomaly detectors to ensemble.
    voting_threshold : int, default=2
        Minimum number of detectors that must flag a sample as anomalous.
    random_state : int, optional
        Random state for reproducibility.
    """

    def __init__(
        self,
        detectors: List[BaseDetector],
        voting_threshold: int = 2,
        random_state: Optional[int] = None,
    ):
        super().__init__(random_state)
        self.detectors = detectors
        self.voting_threshold = voting_threshold

    def fit(self, X: Union[np.ndarray, pd.DataFrame, pd.Series]):
        """
        Fit all detectors in the ensemble.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        """
        for detector in self.detectors:
            detector.fit(X)

    def predict(self, X: Union[np.ndarray, pd.DataFrame, pd.Series]) -> np.ndarray:
        """
        Predict anomalies using voting.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Data to predict.

        Returns
        -------
        predictions : ndarray of shape (n_samples,)
            Anomaly predictions. -1 for anomalies, 1 for normal.
        """
        # Get predictions from all detectors - vectorized
        all_predictions = np.array([detector.predict(X) for detector in self.detectors])

        # Count votes (how many detectors flagged as anomaly) - vectorized
        votes = np.sum(all_predictions == -1, axis=0)

        # Flag as anomaly if voting_threshold or more detectors agree
        ensemble_predictions = np.where(votes >= self.voting_threshold, -1, 1)
        return ensemble_predictions

    def score_samples(self, X: Union[np.ndarray, pd.DataFrame, pd.Series]) -> np.ndarray:
        """
        Compute ensemble scores as mean of individual detector scores.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Data to score.

        Returns
        -------
        scores : ndarray of shape (n_samples,)
            Average anomaly scores across all detectors.
        """
        # Get scores from all detectors - vectorized
        all_scores = np.array([detector.score_samples(X) for detector in self.detectors])

        # Average scores - vectorized
        ensemble_scores = np.mean(all_scores, axis=0)
        return ensemble_scores

    def get_vote_counts(self, X: Union[np.ndarray, pd.DataFrame, pd.Series]) -> np.ndarray:
        """
        Get vote counts for each sample.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Data to analyze.

        Returns
        -------
        vote_counts : ndarray of shape (n_samples,)
            Number of detectors that flagged each sample as anomalous.
        """
        # Get predictions from all detectors - vectorized
        all_predictions = np.array([detector.predict(X) for detector in self.detectors])
        vote_counts = np.sum(all_predictions == -1, axis=0)
        return vote_counts


class EnsembleDetector(BaseDetector):
    """
    General ensemble detector with customizable combination method.

    Parameters
    ----------
    detectors : list of BaseDetector
        List of anomaly detectors to ensemble.
    combination_method : str or callable, default='mean'
        Method to combine scores:
        - 'mean': Average scores
        - 'max': Maximum score
        - 'min': Minimum score
        - 'median': Median score
        - callable: Custom function taking array of scores and returning combined score
    voting_threshold : int, optional
        If provided, uses voting on predictions in addition to score combination.
    random_state : int, optional
        Random state for reproducibility.
    """

    def __init__(
        self,
        detectors: List[BaseDetector],
        combination_method: Union[str, Callable] = "mean",
        voting_threshold: Optional[int] = None,
        random_state: Optional[int] = None,
    ):
        super().__init__(random_state)
        self.detectors = detectors
        self.combination_method = combination_method
        self.voting_threshold = voting_threshold

        # Define combination functions
        self.combination_funcs = {
            "mean": np.mean,
            "max": np.max,
            "min": np.min,
            "median": np.median,
        }

    def fit(self, X: Union[np.ndarray, pd.DataFrame, pd.Series]):
        """
        Fit all detectors in the ensemble.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        """
        for detector in self.detectors:
            detector.fit(X)

    def predict(self, X: Union[np.ndarray, pd.DataFrame, pd.Series]) -> np.ndarray:
        """
        Predict anomalies.

        If voting_threshold is set, uses voting. Otherwise, uses thresholded scores.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Data to predict.

        Returns
        -------
        predictions : ndarray of shape (n_samples,)
            Anomaly predictions. -1 for anomalies, 1 for normal.
        """
        if self.voting_threshold is not None:
            # Use voting - vectorized
            all_predictions = np.array([detector.predict(X) for detector in self.detectors])
            votes = np.sum(all_predictions == -1, axis=0)
            predictions = np.where(votes >= self.voting_threshold, -1, 1)
        else:
            # Use thresholded scores
            scores = self.score_samples(X)
            threshold = np.percentile(scores, 95)  # Top 5% as anomalies
            predictions = np.where(scores > threshold, -1, 1)

        return predictions

    def score_samples(self, X: Union[np.ndarray, pd.DataFrame, pd.Series]) -> np.ndarray:
        """
        Compute ensemble scores using the combination method.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Data to score.

        Returns
        -------
        scores : ndarray of shape (n_samples,)
            Combined anomaly scores.
        """
        # Get scores from all detectors - vectorized
        all_scores = np.array([detector.score_samples(X) for detector in self.detectors])

        # Combine scores
        if isinstance(self.combination_method, str):
            if self.combination_method not in self.combination_funcs:
                raise ValueError(
                    f"Unknown combination method: {self.combination_method}. "
                    f"Available: {list(self.combination_funcs.keys())}"
                )
            combined_scores = self.combination_funcs[self.combination_method](all_scores, axis=0)
        else:
            # Custom function
            combined_scores = np.apply_along_axis(self.combination_method, axis=0, arr=all_scores)

        return combined_scores
