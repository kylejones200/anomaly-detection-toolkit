"""PCA-based anomaly detection for turbomachinery and rotating equipment."""

from typing import Optional, Union

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from .base import BaseDetector


class PCADetector(BaseDetector):
    """
    PCA-based anomaly detector.

    Uses Principal Component Analysis to model healthy operation boundaries.
    Anomalies are detected using either:
    - Mahalanobis distance in the principal component space
    - Reconstruction error (difference between original and reconstructed data)

    This detector is particularly useful for turbomachinery and rotating equipment
    where sensor drift and early failure detection are critical.

    Parameters
    ----------
    n_components : float or int, default=0.95
        Number of components to keep. If 0 < n_components < 1, select the number
        of components such that the amount of variance that needs to be explained
        is greater than the percentage specified by n_components.
    score_method : str, default='reconstruction'
        Method for computing anomaly scores:
        - 'reconstruction': Use reconstruction error
        - 'mahalanobis': Use Mahalanobis distance in PC space
        - 'both': Use both and return average
    contamination : float, default=0.05
        Expected proportion of outliers in the data (used for threshold).
    random_state : int, optional
        Random state for reproducibility.
    """

    def __init__(
        self,
        n_components: Union[float, int] = 0.95,
        score_method: str = "reconstruction",
        contamination: float = 0.05,
        random_state: Optional[int] = None,
    ):
        super().__init__(random_state)
        self.n_components = n_components
        self.score_method = score_method
        self.contamination = contamination
        self.pca_ = None
        self.scaler_ = StandardScaler()
        self.threshold_ = None
        self.mean_ = None
        self.cov_ = None

    def fit(self, X: Union[np.ndarray, pd.DataFrame, pd.Series]):
        """
        Fit the PCA detector on healthy operation data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data (healthy operation data).
        """
        X = self._validate_input(X)
        X_scaled = self.scaler_.fit_transform(X)

        # Fit PCA
        self.pca_ = PCA(n_components=self.n_components, random_state=self.random_state)
        X_pca = self.pca_.fit_transform(X_scaled)

        # Compute statistics in PC space for Mahalanobis distance
        self.mean_ = np.mean(X_pca, axis=0)
        self.cov_ = np.cov(X_pca.T)

        # Compute threshold based on training data
        scores = self._compute_scores(X_scaled, X_pca)
        self.threshold_ = np.percentile(scores, 100 * (1 - self.contamination))

    def predict(self, X: Union[np.ndarray, pd.DataFrame, pd.Series]) -> np.ndarray:
        """
        Predict anomalies.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Data to predict.

        Returns
        -------
        predictions : ndarray of shape (n_samples,)
            Anomaly predictions. -1 for anomalies, 1 for normal.
        """
        scores = self.score_samples(X)
        predictions = np.where(scores > self.threshold_, -1, 1)
        return predictions

    def score_samples(self, X: Union[np.ndarray, pd.DataFrame, pd.Series]) -> np.ndarray:
        """
        Compute anomaly scores.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Data to score.

        Returns
        -------
        scores : ndarray of shape (n_samples,)
            Anomaly scores. Higher values indicate more anomalous samples.
        """
        if self.pca_ is None:
            raise ValueError("Detector must be fitted before scoring.")

        X = self._validate_input(X)
        X_scaled = self.scaler_.transform(X)
        X_pca = self.pca_.transform(X_scaled)

        return self._compute_scores(X_scaled, X_pca)

    def _compute_scores(self, X_scaled: np.ndarray, X_pca: np.ndarray) -> np.ndarray:
        """
        Compute anomaly scores using specified method.

        Parameters
        ----------
        X_scaled : ndarray
            Scaled input data.
        X_pca : ndarray
            Data transformed to principal component space.

        Returns
        -------
        scores : ndarray
            Anomaly scores.
        """
        if self.score_method == "reconstruction":
            # Reconstruction error
            X_reconstructed = self.pca_.inverse_transform(X_pca)
            reconstruction_error = np.sum((X_scaled - X_reconstructed) ** 2, axis=1)
            return reconstruction_error

        elif self.score_method == "mahalanobis":
            # Mahalanobis distance in PC space
            if self.mean_ is None or self.cov_ is None:
                raise ValueError("PCA must be fitted before computing Mahalanobis distance.")

            # Compute Mahalanobis distance
            diff = X_pca - self.mean_
            try:
                inv_cov = np.linalg.inv(self.cov_)
                mahalanobis_dist = np.sqrt(np.sum(diff @ inv_cov * diff, axis=1))
            except np.linalg.LinAlgError:
                # If covariance is singular, use pseudo-inverse
                inv_cov = np.linalg.pinv(self.cov_)
                mahalanobis_dist = np.sqrt(np.sum(diff @ inv_cov * diff, axis=1))
            return mahalanobis_dist

        elif self.score_method == "both":
            # Average of both methods
            recon_scores = self._compute_scores_with_method(X_scaled, X_pca, "reconstruction")
            maha_scores = self._compute_scores_with_method(X_scaled, X_pca, "mahalanobis")
            # Normalize and average
            recon_norm = (recon_scores - recon_scores.min()) / (
                recon_scores.max() - recon_scores.min() + 1e-10
            )
            maha_norm = (maha_scores - maha_scores.min()) / (
                maha_scores.max() - maha_scores.min() + 1e-10
            )
            return (recon_norm + maha_norm) / 2

        else:
            raise ValueError(f"Unknown score_method: {self.score_method}")

    def _compute_scores_with_method(
        self, X_scaled: np.ndarray, X_pca: np.ndarray, method: str
    ) -> np.ndarray:
        """Compute scores with a specific method."""
        if method == "reconstruction":
            X_reconstructed = self.pca_.inverse_transform(X_pca)
            return np.sum((X_scaled - X_reconstructed) ** 2, axis=1)
        elif method == "mahalanobis":
            diff = X_pca - self.mean_
            try:
                inv_cov = np.linalg.inv(self.cov_)
                return np.sqrt(np.sum(diff @ inv_cov * diff, axis=1))
            except np.linalg.LinAlgError:
                inv_cov = np.linalg.pinv(self.cov_)
                return np.sqrt(np.sum(diff @ inv_cov * diff, axis=1))
        else:
            raise ValueError(f"Unknown method: {method}")

    def _validate_input(self, X: Union[np.ndarray, pd.DataFrame, pd.Series]) -> np.ndarray:
        """Validate and convert input to numpy array."""
        if isinstance(X, pd.Series):
            X = X.values.reshape(-1, 1)
        elif isinstance(X, pd.DataFrame):
            X = X.values
        elif isinstance(X, np.ndarray):
            if X.ndim == 1:
                X = X.reshape(-1, 1)
        else:
            X = np.asarray(X)
            if X.ndim == 1:
                X = X.reshape(-1, 1)
        return X

    def get_explained_variance_ratio(self) -> np.ndarray:
        """
        Get explained variance ratio for each principal component.

        Returns
        -------
        explained_variance_ratio : ndarray
            Explained variance ratio for each component.
        """
        if self.pca_ is None:
            raise ValueError("PCA must be fitted first.")
        return self.pca_.explained_variance_ratio_

    def get_n_components(self) -> int:
        """Get the actual number of components used."""
        if self.pca_ is None:
            raise ValueError("PCA must be fitted first.")
        return self.pca_.n_components_

    def transform(self, X: Union[np.ndarray, pd.DataFrame, pd.Series]) -> np.ndarray:
        """
        Transform data to principal component space.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Data to transform.

        Returns
        -------
        X_pca : ndarray of shape (n_samples, n_components)
            Data in principal component space.
        """
        if self.pca_ is None:
            raise ValueError("PCA must be fitted first.")

        X = self._validate_input(X)
        X_scaled = self.scaler_.transform(X)
        return self.pca_.transform(X_scaled)
