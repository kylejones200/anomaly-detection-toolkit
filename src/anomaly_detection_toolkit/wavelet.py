"""Wavelet-based anomaly detection methods."""

from typing import Optional, Tuple, Union

import numpy as np
import pandas as pd
import pywt

from .base import BaseDetector


class WaveletDenoiser:
    """
    Wavelet-based signal denoising.

    Parameters
    ----------
    wavelet : str, default='db4'
        Wavelet type (e.g., 'db4', 'haar', 'bior2.2').
    threshold_mode : str, default='soft'
        Thresholding mode: 'soft' or 'hard'.
    level : int, default=5
        Decomposition level.
    """

    def __init__(self, wavelet: str = "db4", threshold_mode: str = "soft", level: int = 5):
        self.wavelet = wavelet
        self.threshold_mode = threshold_mode
        self.level = level

    def denoise(self, data: np.ndarray) -> np.ndarray:
        """
        Denoise signal using wavelet thresholding.

        Parameters
        ----------
        data : array-like
            Input signal.

        Returns
        -------
        denoised : ndarray
            Denoised signal.
        """
        coeffs = pywt.wavedec(data, self.wavelet, level=self.level)

        # Calculate threshold using universal threshold (Donoho & Johnstone)
        sigma = np.median(np.abs(coeffs[-1])) / 0.6745  # Median absolute deviation
        threshold = sigma * np.sqrt(2 * np.log(len(data)))

        # Apply threshold to detail coefficients (keep approximation)
        coeffs_thresh = [coeffs[0]]  # Keep approximation
        for c in coeffs[1:]:  # Threshold detail coefficients
            coeffs_thresh.append(pywt.threshold(c, threshold, mode=self.threshold_mode))

        # Reconstruct
        denoised = pywt.waverec(coeffs_thresh, self.wavelet)
        return denoised[: len(data)]  # Trim to original length


class WaveletDetector(BaseDetector):
    """
    Wavelet-based anomaly detector for time series.

    Detects anomalies by identifying large coefficients in wavelet
    detail levels, which indicate sudden changes or anomalies.

    Parameters
    ----------
    wavelet : str, default='db4'
        Wavelet type (e.g., 'db4', 'haar', 'bior2.2').
    threshold_factor : float, default=3.0
        Threshold factor for anomaly detection (in terms of MAD).
    level : int, default=5
        Decomposition level.
    random_state : int, optional
        Random state for reproducibility.
    """

    def __init__(
        self,
        wavelet: str = "db4",
        threshold_factor: float = 3.0,
        level: int = 5,
        random_state: Optional[int] = None,
    ):
        super().__init__(random_state)
        self.wavelet = wavelet
        self.threshold_factor = threshold_factor
        self.level = level
        self.anomaly_scores_ = None

    def fit(self, X: Union[np.ndarray, pd.DataFrame, pd.Series]):
        """
        Fit the wavelet detector.

        Parameters
        ----------
        X : array-like of shape (n_samples,)
            Training time series data.
        """
        X = self._validate_input(X)
        if X.ndim > 1 and X.shape[1] > 1:
            raise ValueError("WaveletDetector only supports univariate time series.")

        # Flatten to 1D
        if X.ndim > 1:
            X = X.flatten()

        # Store data for scoring
        self.data_ = X

    def predict(self, X: Union[np.ndarray, pd.DataFrame, pd.Series]) -> np.ndarray:
        """
        Predict anomalies in the data.

        Parameters
        ----------
        X : array-like of shape (n_samples,)
            Time series data to predict.

        Returns
        -------
        predictions : ndarray of shape (n_samples,)
            Anomaly predictions. -1 for anomalies, 1 for normal.
        """
        scores = self.score_samples(X)
        # Threshold based on percentile
        threshold = np.percentile(scores[scores > 0], 95) if np.any(scores > 0) else 0
        predictions = np.where(scores > threshold, -1, 1)
        return predictions

    def score_samples(self, X: Union[np.ndarray, pd.DataFrame, pd.Series]) -> np.ndarray:
        """
        Compute anomaly scores using wavelet decomposition.

        Parameters
        ----------
        X : array-like of shape (n_samples,)
            Time series data to score.

        Returns
        -------
        scores : ndarray of shape (n_samples,)
            Anomaly scores. Higher values indicate more anomalous samples.
        """
        X = self._validate_input(X)
        if X.ndim > 1 and X.shape[1] > 1:
            raise ValueError("WaveletDetector only supports univariate time series.")

        # Flatten to 1D
        if X.ndim > 1:
            X = X.flatten()

        # Perform wavelet decomposition
        coeffs = pywt.wavedec(X, self.wavelet, level=self.level)

        # Focus on detail coefficients (high-frequency anomalies)
        detail_coeffs = coeffs[1:]

        # Calculate threshold for each detail level
        anomaly_scores = np.zeros(len(X))

        for detail in detail_coeffs:
            if len(detail) == 0:
                continue

            # Use robust statistics (median, MAD) - vectorized
            detail_abs = np.abs(detail)
            median_detail = np.median(detail_abs)
            mad = np.median(np.abs(detail_abs - median_detail))
            threshold = median_detail + self.threshold_factor * (mad / 0.6745)

            # Find anomalies in this detail level - vectorized
            anomaly_mask = detail_abs > threshold

            if not np.any(anomaly_mask):
                continue

            # Map back to original time indices - vectorized
            scale_factor = len(X) // len(detail)
            anomaly_indices = np.where(anomaly_mask)[0]

            # Vectorized mapping using broadcasting
            start_indices = anomaly_indices * scale_factor
            end_indices = np.minimum((anomaly_indices + 1) * scale_factor, len(X))

            # Add scores efficiently using vectorized operations
            for start_idx, end_idx, score in zip(
                start_indices, end_indices, detail_abs[anomaly_mask]
            ):
                anomaly_scores[start_idx:end_idx] += score

        self.anomaly_scores_ = anomaly_scores
        return anomaly_scores

    def get_wavelet_coefficients(
        self, X: Union[np.ndarray, pd.DataFrame, pd.Series]
    ) -> Tuple[np.ndarray, list]:
        """
        Get wavelet decomposition coefficients.

        Parameters
        ----------
        X : array-like of shape (n_samples,)
            Time series data.

        Returns
        -------
        approximation : ndarray
            Approximation coefficients (low-frequency trend).
        details : list of ndarray
            Detail coefficients for each level (high-frequency components).
        """
        X = self._validate_input(X)
        if X.ndim > 1:
            X = X.flatten()

        coeffs = pywt.wavedec(X, self.wavelet, level=self.level)
        approximation = coeffs[0]
        details = coeffs[1:]

        return approximation, details

    def get_continuous_wavelet_transform(
        self,
        X: Union[np.ndarray, pd.DataFrame, pd.Series],
        scales: Optional[np.ndarray] = None,
        wavelet: str = "morl",
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute Continuous Wavelet Transform (CWT).

        Parameters
        ----------
        X : array-like of shape (n_samples,)
            Time series data.
        scales : array-like, optional
            Scales for CWT. Defaults to np.arange(1, 65).
        wavelet : str, default='morl'
            Wavelet for CWT (commonly 'morl' for Morlet).

        Returns
        -------
        coefficients : ndarray of shape (n_scales, n_samples)
            CWT coefficients.
        frequencies : ndarray of shape (n_scales,)
            Corresponding frequencies.
        """
        X = self._validate_input(X)
        if X.ndim > 1:
            X = X.flatten()

        if scales is None:
            scales = np.arange(1, 65)

        coefficients, frequencies = pywt.cwt(X, scales, wavelet, 1.0)
        return coefficients, frequencies
