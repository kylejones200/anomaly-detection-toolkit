"""Anomaly Detection Toolkit.

A comprehensive Python library for detecting anomalies in time series and multivariate data.
Supports multiple detection methods including statistical, machine learning, and deep learning
approaches.
"""

__version__ = "0.1.1"
__author__ = "Kyle Jones"
__email__ = "kyletjones@gmail.com"

from .autoencoders import LSTMAutoencoderDetector, PyTorchAutoencoderDetector
from .base import BaseDetector
from .ensemble import EnsembleDetector, VotingEnsemble
from .ml_methods import IsolationForestDetector, LOFDetector, RobustCovarianceDetector
from .statistical import (
    IQROutlierDetector,
    SeasonalBaselineDetector,
    StatisticalDetector,
    ZScoreDetector,
)
from .wavelet import WaveletDenoiser, WaveletDetector

__all__ = [
    # Base class
    "BaseDetector",
    # Statistical methods
    "StatisticalDetector",
    "ZScoreDetector",
    "IQROutlierDetector",
    "SeasonalBaselineDetector",
    # ML methods
    "IsolationForestDetector",
    "LOFDetector",
    "RobustCovarianceDetector",
    # Wavelet methods
    "WaveletDetector",
    "WaveletDenoiser",
    # Autoencoder methods
    "LSTMAutoencoderDetector",
    "PyTorchAutoencoderDetector",
    # Ensemble methods
    "EnsembleDetector",
    "VotingEnsemble",
]
